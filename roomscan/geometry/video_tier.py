"""Video tier pipeline (Phase 7): single handheld .mp4/.mov -> RoomLayout.

Ties together frame sampling, monocular SfM, and scale-ensemble metric
recovery. Every length/area in the resulting layout gets its CI widened by
the scale ensemble's relative spread, since a scale error affects all of
them multiplicatively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from roomscan.config import VIDEO_EMPIRICAL_MIN_REL_HW
from roomscan.geometry.room_layout import RoomLayout, extract_layout
from roomscan.geometry.scale_ensemble import (
    apply_scale,
    combine_estimates,
    estimate_scale_ensemble,
)
from roomscan.geometry.sfm import reconstruct_sparse
from roomscan.io.video_loader import sample_frames
from roomscan.schema_out import Measurement


def _widen_rel(m: Measurement | None, rel_hw: float) -> Measurement | None:
    """Widen a Measurement's CI by an additional relative half-width."""
    if m is None:
        return None
    extra = abs(m.value) * rel_hw
    return Measurement(value=m.value, lo=m.lo - extra, hi=m.hi + extra,
                        confidence_level=m.confidence_level)


def process_video(video_path: Path) -> tuple[RoomLayout, dict[str, Any]]:
    """Reconstruct a RoomLayout from a single handheld video.

    Returns (layout, diagnostics). Raises ValueError if reconstruction fails
    at any stage (too few frames, too few matched points, no scale estimate)
    -- callers should catch this per the "never crash, abstain instead" rule
    and degrade to a stub/warning rather than let it propagate.
    """
    frames = sample_frames(video_path)
    if len(frames) < 3:
        raise ValueError(f"only {len(frames)} usable frame(s) sampled from {video_path}")

    pts_unscaled, poses = reconstruct_sparse(frames)
    if len(pts_unscaled) < 50:
        raise ValueError(f"SfM reconstruction produced only {len(pts_unscaled)} points")

    estimates = estimate_scale_ensemble(pts_unscaled, poses)
    if not estimates:
        raise ValueError("no scale-ensemble estimate could be computed")
    scale, ensemble_rel_hw = combine_estimates(estimates)
    # Floor the CI at the empirically observed error rate (see config comment)
    # rather than trust the ensemble's own spread when it happens to be narrow.
    scale_rel_hw = max(ensemble_rel_hw, VIDEO_EMPIRICAL_MIN_REL_HW)

    pts_scaled = apply_scale(pts_unscaled, scale)
    layout = extract_layout(pts_scaled)

    layout.floor_area_m2 = _widen_rel(layout.floor_area_m2, scale_rel_hw)
    layout.ceiling_height_m = _widen_rel(layout.ceiling_height_m, scale_rel_hw)
    for wall in layout.walls:
        wall.length_m = _widen_rel(wall.length_m, scale_rel_hw)

    layout.capture_warnings.append(
        f"video tier: scale recovered via {len(estimates)}-method ensemble "
        f"(factor={scale:.3f}, +/-{scale_rel_hw:.1%} relative spread); "
        "monocular reconstruction, no absolute depth"
    )

    diagnostics = {
        "scale_factor": scale,
        "scale_rel_half_width": scale_rel_hw,
        "ensemble_rel_half_width": ensemble_rel_hw,
        "scale_estimates": [(e.source, e.factor) for e in estimates],
        "n_frames_sampled": len(frames),
        "n_points_unscaled": len(pts_unscaled),
    }
    return layout, diagnostics
