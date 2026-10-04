"""Video tier pipeline (Phase 7): single handheld .mp4/.mov -> RoomLayout.

Ties together frame sampling, monocular SfM, and scale-ensemble metric
recovery. Every length/area in the resulting layout gets its CI widened by
the scale ensemble's relative spread, since a scale error affects all of
them multiplicatively.

reconstruct_from_frames() is the tier-agnostic core (also reused by the
photo tier in Phase 8 and the drift ablation in bench/ablate.py); it just
needs a list of BGR images, regardless of where they came from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from roomscan.config import VIDEO_EMPIRICAL_MIN_REL_HW, VIDEO_MIN_RECONSTRUCTED_PTS
from roomscan.geometry.room_layout import RoomLayout, extract_layout
from roomscan.geometry.scale_ensemble import (
    apply_scale,
    combine_estimates,
    estimate_scale_ensemble,
)
from roomscan.geometry.sfm import reconstruct_best_pair, reconstruct_sparse
from roomscan.io.video_loader import sample_frames
from roomscan.schema_out import Measurement


def _widen_rel(m: Measurement | None, rel_hw: float) -> Measurement | None:
    """Widen a Measurement's CI by an additional relative half-width."""
    if m is None:
        return None
    extra = abs(m.value) * rel_hw
    return Measurement(value=m.value, lo=m.lo - extra, hi=m.hi + extra,
                        confidence_level=m.confidence_level)


def reconstruct_from_frames(
    frames: list[np.ndarray],
    min_rel_hw: float = VIDEO_EMPIRICAL_MIN_REL_HW,
    detector: str = "orb",
    strategy: str = "chained",
    min_points: int = VIDEO_MIN_RECONSTRUCTED_PTS,
    min_plane_inliers: int = 200,
) -> tuple[RoomLayout, dict[str, Any]]:
    """Reconstruct a RoomLayout from a list of BGR frames (any source).

    strategy="chained" (video: many narrow-baseline frames, consecutive-pair
    chaining, roughly-constant-step-length scale) or "best_pair" (photos: a
    handful of unordered stills, all-pairs search for the one pair with
    enough shared view -- see sfm.reconstruct_best_pair).

    Returns (layout, diagnostics). Raises ValueError if reconstruction fails
    at any stage -- callers should catch this per the "never crash, abstain
    instead" rule and degrade to a stub/warning rather than let it propagate.
    """
    if len(frames) < 2:
        raise ValueError(f"only {len(frames)} usable frame(s) given")

    if strategy == "best_pair":
        pts_unscaled, poses = reconstruct_best_pair(frames, detector=detector)
    else:
        pts_unscaled, poses = reconstruct_sparse(frames, detector=detector)
    if len(pts_unscaled) < min_points:
        raise ValueError(f"SfM reconstruction produced only {len(pts_unscaled)} points")

    estimates = estimate_scale_ensemble(pts_unscaled, poses, min_points=min_points)
    if not estimates:
        raise ValueError("no scale-ensemble estimate could be computed")
    scale, ensemble_rel_hw = combine_estimates(estimates)
    # Floor the CI at the empirically observed error rate (see config comment)
    # rather than trust the ensemble's own spread when it happens to be narrow.
    scale_rel_hw = max(ensemble_rel_hw, min_rel_hw)

    pts_scaled = apply_scale(pts_unscaled, scale)
    layout = extract_layout(pts_scaled, min_plane_inliers=min_plane_inliers)

    layout.floor_area_m2 = _widen_rel(layout.floor_area_m2, scale_rel_hw)
    layout.ceiling_height_m = _widen_rel(layout.ceiling_height_m, scale_rel_hw)
    for wall in layout.walls:
        wall.length_m = _widen_rel(wall.length_m, scale_rel_hw)

    layout.capture_warnings.append(
        f"scale recovered via {len(estimates)}-method ensemble "
        f"(factor={scale:.3f}, +/-{scale_rel_hw:.1%} relative spread); "
        "monocular reconstruction, no absolute depth"
    )

    diagnostics = {
        "scale_factor": scale,
        "scale_rel_half_width": scale_rel_hw,
        "ensemble_rel_half_width": ensemble_rel_hw,
        "scale_estimates": [(e.source, e.factor) for e in estimates],
        "n_frames": len(frames),
        "n_points_unscaled": len(pts_unscaled),
    }
    return layout, diagnostics


def process_video(
    video_path: Path,
    max_frames: int | None = None,
    interval_s: float | None = None,
) -> tuple[RoomLayout, dict[str, Any]]:
    """Reconstruct a RoomLayout from a single handheld video.

    max_frames/interval_s override the config defaults when given (used by
    the drift ablation in bench/ablate.py to vary chain length).
    """
    kwargs: dict[str, Any] = {}
    if max_frames is not None:
        kwargs["max_frames"] = max_frames
    if interval_s is not None:
        kwargs["interval_s"] = interval_s
    frames = sample_frames(video_path, **kwargs)
    if len(frames) < 3:
        raise ValueError(f"only {len(frames)} usable frame(s) sampled from {video_path}")

    layout, diagnostics = reconstruct_from_frames(frames)
    diagnostics["n_frames_sampled"] = diagnostics.pop("n_frames")
    layout.capture_warnings[-1] = "video tier: " + layout.capture_warnings[-1]
    return layout, diagnostics
