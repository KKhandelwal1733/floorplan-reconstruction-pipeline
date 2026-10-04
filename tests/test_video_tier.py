"""Phase 7: video tier tests.

Monocular SfM without bundle adjustment (the lightweight approach chosen for
this phase) is not precision-accurate -- on the one real video tested here,
floor area came out ~8-11x smaller than the LiDAR-tier truth for the same
physical room, and the result changed meaningfully with small prior tweaks.
These tests check structural correctness (never crashes, returns sane types,
CI present and wide, warnings explain the method) rather than tight numeric
accuracy against ground truth, since that accuracy has not been demonstrated.

See config.py VIDEO_EMPIRICAL_MIN_REL_HW and project memory for the full
real-data finding.
"""
from pathlib import Path

import pytest

from roomscan.geometry.scale_ensemble import (
    ScaleEstimate,
    apply_scale,
    combine_estimates,
    estimate_scale_ensemble,
)
from roomscan.geometry.sfm import estimate_intrinsics
from roomscan.io.video_loader import sample_frames

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_ROOM_VIDEO = FIXTURES / "single_room" / "rgb.mp4"


# ---------------------------------------------------------------------------
# Unit tests (no real video needed)
# ---------------------------------------------------------------------------

def test_estimate_intrinsics_sane():
    K = estimate_intrinsics(1920, 1440)
    assert K[0, 0] > 0  # positive focal length
    assert K[0, 2] == pytest.approx(960.0)
    assert K[1, 2] == pytest.approx(720.0)


def test_combine_estimates_single_uses_default_hw():
    from roomscan.config import VIDEO_DEFAULT_SCALE_REL_HW
    median, rel_hw = combine_estimates([ScaleEstimate(0.5, "only_one")])
    assert median == 0.5
    assert rel_hw == VIDEO_DEFAULT_SCALE_REL_HW


def test_combine_estimates_multiple_spread():
    median, rel_hw = combine_estimates([
        ScaleEstimate(0.4, "a"), ScaleEstimate(0.6, "b"),
    ])
    assert median == pytest.approx(0.5)
    assert rel_hw > 0


def test_combine_estimates_empty_raises():
    with pytest.raises(ValueError):
        combine_estimates([])


def test_apply_scale():
    import numpy as np
    pts = np.array([[1.0, 2.0, 3.0]])
    scaled = apply_scale(pts, 2.0)
    assert np.allclose(scaled, [[2.0, 4.0, 6.0]])


def test_scale_ensemble_empty_for_tiny_cloud():
    import numpy as np
    pts = np.zeros((5, 3), dtype=np.float32)
    assert estimate_scale_ensemble(pts) == []


# ---------------------------------------------------------------------------
# Integration (real video, skipped when absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SINGLE_ROOM_VIDEO.exists(), reason="single_room/rgb.mp4 fixture not present")
def test_sample_frames_real_video():
    frames = sample_frames(SINGLE_ROOM_VIDEO, interval_s=1.0, max_frames=10)
    assert len(frames) > 0
    assert frames[0].ndim == 3  # BGR frame


@pytest.mark.skipif(not SINGLE_ROOM_VIDEO.exists(), reason="single_room/rgb.mp4 fixture not present")
def test_process_video_never_crashes_and_returns_sane_layout():
    from roomscan.geometry.video_tier import process_video

    layout, diagnostics = process_video(SINGLE_ROOM_VIDEO)

    assert layout.floor_area_m2.value > 0
    assert layout.floor_area_m2.hi > layout.floor_area_m2.lo
    assert len(layout.walls) > 0
    assert diagnostics["n_points_unscaled"] > 0
    assert diagnostics["scale_factor"] > 0
    assert any("video tier" in w for w in layout.capture_warnings)
    # CI must be meaningfully wide given this method's demonstrated inaccuracy
    assert diagnostics["scale_rel_half_width"] >= 0.5
