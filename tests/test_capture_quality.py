"""Capture-quality flag tests (case-study realignment): mirror/glass/
wet-look-surface/low-light detection. Synthetic frames only -- crude colour
heuristic, same spirit as tests/test_damage.py."""
import numpy as np

from roomscan.geometry.capture_quality import (
    apply_capture_quality_warnings,
    detect_capture_quality_issues,
)
from roomscan.geometry.room_layout import RoomLayout
from roomscan.schema_out import Measurement


def _flat_frame(bgr: tuple[int, int, int], size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _layout() -> RoomLayout:
    m = Measurement(value=10.0, lo=9.5, hi=10.5)
    return RoomLayout(up_axis=np.array([0.0, 0.0, 1.0]), floor_d=0.0, ceiling_d=None,
                       polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
                       floor_area_m2=m, ceiling_height_m=m)


def test_no_issues_for_normal_frame():
    # mid-brightness, saturated (colourful) frame -- neither dark nor glare
    frame = _flat_frame((60, 120, 180))
    assert detect_capture_quality_issues([frame]) == []


def test_low_light_flagged_for_dark_frame():
    frame = _flat_frame((5, 5, 5))
    issues = detect_capture_quality_issues([frame])
    assert any("low light" in w for w in issues)


def test_glare_flagged_for_bright_desaturated_frame():
    # near-white, low-saturation -- the specular-glare proxy
    frame = _flat_frame((250, 250, 250))
    issues = detect_capture_quality_issues([frame])
    assert any("mirror/glass/wet-look" in w for w in issues)


def test_empty_frames_gives_no_issues():
    assert detect_capture_quality_issues([]) == []


def test_apply_warnings_widens_ci_and_lowers_quality_score():
    layout = _layout()
    frame = _flat_frame((5, 5, 5))   # low light
    original_score = layout.quality_score
    original_area_hw = layout.floor_area_m2.hi - layout.floor_area_m2.value

    issues = apply_capture_quality_warnings(layout, [frame])

    assert issues
    assert layout.quality_score < original_score
    assert (layout.floor_area_m2.hi - layout.floor_area_m2.value) > original_area_hw
    assert all(w in layout.capture_warnings for w in issues)


def test_apply_warnings_noop_for_clean_frames():
    layout = _layout()
    frame = _flat_frame((60, 120, 180))
    issues = apply_capture_quality_warnings(layout, [frame])
    assert issues == []
    assert layout.quality_score == 1.0
    assert layout.capture_warnings == []


if __name__ == "__main__":
    test_no_issues_for_normal_frame()
    test_low_light_flagged_for_dark_frame()
    test_glare_flagged_for_bright_desaturated_frame()
    test_empty_frames_gives_no_issues()
    test_apply_warnings_widens_ci_and_lowers_quality_score()
    test_apply_warnings_noop_for_clean_frames()
    print("ok")
