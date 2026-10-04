"""Phase 9: real-data calibration integration test.

With only a handful of real fixtures, both tiers are expected to report
"not achievable" at the project's standard 90% target coverage (needs >= 9
calibration points; see roomscan/calibration/conformal.py). This test
checks the real pipeline runs end to end and reports that honestly, not
that calibration magically succeeds with too little data.
"""
from pathlib import Path

import pytest

from bench.calibrate import gather_calibration_points, run_calibration

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_ROOM = FIXTURES / "single_room"


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="single_room fixture not present")
def test_gather_calibration_points_from_real_fixture():
    points = gather_calibration_points(fixtures=[SINGLE_ROOM])
    assert len(points) > 0
    assert all(p.tier in ("video", "photo") for p in points)
    assert all(p.half_width > 0 for p in points)


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="single_room fixture not present")
def test_run_calibration_reports_honestly_with_few_points():
    result = run_calibration(target_coverage=0.9)
    assert "video" in result["tiers"]
    assert "photo" in result["tiers"]
    for tier_result in result["tiers"].values():
        assert tier_result["min_points_needed_for_target"] == 9
        if tier_result["n_points"] < 9:
            assert not tier_result["achievable"]
            assert tier_result["factor"] is None


def test_load_calibration_factor_missing_file_returns_none(tmp_path):
    from roomscan.calibration.factors import load_calibration_factor
    missing = tmp_path / "does_not_exist.json"
    assert load_calibration_factor("video", path=missing) is None


def test_load_calibration_factor_unachievable_returns_none(tmp_path):
    import json
    from roomscan.calibration.factors import load_calibration_factor

    path = tmp_path / "factors.json"
    path.write_text(json.dumps({
        "tiers": {"video": {"achievable": False, "factor": None}}
    }))
    assert load_calibration_factor("video", path=path) is None


def test_load_calibration_factor_achievable_returns_value(tmp_path):
    import json
    from roomscan.calibration.factors import load_calibration_factor

    path = tmp_path / "factors.json"
    path.write_text(json.dumps({
        "tiers": {"video": {"achievable": True, "factor": 1.75}}
    }))
    assert load_calibration_factor("video", path=path) == pytest.approx(1.75)
