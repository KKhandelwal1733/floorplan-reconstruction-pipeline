"""Phase 8 + case-study realignment: drift ablation tests.

Checks that bench/ablate.py's real-data comparison runs end to end and
reports a result for each requested chain length -- not that drift is
necessarily detected (the real run found it wasn't, in this one example;
see bench/ablate.py and project memory for that honest finding). Also
checks the synthetic pose-graph drift-correction on/off ablation.
"""
from pathlib import Path

import pytest

from bench.ablate import run_drift_ablation, run_pose_graph_drift_ablation

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_ROOM = FIXTURES / "single_room"


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="single_room fixture not present")
def test_drift_ablation_runs_and_reports_all_frame_counts():
    result = run_drift_ablation(SINGLE_ROOM, frame_counts=[5, 20])
    assert len(result["rows"]) == 2
    assert result["lidar_floor_area_m2"] > 0
    assert "drift_detected" in result
    assert isinstance(result["drift_detected"], bool)
    # each row either succeeded (has floor_area_rel_err) or recorded why not
    for row in result["rows"]:
        assert "floor_area_rel_err" in row or "error" in row


def test_pose_graph_drift_ablation_zero_noise_is_exact():
    """With no injected noise, naive and corrected placement must agree --
    there's nothing to correct, so refine=True shouldn't move anything."""
    result = run_pose_graph_drift_ablation(noise_stds_m=[0.0], n_trials=1)
    row = result["rows"][0]
    assert row["loop_closure_gap_off_m"] == pytest.approx(row["loop_closure_gap_on_m"], abs=1e-9)


def test_pose_graph_drift_ablation_correction_helps_as_noise_grows():
    result = run_pose_graph_drift_ablation(noise_stds_m=[0.05, 0.1, 0.2], n_trials=20)
    assert result["synthetic"] is True
    assert len(result["rows"]) == 3
    # naive (off) error should grow with injected noise
    offs = [r["loop_closure_gap_off_m"] for r in result["rows"]]
    assert offs == sorted(offs)
    # joint refinement (on) should beat naive placement at every noise level tested
    for row in result["rows"]:
        assert row["loop_closure_gap_on_m"] < row["loop_closure_gap_off_m"]
    assert result["drift_correction_helps"] is True
