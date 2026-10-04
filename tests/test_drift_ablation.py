"""Phase 8: drift ablation test.

Checks that bench/ablate.py's real-data comparison runs end to end and
reports a result for each requested chain length -- not that drift is
necessarily detected (the real run found it wasn't, in this one example;
see bench/ablate.py and project memory for that honest finding).
"""
from pathlib import Path

import pytest

from bench.ablate import run_drift_ablation

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
