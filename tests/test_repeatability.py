"""Phase 6: split-scan repeatability test.

Splits a real capture into two interleaved frame subsets (not a chronological
prefix/suffix split, which would see different parts of the room since
Stray Scanner frames are time-ordered) and checks that floor area and wall
perimeter agree within tolerance. This is a self-consistency check — it needs
no ground truth, just agreement between two independent measurements of the
same physical room.
"""
from pathlib import Path

import pytest

from bench.harness import PASS, _repeatability_gate

FIXTURES = Path(__file__).parent / "fixtures"
REAL_FLOOR_ONLY = FIXTURES / "real_floor_only"


@pytest.mark.skipif(not REAL_FLOOR_ONLY.exists(), reason="real_floor_only fixture not present")
def test_split_scan_repeatability_real_scan():
    # frame_stride=20 keeps this fast: each half loads ~1/40th of the capture's
    # frames, interleaved so both halves span the whole scan timeline.
    result = _repeatability_gate(REAL_FLOOR_ONLY, frame_stride=20)
    assert result.status == PASS, f"repeatability gate failed: {result.note} ({result.value})"
