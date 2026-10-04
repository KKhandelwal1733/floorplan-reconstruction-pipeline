"""Smoke-test the loader against the real single_room Stray Scanner capture.

Uses max_frames=50 to keep the test fast (~1 s); the full 1715-frame fusion
is exercised by the CLI / manual runs.
"""
import numpy as np
import pytest
from pathlib import Path

from roomscan.io.stray_scanner import load_scan, fuse_scan

SINGLE_ROOM = Path(__file__).parent / "fixtures" / "single_room"


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="real scan fixture not present")
def test_real_scan_loads():
    pts = load_scan(SINGLE_ROOM, max_frames=50)
    assert pts.shape[1] == 3
    assert len(pts) > 1000, f"too few points: {len(pts)}"


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="real scan fixture not present")
def test_real_scan_ply(tmp_path):
    out = tmp_path / "scan.ply"
    fuse_scan(SINGLE_ROOM, out, max_frames=50)
    assert out.exists() and out.stat().st_size > 0
    header = out.read_text(errors="replace").split("\n")[:8]
    assert any("element vertex" in l for l in header)
