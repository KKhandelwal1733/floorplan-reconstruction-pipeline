"""Phase 3: room layout extraction tests.

Uses synthetic fixtures (fast) + real scan (skipped when absent).
"""
import numpy as np
import pytest
from pathlib import Path

from roomscan.geometry.planes import detect_gravity, ransac_horizontal_plane
from roomscan.geometry.room_layout import extract_layout
from roomscan.render.svg_plan import render_plan

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_ROOM = FIXTURES / "single_room"


# ---------------------------------------------------------------------------
# Unit: gravity detection on synthetic floor-only cloud
# ---------------------------------------------------------------------------

def _make_floor_cloud(up_axis: int = 2, n: int = 5000) -> np.ndarray:
    """Flat point cloud at height 0 along `up_axis`, with some noise."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
    pts[:, up_axis] = rng.normal(0, 0.01, n).astype(np.float32)
    return pts


@pytest.mark.parametrize("up_axis", [0, 1, 2])
def test_detect_gravity_canonical(up_axis):
    pts = _make_floor_cloud(up_axis)
    g = detect_gravity(pts)
    assert abs(float(g[up_axis])) > 0.9, f"gravity={g} not aligned with axis {up_axis}"


def test_ransac_finds_floor():
    pts = _make_floor_cloud(up_axis=2)
    gravity = np.array([0., 0., 1.])
    result = ransac_horizontal_plane(pts, gravity, thresh=0.05)
    assert result is not None
    n, d, mask = result
    assert mask.sum() > len(pts) * 0.8, "too few inliers for a clean flat cloud"


# ---------------------------------------------------------------------------
# Integration: extract_layout on synthetic cloud
# ---------------------------------------------------------------------------

def test_extract_layout_synthetic():
    rng = np.random.default_rng(0)
    # Floor at z=0 ± 5 mm
    floor = rng.uniform(-3, 3, (8000, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, 8000).astype(np.float32)
    # Wall at x=3, rising from floor (z 0 to 2.5 m) — realistic room geometry
    wall = rng.uniform(-3, 3, (2000, 3)).astype(np.float32)
    wall[:, 0] = 3.0
    wall[:, 2] = rng.uniform(0, 2.5, 2000).astype(np.float32)
    pts = np.concatenate([floor, wall])

    layout = extract_layout(pts)
    assert layout.floor_area_m2.value > 1.0
    assert len(layout.polygon) >= 4
    assert len(layout.walls) >= 3


def test_render_svg_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    floor = rng.uniform(-2, 2, (5000, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, 5000).astype(np.float32)
    # add some points above floor so cloud isn't 100% floor (gravity detection needs contrast)
    above = rng.uniform(-2, 2, (500, 3)).astype(np.float32)
    above[:, 2] = rng.uniform(0.5, 2.5, 500).astype(np.float32)
    pts = np.concatenate([floor, above])
    layout = extract_layout(pts)
    out = tmp_path / "plan.svg"
    render_plan(layout, out)
    content = out.read_text()
    assert "<polygon" in content
    assert "Floor area" in content


# ---------------------------------------------------------------------------
# Integration: real scan (skipped when fixture absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="real scan fixture not present")
def test_real_scan_layout():
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(SINGLE_ROOM, max_frames=100)
    layout = extract_layout(pts)
    assert layout.floor_area_m2.value > 0
    assert len(layout.polygon) >= 4


if __name__ == "__main__":
    test_detect_gravity_canonical(0)
    test_detect_gravity_canonical(1)
    test_detect_gravity_canonical(2)
    test_ransac_finds_floor()
    test_extract_layout_synthetic()
    print("All layout checks passed.")
