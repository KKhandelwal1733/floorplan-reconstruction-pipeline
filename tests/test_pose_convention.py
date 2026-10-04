"""Pose-axis-convention tests.

Checks:
  1. Floor-only scan: fitted plane normal ≈ world-Z, residual < 2 cm
  2. Ceiling scan: floor and ceiling normals are parallel (dot > 0.99)
"""
import numpy as np
import pytest
from pathlib import Path

from roomscan.io.stray_scanner import load_scan

FIXTURES = Path(__file__).parent / "fixtures"


def _fit_plane_normal(pts: np.ndarray) -> np.ndarray:
    """PCA on centred points → unit normal of dominant plane (smallest eigenvalue)."""
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[-1]


def test_floor_only_normal_is_vertical():
    pts = load_scan(FIXTURES / "single_scan_floor_only")
    assert len(pts) > 0, "loader returned empty cloud"
    normal = _fit_plane_normal(pts)
    # floor normal must be (nearly) aligned with world Z
    assert abs(abs(normal[2]) - 1.0) < 0.01, f"floor normal z-component off: {normal}"


def test_floor_only_residual_under_2cm():
    pts = load_scan(FIXTURES / "single_scan_floor_only")
    residual = float(np.std(pts[:, 2]))
    assert residual < 0.02, f"floor residual {residual:.4f} m > 2 cm"


def test_ceiling_parallel_to_floor():
    pts = load_scan(FIXTURES / "single_scan_with_ceiling")
    z = pts[:, 2]
    mid = float(z.mean())
    floor_pts = pts[z < mid]
    ceil_pts = pts[z >= mid]
    assert len(floor_pts) > 10 and len(ceil_pts) > 10, "not enough points in each cluster"

    fn = _fit_plane_normal(floor_pts)
    cn = _fit_plane_normal(ceil_pts)
    dot = float(abs(np.dot(fn, cn)))
    assert dot > 0.99, f"floor/ceiling normals not parallel: dot={dot:.4f}, fn={fn}, cn={cn}"


# ponytail: self-check — run without pytest
if __name__ == "__main__":
    test_floor_only_normal_is_vertical()
    test_floor_only_residual_under_2cm()
    test_ceiling_parallel_to_floor()
    print("All pose-convention checks passed.")
