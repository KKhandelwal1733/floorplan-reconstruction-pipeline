"""Phase 6: quality gate tests.

A sparse/low-density scan should get its CIs widened (quality_score < 1.0)
rather than silently reporting a confident number for incomplete coverage.
"""
import numpy as np

from roomscan.config import QUALITY_CI_WIDEN_FACTOR
from roomscan.geometry.room_layout import extract_layout


def _flat_room_cloud(n: int, room: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, room, (n, 3)).astype(np.float32)
    pts[:, 2] = rng.normal(0, 0.005, n).astype(np.float32)
    return pts


def test_dense_scan_full_quality():
    pts = _flat_room_cloud(30_000)
    layout = extract_layout(pts)
    assert layout.quality_score == 1.0


def test_sparse_scan_widened_ci():
    dense = extract_layout(_flat_room_cloud(30_000))
    sparse = extract_layout(_flat_room_cloud(500))

    assert sparse.quality_score < 1.0
    assert any("quality" not in w and ("low floor point count" in w or "low point density" in w)
               for w in sparse.capture_warnings)

    dense_hw = dense.floor_area_m2.hi - dense.floor_area_m2.value
    sparse_hw = sparse.floor_area_m2.hi - sparse.floor_area_m2.value
    # Same relative base CI (±3%), so the widen factor should show up directly
    # in the ratio between the two half-widths (allowing for area differing).
    base_frac = 0.03
    expected_sparse_hw = sparse.floor_area_m2.value * base_frac * QUALITY_CI_WIDEN_FACTOR
    assert abs(sparse_hw - expected_sparse_hw) < 1e-6
    assert sparse_hw > dense_hw * 0.5  # sanity: materially wider per unit area


def test_quality_score_bounded():
    pts = _flat_room_cloud(10)
    try:
        layout = extract_layout(pts)
    except ValueError:
        return  # too few points for RANSAC to find a floor at all - acceptable
    assert 0.0 <= layout.quality_score <= 1.0
