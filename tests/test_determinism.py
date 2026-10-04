"""Phase 6: determinism tests.

Guards against the RNG-sharing bug previously found in planes.py (a module-
level RNG singleton made results depend on prior calls in the same process).
Every call must now seed fresh from config.SEED, so two extract_layout()
calls on the identical point cloud must agree exactly, not just approximately.
"""
import numpy as np

from roomscan.geometry.room_layout import extract_layout


def _room_cloud() -> np.ndarray:
    rng = np.random.default_rng(3)
    floor = rng.uniform(-3, 3, (8000, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, 8000).astype(np.float32)
    wall = rng.uniform(-3, 3, (2000, 3)).astype(np.float32)
    wall[:, 0] = 3.0
    wall[:, 2] = rng.uniform(0, 2.5, 2000).astype(np.float32)
    return np.concatenate([floor, wall])


def test_extract_layout_exactly_repeatable():
    pts = _room_cloud()
    a = extract_layout(pts)
    b = extract_layout(pts)

    assert a.floor_area_m2.value == b.floor_area_m2.value
    assert a.floor_area_m2.lo == b.floor_area_m2.lo
    assert a.floor_area_m2.hi == b.floor_area_m2.hi
    assert a.floor_d == b.floor_d
    assert a.ceiling_unobserved == b.ceiling_unobserved
    assert len(a.walls) == len(b.walls)
    for wa, wb in zip(a.walls, b.walls):
        assert wa.length_m.value == wb.length_m.value
        assert wa.p0 == wb.p0
        assert wa.p1 == wb.p1


def test_extract_layout_repeatable_interleaved_with_other_calls():
    """Calling unrelated extract_layout()s in between must not change the result
    (regression guard for the old module-level shared-RNG bug)."""
    pts = _room_cloud()
    first = extract_layout(pts)

    # Burn through RNG draws via unrelated calls, as a different test file
    # running earlier in the same process would have done under the old bug.
    for _ in range(5):
        extract_layout(_room_cloud())

    second = extract_layout(pts)
    assert first.floor_area_m2.value == second.floor_area_m2.value
    assert first.floor_d == second.floor_d


def test_load_scan_repeatable(tmp_path):
    from tests.fixtures.make_fixtures import make_floor_only
    from roomscan.io.stray_scanner import load_scan

    make_floor_only(tmp_path)
    scan_dir = tmp_path / "single_scan_floor_only"
    pts1 = load_scan(scan_dir)
    pts2 = load_scan(scan_dir)
    assert np.array_equal(pts1, pts2)
