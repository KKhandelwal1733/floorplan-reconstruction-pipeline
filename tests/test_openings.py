"""Phase 4: openings detection tests.

Uses a synthetic room with a clear rectangular gap in one wall to verify that
the occupancy-grid detector finds it and classifies it correctly.
"""
import numpy as np
import pytest

from roomscan.geometry.room_layout import extract_layout
from roomscan.geometry.openings import detect_openings


def _make_room_with_door(
    room_w: float = 4.0,
    room_d: float = 5.0,
    ceiling_h: float = 2.6,
    door_start: float = 1.0,   # m along x-wall from corner
    door_w: float = 0.9,
    door_h: float = 2.1,
    n_wall: int = 3000,
    n_floor: int = 6000,
) -> np.ndarray:
    """Synthetic box room (Z-up) with a door opening on the +X wall."""
    rng = np.random.default_rng(42)
    pts: list[np.ndarray] = []

    # Floor at z=0
    fx = rng.uniform(0, room_w, n_floor)
    fy = rng.uniform(0, room_d, n_floor)
    pts.append(np.stack([fx, fy, np.zeros(n_floor)], axis=1))

    # Ceiling
    cx = rng.uniform(0, room_w, n_floor // 2)
    cy = rng.uniform(0, room_d, n_floor // 2)
    pts.append(np.stack([cx, cy, np.full(n_floor // 2, ceiling_h)], axis=1))

    # Walls — simple boxes, skip the door gap on the +X wall
    # +X wall (x = room_w, y ∈ [0, room_d], z ∈ [0, ceiling_h])
    wy = rng.uniform(0, room_d, n_wall)
    wz = rng.uniform(0, ceiling_h, n_wall)
    # keep only points outside the door gap
    in_door = (wy >= door_start) & (wy <= door_start + door_w) & (wz <= door_h)
    wall_pts = np.stack([np.full(n_wall, room_w), wy, wz], axis=1)
    pts.append(wall_pts[~in_door])

    # -X wall
    wy2 = rng.uniform(0, room_d, n_wall)
    wz2 = rng.uniform(0, ceiling_h, n_wall)
    pts.append(np.stack([np.zeros(n_wall), wy2, wz2], axis=1))

    # +Y and -Y walls
    for yval in [room_d, 0.0]:
        wx = rng.uniform(0, room_w, n_wall)
        wz3 = rng.uniform(0, ceiling_h, n_wall)
        pts.append(np.stack([wx, np.full(n_wall, yval), wz3], axis=1))

    all_pts = np.concatenate(pts).astype(np.float32)
    noise = rng.normal(0, 0.005, all_pts.shape).astype(np.float32)
    return all_pts + noise


def test_detect_door_in_synthetic_room():
    pts = _make_room_with_door(door_w=0.9, door_h=2.1)
    layout = extract_layout(pts)
    openings = detect_openings(pts, layout)

    doors = [o for o in openings if o.kind == "door"]
    assert len(doors) >= 1, f"expected ≥1 door, got {openings}"

    door = doors[0]
    # Width should be roughly 0.9 m (within 20 cm given cell resolution)
    assert abs(door.width_m.value - 0.9) < 0.20, f"door width {door.width_m.value:.2f} m off"
    # Sill should be near floor
    assert door.sill_height_m < 0.20, f"sill height {door.sill_height_m:.2f} m too high for a door"


def _make_room_with_window(
    room_w: float = 4.0,
    room_d: float = 5.0,
    ceiling_h: float = 2.6,
    win_start: float = 1.2,
    win_w: float = 1.2,
    win_sill: float = 0.9,
    win_h: float = 1.0,
    n_wall: int = 3000,
    n_floor: int = 6000,
) -> np.ndarray:
    rng = np.random.default_rng(7)
    pts: list[np.ndarray] = []

    fx = rng.uniform(0, room_w, n_floor)
    fy = rng.uniform(0, room_d, n_floor)
    pts.append(np.stack([fx, fy, np.zeros(n_floor)], axis=1))

    cx = rng.uniform(0, room_w, n_floor // 2)
    cy = rng.uniform(0, room_d, n_floor // 2)
    pts.append(np.stack([cx, cy, np.full(n_floor // 2, ceiling_h)], axis=1))

    wy = rng.uniform(0, room_d, n_wall)
    wz = rng.uniform(0, ceiling_h, n_wall)
    in_win = (
        (wy >= win_start) & (wy <= win_start + win_w)
        & (wz >= win_sill) & (wz <= win_sill + win_h)
    )
    wall_pts = np.stack([np.full(n_wall, room_w), wy, wz], axis=1)
    pts.append(wall_pts[~in_win])

    for yval in [room_d, 0.0]:
        wx = rng.uniform(0, room_w, n_wall)
        wz3 = rng.uniform(0, ceiling_h, n_wall)
        pts.append(np.stack([wx, np.full(n_wall, yval), wz3], axis=1))

    wy2 = rng.uniform(0, room_d, n_wall)
    wz2 = rng.uniform(0, ceiling_h, n_wall)
    pts.append(np.stack([np.zeros(n_wall), wy2, wz2], axis=1))

    all_pts = np.concatenate(pts).astype(np.float32)
    noise = rng.normal(0, 0.005, all_pts.shape).astype(np.float32)
    return all_pts + noise


def test_detect_window_in_synthetic_room():
    pts = _make_room_with_window(win_sill=0.9, win_w=1.2, win_h=1.0)
    layout = extract_layout(pts)
    openings = detect_openings(pts, layout)

    windows = [o for o in openings if o.kind == "window"]
    assert len(windows) >= 1, f"expected ≥1 window, got {openings}"

    win = windows[0]
    assert win.sill_height_m > 0.20, f"sill height {win.sill_height_m:.2f} m too low for a window"


def test_no_openings_in_closed_box():
    """Fully closed box room (4 m x 5 m) should report no openings."""
    room_w, room_d, ceiling_h = 4.0, 5.0, 2.5
    rng = np.random.default_rng(99)
    pts: list[np.ndarray] = []

    # walls: (fixed_axis, fixed_value, free_axis, free_span)
    for axis, val, free_span in [
        (0, 0.0, room_d), (0, room_w, room_d),
        (1, 0.0, room_w), (1, room_d, room_w),
    ]:
        free_axis = 1 if axis == 0 else 0
        n_wall = 3000
        wall = np.zeros((n_wall, 3), dtype=np.float32)
        wall[:, axis] = val
        wall[:, free_axis] = rng.uniform(0, free_span, n_wall).astype(np.float32)
        wall[:, 2] = rng.uniform(0, ceiling_h, n_wall).astype(np.float32)
        pts.append(wall)

    n_floor = 6000
    fx = rng.uniform(0, room_w, n_floor).astype(np.float32)
    fy = rng.uniform(0, room_d, n_floor).astype(np.float32)
    pts.append(np.stack([fx, fy, np.zeros(n_floor)], axis=1))
    pts.append(np.stack([fx, fy, np.full(n_floor, ceiling_h)], axis=1))

    all_pts = np.concatenate(pts)
    layout = extract_layout(all_pts)
    assert not layout.ceiling_unobserved, "closed box should detect its ceiling"
    openings = detect_openings(all_pts, layout)
    assert len(openings) == 0, f"expected no openings in closed box, got {openings}"
