"""Detect door and window openings in walls via occupancy-grid hole detection.

For each WallSegment we:
  1. Extract nearby points (within OPENING_MAX_DIST_M of the wall line).
  2. Project onto (u_wall=along-wall, v_height=above-floor) coordinates.
  3. Build a binary occupancy grid.
  4. Find runs of empty columns → candidate openings.
  5. Classify door (bottom near floor) vs. window (bottom elevated).

Output matches the Opening Pydantic model from schema_out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.config import (
    OPENING_CELL_M,
    OPENING_DOOR_GAP_M,
    OPENING_MAX_DIST_M,
    OPENING_MAX_W_M,
    OPENING_MIN_H_M,
    OPENING_MIN_W_M,
    OPENING_MIN_WALL_PTS,
    OPENING_OCCUPANCY_THRESH,
)
from roomscan.geometry.room_layout import RoomLayout, WallSegment
from roomscan.schema_out import Measurement


@dataclass
class DetectedOpening:
    wall_idx: int
    kind: str           # "door" | "window"
    u_start_m: float    # distance along wall from p0
    width_m: Measurement
    height_m: Measurement
    sill_height_m: float   # bottom of opening above floor


def _wall_frame(
    wall: WallSegment,
    layout: RoomLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (p0_3d, wall_dir_3d, wall_normal_3d, wall_length) for a wall segment."""
    p0_2d = np.array(wall.p0)
    p1_2d = np.array(wall.p1)
    up = layout.up_axis
    u3, v3 = layout.plan_u, layout.plan_v

    p0_3d = p0_2d[0] * u3 + p0_2d[1] * v3 + layout.floor_level * up
    p1_3d = p1_2d[0] * u3 + p1_2d[1] * v3 + layout.floor_level * up

    d3 = p1_3d - p0_3d
    wall_len = float(np.linalg.norm(d3))
    if wall_len < 1e-6:
        return p0_3d, np.array([1., 0., 0.]), np.array([0., 1., 0.]), 0.0
    d3 /= wall_len
    # inward-facing normal: cross(wall_dir, up) then normalise
    n3 = np.cross(d3, up)
    n3 /= np.linalg.norm(n3)
    return p0_3d, d3, n3, wall_len


def _occupancy_grid(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    wall_len: float,
    ceil_h: float,
    cell: float,
) -> np.ndarray:
    """Build a (n_height, n_width) bool occupancy grid from projected points."""
    n_w = max(1, int(np.ceil(wall_len / cell)))
    n_h = max(1, int(np.ceil(ceil_h / cell)))
    grid = np.zeros((n_h, n_w), dtype=bool)
    ui = np.clip((u_coords / cell).astype(int), 0, n_w - 1)
    vi = np.clip((v_coords / cell).astype(int), 0, n_h - 1)
    grid[vi, ui] = True
    return grid


def detect_openings(
    pts: np.ndarray,
    layout: RoomLayout,
) -> list[DetectedOpening]:
    """Detect openings (doors/windows) in all wall segments of a room layout.

    Args:
        pts:    World-frame (N, 3) point cloud.
        layout: RoomLayout from extract_layout().

    Returns:
        List of DetectedOpening; may be empty if scan coverage is insufficient.
    """
    up = layout.up_axis
    ceil_h = layout.ceiling_height_m.value if layout.ceiling_height_m else 3.0
    results: list[DetectedOpening] = []

    for wall_idx, wall in enumerate(layout.walls):
        p0_3d, d3, n3, wall_len = _wall_frame(wall, layout)
        if wall_len < OPENING_MIN_W_M:
            continue

        # --- filter points near this wall ---
        # signed distance along wall normal from wall plane
        wall_plane_d = float(p0_3d @ n3)
        dist_from_plane = np.abs((pts @ n3) - wall_plane_d)

        # projection along wall axis
        u_wall = (pts - p0_3d) @ d3
        v_height = (pts @ up) - layout.floor_level

        near = (
            (dist_from_plane < OPENING_MAX_DIST_M)
            & (u_wall >= 0) & (u_wall <= wall_len)
            & (v_height >= 0) & (v_height <= ceil_h)
        )
        if near.sum() < OPENING_MIN_WALL_PTS:
            continue

        u_near = u_wall[near]
        v_near = v_height[near]

        grid = _occupancy_grid(u_near, v_near, wall_len, ceil_h, OPENING_CELL_M)
        n_h, n_w = grid.shape
        cell = OPENING_CELL_M

        # A cell is "empty" if it falls well short of its ROW's typical (median
        # across columns) density. Row-relative, not column-aggregate: a door
        # with a solid lintel above it still has a normal-density top band, so
        # comparing whole-column totals washes out the gap. Comparing per row
        # isolates exactly the band that's actually missing.
        row_typical = np.median(grid, axis=1)
        cell_empty = np.where(
            row_typical[:, None] > 0,
            grid <= OPENING_OCCUPANCY_THRESH * row_typical[:, None],
            grid == 0,
        )

        min_empty_rows = max(1, int(np.ceil(OPENING_MIN_H_M / cell)))

        def _longest_empty_run(col_mask: np.ndarray) -> tuple[int, int]:
            """Return (start_row, length) of the longest contiguous empty run."""
            best_start = best_len = cur_start = cur_len = 0
            for i, empty in enumerate(col_mask):
                if empty:
                    if cur_len == 0:
                        cur_start = i
                    cur_len += 1
                    if cur_len > best_len:
                        best_len, best_start = cur_len, cur_start
                else:
                    cur_len = 0
            return best_start, best_len

        col_has_gap = np.array([
            _longest_empty_run(cell_empty[:, c])[1] >= min_empty_rows
            for c in range(n_w)
        ])

        # Find runs of gap-bearing columns (potential openings)
        run_starts: list[int] = []
        run_ends: list[int] = []
        in_run = False
        for ci in range(n_w):
            if col_has_gap[ci]:
                if not in_run:
                    run_starts.append(ci)
                    in_run = True
            else:
                if in_run:
                    run_ends.append(ci - 1)
                    in_run = False
        if in_run:
            run_ends.append(n_w - 1)

        for rs, re in zip(run_starts, run_ends):
            width = (re - rs + 1) * cell
            if not (OPENING_MIN_W_M <= width <= OPENING_MAX_W_M):
                continue

            # Height of opening: rows empty across (nearly) all columns in the run
            empty_rows = cell_empty[:, rs:re + 1].all(axis=1)
            if not empty_rows.any():
                continue

            first_empty = int(np.argmax(empty_rows))
            last_empty = int(len(empty_rows) - 1 - np.argmax(empty_rows[::-1]))
            open_height = (last_empty - first_empty + 1) * cell
            if open_height < OPENING_MIN_H_M:
                continue

            sill_h = first_empty * cell
            kind = "door" if sill_h < OPENING_DOOR_GAP_M else "window"

            # ponytail: ±1 cell CI is the simplest defensible bound here
            results.append(DetectedOpening(
                wall_idx=wall_idx,
                kind=kind,
                u_start_m=rs * cell,
                width_m=Measurement(
                    value=width, lo=width - cell, hi=width + cell,
                    confidence_level=0.9,
                ),
                height_m=Measurement(
                    value=open_height,
                    lo=open_height - cell,
                    hi=open_height + cell,
                    confidence_level=0.9,
                ),
                sill_height_m=sill_h,
            ))

    return results
