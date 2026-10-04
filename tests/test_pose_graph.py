"""Pose-graph door-to-door stitching tests (case-study realignment, see
COMPLIANCE_MATRIX.md). Synthetic layouts only -- no real multi-room fixture
exists (same limitation as multi_room.py).

The asymmetric case below is the one that actually distinguishes "align
doors" from the earlier, wrong "align wall midpoints" implementation: with
differing door offsets and differing wall lengths, a wall-midpoint aligner
and a door aligner disagree, so this is the regression test for that bug.
"""
import numpy as np
import pytest

from roomscan.geometry.openings import DetectedOpening
from roomscan.geometry.room_layout import RoomLayout, WallSegment
from roomscan.schema_out import Measurement
from roomscan.geometry.pose_graph import (
    POSE_GRAPH_WALL_THICKNESS_M,
    _apply_pose,
    _door_point,
    find_door_correspondences,
    solve_pose_graph,
)


def _m(v: float) -> Measurement:
    return Measurement(value=v, lo=v, hi=v)


def _rect_layout(w: float, h: float) -> RoomLayout:
    walls = [
        WallSegment(p0=(0, 0), p1=(w, 0), length_m=_m(w)),
        WallSegment(p0=(w, 0), p1=(w, h), length_m=_m(h)),
        WallSegment(p0=(w, h), p1=(0, h), length_m=_m(w)),
        WallSegment(p0=(0, h), p1=(0, 0), length_m=_m(h)),
    ]
    return RoomLayout(up_axis=np.array([0.0, 0.0, 1.0]), floor_d=0.0, ceiling_d=None,
                       polygon=[(0, 0), (w, 0), (w, h), (0, h)],
                       floor_area_m2=_m(w * h), ceiling_height_m=None, walls=walls)


def _door(wall_idx: int, u_start: float, width: float) -> DetectedOpening:
    return DetectedOpening(wall_idx=wall_idx, kind="door", u_start_m=u_start,
                            width_m=_m(width), height_m=_m(2.0), sill_height_m=0.0)


def _door_gap(room_a, corr, pose_a, room_b, pose_b) -> float:
    d_a = _apply_pose(np.array(_door_point(room_a, corr.wall_a, corr.door_u_a)), pose_a)
    d_b = _apply_pose(np.array(_door_point(room_b, corr.wall_b, corr.door_u_b)), pose_b)
    return float(np.linalg.norm(d_a - d_b))


def test_symmetric_two_room_doors_coincide():
    room_a, room_b = _rect_layout(4, 5), _rect_layout(3, 5)
    openings_a = [_door(1, 2.0, 0.9)]   # right wall of A
    openings_b = [_door(3, 2.0, 0.9)]   # left wall of B
    corr = find_door_correspondences([room_a, room_b], [openings_a, openings_b], 0, 1)
    assert corr is not None
    poses = solve_pose_graph([room_a, room_b], [corr])
    gap = _door_gap(room_a, corr, poses[0], room_b, poses[1])
    assert abs(gap - POSE_GRAPH_WALL_THICKNESS_M) < 0.01


def test_asymmetric_doors_and_wall_lengths_still_coincide():
    """Regression test: catches the wall-midpoint-alignment bug. Doors are
    offset differently on each wall and the walls are different lengths, so
    wall midpoints and door midpoints are nowhere near each other."""
    room_a, room_b = _rect_layout(6, 5), _rect_layout(3, 5)
    openings_a = [_door(1, 4.0, 0.9)]   # door_u = 4.45, wall midpoint = 2.5
    openings_b = [_door(3, 0.2, 0.9)]   # door_u = 0.65, wall midpoint = 2.5
    corr = find_door_correspondences([room_a, room_b], [openings_a, openings_b], 0, 1)
    assert corr is not None
    poses = solve_pose_graph([room_a, room_b], [corr])
    gap = _door_gap(room_a, corr, poses[0], room_b, poses[1])
    assert abs(gap - POSE_GRAPH_WALL_THICKNESS_M) < 0.01, (
        f"doors should coincide up to the wall-thickness prior, got gap={gap}"
    )
    # the rooms must NOT be placed as if wall midpoints aligned
    assert poses[1].x > 5.0


def test_no_matching_door_returns_none():
    room_a, room_b = _rect_layout(4, 5), _rect_layout(3, 5)
    openings_a = [_door(1, 2.0, 0.9)]
    openings_b = [_door(3, 2.0, 2.5)]   # width disagrees far beyond tolerance
    assert find_door_correspondences([room_a, room_b], [openings_a, openings_b], 0, 1) is None


def test_disconnected_room_stays_at_origin():
    rooms = [_rect_layout(4, 5), _rect_layout(3, 5)]
    poses = solve_pose_graph(rooms, [])
    assert all(p.x == 0.0 and p.y == 0.0 for p in poses)


def test_refine_false_matches_refine_true_on_a_tree_with_no_redundant_edge():
    """A single correspondence is a 1-edge spanning tree with nothing left
    over to jointly refine -- "drift correction off" (refine=False) and
    "on" (refine=True) must agree exactly, since there's no extra
    constraint to use either way. (bench/ablate.py's drift ablation relies
    on this base case: refine only matters once a redundant/loop-closure
    edge exists.)"""
    room_a, room_b = _rect_layout(4, 5), _rect_layout(3, 5)
    openings_a = [_door(1, 2.0, 0.9)]
    openings_b = [_door(3, 2.0, 0.9)]
    corr = find_door_correspondences([room_a, room_b], [openings_a, openings_b], 0, 1)
    poses_off = solve_pose_graph([room_a, room_b], [corr], refine=False)
    poses_on = solve_pose_graph([room_a, room_b], [corr], refine=True)
    for p_off, p_on in zip(poses_off, poses_on):
        assert p_off.x == pytest.approx(p_on.x, abs=1e-9)
        assert p_off.y == pytest.approx(p_on.y, abs=1e-9)
        assert p_off.yaw == pytest.approx(p_on.yaw, abs=1e-9)


if __name__ == "__main__":
    test_symmetric_two_room_doors_coincide()
    test_asymmetric_doors_and_wall_lengths_still_coincide()
    test_no_matching_door_returns_none()
    test_disconnected_room_stays_at_origin()
    test_refine_false_matches_refine_true_on_a_tree_with_no_redundant_edge()
    print("ok")
