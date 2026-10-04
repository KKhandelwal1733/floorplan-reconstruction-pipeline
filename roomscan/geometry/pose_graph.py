"""Multi-room pose-graph solver: door-to-door correspondences (case-study
realignment, see COMPLIANCE_MATRIX.md).

Rooms are nodes with a 2-D pose (x, y, yaw) in a shared property-level
frame. A door/opening detected in two adjacent rooms' own local frames
creates a relative-pose edge: the two door midpoints should coincide (up to
a wall-thickness gap) and the two walls' normals should point toward each
other (anti-parallel). Solved with robust nonlinear least squares
(`scipy.optimize.least_squares`, Huber loss) for translation, after a
discrete Manhattan (90-degree) search fixes each room's rotation relative
to its neighbours -- matching the brief's specified approach.

Known limitation (disclosed, not hidden): this assumes each room is
captured as a SEPARATE input (its own LiDAR scan / video / photo folder),
stitched via visual/geometric door correspondence -- not the brief's
alternate architecture for LiDAR/video (one continuous capture, poses
refined via loop closure). No real continuous multi-room LiDAR/video
capture exists to develop or validate that approach against; this one
does, and is applied uniformly across all three tiers since every tier
already produces the RoomLayout + detected-openings inputs it needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.config import (
    POSE_GRAPH_DOOR_WIDTH_TOL_FRAC,
    POSE_GRAPH_HUBER_DELTA,
    POSE_GRAPH_MIN_OVERLAP_FRAC,
    POSE_GRAPH_WALL_THICKNESS_M,
)

MANHATTAN_ANGLES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]


@dataclass
class DoorCorrespondence:
    room_a: int        # index into the rooms list
    room_b: int
    wall_a: int        # wall index within room_a's layout
    wall_b: int        # wall index within room_b's layout
    door_u_a: float     # door midpoint, distance along wall_a from its p0 (metres)
    door_u_b: float     # door midpoint, distance along wall_b from its p0 (metres)
    width_diff_frac: float = 0.0   # relative door-width disagreement (0 = perfect match)


@dataclass
class RoomPose:
    x: float
    y: float
    yaw: float     # radians, Manhattan-snapped


def find_door_correspondences(
    layouts: list,
    openings_per_room: list[list],
    room_a: int,
    room_b: int,
) -> DoorCorrespondence | None:
    """Best-guess match between a door in room_a and a door in room_b, by
    width agreement -- the geometric verifier the brief calls for in place
    of a learned one ("door-width agreement... plus the doorway-photo
    matching"). Returns None if no two doors agree closely enough.
    """
    best: tuple[float, int, int, float, float] | None = None
    for o_a in openings_per_room[room_a]:
        if o_a.kind != "door":
            continue
        for o_b in openings_per_room[room_b]:
            if o_b.kind != "door":
                continue
            w_a, w_b = o_a.width_m.value, o_b.width_m.value
            diff_frac = abs(w_a - w_b) / max(w_a, w_b, 1e-6)
            if diff_frac <= POSE_GRAPH_DOOR_WIDTH_TOL_FRAC:
                door_u_a = o_a.u_start_m + w_a / 2
                door_u_b = o_b.u_start_m + w_b / 2
                if best is None or diff_frac < best[0]:
                    best = (diff_frac, o_a.wall_idx, o_b.wall_idx, door_u_a, door_u_b)
    if best is None:
        return None
    return DoorCorrespondence(
        room_a=room_a, room_b=room_b, wall_a=best[1], wall_b=best[2],
        door_u_a=best[3], door_u_b=best[4], width_diff_frac=best[0],
    )


def _wall_dir_and_normal(layout, wall_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p0, unit direction, outward normal) of a wall, in the room's own local frame."""
    wall = layout.walls[wall_idx]
    p0, p1 = np.array(wall.p0), np.array(wall.p1)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return p0, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    d = d / length
    normal = np.array([-d[1], d[0]])
    return p0, d, normal


def _door_point(layout, wall_idx: int, door_u: float) -> np.ndarray:
    """The door's own midpoint along a wall (distance door_u from the wall's
    p0), NOT the wall's own midpoint -- these differ whenever the door isn't
    centred on the wall or the two matched walls have different lengths."""
    p0, d, _ = _wall_dir_and_normal(layout, wall_idx)
    return p0 + d * door_u


def _rotate(v: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]]) @ v


def _apply_pose(pt: np.ndarray, pose: RoomPose) -> np.ndarray:
    return _rotate(pt, pose.yaw) + np.array([pose.x, pose.y])


def solve_pose_graph(
    layouts: list,
    correspondences: list[DoorCorrespondence],
    wall_thickness_m: float = POSE_GRAPH_WALL_THICKNESS_M,
    refine: bool = True,
) -> list[RoomPose]:
    """Jointly solve (x, y, yaw) per room from door-to-door correspondences.

    Room 0 is fixed at the origin. Returns one RoomPose per layout, in
    layouts' order. Disconnected rooms (no path of correspondences back to
    room 0) are returned at the origin with yaw=0 -- callers should treat
    these as unplaced and fall back to grid placement for them.

    refine=False skips the final joint robust-least-squares step and
    returns the naive BFS placement instead: each room placed only from its
    one parent edge in the BFS spanning tree, ignoring every other edge
    (e.g. a loop-closure correspondence back to an earlier room). This is
    "drift correction off" -- used by bench/ablate.py's pose-graph drift
    ablation to measure what the joint solve (refine=True, "on") actually
    buys you when a capture loops back on itself. On a pure chain/tree of
    correspondences (no loop) the two are identical, since every edge is
    already in the spanning tree and already satisfied exactly.
    """
    from scipy.optimize import least_squares

    n = len(layouts)
    if n == 0:
        return []
    if n == 1 or not correspondences:
        return [RoomPose(0.0, 0.0, 0.0) for _ in range(n)]

    edges_by_room: dict[int, list[DoorCorrespondence]] = {i: [] for i in range(n)}
    for c in correspondences:
        edges_by_room[c.room_a].append(c)
        edges_by_room[c.room_b].append(c)

    # Discrete Manhattan yaw search via BFS from room 0: each newly-reached
    # room's yaw is picked from the 4 cardinal angles to make its door's
    # outward normal most nearly anti-parallel to the connecting room's.
    yaws = [0.0] * n
    visited = {0}
    queue = [0]
    while queue:
        cur = queue.pop(0)
        for c in edges_by_room[cur]:
            other = c.room_b if c.room_a == cur else c.room_a
            if other in visited:
                continue
            wall_cur = c.wall_a if c.room_a == cur else c.wall_b
            wall_other = c.wall_b if c.room_a == cur else c.wall_a
            _, _, normal_cur = _wall_dir_and_normal(layouts[cur], wall_cur)
            _, _, normal_other = _wall_dir_and_normal(layouts[other], wall_other)
            normal_cur_world = _rotate(normal_cur, yaws[cur])

            best_angle, best_score = 0.0, -np.inf
            for extra in MANHATTAN_ANGLES:
                normal_other_world = _rotate(normal_other, extra)
                score = -float(normal_cur_world @ normal_other_world)
                if score > best_score:
                    best_score, best_angle = score, extra
            yaws[other] = best_angle
            visited.add(other)
            queue.append(other)

    # Initial translation via BFS placement (door midpoints coincide + gap).
    xs, ys = [0.0] * n, [0.0] * n
    placed = {0}
    queue = [0]
    while queue:
        cur = queue.pop(0)
        for c in edges_by_room[cur]:
            other = c.room_b if c.room_a == cur else c.room_a
            if other in placed:
                continue
            wall_cur = c.wall_a if c.room_a == cur else c.wall_b
            wall_other = c.wall_b if c.room_a == cur else c.wall_a
            door_u_cur = c.door_u_a if c.room_a == cur else c.door_u_b
            door_u_other = c.door_u_b if c.room_a == cur else c.door_u_a
            _, _, normal_cur = _wall_dir_and_normal(layouts[cur], wall_cur)
            door_cur_local = _door_point(layouts[cur], wall_cur, door_u_cur)
            door_other_local = _door_point(layouts[other], wall_other, door_u_other)

            pose_cur = RoomPose(xs[cur], ys[cur], yaws[cur])
            door_cur_world = _apply_pose(door_cur_local, pose_cur)
            normal_cur_world = _rotate(normal_cur, yaws[cur])
            target = door_cur_world + normal_cur_world * wall_thickness_m

            door_other_rot = _rotate(door_other_local, yaws[other])
            xs[other] = target[0] - door_other_rot[0]
            ys[other] = target[1] - door_other_rot[1]
            placed.add(other)
            queue.append(other)

    if not refine:
        return [RoomPose(xs[i], ys[i], yaws[i]) for i in range(n)]

    # Continuous refinement: translations only (yaw stays Manhattan-snapped),
    # joint robust least squares over every correspondence simultaneously.
    def residuals(params: np.ndarray) -> np.ndarray:
        xs_r = [0.0] + list(params[0::2])
        ys_r = [0.0] + list(params[1::2])
        res = []
        for c in correspondences:
            pose_a = RoomPose(xs_r[c.room_a], ys_r[c.room_a], yaws[c.room_a])
            pose_b = RoomPose(xs_r[c.room_b], ys_r[c.room_b], yaws[c.room_b])
            _, _, normal_a = _wall_dir_and_normal(layouts[c.room_a], c.wall_a)
            door_a_local = _door_point(layouts[c.room_a], c.wall_a, c.door_u_a)
            door_b_local = _door_point(layouts[c.room_b], c.wall_b, c.door_u_b)
            door_a_world = _apply_pose(door_a_local, pose_a)
            door_b_world = _apply_pose(door_b_local, pose_b)
            normal_a_world = _rotate(normal_a, pose_a.yaw)
            target = door_a_world + normal_a_world * wall_thickness_m
            res.extend((door_b_world - target).tolist())
        return np.array(res)

    x0 = []
    for i in range(1, n):
        x0.extend([xs[i], ys[i]])
    result = least_squares(residuals, x0, loss="huber", f_scale=POSE_GRAPH_HUBER_DELTA)
    xs = [0.0] + list(result.x[0::2])
    ys = [0.0] + list(result.x[1::2])

    return [RoomPose(xs[i], ys[i], yaws[i]) for i in range(n)]


def connected_rooms(n: int, correspondences: list[DoorCorrespondence]) -> set[int]:
    """Room indices reachable from room 0 via correspondence edges."""
    edges_by_room: dict[int, list[DoorCorrespondence]] = {i: [] for i in range(n)}
    for c in correspondences:
        edges_by_room[c.room_a].append(c)
        edges_by_room[c.room_b].append(c)
    visited = {0}
    queue = [0]
    while queue:
        cur = queue.pop(0)
        for c in edges_by_room[cur]:
            other = c.room_b if c.room_a == cur else c.room_a
            if other not in visited:
                visited.add(other)
                queue.append(other)
    return visited


def _room_aabb(layout, pose: RoomPose) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array([_apply_pose(np.array(p), pose) for p in layout.polygon])
    return pts.min(axis=0), pts.max(axis=0)


def has_overlap(
    layout_a, pose_a: RoomPose, layout_b, pose_b: RoomPose,
    min_overlap_frac: float = POSE_GRAPH_MIN_OVERLAP_FRAC,
) -> bool:
    """Axis-aligned-bounding-box overlap check (a deliberately simple proxy
    for full polygon intersection -- lazy but inspectable; upgrade to a real
    polygon-intersection test if AABB false positives from non-rectangular
    rooms turn out to matter in practice)."""
    min_a, max_a = _room_aabb(layout_a, pose_a)
    min_b, max_b = _room_aabb(layout_b, pose_b)
    overlap_min = np.maximum(min_a, min_b)
    overlap_max = np.minimum(max_a, max_b)
    overlap_size = np.maximum(overlap_max - overlap_min, 0)
    overlap_area = overlap_size[0] * overlap_size[1]
    area_a = (max_a[0] - min_a[0]) * (max_a[1] - min_a[1])
    area_b = (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])
    smaller = min(area_a, area_b, 1e-6)
    return (overlap_area / smaller) > min_overlap_frac
