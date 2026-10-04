"""Multi-room stitcher (Phase 8, extended for case-study realignment -- see
COMPLIANCE_MATRIX.md).

Each room is reconstructed independently (its own local floor-plane
coordinate frame, its own scale) -- nothing inherently connects one room's
coordinate frame to another's. Two placement strategies are tried, per room
pair, in order:

  1. Door-to-door pose-graph stitch (roomscan/geometry/pose_graph.py): if a
     door detected in one room's walls width-matches a door in another's,
     solve_pose_graph places every connected room so that corresponding
     doors coincide (up to a wall-thickness gap) and walls are Manhattan-
     aligned. This is the real geometric stitch the case study asks for.
  2. Schematic grid fallback: when no door correspondence exists for a room
     (or the resulting placement would make two rooms implausibly overlap --
     has_overlap as a lazy AABB proxy), it's placed left-to-right on a
     simple non-overlapping grid instead, making no spatial claim beyond
     "this is a different room, shown separately."

Honesty note: there is no real multi-room test fixture (LiDAR, video, or
photo) to validate either path end-to-end against (unlike the single-room
tiers, which have real or simulated-from-real data -- see
bench/derive_tiers.py). The door-to-door approach is applied uniformly
across all three tiers since every tier already produces the RoomLayout +
detected-openings inputs it needs; treat any pose-graph placement as
internally self-consistent (doors really do coincide), not externally
validated against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from roomscan.config import MULTI_ROOM_GRID_GAP_M, MULTI_ROOM_MIN_MATCHES
from roomscan.geometry.pose_graph import (
    DoorCorrespondence,
    RoomPose,
    _room_aabb,
    connected_rooms,
    find_door_correspondences,
    has_overlap,
    solve_pose_graph,
)
from roomscan.geometry.room_layout import RoomLayout
from roomscan.geometry.sfm import _match_features


@dataclass
class PlacedRoom:
    name: str
    layout: RoomLayout
    offset: tuple[float, float]   # added to the room's own 2-D polygon coords
    yaw: float = 0.0               # Manhattan-snapped rotation (radians), 0 for grid fallback
    connected: bool = False         # True if placed via pose-graph or flagged-adjacent photos,
                                      # False if placed on the grid fallback with no signal at all


@dataclass
class PropertyLayout:
    rooms: list[PlacedRoom] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Door correspondences actually trusted for placement (empty if the pose
    # graph found none, or its result was rejected for implausible overlap).
    correspondences: list[DoorCorrespondence] = field(default_factory=list)


def _room_bbox(layout: RoomLayout) -> tuple[float, float]:
    """(width, height) of a room's 2-D floor polygon."""
    if not layout.polygon:
        return (1.0, 1.0)
    xs = [p[0] for p in layout.polygon]
    ys = [p[1] for p in layout.polygon]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _try_connect(room_a_photos, room_b_photos) -> bool:
    """Best-effort check for shared visual structure between two rooms'
    photo sets (e.g. a doorway photographed from both sides).

    Returns True if any cross-room photo pair has enough matched features to
    suggest real overlap. Does NOT compute a relative transform -- with this
    little guaranteed structure (at most 2-8 photos per room, no guarantee
    any pair actually shows the connection), attempting a precise geometric
    stitch here would be unvalidated speculation; this only decides whether
    to report "likely adjacent" versus falling back to the grid.
    """
    import cv2

    for img_a in room_a_photos:
        for img_b in room_b_photos:
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            matched = _match_features(gray_a, gray_b, detector="sift")
            if matched is not None and len(matched[0]) >= MULTI_ROOM_MIN_MATCHES:
                return True
    return False


def stitch_rooms(
    room_results: list[tuple[str, RoomLayout, list]],
    openings_per_room: list[list] | None = None,
) -> PropertyLayout:
    """Arrange independently-reconstructed rooms into one property layout.

    Args:
        room_results: list of (room_name, layout, photos) for each
            successfully-reconstructed room, in input order. photos is the
            list of BGR images used for that room (needed for the best-
            effort cross-room connection check used by the grid fallback).
        openings_per_room: per-room list of DetectedOpening (same order as
            room_results), used to find door-to-door correspondences. Pass
            None to skip the pose-graph attempt entirely and always use the
            grid (e.g. when openings weren't detected for this tier).

    Grid fallback: rooms without a trusted door placement are placed
    left-to-right, past the bounding box of every room placed so far
    (pose-graph or grid) plus a fixed gap -- purely schematic, makes no claim
    about true adjacency or orientation.
    """
    property_layout = PropertyLayout()
    if not room_results:
        return property_layout

    layouts = [layout for _, layout, _ in room_results]
    n = len(layouts)

    correspondences: list[DoorCorrespondence] = []
    if openings_per_room is not None:
        for i in range(n):
            for j in range(i + 1, n):
                corr = find_door_correspondences(layouts, openings_per_room, i, j)
                if corr is not None:
                    correspondences.append(corr)

    poses: list[RoomPose] | None = None
    reachable: set[int] = set()
    if correspondences:
        candidate_poses = solve_pose_graph(layouts, correspondences)
        candidate_reachable = sorted(connected_rooms(n, correspondences))
        overlap_found = any(
            has_overlap(layouts[i], candidate_poses[i], layouts[j], candidate_poses[j])
            for a, i in enumerate(candidate_reachable)
            for j in candidate_reachable[a + 1:]
        )
        if not overlap_found:
            poses, reachable = candidate_poses, set(candidate_reachable)
            property_layout.correspondences = [
                c for c in correspondences if c.room_a in reachable and c.room_b in reachable
            ]

    max_x_so_far = 0.0
    prev_photos = None
    for i, (name, layout, photos) in enumerate(room_results):
        if poses is not None and i in reachable:
            pose = poses[i]
            placed = PlacedRoom(name=name, layout=layout, offset=(pose.x, pose.y),
                                 yaw=pose.yaw, connected=True)
            if i > 0:
                property_layout.warnings.append(
                    f"room '{name}': placed via door-to-door pose-graph stitch "
                    "(see roomscan/geometry/pose_graph.py)"
                )
        else:
            connected = False
            if prev_photos is not None:
                connected = _try_connect(prev_photos, photos)
            placed = PlacedRoom(name=name, layout=layout, offset=(max_x_so_far, 0.0),
                                 yaw=0.0, connected=connected)
            if not connected and i > 0:
                property_layout.warnings.append(
                    f"room '{name}': no door correspondence or shared visual structure "
                    "found -- placed on a schematic grid, not a geometric stitch"
                )
            elif connected:
                property_layout.warnings.append(
                    f"room '{name}': shared visual structure found with the previous room's "
                    "photos, but no door-width correspondence to anchor a pose-graph placement "
                    "-- still placed on the schematic grid (see module docstring)"
                )
        property_layout.rooms.append(placed)

        _, max_pt = _room_aabb(layout, RoomPose(placed.offset[0], placed.offset[1], placed.yaw))
        max_x_so_far = max(max_x_so_far, float(max_pt[0]) + MULTI_ROOM_GRID_GAP_M)
        prev_photos = photos

    if n > 1 and not reachable:
        property_layout.warnings.append(
            "multi-room layout is schematic: room shapes/areas are individually "
            "reconstructed, but relative room position/orientation is not "
            "determined by this method (no door correspondence found, or the "
            "resulting placement overlapped implausibly -- see module docstring)"
        )
    elif reachable and len(reachable) < n:
        property_layout.warnings.append(
            f"pose-graph stitch placed {len(reachable)}/{n} rooms via door "
            "correspondence; remaining rooms had no matching door and are shown "
            "on a schematic grid instead"
        )
    return property_layout
