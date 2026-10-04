"""Multi-room photo stitcher (Phase 8).

Each room is reconstructed independently (its own local floor-plane
coordinate frame, its own scale) -- nothing inherently connects one room's
coordinate frame to another's. This makes a best-effort attempt to connect
adjacent rooms via shared visual features (e.g. a doorway photographed from
both sides), falling back to a simple non-overlapping grid placement when no
such connection is found.

Honesty note: there is no real multi-room test fixture to validate the
cross-room matching path against (unlike the single-room photo/video tiers,
which have real or simulated-from-real data to check against -- see
bench/derive_tiers.py). Treat any "connected" relative room placement here
as unverified; the grid fallback makes no spatial claim at all beyond
"these are different rooms, shown separately."
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roomscan.config import MULTI_ROOM_GRID_GAP_M, MULTI_ROOM_MIN_MATCHES
from roomscan.geometry.room_layout import RoomLayout
from roomscan.geometry.sfm import _match_features


@dataclass
class PlacedRoom:
    name: str
    layout: RoomLayout
    offset: tuple[float, float]   # added to the room's own 2-D polygon coords
    connected: bool               # True if placed via cross-room feature match,
                                   # False if placed on the grid fallback


@dataclass
class PropertyLayout:
    rooms: list[PlacedRoom] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
) -> PropertyLayout:
    """Arrange independently-reconstructed rooms into one property layout.

    Args:
        room_results: list of (room_name, layout, photos) for each
            successfully-reconstructed room, in input order. photos is the
            list of BGR images used for that room (needed for the best-
            effort cross-room connection check).

    Grid fallback: rooms are placed left-to-right in input order, each
    offset by the running width of previous rooms plus a fixed gap -- purely
    schematic, makes no claim about true adjacency or orientation.
    """
    property_layout = PropertyLayout()
    if not room_results:
        return property_layout

    x_offset = 0.0
    prev_photos = None
    for i, (name, layout, photos) in enumerate(room_results):
        connected = False
        if prev_photos is not None:
            connected = _try_connect(prev_photos, photos)

        placed = PlacedRoom(name=name, layout=layout, offset=(x_offset, 0.0), connected=connected)
        property_layout.rooms.append(placed)

        if not connected and i > 0:
            property_layout.warnings.append(
                f"room '{name}': no shared visual structure found with the previous "
                "room's photos -- placed on a schematic grid, not a geometric stitch"
            )
        elif connected:
            property_layout.warnings.append(
                f"room '{name}': shared visual structure found with the previous room's "
                "photos, but only used to flag likely adjacency -- still placed on the "
                "schematic grid (no validated cross-room transform; see module docstring)"
            )

        w, _ = _room_bbox(layout)
        x_offset += w + MULTI_ROOM_GRID_GAP_M
        prev_photos = photos

    if len(room_results) > 1:
        property_layout.warnings.append(
            "multi-room layout is schematic: room shapes/areas are individually "
            "reconstructed, but relative room position/orientation is not "
            "determined by this method (no real multi-room fixture exists to "
            "validate cross-room stitching against)"
        )
    return property_layout
