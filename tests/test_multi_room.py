"""Phase 8: multi-room stitcher tests.

No real multi-room photo fixture exists to validate cross-room matching
against (see multi_room.py's module docstring), so these tests use
synthetic RoomLayout objects to check the grid-placement and warning-
surfacing logic -- the part of this module that IS fully deterministic and
testable without real data.
"""
import numpy as np
import pytest

from roomscan.geometry.multi_room import stitch_rooms
from roomscan.geometry.room_layout import extract_layout


def _simple_room_layout(offset_x: float = 0.0):
    rng = np.random.default_rng(1)
    floor = rng.uniform(0, 3, (6000, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, 6000).astype(np.float32)
    floor[:, 0] += offset_x
    return extract_layout(floor)


def test_stitch_rooms_empty():
    result = stitch_rooms([])
    assert result.rooms == []
    assert result.warnings == []


def test_stitch_single_room_no_warning_about_connectivity():
    layout = _simple_room_layout()
    result = stitch_rooms([("only_room", layout, [])])
    assert len(result.rooms) == 1
    assert result.rooms[0].name == "only_room"
    assert result.rooms[0].offset == (0.0, 0.0)
    # single room: no multi-room schematic-layout caveat needed
    assert not any("schematic" in w for w in result.warnings)


def test_stitch_multiple_rooms_placed_on_grid_without_photos():
    """With no photos given (empty lists), cross-room connection can never
    be found, so every room after the first must fall back to the grid."""
    layout_a = _simple_room_layout()
    layout_b = _simple_room_layout()
    result = stitch_rooms([("room_a", layout_a, []), ("room_b", layout_b, [])])

    assert len(result.rooms) == 2
    assert result.rooms[0].offset == (0.0, 0.0)
    # room_b must be offset past room_a's width (non-overlapping grid)
    assert result.rooms[1].offset[0] > 0.0
    assert result.rooms[1].connected is False
    assert any("schematic" in w for w in result.warnings)


def test_stitch_rooms_grid_offsets_increase_monotonically():
    layouts = [_simple_room_layout() for _ in range(4)]
    results = [(f"room_{i}", layouts[i], []) for i in range(4)]
    result = stitch_rooms(results)
    offsets = [r.offset[0] for r in result.rooms]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == 4  # all distinct, no overlap
