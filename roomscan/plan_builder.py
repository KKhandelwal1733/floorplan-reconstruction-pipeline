"""Builds a schema v0.1 Plan (plan.json) from a RoomLayout + detected
openings/damage/scope (Phase 10 closes the loop: CLAUDE.md documents
plan.json as a per-capture output, but nothing wrote one until damage/scope
data existed to put in it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from roomscan.geometry.room_layout import RoomLayout
from roomscan.schema_out import (
    Capture,
    Damage,
    Measurement,
    Opening,
    Plan,
    Pose,
    PropertyBlock,
    Room,
    ScopeItem,
    Wall,
    write_plan,
)


def _opening_confidence(width_m) -> float:
    """Crude confidence proxy from the opening's own CI relative width --
    no other confidence signal exists for openings yet."""
    if width_m.value <= 0:
        return 0.1
    rel_hw = (width_m.hi - width_m.lo) / (2 * width_m.value)
    return max(0.1, min(0.95, 1.0 - rel_hw))


def build_room(
    layout: RoomLayout,
    room_id: str,
    damages: list[Damage] | None = None,
    scope_items: list[ScopeItem] | None = None,
    detected_openings: list[Any] | None = None,
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Room:
    """Convert a RoomLayout (+ Phase 4/10 outputs) into a schema Room."""
    walls = [
        Wall(id=f"wall_{i}", length_m=w.length_m)
        for i, w in enumerate(layout.walls)
    ]

    openings = []
    for o in (detected_openings or []):
        openings.append(Opening(
            id=f"opening_{o.wall_idx}_{round(o.u_start_m, 2)}",
            type=o.kind if o.kind in ("door", "window") else "door",
            wall=f"wall_{o.wall_idx}",
            width_m=o.width_m,
            confidence=_opening_confidence(o.width_m),
        ))

    ceiling_height_m = layout.ceiling_height_m
    if ceiling_height_m is None:
        # schema requires ceiling_height_m; emit a maximally wide abstention
        # rather than omit the field (never crash, always output something).
        from roomscan.schema_out import Measurement
        ceiling_height_m = Measurement(value=0.0, lo=0.0, hi=0.0, confidence_level=0.0)

    x, y, yaw = pose
    return Room(
        id=room_id,
        polygon=[tuple(p) for p in layout.polygon],
        pose=Pose(x=x, y=y, yaw=yaw),
        walls=walls,
        ceiling_height_m=ceiling_height_m,
        floor_area_m2=layout.floor_area_m2,
        openings=openings,
        damage=damages or [],
        scope=scope_items or [],
    )


def build_single_room_plan(
    tier: str,
    layout: RoomLayout,
    damages: list[Damage] | None = None,
    scope_items: list[ScopeItem] | None = None,
    detected_openings: list[Any] | None = None,
) -> Plan:
    """A single-room capture's Plan (LiDAR or video tier)."""
    room = build_room(layout, "room_0", damages, scope_items, detected_openings)
    status = "degraded" if (layout.quality_score < 1.0 or layout.ceiling_unobserved) else "ok"
    return Plan(
        capture=Capture(tier=tier, status=status, warnings=layout.capture_warnings),
        property=PropertyBlock(
            rooms=[room.id], adjacency=[], footprint_m2=layout.floor_area_m2,
        ),
        rooms=[room],
    )


def build_abstained_plan(tier: str, reason: str) -> Plan:
    """A minimal, schema-valid Plan for when reconstruction fails entirely --
    never skip writing plan.json just because there's no geometry (hard
    rule: always output something, even if that something is an honest
    abstention)."""
    from roomscan.schema_out import Measurement
    zero = Measurement(value=0.0, lo=0.0, hi=0.0, confidence_level=0.0)
    return Plan(
        capture=Capture(tier=tier, status="abstained", warnings=[reason]),
        property=PropertyBlock(rooms=[], adjacency=[], footprint_m2=zero),
        rooms=[],
    )


def write_abstained_plan(tier: str, reason: str, out_path: Path) -> None:
    write_plan(build_abstained_plan(tier, reason), out_path)


def build_property_plan(
    property_layout: Any,
    per_room_damage: dict[str, tuple[list[Damage], list[ScopeItem]]] | None = None,
) -> Plan:
    """A multi-room (photo tier) capture's Plan.

    property_layout is a geometry.multi_room.PropertyLayout. Room pose uses
    its schematic grid offset (x, y) with yaw=0 -- see multi_room.py: this
    is NOT a geometrically validated position, just the fallback grid
    placement, carried through honestly rather than invented as more
    precise than it is.
    """
    per_room_damage = per_room_damage or {}
    rooms = []
    for placed in property_layout.rooms:
        damages, scope_items = per_room_damage.get(placed.name, ([], []))
        x, y = placed.offset
        rooms.append(build_room(
            placed.layout, placed.name, damages, scope_items, pose=(x, y, 0.0),
        ))

    total_footprint = sum((r.floor_area_m2.value for r in rooms), 0.0)
    footprint_m2 = (
        Measurement(value=total_footprint, lo=total_footprint * 0.9,
                    hi=total_footprint * 1.1, confidence_level=0.9)
        if rooms else
        Measurement(value=0.0, lo=0.0, hi=0.0, confidence_level=0.0)
    )
    status = "ok" if rooms else "abstained"
    return Plan(
        capture=Capture(tier="photo", status=status, warnings=property_layout.warnings),
        property=PropertyBlock(
            rooms=[r.id for r in rooms], adjacency=[], footprint_m2=footprint_m2,
        ),
        rooms=rooms,
    )


def write_property_plan(
    property_layout: Any,
    out_path: Path,
    per_room_damage: dict[str, tuple[list[Damage], list[ScopeItem]]] | None = None,
) -> None:
    write_plan(build_property_plan(property_layout, per_room_damage), out_path)


def write_single_room_plan(
    tier: str,
    layout: RoomLayout,
    out_path: Path,
    damages: list[Damage] | None = None,
    scope_items: list[ScopeItem] | None = None,
    detected_openings: list[Any] | None = None,
) -> None:
    plan = build_single_room_plan(tier, layout, damages, scope_items, detected_openings)
    write_plan(plan, out_path)
