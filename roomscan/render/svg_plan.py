"""SVG floor-plan renderer.

render_plan() produces a dimensioned floor-plan SVG from a RoomLayout.
render_stub() is kept for CLI fallback when geometry has not run yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomscan.geometry.multi_room import PropertyLayout
    from roomscan.geometry.room_layout import RoomLayout

_PAD = 60          # px margin around the drawing
_PAGE_W = 900      # px
_PAGE_H = 700      # px


def render_stub(out_path: Path, message: str) -> None:
    """Minimal valid SVG placeholder."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="10" y="60" font-family="sans-serif" font-size="14" fill="black">'
        f'{message}</text></svg>'
    )
    Path(out_path).write_text(svg, encoding="utf-8")


def render_plan(layout: "RoomLayout", out_path: Path) -> None:
    """Render a dimensioned floor-plan SVG from a RoomLayout."""
    poly = layout.polygon
    if not poly:
        render_stub(out_path, "No floor polygon detected.")
        return

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    room_w = max_x - min_x or 1.0
    room_h = max_y - min_y or 1.0

    draw_w = _PAGE_W - 2 * _PAD
    draw_h = _PAGE_H - 2 * _PAD
    scale = min(draw_w / room_w, draw_h / room_h)

    def tx(x: float) -> float:
        return _PAD + (x - min_x) * scale

    def ty(y: float) -> float:
        return _PAGE_H - _PAD - (y - min_y) * scale

    # Polygon points string
    pts_str = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in poly)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_PAGE_W}" height="{_PAGE_H}" '
        f'font-family="sans-serif" font-size="11">',
        '<rect width="100%" height="100%" fill="#f8f8f8"/>',
        # Floor polygon
        f'<polygon points="{pts_str}" fill="#dce8f5" stroke="#2060a0" stroke-width="2"/>',
    ]

    # Wall labels
    for wall in layout.walls:
        mx = (wall.p0[0] + wall.p1[0]) / 2
        my = (wall.p0[1] + wall.p1[1]) / 2
        label = f"{wall.length_m.value:.2f} m"
        lines.append(
            f'<text x="{tx(mx):.1f}" y="{ty(my):.1f}" '
            f'text-anchor="middle" fill="#204080" font-size="10">{label}</text>'
        )

    # Stats block
    area = layout.floor_area_m2.value
    stats = [f"Floor area: {area:.1f} m&#178;  (±{(layout.floor_area_m2.hi - area):.2f} m&#178;)"]
    if layout.ceiling_height_m:
        h = layout.ceiling_height_m.value
        stats.append(f"Ceiling height: {h:.2f} m")
    else:
        stats.append("Ceiling: not captured")

    for i, s in enumerate(stats):
        lines.append(
            f'<text x="{_PAD}" y="{20 + i * 18}" fill="#333" font-size="13">{s}</text>'
        )

    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def render_property(property_layout: "PropertyLayout", out_path: Path) -> None:
    """Render a schematic multi-room property SVG (Phase 8).

    Each room's own polygon is drawn at its own (independently reconstructed)
    shape and size, placed at the grid offset computed by
    geometry/multi_room.py. This is NOT a geometrically precise stitch --
    see that module's docstring for why relative room position/orientation
    isn't determined by this method.
    """
    rooms = property_layout.rooms
    if not rooms:
        render_stub(out_path, "No rooms reconstructed.")
        return

    all_xs: list[float] = []
    all_ys: list[float] = []
    for room in rooms:
        ox, oy = room.offset
        for x, y in room.layout.polygon:
            all_xs.append(x + ox)
            all_ys.append(y + oy)
    if not all_xs:
        render_stub(out_path, "No room polygons to render.")
        return

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    total_w = max_x - min_x or 1.0
    total_h = max_y - min_y or 1.0

    draw_w = _PAGE_W - 2 * _PAD
    draw_h = _PAGE_H - 2 * _PAD
    scale = min(draw_w / total_w, draw_h / total_h)

    def tx(x: float) -> float:
        return _PAD + (x - min_x) * scale

    def ty(y: float) -> float:
        return _PAGE_H - _PAD - (y - min_y) * scale

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_PAGE_W}" height="{_PAGE_H}" '
        f'font-family="sans-serif" font-size="11">',
        '<rect width="100%" height="100%" fill="#f8f8f8"/>',
    ]

    total_area = 0.0
    for room in rooms:
        ox, oy = room.offset
        poly = room.layout.polygon
        pts_str = " ".join(f"{tx(x + ox):.1f},{ty(y + oy):.1f}" for x, y in poly)
        lines.append(
            f'<polygon points="{pts_str}" fill="#dce8f5" stroke="#2060a0" stroke-width="2"/>'
        )
        cx = sum(x for x, _ in poly) / len(poly) + ox
        cy = sum(y for _, y in poly) / len(poly) + oy
        area = room.layout.floor_area_m2.value
        total_area += area
        lines.append(
            f'<text x="{tx(cx):.1f}" y="{ty(cy):.1f}" text-anchor="middle" '
            f'fill="#204080" font-size="12">{room.name} ({area:.1f} m&#178;)</text>'
        )

    stats = [f"Property: {len(rooms)} room(s), total floor area {total_area:.1f} m&#178; (schematic layout)"]
    for i, s in enumerate(stats):
        lines.append(
            f'<text x="{_PAD}" y="{20 + i * 18}" fill="#333" font-size="13">{s}</text>'
        )

    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
