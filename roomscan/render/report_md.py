"""Per-capture report.md: a human-readable summary of a Plan, for the live
walk-in test (documented as a standard CLI output since Phase 1 -- see
CLAUDE.md -- but not implemented until now, see COMPLIANCE_MATRIX.md).

Pure presentation over the already-validated Plan (roomscan.schema_out);
carries no new data or claims beyond what plan.json already states.
"""
from __future__ import annotations

from pathlib import Path

from roomscan.schema_out import Plan


def _fmt_m(m) -> str:
    return f"{m.value:.2f} m (90% CI: {m.lo:.2f}-{m.hi:.2f})"


def _fmt_m2(m) -> str:
    return f"{m.value:.1f} m² (90% CI: {m.lo:.1f}-{m.hi:.1f})"


def render_report_md(plan: Plan) -> str:
    """Build the report.md text for a Plan. Pure function (no I/O) so tests
    can check content without touching the filesystem."""
    lines = [
        "# Floor Plan Report",
        "",
        f"**Tier:** {plan.capture.tier}  **Status:** {plan.capture.status}",
        "",
    ]

    if plan.capture.warnings:
        lines.append("## Warnings")
        lines.extend(f"- {w}" for w in plan.capture.warnings)
        lines.append("")

    if not plan.rooms:
        lines.append("No rooms reconstructed -- see Warnings above for why.")
        return "\n".join(lines) + "\n"

    lines.append("## Property")
    lines.append(f"- Footprint: {_fmt_m2(plan.property.footprint_m2)}")
    lines.append(f"- Rooms: {', '.join(plan.property.rooms) or '(none)'}")
    if plan.property.adjacency:
        lines.append("- Adjacency:")
        for adj in plan.property.adjacency:
            lines.append(f"  - {adj.a} <-> {adj.b} via {adj.via} (confidence {adj.confidence:.2f})")
    lines.append("")

    lines.append("## Rooms")
    for room in plan.rooms:
        lines.append(f"\n### {room.id}")
        lines.append(f"- Floor area: {_fmt_m2(room.floor_area_m2)}")
        lines.append(f"- Ceiling height: {_fmt_m(room.ceiling_height_m)}")
        lines.append(
            f"- Pose: x={room.pose.x:.2f}, y={room.pose.y:.2f}, "
            f"yaw={room.pose.yaw * 180 / 3.14159265:.0f} deg"
        )
        lines.append(f"- Walls: {len(room.walls)}")

        if room.openings:
            lines.append("- Openings:")
            for o in room.openings:
                lines.append(
                    f"  - {o.type} on {o.wall}: {o.width_m.value:.2f} m wide "
                    f"(confidence {o.confidence:.2f})"
                )
        else:
            lines.append("- Openings: none detected")

        if room.damage:
            lines.append(f"- Damage ({len(room.damage)} flagged, advisory -- human review required):")
            for d in room.damage:
                rule_ids = ", ".join(f.rule_id for f in d.flags) or "none"
                lines.append(
                    f"  - {d.class_} on {d.surface_id}: ~{d.area_m2.value:.2f} m² "
                    f"[rules: {rule_ids}]"
                )
        else:
            lines.append("- Damage: none flagged")

        if room.scope:
            lines.append("- Scope line items:")
            for s in room.scope:
                lines.append(f"  - {s.item}: {s.qty:g} {s.unit} on {s.surface_id}")

    return "\n".join(lines) + "\n"


def write_report(plan: Plan, out_path: Path) -> None:
    Path(out_path).write_text(render_report_md(plan), encoding="utf-8")
