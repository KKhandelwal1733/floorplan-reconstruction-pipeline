"""Rule-based classification, flagging, and scope generation from detected
damage candidates (Phase 10).

Simple, explainable if/then rules -- no ML, no fabricated cost estimates
(no real pricing data exists to ground a cost figure in; see the project's
hard rule against fabricating numbers). Every generated flag/scope item
traces back to a concrete, inspectable signal via its evidence dict -- this
is advisory input for a human reviewer, not a diagnosis.
"""
from __future__ import annotations

from roomscan.config import (
    DAMAGE_CEILING_ZONE_FRAC,
    DAMAGE_FLOOR_ZONE_FRAC,
    DAMAGE_LARGE_STAIN_AREA_FRAC,
    DAMAGE_VERY_DARK_VALUE,
)
from roomscan.damage.detector import DamageCandidate
from roomscan.schema_out import DamageFlag, ScopeItem


def classify_damage(candidate: DamageCandidate, frame_height: int) -> str:
    """Heuristic class from the blob's vertical position in-frame.

    A crude proxy for floor/ceiling proximity, not a real 3-D projection
    (this pipeline doesn't build a colour-fused point cloud -- see
    pipeline.py's module docstring for why).
    """
    _, y, _, h = candidate.bbox_px
    center_frac = (y + h / 2) / frame_height
    if center_frac > (1 - DAMAGE_FLOOR_ZONE_FRAC):
        return "possible_water_staining"
    if center_frac < DAMAGE_CEILING_ZONE_FRAC:
        return "possible_ceiling_leak"
    return "discoloration"


def flags_for_damage(candidate: DamageCandidate, damage_class: str) -> list[DamageFlag]:
    """Rule evaluation -> DamageFlag list. Each rule is a concrete, stated
    if/then condition over the detector's own measured signals."""
    flags: list[DamageFlag] = []
    if damage_class == "possible_water_staining" and candidate.area_frac > DAMAGE_LARGE_STAIN_AREA_FRAC:
        flags.append(DamageFlag(
            rule_id="R001_large_floor_stain",
            text="Large discoloured area near floor level -- check for concealed "
                 "water damage behind baseboard/drywall, not just surface staining.",
            evidence={"area_frac": candidate.area_frac, "mean_value": candidate.mean_value},
        ))
    if candidate.mean_value < DAMAGE_VERY_DARK_VALUE:
        flags.append(DamageFlag(
            rule_id="R002_very_dark_region",
            text="Very dark, low-saturation region -- possible mould; recommend a "
                 "moisture-meter reading before remediation.",
            evidence={"mean_value": candidate.mean_value, "mean_saturation": candidate.mean_saturation},
        ))
    return flags


_SCOPE_TEXT = {
    "possible_water_staining": "Inspect and treat suspected water staining; "
                                "clean/reseal or replace affected surface section",
    "possible_ceiling_leak": "Inspect ceiling for active leak source; "
                              "patch/repaint affected area after drying",
    "discoloration": "Inspect and clean/repaint discoloured area",
}


def scope_for_damage(damage_class: str, area_m2: float, surface_id: str) -> list[ScopeItem]:
    """Damage class + area -> recommended scope line item. No cost figures:
    this project has no real pricing data to ground one in."""
    text = _SCOPE_TEXT.get(damage_class, _SCOPE_TEXT["discoloration"])
    return [ScopeItem(item=text, surface_id=surface_id, qty=round(area_m2, 2), unit="m2")]
