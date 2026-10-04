"""Ties candidate detection + rules into schema-ready Damage/ScopeItem lists
(Phase 10).

Honesty note on surface_id: this pipeline does NOT build a colour-fused 3-D
point cloud, so a detected 2-D blob can't be projected onto a specific wall
with real geometric confidence. surface_id here is a round-robin guess
(frame i -> wall i % n_walls), not a projection -- treat it as "roughly
which wall was probably in view for this frame", not a precise mapping.
Likewise, area_m2 converts the blob's pixel-area fraction to square metres
via the room's own average wall area as a rough proxy scale, not a true
per-wall measurement -- hence the wide, asymmetric CI (see config.py
DAMAGE_AREA_CI_LOW_MULT / DAMAGE_AREA_CI_HIGH_MULT).
"""
from __future__ import annotations

import numpy as np

from roomscan.config import DAMAGE_AREA_CI_HIGH_MULT, DAMAGE_AREA_CI_LOW_MULT
from roomscan.damage.detector import detect_damage_candidates
from roomscan.damage.rules import classify_damage, flags_for_damage, scope_for_damage
from roomscan.geometry.room_layout import RoomLayout
from roomscan.schema_out import Damage, Measurement, ScopeItem


def _avg_wall_area_m2(layout: RoomLayout) -> float:
    if not layout.walls:
        return layout.floor_area_m2.value
    ceiling_h = layout.ceiling_height_m.value if layout.ceiling_height_m else 2.4
    total_wall_length = sum(w.length_m.value for w in layout.walls)
    return (total_wall_length / len(layout.walls)) * ceiling_h


def detect_damage_for_room(
    frames: list[np.ndarray],
    layout: RoomLayout,
) -> tuple[list[Damage], list[ScopeItem]]:
    """Run damage detection across a room's available RGB frames, returning
    schema-ready (Damage, ScopeItem) lists. Never raises -- an empty frame
    list, or a frame with no candidates, just yields empty results (no
    damage found is a valid, common outcome, not an error)."""
    n_walls = max(len(layout.walls), 1)
    avg_wall_area = _avg_wall_area_m2(layout)

    damages: list[Damage] = []
    scope_items: list[ScopeItem] = []

    for frame_idx, frame in enumerate(frames):
        surface_id = f"wall_{frame_idx % n_walls}" if layout.walls else "unknown"
        candidates = detect_damage_candidates(frame, frame_index=frame_idx)

        for ci, candidate in enumerate(candidates):
            damage_class = classify_damage(candidate, frame.shape[0])
            area_est = candidate.area_frac * avg_wall_area
            area_m2 = Measurement(
                value=area_est,
                lo=area_est * DAMAGE_AREA_CI_LOW_MULT,
                hi=area_est * DAMAGE_AREA_CI_HIGH_MULT,
                confidence_level=0.9,
            )
            flags = flags_for_damage(candidate, damage_class)

            damages.append(Damage(
                id=f"dmg_{frame_idx}_{ci}",
                surface_id=surface_id,
                class_=damage_class,
                area_m2=area_m2,
                extent={"frame_index": frame_idx, "bbox_px": list(candidate.bbox_px)},
                flags=flags,
            ))
            scope_items.extend(scope_for_damage(damage_class, area_est, surface_id))

    return damages, scope_items
