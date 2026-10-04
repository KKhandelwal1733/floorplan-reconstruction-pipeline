"""Photo tier pipeline (Phase 8): 2-8 unordered stills per room -> RoomLayout.

Reuses the exact same monocular SfM + scale-ensemble core as the video tier
(reconstruct_from_frames) -- a short photo set is treated like a very sparse
"video" (consecutive-pair matching over filename-sorted photos). Expect
materially worse accuracy than the video tier: far fewer views means far
less triangulated structure and a shakier scale estimate. No real photo-tier
data exists yet to calibrate this empirically (see config.py
PHOTO_EMPIRICAL_MIN_REL_HW) -- treat this tier's numbers with real caution
until validated against real captures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from roomscan.config import (
    PHOTO_EMPIRICAL_MIN_REL_HW,
    PHOTO_MIN_PHOTOS,
    PHOTO_MIN_PLANE_INLIERS,
    PHOTO_MIN_RECONSTRUCTED_PTS,
)
from roomscan.geometry.multi_room import PropertyLayout, stitch_rooms
from roomscan.geometry.room_layout import RoomLayout
from roomscan.geometry.video_tier import reconstruct_from_frames
from roomscan.io.photo_loader import list_room_dirs, load_photos


def process_room_photos(room_dir: Path) -> tuple[RoomLayout, dict[str, Any]]:
    """Reconstruct a single room's RoomLayout from its photo folder.

    Raises ValueError if reconstruction fails at any stage -- callers should
    catch this and degrade to a stub/warning rather than let it propagate.
    """
    photos = load_photos(room_dir)
    if len(photos) < PHOTO_MIN_PHOTOS:
        raise ValueError(f"only {len(photos)} photo(s) found in {room_dir}, need >= {PHOTO_MIN_PHOTOS}")

    layout, diagnostics = reconstruct_from_frames(
        photos, min_rel_hw=PHOTO_EMPIRICAL_MIN_REL_HW, detector="sift",
        strategy="best_pair", min_points=PHOTO_MIN_RECONSTRUCTED_PTS,
        min_plane_inliers=PHOTO_MIN_PLANE_INLIERS, calibration_tier="photo",
        tier_label="photo",
    )
    diagnostics["n_photos"] = diagnostics.pop("n_frames")
    diagnostics["room_dir"] = str(room_dir)
    return layout, diagnostics


def process_property(property_dir: Path) -> tuple[PropertyLayout, dict[str, Any]]:
    """Process every room subfolder of a multi-room property and stitch them.

    Per the "never crash, abstain instead" rule, a room that fails to
    reconstruct (insufficient photos, insufficient visual overlap) is
    skipped with a warning rather than aborting the whole property.
    """
    room_dirs = list_room_dirs(property_dir)
    results: list[tuple[str, RoomLayout, list]] = []
    failures: dict[str, str] = {}

    for room_dir in room_dirs:
        try:
            layout, _ = process_room_photos(room_dir)
            photos = load_photos(room_dir)
            results.append((room_dir.name, layout, photos))
        except ValueError as e:
            failures[room_dir.name] = str(e)

    property_layout = stitch_rooms(results)
    for name, reason in failures.items():
        property_layout.warnings.append(f"room '{name}' skipped: {reason}")

    diagnostics = {
        "n_rooms_found": len(room_dirs),
        "n_rooms_reconstructed": len(results),
        "n_rooms_failed": len(failures),
        "failures": failures,
    }
    return property_layout, diagnostics
