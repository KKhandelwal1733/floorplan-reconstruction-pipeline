"""Heuristic damage-candidate detection (Phase 10).

No trained classifier exists or is claimed here -- see config.py's comment
block above the DAMAGE_* constants. This flags dark, desaturated blobs that
deviate from a frame's own median brightness via HSV thresholding and
connected components -- a crude, explainable heuristic that WILL also flag
shadows, dark furniture, and normal wall texture as false positives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.config import (
    DAMAGE_MAX_BLOB_AREA_FRAC,
    DAMAGE_MIN_BLOB_AREA_FRAC,
    DAMAGE_SATURATION_THRESH,
    DAMAGE_VALUE_DROP_THRESH,
)


@dataclass
class DamageCandidate:
    frame_index: int
    bbox_px: tuple[int, int, int, int]   # x, y, w, h in frame pixels
    area_frac: float                      # fraction of frame area
    mean_value: float                     # mean HSV "V" (0-1, darkness indicator)
    mean_saturation: float                # mean HSV "S" (0-1)


def detect_damage_candidates(frame: np.ndarray, frame_index: int = 0) -> list[DamageCandidate]:
    """Find dark/discoloured blobs in a single BGR frame.

    Relative (not absolute) brightness threshold, since lighting varies
    hugely between captures -- a pixel is flagged only if it's notably
    darker than THIS frame's own median, and desaturated (stains/mould are
    usually duller than a lit, painted wall, not vividly coloured).
    """
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    median_v = float(np.median(v))
    mask = ((median_v - v) > DAMAGE_VALUE_DROP_THRESH) & (s < DAMAGE_SATURATION_THRESH)
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    frame_area = frame.shape[0] * frame.shape[1]

    candidates: list[DamageCandidate] = []
    for label in range(1, n_labels):  # label 0 is background
        x, y, w, h, area = stats[label]
        area_frac = area / frame_area
        if not (DAMAGE_MIN_BLOB_AREA_FRAC <= area_frac <= DAMAGE_MAX_BLOB_AREA_FRAC):
            continue
        region = labels == label
        candidates.append(DamageCandidate(
            frame_index=frame_index,
            bbox_px=(int(x), int(y), int(w), int(h)),
            area_frac=float(area_frac),
            mean_value=float(v[region].mean()),
            mean_saturation=float(s[region].mean()),
        ))
    return candidates


def detect_damage_candidates_multi(frames: list[np.ndarray]) -> list[DamageCandidate]:
    """Run detect_damage_candidates across several frames, tagging each
    result with its frame index."""
    candidates: list[DamageCandidate] = []
    for i, frame in enumerate(frames):
        candidates.extend(detect_damage_candidates(frame, frame_index=i))
    return candidates
