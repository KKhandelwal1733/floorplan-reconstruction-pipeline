"""Capture-quality flags (case-study realignment, see COMPLIANCE_MATRIX.md):
mirror/glass/wet-look-surface/low-light detection in submitted frames.

Crude colour heuristics, same spirit as roomscan/damage/detector.py: flags
candidates for human review, not a trained classifier -- no labelled imagery
exists to train or validate one against. Never raises; a frame this can't
score is simply skipped (quality flagging is advisory, not core geometry).
"""
from __future__ import annotations

import numpy as np

from roomscan.config import (
    CAPTURE_QUALITY_GLARE_FRAC_THRESH,
    CAPTURE_QUALITY_GLARE_S_THRESH,
    CAPTURE_QUALITY_GLARE_V_THRESH,
    CAPTURE_QUALITY_LOW_LIGHT_V_THRESH,
    QUALITY_CI_WIDEN_FACTOR,
)
from roomscan.geometry.room_layout import RoomLayout, _widen


def detect_capture_quality_issues(frames: list[np.ndarray]) -> list[str]:
    """Scan a handful of BGR frames for low light and specular-glare
    (mirror/glass/wet-look surface) conditions.

    Low light: a frame's own mean brightness (HSV V) below threshold.
    Glare: fraction of near-saturated-bright + desaturated pixels above
    threshold -- light bouncing straight back at the camera (glass,
    mirrors, wet surfaces, direct reflections) blows out brightness while
    washing out colour, unlike a normally lit, saturated surface.

    Returns a list of human-readable warning strings (one entry per frame
    that trips a check), or [] if nothing was flagged or frames is empty.
    """
    import cv2

    warnings: list[str] = []
    for i, frame in enumerate(frames):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            v = hsv[:, :, 2] / 255.0
            s = hsv[:, :, 1] / 255.0

            mean_v = float(v.mean())
            if mean_v < CAPTURE_QUALITY_LOW_LIGHT_V_THRESH:
                warnings.append(
                    f"frame {i}: low light (mean brightness {mean_v:.2f} < "
                    f"{CAPTURE_QUALITY_LOW_LIGHT_V_THRESH}) -- geometry/damage "
                    "detection confidence may be reduced"
                )

            glare = (v > CAPTURE_QUALITY_GLARE_V_THRESH) & (s < CAPTURE_QUALITY_GLARE_S_THRESH)
            glare_frac = float(glare.mean())
            if glare_frac > CAPTURE_QUALITY_GLARE_FRAC_THRESH:
                warnings.append(
                    f"frame {i}: possible mirror/glass/wet-look surface "
                    f"(specular glare covers {glare_frac:.1%} of frame) -- "
                    "affected region's geometry/damage readings are unreliable"
                )
        except Exception:
            continue   # advisory check only -- never let this abstain a run
    return warnings


def apply_capture_quality_warnings(layout: RoomLayout, frames: list[np.ndarray]) -> list[str]:
    """Run detect_capture_quality_issues on frames and fold the result into
    layout in place: appends warnings, and -- same pattern as
    room_layout.py's own quality gate -- widens the floor-area/ceiling-height
    CIs and lowers quality_score when anything was flagged, rather than
    reporting a confident number over a mirror/glass/wet/low-light capture.

    Returns the warnings found (also already appended to
    layout.capture_warnings) so callers can print/log them.
    """
    issues = detect_capture_quality_issues(frames)
    if not issues:
        return issues

    layout.capture_warnings.extend(issues)
    layout.quality_score = min(layout.quality_score, 1.0 / QUALITY_CI_WIDEN_FACTOR)
    layout.floor_area_m2 = _widen(layout.floor_area_m2, QUALITY_CI_WIDEN_FACTOR)
    if layout.ceiling_height_m is not None:
        layout.ceiling_height_m = _widen(layout.ceiling_height_m, QUALITY_CI_WIDEN_FACTOR)
    return issues
