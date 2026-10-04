"""Scale-ensemble metric recovery for monocular video reconstruction (Phase 7).

Monocular SfM is scale-ambiguous: the sparse cloud from sfm.py is in an
arbitrary unit (assumed roughly-constant per-sample camera step length).
This estimates several INDEPENDENT candidate scale factors from physical
priors; their spread becomes the confidence interval rather than a single
unverifiable number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.config import (
    VIDEO_ASSUMED_CAMERA_HEIGHT_M,
    VIDEO_ASSUMED_CEILING_M,
    VIDEO_ASSUMED_DIAGONAL_M,
    VIDEO_DEFAULT_SCALE_REL_HW,
    VIDEO_MIN_RECONSTRUCTED_PTS,
    VIDEO_PLANE_THRESH_FRAC,
)
from roomscan.geometry.planes import detect_gravity, find_floor_ceiling
from roomscan.geometry.sfm import FramePose


@dataclass
class ScaleEstimate:
    factor: float
    source: str


def estimate_scale_ensemble(
    pts_unscaled: np.ndarray,
    poses: list[FramePose] | None = None,
    min_points: int = VIDEO_MIN_RECONSTRUCTED_PTS,
) -> list[ScaleEstimate]:
    """Return independent scale-factor candidates (unscaled-units -> metres)."""
    estimates: list[ScaleEstimate] = []
    if len(pts_unscaled) < min_points:
        return estimates

    # (1) room-footprint-diagonal prior: crude but always available.
    bbox_min, bbox_max = pts_unscaled.min(axis=0), pts_unscaled.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag > 1e-6:
        estimates.append(ScaleEstimate(VIDEO_ASSUMED_DIAGONAL_M / diag, "room_diagonal_prior"))

    gravity = None
    floor = ceiling = None
    try:
        gravity = detect_gravity(pts_unscaled)
        # The unscaled cloud has no metric units yet, so the plane-fit
        # threshold must scale with the cloud's own extent rather than use
        # the fixed-metre default (config.RANSAC_THRESH_M) tuned for LiDAR data.
        plane_thresh = VIDEO_PLANE_THRESH_FRAC * diag
        floor, ceiling = find_floor_ceiling(pts_unscaled, gravity=gravity, thresh=plane_thresh)
    except Exception:
        pass

    # (2) ceiling-height prior: only if both floor and ceiling planes are found.
    if floor is not None and ceiling is not None:
        floor_h = float(np.median((pts_unscaled[floor[2]] @ gravity)))
        ceil_h = float(np.median((pts_unscaled[ceiling[2]] @ gravity)))
        room_h = abs(ceil_h - floor_h)
        if room_h > 1e-6:
            estimates.append(ScaleEstimate(VIDEO_ASSUMED_CEILING_M / room_h, "ceiling_height_prior"))

    # (3) camera-to-floor height prior: needs a floor plane and camera poses.
    if floor is not None and poses:
        floor_h = float(np.median((pts_unscaled[floor[2]] @ gravity)))
        cam_heights = [abs(float(p.t @ gravity) - floor_h) for p in poses]
        avg_h = float(np.median(cam_heights))
        if avg_h > 1e-6:
            estimates.append(ScaleEstimate(VIDEO_ASSUMED_CAMERA_HEIGHT_M / avg_h, "camera_height_prior"))

    return estimates


def apply_scale(pts_unscaled: np.ndarray, factor: float) -> np.ndarray:
    return pts_unscaled * factor


def combine_estimates(estimates: list[ScaleEstimate]) -> tuple[float, float]:
    """Return (median_scale, relative_half_width) from ensemble spread.

    A single estimate has no internal disagreement signal to measure, so it
    falls back to a fixed conservative relative half-width instead of
    claiming false precision.
    """
    if not estimates:
        raise ValueError("no scale estimates available")
    factors = np.array([e.factor for e in estimates])
    median = float(np.median(factors))
    if len(factors) > 1:
        rel_hw = float((factors.max() - factors.min()) / 2 / median)
    else:
        rel_hw = VIDEO_DEFAULT_SCALE_REL_HW
    return median, rel_hw
