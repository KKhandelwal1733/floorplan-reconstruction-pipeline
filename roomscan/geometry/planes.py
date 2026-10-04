"""Plane fitting utilities used by room_layout.py.

Convention: planes represented as (normal, d) where normal @ x + d = 0,
normal is a unit vector.
"""
from __future__ import annotations

import numpy as np

from roomscan.config import SEED

_RNG = np.random.default_rng(SEED)
_CANONICAL = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]


# ---------------------------------------------------------------------------
# Gravity auto-detection
# ---------------------------------------------------------------------------

def detect_gravity(pts: np.ndarray, n_sample: int = 8_000) -> np.ndarray:
    """Estimate the world-up direction as the canonical axis whose bottom-5%
    slice is flattest (smallest std of residuals from a constant-height plane).

    Returns a unit vector such that pts @ gravity gives height.
    """
    if len(pts) > n_sample:
        idx = _RNG.choice(len(pts), n_sample, replace=False)
        sub = pts[idx]
    else:
        sub = pts

    best_score = np.inf
    best_axis = np.array([0., 1., 0.])   # ARKit Y-up as default

    for ax in _CANONICAL:
        proj = sub @ ax
        thresh = float(np.percentile(proj, 12))
        bottom = proj[proj < thresh]
        if len(bottom) < 30:
            continue
        score = float(bottom.std())
        if score < best_score:
            best_score = score
            best_axis = ax.copy()

    return best_axis


# ---------------------------------------------------------------------------
# RANSAC biased toward planes perpendicular to gravity
# ---------------------------------------------------------------------------

def ransac_horizontal_plane(
    pts: np.ndarray,
    gravity: np.ndarray,
    n_iter: int = 500,
    thresh: float = 0.05,
    horiz_tol: float = 0.30,      # radians: max angle from gravity direction
    min_inliers: int = 200,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """RANSAC for a nearly-horizontal plane.

    Only accepts candidate planes whose normal is within `horiz_tol` radians
    of `gravity`.  Returns (unit_normal, d, inlier_mask) or None.
    """
    cos_tol = float(np.cos(horiz_tol))
    best_count = min_inliers - 1
    best: tuple[np.ndarray, float, np.ndarray] | None = None

    for _ in range(n_iter):
        idx = _RNG.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            continue
        n /= norm
        # Skip if not near-horizontal
        if abs(float(n @ gravity)) < cos_tol:
            continue
        d = float(-n @ p0)
        dists = np.abs(pts @ n + d)
        mask = dists < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best = (n.copy(), d, mask)

    if best is None:
        return None

    # Refit on inliers
    n, d, mask = best
    inlier_pts = pts[mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    n = Vt[-1]
    if float(n @ gravity) < 0:
        n = -n
    d = float(-n @ centroid)
    mask = np.abs(pts @ n + d) < thresh
    return n, d, mask


# ---------------------------------------------------------------------------
# Floor / ceiling detection
# ---------------------------------------------------------------------------

def find_floor_ceiling(
    pts: np.ndarray,
    gravity: np.ndarray | None = None,
    subsample: int = 50_000,
    thresh: float = 0.05,
) -> tuple[
    tuple[np.ndarray, float, np.ndarray] | None,
    tuple[np.ndarray, float, np.ndarray] | None,
]:
    """Detect floor and ceiling horizontal planes.

    Returns (floor, ceiling): each is (normal, d, inlier_mask on full pts)
    or None if not detected.  Floor has the lower median height; ceiling higher.
    """
    if gravity is None:
        gravity = detect_gravity(pts)

    if len(pts) > subsample:
        idx = _RNG.choice(len(pts), subsample, replace=False)
        sub = pts[idx]
    else:
        sub = pts.copy()

    planes: list[tuple[np.ndarray, float, np.ndarray]] = []
    used = np.zeros(len(sub), dtype=bool)

    for _ in range(4):  # find up to 4 horizontal planes
        available = ~used
        if available.sum() < 500:
            break
        result = ransac_horizontal_plane(
            sub[available], gravity, thresh=thresh
        )
        if result is None:
            break
        n, d, mask_avail = result
        # Map mask back to sub indices
        avail_idx = np.where(available)[0]
        sub_mask = np.zeros(len(sub), dtype=bool)
        sub_mask[avail_idx[mask_avail]] = True
        used |= sub_mask
        planes.append((n, d, sub_mask))

    if not planes:
        return None, None

    # Build full-cloud inlier masks
    full_planes = []
    for n, d, _ in planes:
        full_mask = np.abs(pts @ n + d) < thresh
        median_h = float(np.median((pts @ gravity)[full_mask]))
        full_planes.append((n, d, full_mask, median_h))

    full_planes.sort(key=lambda p: p[3])   # ascending by height

    floor = (full_planes[0][0], full_planes[0][1], full_planes[0][2])
    ceiling = (
        (full_planes[-1][0], full_planes[-1][1], full_planes[-1][2])
        if len(full_planes) > 1 else None
    )
    return floor, ceiling
