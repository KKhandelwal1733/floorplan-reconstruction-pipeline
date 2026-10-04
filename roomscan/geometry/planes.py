"""Plane fitting utilities used by room_layout.py.

Convention: planes represented as (normal, d) where normal @ x + d = 0,
normal is a unit vector.
"""
from __future__ import annotations

import numpy as np

from roomscan.config import CEIL_MIN_FILL_RATIO, GRAVITY_SAMPLE, SEED

_CANONICAL = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]


# ---------------------------------------------------------------------------
# Gravity auto-detection
# ---------------------------------------------------------------------------

def detect_gravity(pts: np.ndarray, n_sample: int = GRAVITY_SAMPLE) -> np.ndarray:
    """Estimate the world-up direction as the canonical axis whose bottom-5%
    slice is flattest (smallest std of residuals from a constant-height plane).

    Returns a unit vector such that pts @ gravity gives height.
    """
    if len(pts) > n_sample:
        idx = np.random.default_rng(SEED).choice(len(pts), n_sample, replace=False)
        sub = pts[idx]
    else:
        sub = pts

    best_score = np.inf
    best_axis = np.array([0., 1., 0.])   # ARKit Y-up as default

    for ax in _CANONICAL:
        proj = sub @ ax
        thresh = float(np.percentile(proj, 12))
        # <=, not <: a perfectly flat axis (e.g. floor-only scan) has every
        # value tied at the percentile, so strict '<' would wrongly yield an
        # empty slice and skip the very axis that should win.
        bottom = proj[proj <= thresh]
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
    rng = np.random.default_rng(SEED)

    for _ in range(n_iter):
        idx = rng.choice(len(pts), 3, replace=False)
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
    min_inliers: int = 200,
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
        idx = np.random.default_rng(SEED).choice(len(pts), subsample, replace=False)
        sub = pts[idx]
    else:
        sub = pts.copy()

    planes: list[tuple[np.ndarray, float, np.ndarray]] = []
    used = np.zeros(len(sub), dtype=bool)

    for _ in range(4):  # find up to 4 horizontal planes
        available = ~used
        if available.sum() < min_inliers:
            break
        result = ransac_horizontal_plane(
            sub[available], gravity, thresh=thresh, min_inliers=min_inliers
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

    ceiling = None
    if len(full_planes) > 1:
        # Reject ceiling candidates whose 2D inlier coverage looks like wall-tops
        # (perimeter-only ring) rather than an actual ceiling (interior filled).
        ceil_candidate = full_planes[-1]
        fill = _ceiling_fill_ratio(pts[ceil_candidate[2]], gravity)
        if fill >= CEIL_MIN_FILL_RATIO:
            ceiling = (ceil_candidate[0], ceil_candidate[1], ceil_candidate[2])

    return floor, ceiling


def _ceiling_fill_ratio(
    inlier_pts: np.ndarray,
    gravity: np.ndarray,
    cell: float = 0.50,
) -> float:
    """Fraction of coarse 2-D grid cells occupied by ceiling inlier points.

    A real ceiling fills the interior; wall-top rings only fill the perimeter,
    giving a much lower ratio.
    """
    ref = np.array([1., 0., 0.]) if abs(float(gravity[0])) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(gravity, ref); u /= np.linalg.norm(u)
    v = np.cross(gravity, u)
    pts2 = np.stack([inlier_pts @ u, inlier_pts @ v], axis=1)
    u_min, v_min = pts2.min(axis=0)
    u_max, v_max = pts2.max(axis=0)
    span_u = max(float(u_max - u_min), cell)
    span_v = max(float(v_max - v_min), cell)
    nu = max(1, int(np.ceil(span_u / cell)))
    nv = max(1, int(np.ceil(span_v / cell)))
    ui = np.clip(((pts2[:, 0] - u_min) / cell).astype(int), 0, nu - 1)
    vi = np.clip(((pts2[:, 1] - v_min) / cell).astype(int), 0, nv - 1)
    grid = np.zeros((nv, nu), dtype=bool)
    grid[vi, ui] = True
    return float(grid.sum()) / (nv * nu)
