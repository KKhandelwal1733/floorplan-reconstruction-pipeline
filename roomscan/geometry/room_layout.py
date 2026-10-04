"""Extract room geometry (floor polygon, area, ceiling height, walls) from a
fused point cloud.

Output is a RoomLayout dataclass that carries both the raw geometry and the
schema-ready Measurement objects for plan.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roomscan.config import (
    CEIL_UNOBSERVED_HALF_WIDTH_M,
    CEIL_UNOBSERVED_MARGIN_M,
    QUALITY_CI_WIDEN_FACTOR,
    QUALITY_MIN_FLOOR_INLIERS,
    QUALITY_MIN_PTS_PER_M2,
    WALL_MERGE_ANGLE_DEG,
)
from roomscan.geometry.planes import detect_gravity, find_floor_ceiling
from roomscan.geometry.uncertainty import measurement_from_inliers
from roomscan.schema_out import Measurement


@dataclass
class WallSegment:
    """One wall: start/end in 2-D floor-plane coords (metres)."""
    p0: tuple[float, float]
    p1: tuple[float, float]
    length_m: Measurement


@dataclass
class RoomLayout:
    """All geometry extracted from a single-room scan."""
    up_axis: np.ndarray                  # unit vector pointing up in world coords
    floor_d: float                        # plane offset: floor is {x: up_axis@x = -floor_d}
    ceiling_d: float | None              # None → ceiling not captured
    polygon: list[tuple[float, float]]   # floor outline in 2-D (metres)
    floor_area_m2: Measurement
    ceiling_height_m: Measurement | None
    walls: list[WallSegment] = field(default_factory=list)
    # 3-D basis for back-projecting 2-D polygon points (set by extract_layout)
    plan_u: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    plan_v: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    floor_level: float = 0.0             # median floor height in up_axis direction
    ceiling_unobserved: bool = False     # True when ceiling plane not detected
    capture_warnings: list[str] = field(default_factory=list)
    quality_score: float = 1.0           # 1.0 = full confidence; <1.0 -> CIs were widened


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plan_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two in-plane orthonormal 3-D vectors perpendicular to `up`."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(up, ref); u /= np.linalg.norm(u)
    v = np.cross(up, u)
    return u, v


def _to_2d(pts3: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Project 3-D points onto the floor plane → (N, 2)."""
    u, v = _plan_basis(up)
    return np.stack([pts3 @ u, pts3 @ v], axis=1)


def _convex_hull_2d(pts2: np.ndarray) -> list[tuple[float, float]]:
    """Graham-scan convex hull.  Returns vertices in CCW order."""
    pts = pts2.tolist()
    pts.sort(key=lambda p: (p[0], p[1]))
    pts = [tuple(p) for p in pts]

    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower: list[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _simplify_polygon(
    verts: list[tuple[float, float]],
    angle_tol_deg: float = WALL_MERGE_ANGLE_DEG,
) -> list[tuple[float, float]]:
    """Merge near-collinear hull vertices so each real wall is one straight run.

    Raw convex hulls of noisy point clouds have dozens of micro-segments per
    physical wall; this collapses runs that turn by less than `angle_tol_deg`.
    """
    if len(verts) <= 3:
        return verts
    tol = np.deg2rad(angle_tol_deg)
    pts = list(verts)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        n = len(pts)
        for i in range(n):
            prev = np.array(pts[(i - 1) % n])
            cur = np.array(pts[i])
            nxt = np.array(pts[(i + 1) % n])
            v1, v2 = cur - prev, nxt - cur
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos_a = np.clip((v1 @ v2) / (n1 * n2), -1.0, 1.0)
            if np.arccos(cos_a) < tol:
                pts.pop(i)
                changed = True
                break
    return pts


def _polygon_area(verts: list[tuple[float, float]]) -> float:
    """Shoelace formula."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _widen(m: Measurement, factor: float) -> Measurement:
    """Scale a Measurement's CI half-width by `factor`, keeping value fixed."""
    half = (m.hi - m.lo) / 2 * factor
    return Measurement(value=m.value, lo=m.value - half, hi=m.value + half,
                        confidence_level=m.confidence_level)


def _wall_segments(hull: list[tuple[float, float]]) -> list[WallSegment]:
    walls = []
    n = len(hull)
    for i in range(n):
        p0 = hull[i]
        p1 = hull[(i + 1) % n]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = float(np.hypot(dx, dy))
        # CI: ±1 cm for clean LiDAR walls (no inlier residuals at this stage)
        # ponytail: simplified 1 cm uncertainty until per-wall RANSAC lands
        m = Measurement(value=length, lo=length - 0.01, hi=length + 0.01,
                        confidence_level=0.9)
        walls.append(WallSegment(p0=p0, p1=p1, length_m=m))
    return walls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_layout(
    pts: np.ndarray,
    floor_subsample: int = 20_000,
    min_plane_inliers: int = 200,
) -> RoomLayout:
    """Given a fused (N,3) point cloud, return the room layout.

    Args:
        pts:               World-frame point cloud from load_scan().
        floor_subsample:   How many floor inliers to use for polygon computation.
        min_plane_inliers: Minimum RANSAC inliers to accept a floor/ceiling
                            plane. The default (200) assumes a dense LiDAR/
                            video point cloud; a sparse photo-tier best-pair
                            reconstruction (tens of points) needs this much
                            lower to find any plane at all -- callers using
                            that sparse a cloud should lower it explicitly
                            (and expect correspondingly low-confidence output).
    """
    gravity = detect_gravity(pts)
    floor, ceiling = find_floor_ceiling(pts, gravity=gravity, min_inliers=min_plane_inliers)

    if floor is None:
        raise ValueError("Could not detect floor plane in point cloud.")

    floor_normal, floor_d_val, floor_mask = floor
    warnings: list[str] = []

    # Use detected gravity as up direction (more stable than RANSAC normal alone)
    up = gravity.copy()

    # Ensure floor_normal points in the same half-space as gravity
    if float(floor_normal @ up) < 0:
        floor_normal = -floor_normal
        floor_d_val = -floor_d_val

    # Floor height measurement (positions of inliers projected onto up)
    floor_heights = (pts[floor_mask] @ up)
    floor_height_m = measurement_from_inliers(floor_heights)
    floor_level = float(floor_height_m.value)

    # Ceiling — Phase 5: fall back to wide-CI estimate when ceiling not detected
    ceil_d: float | None = None
    ceil_height_m: Measurement | None = None
    ceiling_unobserved = False

    if ceiling is not None:
        _ceil_normal, ceil_d_raw, ceil_mask = ceiling
        ceil_heights = (pts[ceil_mask] @ up)
        ceil_pos_m = measurement_from_inliers(ceil_heights)
        room_height = ceil_pos_m.value - floor_level
        ceil_height_m = Measurement(
            value=abs(room_height),
            lo=abs(room_height) - 0.05,
            hi=abs(room_height) + 0.05,
            confidence_level=0.9,
        )
        ceil_d = ceil_d_raw
    else:
        # Use 98th-percentile of observed heights as an upper-bound estimate
        obs_top = float(np.percentile(pts @ up, 98)) - floor_level
        est_h = obs_top + CEIL_UNOBSERVED_MARGIN_M
        hw = CEIL_UNOBSERVED_HALF_WIDTH_M
        # Keep the interval symmetric even if lo goes negative: a near-zero
        # estimate (e.g. a floor-only scan with no height signal at all) is
        # exactly the case that most needs the full configured half-width —
        # clamping lo to 0 would silently shrink it below CEIL_UNOBSERVED_HALF_WIDTH_M.
        ceil_height_m = Measurement(
            value=est_h,
            lo=est_h - hw,
            hi=est_h + hw,
            confidence_level=0.9,
        )
        ceiling_unobserved = True
        warnings.append(
            "ceiling plane not detected - height estimated from point-cloud upper "
            f"bound ({obs_top:.2f} m obs + {CEIL_UNOBSERVED_MARGIN_M} m margin); "
            f"CI is +/-{hw} m"
        )

    # Floor polygon: project floor inliers into 2-D
    plan_u, plan_v = _plan_basis(up)
    floor_pts = pts[floor_mask]
    if len(floor_pts) > floor_subsample:
        idx = np.random.default_rng(0).choice(len(floor_pts), floor_subsample,
                                               replace=False)
        floor_pts = floor_pts[idx]

    pts2d = np.stack([floor_pts @ plan_u, floor_pts @ plan_v], axis=1)
    hull = _convex_hull_2d(pts2d)
    hull = _simplify_polygon(hull)

    area = _polygon_area(hull)
    area_m = Measurement(value=area, lo=area * 0.97, hi=area * 1.03,
                         confidence_level=0.9)

    walls = _wall_segments(hull)

    # Quality gate: flag and widen CIs when coverage looks partial/sparse rather
    # than silently reporting a confident number for an incomplete scan.
    floor_inlier_count = int(floor_mask.sum())
    pts_per_m2 = floor_inlier_count / max(area, 1e-6)
    quality_score = 1.0
    if floor_inlier_count < QUALITY_MIN_FLOOR_INLIERS:
        quality_score = min(quality_score, floor_inlier_count / QUALITY_MIN_FLOOR_INLIERS)
        warnings.append(
            f"low floor point count ({floor_inlier_count} < {QUALITY_MIN_FLOOR_INLIERS}) "
            "- CIs widened, possible partial scan coverage"
        )
    if pts_per_m2 < QUALITY_MIN_PTS_PER_M2:
        quality_score = min(quality_score, pts_per_m2 / QUALITY_MIN_PTS_PER_M2)
        warnings.append(
            f"low point density ({pts_per_m2:.0f} pts/sq m < {QUALITY_MIN_PTS_PER_M2}) "
            "- CIs widened, possible partial scan coverage"
        )
    quality_score = max(0.0, min(1.0, quality_score))

    if quality_score < 1.0:
        area_m = _widen(area_m, QUALITY_CI_WIDEN_FACTOR)
        if ceil_height_m is not None:
            ceil_height_m = _widen(ceil_height_m, QUALITY_CI_WIDEN_FACTOR)

    return RoomLayout(
        up_axis=up,
        floor_d=floor_d_val,
        ceiling_d=ceil_d,
        polygon=hull,
        floor_area_m2=area_m,
        ceiling_height_m=ceil_height_m,
        walls=walls,
        plan_u=plan_u,
        plan_v=plan_v,
        floor_level=floor_level,
        ceiling_unobserved=ceiling_unobserved,
        capture_warnings=warnings,
        quality_score=quality_score,
    )
