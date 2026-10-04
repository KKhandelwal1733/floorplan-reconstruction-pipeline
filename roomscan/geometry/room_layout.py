"""Extract room geometry (floor polygon, area, ceiling height, walls) from a
fused point cloud.

Output is a RoomLayout dataclass that carries both the raw geometry and the
schema-ready Measurement objects for plan.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
    up_axis: np.ndarray           # unit vector pointing up in world coords
    floor_d: float                 # plane offset: floor is {x: up_axis@x = -floor_d}
    ceiling_d: float | None        # None → ceiling not captured
    polygon: list[tuple[float, float]]   # floor outline in 2-D (metres)
    floor_area_m2: Measurement
    ceiling_height_m: Measurement | None
    walls: list[WallSegment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_2d(pts3: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Project 3-D points onto the plane perpendicular to `up`.

    Returns (N, 2) array in an arbitrary but consistent 2-D coordinate system.
    """
    # pick two in-plane orthonormal vectors
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(up, ref)
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    v /= np.linalg.norm(v)
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


def _polygon_area(verts: list[tuple[float, float]]) -> float:
    """Shoelace formula."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


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
) -> RoomLayout:
    """Given a fused (N,3) point cloud, return the room layout.

    Args:
        pts:             World-frame point cloud from load_scan().
        floor_subsample: How many floor inliers to use for polygon computation.
    """
    gravity = detect_gravity(pts)
    floor, ceiling = find_floor_ceiling(pts, gravity=gravity)

    if floor is None:
        raise ValueError("Could not detect floor plane in point cloud.")

    floor_normal, floor_d_val, floor_mask = floor

    # Use detected gravity as up direction (more stable than RANSAC normal alone)
    up = gravity.copy()

    # Ensure floor_normal points in the same half-space as gravity
    if float(floor_normal @ up) < 0:
        floor_normal = -floor_normal
        floor_d_val = -floor_d_val

    # Floor height measurement (positions of inliers projected onto up)
    floor_heights = (pts[floor_mask] @ up)
    floor_height_m = measurement_from_inliers(floor_heights)

    # Ceiling
    ceil_d: float | None = None
    ceil_height_m: Measurement | None = None
    if ceiling is not None:
        _ceil_normal, ceil_d_raw, ceil_mask = ceiling
        ceil_heights = (pts[ceil_mask] @ up)
        ceil_pos_m = measurement_from_inliers(ceil_heights)
        room_height = ceil_pos_m.value - floor_height_m.value
        ceil_height_m = Measurement(
            value=abs(room_height),
            lo=abs(room_height) - 0.05,
            hi=abs(room_height) + 0.05,
            confidence_level=0.9,
        )
        ceil_d = ceil_d_raw

    # Floor polygon: project floor inliers into 2-D
    floor_pts = pts[floor_mask]
    if len(floor_pts) > floor_subsample:
        idx = np.random.default_rng(0).choice(len(floor_pts), floor_subsample,
                                               replace=False)
        floor_pts = floor_pts[idx]

    pts2d = _to_2d(floor_pts, up)
    hull = _convex_hull_2d(pts2d)

    area = _polygon_area(hull)
    area_m = Measurement(value=area, lo=area * 0.97, hi=area * 1.03,
                         confidence_level=0.9)

    walls = _wall_segments(hull)

    return RoomLayout(
        up_axis=up,
        floor_d=floor_d_val,
        ceiling_d=ceil_d,
        polygon=hull,
        floor_area_m2=area_m,
        ceiling_height_m=ceil_height_m,
        walls=walls,
    )
