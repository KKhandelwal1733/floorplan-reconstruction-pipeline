"""Lightweight monocular structure-from-motion for the video tier (Phase 7).

No depth, no poses, no known camera intrinsics -- everything here is a
CPU-only approximation, not a full bundle-adjusted SfM pipeline:

  - Intrinsics: assumed horizontal FOV (config.VIDEO_ASSUMED_FOV_DEG), since
    the true focal length of an arbitrary handheld video is unknown.
  - Pose chaining assumes roughly constant inter-sample step length (a
    walkaround capture at fairly steady pace), since each pairwise essential-
    matrix decomposition only recovers a unit-norm relative translation
    (monocular scale ambiguity) -- full bundle adjustment / per-step scale
    recovery is out of scope for this phase.
  - Absolute metric scale is NOT resolved here; see geometry/scale_ensemble.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.config import (
    VIDEO_ASSUMED_FOV_DEG,
    VIDEO_MATCH_RATIO,
    VIDEO_MAX_POINT_DIST,
    VIDEO_MIN_MATCHES,
    VIDEO_ORB_FEATURES,
    VIDEO_OUTLIER_MEDIAN_MULT,
    VIDEO_RANSAC_PROB,
    VIDEO_RANSAC_THRESH_PX,
)


def estimate_intrinsics(width: int, height: int, fov_deg: float = VIDEO_ASSUMED_FOV_DEG) -> np.ndarray:
    """Assume a horizontal FOV and build a pinhole K (true focal length of an
    arbitrary handheld video is unknown)."""
    fx = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])


@dataclass
class FramePose:
    R: np.ndarray   # 3x3, world-from-camera rotation
    t: np.ndarray   # 3, world-from-camera translation (unknown overall scale)


def _match_orb(img_a: np.ndarray, img_b: np.ndarray):
    """ORB feature match with Lowe's ratio test. Returns (pts_a, pts_b) pixel
    coordinate arrays or None if too few good matches."""
    import cv2

    orb = cv2.ORB_create(nfeatures=VIDEO_ORB_FEATURES)
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)
    if des_a is None or des_b is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < VIDEO_MATCH_RATIO * n.distance]
    if len(good) < VIDEO_MIN_MATCHES:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def reconstruct_sparse(frames: list[np.ndarray]) -> tuple[np.ndarray, list[FramePose]]:
    """Chain consecutive-frame two-view reconstructions into one sparse cloud.

    Returns (points (N,3) in an UNKNOWN-SCALE world frame anchored at frame 0,
    poses per sampled frame). Points combine triangulated matches from every
    consecutive pair, assuming roughly equal inter-sample step length.
    """
    import cv2

    if len(frames) < 2:
        return np.empty((0, 3), dtype=np.float32), []

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = gray[0].shape
    K = estimate_intrinsics(w, h)

    poses = [FramePose(R=np.eye(3), t=np.zeros(3))]
    all_points: list[np.ndarray] = []
    R_global, t_global = np.eye(3), np.zeros(3)

    for i in range(len(gray) - 1):
        matched = _match_orb(gray[i], gray[i + 1])
        if matched is None:
            poses.append(FramePose(R=R_global.copy(), t=t_global.copy()))
            continue
        pts_a, pts_b = matched

        E, mask = cv2.findEssentialMat(
            pts_a, pts_b, K, method=cv2.RANSAC,
            prob=VIDEO_RANSAC_PROB, threshold=VIDEO_RANSAC_THRESH_PX,
        )
        if E is None or E.shape != (3, 3):
            poses.append(FramePose(R=R_global.copy(), t=t_global.copy()))
            continue

        _, R_rel, t_rel, pose_mask = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
        t_rel = t_rel.ravel()

        R_new = R_global @ R_rel
        t_new = t_global + R_global @ t_rel
        poses.append(FramePose(R=R_new.copy(), t=t_new.copy()))

        inlier = pose_mask.ravel().astype(bool)
        if inlier.sum() >= 8:
            P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P1 = K @ np.hstack([R_rel, t_rel.reshape(3, 1)])
            pts4d = cv2.triangulatePoints(P0, P1, pts_a[inlier].T, pts_b[inlier].T)
            pts3d_local = (pts4d[:3] / pts4d[3]).T

            valid = (pts3d_local[:, 2] > 0) & (np.linalg.norm(pts3d_local, axis=1) < VIDEO_MAX_POINT_DIST)
            pts3d_local = pts3d_local[valid]
            if len(pts3d_local):
                pts3d_world = (R_global @ pts3d_local.T).T + t_global
                all_points.append(pts3d_world)

        R_global, t_global = R_new, t_new

    if not all_points:
        return np.empty((0, 3), dtype=np.float32), poses
    points = np.concatenate(all_points, axis=0).astype(np.float32)
    return _drop_outliers(points), poses


def _drop_outliers(points: np.ndarray, mult: float = VIDEO_OUTLIER_MEDIAN_MULT) -> np.ndarray:
    """Reject points far from the robust centre -- consecutive-pair chaining
    with no bundle adjustment produces a long tail of badly-triangulated
    points (near-zero parallax, bad matches) that otherwise dominate any
    bounding-box-based scale estimate.
    """
    if len(points) < 10:
        return points
    median_pt = np.median(points, axis=0)
    dist = np.linalg.norm(points - median_pt, axis=1)
    med_dist = np.median(dist)
    if med_dist < 1e-9:
        return points
    return points[dist <= mult * med_dist]
