"""Stray Scanner directory loader → fused world-frame point cloud.

Supports two odometry column layouts:
  Synthetic fixtures: timestamp,tx,ty,tz,qx,qy,qz,qw
  Real Stray Scanner: timestamp,frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy,...

Depth images are 16-bit uint PNG (millimetres, real device) or float32 NPY
(metres, synthetic fixtures). Camera intrinsics in camera_matrix.csv are at
RGB resolution; for PNG depth the loader scales K to the depth frame size.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from roomscan.config import CONFIDENCE_MIN, DEPTH_MAX_M, DEPTH_MIN_M


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton quaternion → 3×3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _scale_K(K: np.ndarray, depth_w: int, depth_h: int) -> np.ndarray:
    """Scale RGB intrinsics to depth-image resolution."""
    rgb_w = K[0, 2] * 2   # cx ≈ width/2
    rgb_h = K[1, 2] * 2
    sx, sy = depth_w / rgb_w, depth_h / rgb_h
    return np.array([
        [K[0, 0] * sx,          0, K[0, 2] * sx],
        [          0, K[1, 1] * sy, K[1, 2] * sy],
        [          0,           0,             1],
    ])


def _backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Depth map (H, W) metres → camera-frame points (N, 3), invalid pixels dropped."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    d = depth.ravel()
    valid = d > 0
    x = (uu.ravel()[valid] - cx) * d[valid] / fx
    y = (vv.ravel()[valid] - cy) * d[valid] / fy
    return np.stack([x, y, d[valid]], axis=1)


def _load_depth(path: Path) -> np.ndarray:
    """Return depth in metres regardless of format (.npy or .png)."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    try:
        from PIL import Image
        return np.array(Image.open(path), dtype=np.float32) / 1000.0  # mm → m
    except ImportError:
        raise RuntimeError(
            "Pillow is required to read PNG depth maps: pip install Pillow"
        )


def load_scan(
    scan_dir: Path,
    max_frames: int | None = None,
    frame_stride: int = 1,
    frame_offset: int = 0,
) -> np.ndarray:
    """Load a Stray Scanner capture → (N, 3) fused world-frame point cloud.

    Args:
        scan_dir:     Path to a Stray Scanner export directory.
        max_frames:   Cap the number of depth frames loaded (useful for large scans).
        frame_stride: Take every Nth frame (e.g. 2 = every other frame). Frames are
                      chronological, so a stride spans the whole capture — unlike
                      max_frames, which only sees a chronological prefix. Used for
                      fast representative previews and split-scan repeatability
                      checks (two interleaved halves of the same capture).
        frame_offset: Starting index before striding (0-based).
    """
    scan_dir = Path(scan_dir)
    K_rgb = np.loadtxt(scan_dir / "camera_matrix.csv", delimiter=",")

    # Parse odometry — handle both synthetic (tx/ty/tz) and real (x/y/z) layouts.
    # Strip whitespace from keys: real Stray Scanner CSVs use "timestamp, frame, x, ..."
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    with open(scan_dir / "odometry.csv", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [k.strip() for k in reader.fieldnames] if reader.fieldnames else None
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            tx = float(row.get("tx") or row["x"])
            ty = float(row.get("ty") or row["y"])
            tz = float(row.get("tz") or row["z"])
            t = np.array([tx, ty, tz])
            R = _quat_to_rot(float(row["qx"]), float(row["qy"]),
                             float(row["qz"]), float(row["qw"]))
            poses.append((R, t))

    depth_dir = scan_dir / "depth"
    # .npy wins over .png for the same stem (synthetic fixtures take priority)
    seen: dict[str, Path] = {}
    for p in sorted(depth_dir.glob("*.png")):
        seen[p.stem] = p
    for p in sorted(depth_dir.glob("*.npy")):
        seen[p.stem] = p
    depth_files = [seen[k] for k in sorted(seen)]

    if frame_stride != 1 or frame_offset != 0:
        depth_files = depth_files[frame_offset::frame_stride]
        poses = poses[frame_offset::frame_stride]

    if max_frames is not None:
        depth_files = depth_files[:max_frames]
        poses = poses[:max_frames]

    # Determine whether to scale K (only needed for PNG depth maps)
    K_scaled: np.ndarray | None = None
    conf_dir = scan_dir / "confidence"

    all_pts: list[np.ndarray] = []
    for (R, t), df in zip(poses, depth_files):
        depth = _load_depth(df)
        if df.suffix == ".png" and K_scaled is None:
            K_scaled = _scale_K(K_rgb, depth.shape[1], depth.shape[0])
        K = K_scaled if df.suffix == ".png" else K_rgb

        # Range filter (config.DEPTH_MIN_M / DEPTH_MAX_M)
        depth[(depth < DEPTH_MIN_M) | (depth > DEPTH_MAX_M)] = 0.0

        # Confidence filter — only for real scans that have a confidence dir
        if conf_dir.exists():
            conf_file = conf_dir / (df.stem + ".png")
            if conf_file.exists():
                from PIL import Image  # already imported for depth PNGs
                conf = np.array(Image.open(conf_file), dtype=np.uint8)
                depth[conf < CONFIDENCE_MIN] = 0.0

        cam_pts = _backproject(depth, K)
        if cam_pts.size:
            all_pts.append((R @ cam_pts.T).T + t)

    return np.concatenate(all_pts, axis=0) if all_pts else np.empty((0, 3), dtype=np.float32)


def save_ply(points: np.ndarray, out_path: Path) -> None:
    """Write (N, 3) float32 array to an ASCII PLY file."""
    out_path = Path(out_path)
    n = len(points)
    lines = [
        "ply", "format ascii 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
        "end_header",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def fuse_scan(scan_dir: Path, out_path: Path, max_frames: int | None = None) -> np.ndarray:
    """Load scan, write fused PLY, return point cloud."""
    pts = load_scan(scan_dir, max_frames=max_frames)
    save_ply(pts, out_path)
    return pts
