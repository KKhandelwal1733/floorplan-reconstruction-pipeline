"""Stray Scanner directory loader → fused world-frame point cloud.

Directory layout (Stray Scanner export):
    scan/
      camera_matrix.csv   3×3 intrinsics (fx 0 cx / 0 fy cy / 0 0 1)
      odometry.csv        timestamp,tx,ty,tz,qx,qy,qz,qw  (camera-to-world)
      depth/
        000000.png        16-bit uint, millimetres  (real device)
        000000.npy        float32, metres           (synthetic fixtures)
      confidence/         optional, ignored for now
      rgb/                optional, ignored for now
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton quaternion → 3×3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
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


def load_scan(scan_dir: Path) -> np.ndarray:
    """Load a Stray Scanner capture → (N, 3) fused world-frame point cloud."""
    scan_dir = Path(scan_dir)
    K = np.loadtxt(scan_dir / "camera_matrix.csv", delimiter=",")

    poses: list[tuple[np.ndarray, np.ndarray]] = []
    with open(scan_dir / "odometry.csv", newline="") as f:
        for row in csv.DictReader(f):
            t = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            R = _quat_to_rot(float(row["qx"]), float(row["qy"]),
                             float(row["qz"]), float(row["qw"]))
            poses.append((R, t))

    depth_dir = scan_dir / "depth"
    depth_files = sorted(depth_dir.glob("*.npy")) + sorted(depth_dir.glob("*.png"))
    # deduplicate by stem so .npy wins over .png for same frame
    seen: dict[str, Path] = {}
    for p in depth_files:
        seen.setdefault(p.stem, p)
    depth_files = [seen[k] for k in sorted(seen)]

    all_pts: list[np.ndarray] = []
    for (R, t), df in zip(poses, depth_files):
        depth = _load_depth(df)
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


def fuse_scan(scan_dir: Path, out_path: Path) -> np.ndarray:
    """Load scan, write fused PLY, return point cloud."""
    pts = load_scan(scan_dir)
    save_ply(pts, out_path)
    return pts
