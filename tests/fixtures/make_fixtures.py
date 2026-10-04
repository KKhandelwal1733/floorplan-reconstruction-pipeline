"""Generate synthetic Stray Scanner fixtures for pose-convention tests.

Scene convention (matches Stray Scanner export):
  World: Z-up, X-right, Y-forward
  Camera: Z forward (optical axis), Y down, X right
  Pose in odometry.csv: camera-to-world transform (R, t)

Quaternion (1,0,0,0) = 180° around X → camera Z points world -Z (looking down).
Quaternion (0,0,0,1) = identity          → camera Z points world +Z (looking up).
"""
import numpy as np
from pathlib import Path

H, W = 32, 32          # small depth maps; enough for plane-fitting
FX = FY = 30.0
CX, CY = W / 2.0, H / 2.0


def _write_scan(path: Path, frames: list) -> None:
    """Write a minimal Stray Scanner directory.

    frames: list of (depth_m, [tx,ty,tz], [qx,qy,qz,qw])
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "depth").mkdir(exist_ok=True)

    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    np.savetxt(path / "camera_matrix.csv", K, delimiter=",")

    header = "timestamp,tx,ty,tz,qx,qy,qz,qw"
    rows = []
    for i, (depth_val, t, q) in enumerate(frames):
        rows.append([i * 0.1] + list(t) + list(q))
        np.save(path / "depth" / f"{i:06d}.npy",
                np.full((H, W), depth_val, dtype=np.float32))

    with open(path / "odometry.csv", "w", newline="") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def make_floor_only(root: Path) -> None:
    # Camera 1.5 m above floor, looking straight down.
    # After back-projection all world points land at z ≈ 0.
    _write_scan(root / "single_scan_floor_only", [
        (1.5, [0.0, 0.0, 1.5], [1.0, 0.0, 0.0, 0.0]),
    ])


def make_with_ceiling(root: Path) -> None:
    # Frame 0: camera at z=1.2, looking down  → floor points at world z=0
    # Frame 1: camera at z=1.2, looking up    → ceiling points at world z=2.4
    _write_scan(root / "single_scan_with_ceiling", [
        (1.2, [0.0, 0.0, 1.2], [1.0, 0.0, 0.0, 0.0]),
        (1.2, [0.0, 0.0, 1.2], [0.0, 0.0, 0.0, 1.0]),
    ])


if __name__ == "__main__":
    root = Path(__file__).parent
    make_floor_only(root)
    make_with_ceiling(root)
    print("Fixtures written to", root)
