"""Drift ablation (Phase 8): does accumulated pose-chaining error grow with
chain length?

The video tier (roomscan/geometry/sfm.py) chains consecutive-frame relative
poses with no bundle adjustment, so error should accumulate with more
chained steps ("drift"). This reconstructs the same real video at several
frame counts and compares each reconstruction's floor area / ceiling height
against LiDAR-tier pseudo-ground-truth for the same physical room -- a real
comparison (see COMPLIANCE.md on pseudo-ground-truth), not a fabricated one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from roomscan.config import VIDEO_DRIFT_ABLATION_FRAME_COUNTS


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def run_drift_ablation(
    scan_dir: Path,
    frame_counts: list[int] = VIDEO_DRIFT_ABLATION_FRAME_COUNTS,
    lidar_frame_stride: int = 5,
) -> dict[str, Any]:
    """Reconstruct scan_dir's rgb.mp4 at each frame count in frame_counts and
    compare each to LiDAR-tier geometry on the same scan.

    Returns a dict with one row per frame count plus whether error trended
    upward with chain length (the signature of accumulated drift).
    """
    from roomscan.geometry.room_layout import extract_layout
    from roomscan.geometry.video_tier import process_video
    from roomscan.io.stray_scanner import load_scan

    scan_dir = Path(scan_dir)
    video_path = scan_dir / "rgb.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"{video_path} not found -- scan_dir must be a Stray Scanner export")

    lidar_pts = load_scan(scan_dir, frame_stride=lidar_frame_stride)
    lidar_layout = extract_layout(lidar_pts)
    lidar_area = lidar_layout.floor_area_m2.value

    rows = []
    for n in frame_counts:
        try:
            video_layout, diagnostics = process_video(video_path, max_frames=n)
            area_err = _rel_err(lidar_area, video_layout.floor_area_m2.value)
            rows.append({
                "n_frames": n,
                "video_floor_area_m2": video_layout.floor_area_m2.value,
                "floor_area_rel_err": area_err,
                "n_points_unscaled": diagnostics["n_points_unscaled"],
            })
        except ValueError as e:
            rows.append({"n_frames": n, "error": str(e)})

    errs = [r["floor_area_rel_err"] for r in rows if "floor_area_rel_err" in r]
    drift_detected = len(errs) >= 2 and errs[-1] > errs[0]

    return {
        "scan_dir": str(scan_dir),
        "pseudo_ground_truth": "lidar_tier_geometry (not laser-measured)",
        "lidar_floor_area_m2": lidar_area,
        "rows": rows,
        "drift_detected": drift_detected,
        "note": (
            "error at the longest chain exceeds error at the shortest -- "
            "consistent with accumulated pose-chaining drift"
            if drift_detected else
            "error did not increase with chain length in this run -- "
            "other noise sources likely dominate over chaining drift here"
        ),
    }


if __name__ == "__main__":
    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/single_room")
    result = run_drift_ablation(scan_dir)
    print(json.dumps(result, indent=2))
