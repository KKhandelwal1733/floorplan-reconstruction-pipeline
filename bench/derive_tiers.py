"""Derive/compare the video tier against LiDAR-tier pseudo-ground-truth.

The brief's benchmark data situation note anticipates deriving a *simulated*
video tier from a LiDAR scan when no real video exists. Our Stray Scanner
captures happen to include a genuinely synchronized rgb.mp4 alongside the
depth data, so this is a REAL comparison (same physical capture, two
independent tiers), not a simulated one -- labelled simulated=False below.

Per the hard rule against fabricating benchmark numbers: this pseudo-ground-
truth is LiDAR-tier geometry on the full scan, not laser-measured truth (see
COMPLIANCE.md). Known finding from the one real example run so far: the video
tier's lightweight (non-bundle-adjusted) SfM is substantially inaccurate --
see config.py VIDEO_EMPIRICAL_MIN_REL_HW and project memory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def compare_video_to_lidar(scan_dir: Path, lidar_frame_stride: int = 5) -> dict[str, Any]:
    """Run both tiers on the same physical capture and compare geometry.

    scan_dir must be a Stray Scanner export containing both depth/ (LiDAR
    tier) and rgb.mp4 (video tier) -- any of the real fixtures qualify.
    lidar_frame_stride keeps the LiDAR-tier load fast and representative
    (see project memory: a chronological prefix can see only part of a
    room, a stride spans the whole capture).
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

    video_layout, diagnostics = process_video(video_path)

    result: dict[str, Any] = {
        "scan_dir": str(scan_dir),
        "simulated": False,  # both tiers ran on the SAME real capture
        "pseudo_ground_truth": "lidar_tier_geometry (not laser-measured)",
        "lidar_floor_area_m2": lidar_layout.floor_area_m2.value,
        "video_floor_area_m2": video_layout.floor_area_m2.value,
        "floor_area_rel_err": _rel_err(
            lidar_layout.floor_area_m2.value, video_layout.floor_area_m2.value
        ),
        "video_scale_factor": diagnostics["scale_factor"],
        "video_scale_rel_half_width": diagnostics["scale_rel_half_width"],
        "video_scale_estimates": diagnostics["scale_estimates"],
        "video_n_points_unscaled": diagnostics["n_points_unscaled"],
    }
    if lidar_layout.ceiling_height_m and video_layout.ceiling_height_m:
        result["lidar_ceiling_height_m"] = lidar_layout.ceiling_height_m.value
        result["video_ceiling_height_m"] = video_layout.ceiling_height_m.value
        result["ceiling_height_rel_err"] = _rel_err(
            lidar_layout.ceiling_height_m.value, video_layout.ceiling_height_m.value
        )
    return result


if __name__ == "__main__":
    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/single_room")
    result = compare_video_to_lidar(scan_dir)
    print(json.dumps(result, indent=2))
