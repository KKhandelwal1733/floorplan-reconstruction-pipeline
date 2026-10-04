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

    # calibration_tier=None: this comparison IS the raw data used to compute
    # calibration factors (bench/calibrate.py) -- applying a prior factor
    # here would be circular.
    video_layout, diagnostics = process_video(video_path, calibration_tier=None)

    result: dict[str, Any] = {
        "scan_dir": str(scan_dir),
        "simulated": False,  # both tiers ran on the SAME real capture
        "pseudo_ground_truth": "lidar_tier_geometry (not laser-measured)",
        "lidar_floor_area_m2": lidar_layout.floor_area_m2.value,
        "video_floor_area_m2": video_layout.floor_area_m2.value,
        "video_floor_area_lo": video_layout.floor_area_m2.lo,
        "video_floor_area_hi": video_layout.floor_area_m2.hi,
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
        result["video_ceiling_height_lo"] = video_layout.ceiling_height_m.lo
        result["video_ceiling_height_hi"] = video_layout.ceiling_height_m.hi
        result["ceiling_height_rel_err"] = _rel_err(
            lidar_layout.ceiling_height_m.value, video_layout.ceiling_height_m.value
        )
    return result


def derive_simulated_photos(video_path: Path, n_photos: int = 6) -> list:
    """Extract n_photos evenly-spaced stills from a real video as a stand-in
    for a real photo-tier capture (no real multi-room photo fixture exists).
    Clearly labelled simulated=True wherever used -- these are real frames
    from a real room, but not an actual independent still-camera capture.
    """
    from roomscan.io.video_loader import sample_frames

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    finally:
        cap.release()
    interval_s = duration_s / n_photos if n_photos > 0 else duration_s
    return sample_frames(video_path, interval_s=interval_s, max_frames=n_photos)


def compare_photo_to_lidar(
    scan_dir: Path, n_photos: int = 6, lidar_frame_stride: int = 5,
) -> dict[str, Any]:
    """Compare a SIMULATED photo-tier reconstruction (stills derived from the
    scan's own rgb.mp4) against LiDAR-tier pseudo-ground-truth.

    simulated=True: unlike compare_video_to_lidar, this is NOT a genuine
    independent photo capture -- it's the fallback the brief anticipated,
    used because no real multi-room photo fixture exists yet.
    """
    from roomscan.config import (
        PHOTO_EMPIRICAL_MIN_REL_HW,
        PHOTO_MIN_PLANE_INLIERS,
        PHOTO_MIN_RECONSTRUCTED_PTS,
    )
    from roomscan.geometry.room_layout import extract_layout
    from roomscan.geometry.video_tier import reconstruct_from_frames
    from roomscan.io.stray_scanner import load_scan

    scan_dir = Path(scan_dir)
    video_path = scan_dir / "rgb.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"{video_path} not found -- scan_dir must be a Stray Scanner export")

    lidar_pts = load_scan(scan_dir, frame_stride=lidar_frame_stride)
    lidar_layout = extract_layout(lidar_pts)

    photos = derive_simulated_photos(video_path, n_photos=n_photos)
    # calibration_tier=None: see the comment in compare_video_to_lidar above.
    photo_layout, diagnostics = reconstruct_from_frames(
        photos, min_rel_hw=PHOTO_EMPIRICAL_MIN_REL_HW, detector="sift",
        strategy="best_pair", min_points=PHOTO_MIN_RECONSTRUCTED_PTS,
        min_plane_inliers=PHOTO_MIN_PLANE_INLIERS, calibration_tier=None,
        tier_label="photo",
    )

    result: dict[str, Any] = {
        "scan_dir": str(scan_dir),
        "simulated": True,  # stills derived from rgb.mp4, not an independent photo capture
        "n_photos": n_photos,
        "pseudo_ground_truth": "lidar_tier_geometry (not laser-measured)",
        "lidar_floor_area_m2": lidar_layout.floor_area_m2.value,
        "photo_floor_area_m2": photo_layout.floor_area_m2.value,
        "photo_floor_area_lo": photo_layout.floor_area_m2.lo,
        "photo_floor_area_hi": photo_layout.floor_area_m2.hi,
        "floor_area_rel_err": _rel_err(
            lidar_layout.floor_area_m2.value, photo_layout.floor_area_m2.value
        ),
        "photo_scale_factor": diagnostics["scale_factor"],
        "photo_scale_rel_half_width": diagnostics["scale_rel_half_width"],
        "photo_n_points_unscaled": diagnostics["n_points_unscaled"],
    }
    if lidar_layout.ceiling_height_m and photo_layout.ceiling_height_m:
        result["lidar_ceiling_height_m"] = lidar_layout.ceiling_height_m.value
        result["photo_ceiling_height_m"] = photo_layout.ceiling_height_m.value
        result["photo_ceiling_height_lo"] = photo_layout.ceiling_height_m.lo
        result["photo_ceiling_height_hi"] = photo_layout.ceiling_height_m.hi
        result["ceiling_height_rel_err"] = _rel_err(
            lidar_layout.ceiling_height_m.value, photo_layout.ceiling_height_m.value
        )
    return result


if __name__ == "__main__":
    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/single_room")
    print("--- video tier vs LiDAR (real) ---")
    print(json.dumps(compare_video_to_lidar(scan_dir), indent=2))
    print("\n--- photo tier vs LiDAR (simulated stills) ---")
    print(json.dumps(compare_photo_to_lidar(scan_dir), indent=2))
