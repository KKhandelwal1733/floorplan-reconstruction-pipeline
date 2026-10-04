"""Gather real calibration points across available fixtures and run
conformal calibration for the video and photo tiers (Phase 9).

Pseudo-ground-truth is LiDAR-tier geometry on the same physical capture
(see COMPLIANCE.md) -- never laser-measured truth. With only a handful of
real fixtures, this honestly reports when the standard conformal coverage
guarantee is mathematically unreachable at the target confidence level,
rather than quietly computing an undersized factor (see
roomscan/calibration/conformal.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roomscan.calibration.conformal import CalibrationPoint, calibrate_tier

FIXTURES = [
    Path("tests/fixtures/single_room"),
    Path("tests/fixtures/real_floor_only"),
    Path("tests/fixtures/real_with_ceiling"),
]


def _points_from_comparison(tier: str, cmp: dict[str, Any]) -> list[CalibrationPoint]:
    points = [CalibrationPoint(
        tier=tier, metric="floor_area_m2",
        predicted_value=cmp[f"{tier}_floor_area_m2"],
        predicted_lo=cmp[f"{tier}_floor_area_lo"],
        predicted_hi=cmp[f"{tier}_floor_area_hi"],
        true_value=cmp["lidar_floor_area_m2"],
    )]
    if f"{tier}_ceiling_height_m" in cmp:
        points.append(CalibrationPoint(
            tier=tier, metric="ceiling_height_m",
            predicted_value=cmp[f"{tier}_ceiling_height_m"],
            predicted_lo=cmp[f"{tier}_ceiling_height_lo"],
            predicted_hi=cmp[f"{tier}_ceiling_height_hi"],
            true_value=cmp["lidar_ceiling_height_m"],
        ))
    return points


def gather_calibration_points(fixtures: list[Path] = FIXTURES) -> list[CalibrationPoint]:
    from bench.derive_tiers import compare_photo_to_lidar, compare_video_to_lidar

    points: list[CalibrationPoint] = []
    for scan_dir in fixtures:
        if not scan_dir.exists() or not (scan_dir / "rgb.mp4").exists():
            continue
        try:
            points += _points_from_comparison("video", compare_video_to_lidar(scan_dir))
        except (ValueError, FileNotFoundError):
            pass
        try:
            points += _points_from_comparison("photo", compare_photo_to_lidar(scan_dir))
        except (ValueError, FileNotFoundError):
            pass
    return points


def run_calibration(target_coverage: float = 0.9) -> dict[str, Any]:
    points = gather_calibration_points()
    tiers: dict[str, Any] = {}
    for tier in ["video", "photo"]:
        cal = calibrate_tier(points, tier, target_coverage)
        tiers[tier] = {
            "n_points": cal.n_points,
            "target_coverage": cal.target_coverage,
            "min_points_needed_for_target": cal.min_points_needed,
            "factor": cal.factor if cal.achievable else None,
            "achievable": cal.achievable,
            "note": (
                f"factor {cal.factor:.2f}x achieves {target_coverage:.0%} marginal coverage "
                "(assuming calibration and future captures are exchangeable)"
                if cal.achievable else
                f"only {cal.n_points} calibration point(s) available; need >= "
                f"{cal.min_points_needed} for a finite {target_coverage:.0%}-coverage factor. "
                "Existing hand-set empirical floors (VIDEO/PHOTO_EMPIRICAL_MIN_REL_HW) remain "
                "the active mechanism until more real captures are available."
            ),
        }
    return {"n_total_calibration_points": len(points), "tiers": tiers}


FACTORS_PATH = Path(__file__).parents[1] / "roomscan" / "calibration" / "factors.json"


def save_factors(result: dict[str, Any], path: Path = FACTORS_PATH) -> None:
    """Persist calibration results for roomscan.calibration.factors to load.

    Only ever read by production code when a tier is actually achievable
    (see factors.load_calibration_factor) -- an unreachable-at-this-n result
    is saved too, purely for visibility/history, but never applied.
    """
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    result = run_calibration()
    print(json.dumps(result, indent=2))
    save_factors(result)
    print(f"\nSaved to {FACTORS_PATH}")
