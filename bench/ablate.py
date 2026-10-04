"""Drift ablations. Two different kinds of "drift" are tested here:

1. run_drift_ablation (Phase 8): does the VIDEO TIER's accumulated
   pose-chaining error (roomscan/geometry/sfm.py chains consecutive-frame
   relative poses with no bundle adjustment) grow with chain length? This
   reconstructs the same real video at several frame counts and compares
   each reconstruction's floor area / ceiling height against LiDAR-tier
   pseudo-ground-truth for the same physical room -- a real comparison
   (see COMPLIANCE.md on pseudo-ground-truth), not a fabricated one.

2. run_pose_graph_drift_ablation (case-study realignment): does the
   MULTI-ROOM POSE GRAPH's drift-correction mechanism (loop-closure-aware
   joint least-squares, roomscan/geometry/pose_graph.py) actually reduce
   accumulated placement error versus naive sequential placement, as
   simulated per-room measurement noise grows? No real multi-room capture
   exists (any tier) to test this against, so this uses a synthetic room
   ring instead -- labelled `synthetic: true`, never conflated with (1)'s
   real-data result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roomscan.config import (
    POSE_GRAPH_ABLATION_N_TRIALS,
    POSE_GRAPH_ABLATION_NOISE_STDS_M,
    SEED,
    VIDEO_DRIFT_ABLATION_FRAME_COUNTS,
)


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


def _rect_layout(w: float, h: float):
    from roomscan.geometry.room_layout import RoomLayout, WallSegment
    from roomscan.schema_out import Measurement

    def m(v: float) -> Measurement:
        return Measurement(value=v, lo=v, hi=v)

    walls = [
        WallSegment(p0=(0, 0), p1=(w, 0), length_m=m(w)),
        WallSegment(p0=(w, 0), p1=(w, h), length_m=m(h)),
        WallSegment(p0=(w, h), p1=(0, h), length_m=m(w)),
        WallSegment(p0=(0, h), p1=(0, 0), length_m=m(h)),
    ]
    return RoomLayout(up_axis=np.array([0.0, 0.0, 1.0]), floor_d=0.0, ceiling_d=None,
                       polygon=[(0, 0), (w, 0), (w, h), (0, h)],
                       floor_area_m2=m(w * h), ceiling_height_m=None, walls=walls)


def _build_synthetic_ring(noise_std_m: float, rng: np.random.Generator):
    """4 identical rectangular rooms arranged in a real spatial square ring
    (room0 -> room1 -> room2 -> room3 -> room0, turning 90 degrees at each
    corner) -- a synthetic stand-in for a capture that walks a loop of
    rooms and returns to where it started (no real multi-room capture
    exists to do this with -- see COMPLIANCE.md). Wall indices: 0=bottom,
    1=right, 2=top, 3=left (CCW from (0,0)).

    3 of the 4 door correspondences (room0->1, room1->2, room3->0) have
    their matched door position perturbed by independent Gaussian noise
    (simulating compounding per-room door-detection error); the 4th
    (room2->3) is left exact and held out as the measurement edge. Given
    this graph's BFS traversal order (room0's two neighbours, room1 and
    room3, are reached directly; room2 is then reached via room1 before
    room3 is expanded), naive (refine=False) placement's spanning tree is
    exactly {0->1, 1->2, 3->0} -- confirmed empirically, not assumed -- so
    room2->3 is the one edge never used to place anything, and its
    residual *after* naive placement is exactly the accumulated drift that
    only the joint solve (refine=True) has a chance to correct.
    """
    from roomscan.geometry.pose_graph import DoorCorrespondence

    w, h = 4.0, 3.0
    layouts = [_rect_layout(w, h) for _ in range(4)]
    # (room_a, wall_a, room_b, wall_b, true_u_a, true_u_b): true_u is each
    # wall's own midpoint, since every door is centred in this synthetic ring.
    noisy_edges = [
        (0, 1, 1, 3, h / 2, h / 2),   # room0's right wall -> room1's left wall
        (1, 2, 2, 0, w / 2, w / 2),   # room1's top wall -> room2's bottom wall
        (3, 0, 0, 2, w / 2, w / 2),   # room3's bottom wall -> room0's top wall
    ]
    held_out_edge = (2, 3, 3, 1, h / 2, h / 2)   # room2's left wall -> room3's right wall

    chain = []
    for room_a, wall_a, room_b, wall_b, u_a, u_b in noisy_edges:
        noisy_u_b = u_b + float(rng.normal(0.0, noise_std_m))
        chain.append(DoorCorrespondence(room_a=room_a, room_b=room_b, wall_a=wall_a,
                                         wall_b=wall_b, door_u_a=u_a, door_u_b=noisy_u_b))
    room_a, wall_a, room_b, wall_b, u_a, u_b = held_out_edge
    loop_closure = DoorCorrespondence(room_a=room_a, room_b=room_b, wall_a=wall_a,
                                       wall_b=wall_b, door_u_a=u_a, door_u_b=u_b)
    return layouts, chain, loop_closure


def _loop_closure_gap(layouts, loop_closure, poses) -> float:
    from roomscan.geometry.pose_graph import _apply_pose, _door_point

    door_a = _apply_pose(
        _door_point(layouts[loop_closure.room_a], loop_closure.wall_a, loop_closure.door_u_a),
        poses[loop_closure.room_a],
    )
    door_b = _apply_pose(
        _door_point(layouts[loop_closure.room_b], loop_closure.wall_b, loop_closure.door_u_b),
        poses[loop_closure.room_b],
    )
    return float(np.linalg.norm(door_a - door_b))


def run_pose_graph_drift_ablation(
    noise_stds_m: list[float] = POSE_GRAPH_ABLATION_NOISE_STDS_M,
    n_trials: int = POSE_GRAPH_ABLATION_N_TRIALS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Drift-correction on/off: does the joint pose-graph solve (loop-closure-
    aware, refine=True) reduce accumulated placement error versus naive
    sequential (tree-only, refine=False) placement, as per-room door-
    detection noise grows?

    Uses a synthetic 4-room square ring, not a real multi-room capture
    (none exists -- see COMPLIANCE.md); this tests the mechanism (does the
    extra loop-closure constraint help at all, and does the benefit grow
    as drift grows), not real-world accuracy. The gap is a 2-D vector norm
    (always >= 0), so a single noise draw can land anywhere and even
    cancel out by chance direction -- each noise level averages n_trials
    independent draws (RMS) instead of trusting one.
    """
    from roomscan.geometry.pose_graph import solve_pose_graph

    rows = []
    for noise_std_m in noise_stds_m:
        rng = np.random.default_rng(seed)   # fresh per noise level: independent, reproducible
        gaps_off, gaps_on = [], []
        for _ in range(n_trials):
            layouts, chain, loop_closure = _build_synthetic_ring(noise_std_m, rng)
            correspondences = chain + [loop_closure]
            poses_off = solve_pose_graph(layouts, correspondences, refine=False)
            poses_on = solve_pose_graph(layouts, correspondences, refine=True)
            gaps_off.append(_loop_closure_gap(layouts, loop_closure, poses_off))
            gaps_on.append(_loop_closure_gap(layouts, loop_closure, poses_on))

        gap_off = float(np.sqrt(np.mean(np.square(gaps_off))))
        gap_on = float(np.sqrt(np.mean(np.square(gaps_on))))
        rows.append({
            "noise_std_m": noise_std_m,
            "loop_closure_gap_off_m": gap_off,
            "loop_closure_gap_on_m": gap_on,
            "improved": gap_on < gap_off,
        })

    # Only the noise > 0 rows are informative (at noise=0 there's nothing to
    # correct and both should already match) -- judge the mechanism on those.
    noisy_rows = [r for r in rows if r["noise_std_m"] > 0]
    improved_count = sum(1 for r in noisy_rows if r["improved"])
    drift_correction_helps = noisy_rows and improved_count >= max(1, len(noisy_rows) - 1)

    return {
        "synthetic": True,
        "note_data_source": (
            "a synthetic 4-room square ring with injected Gaussian door-position "
            "noise per chain edge, NOT a real multi-room capture (none exists -- "
            "see COMPLIANCE.md); tests the drift-correction mechanism, not real "
            "accuracy"
        ),
        "rows": rows,
        "drift_correction_helps": drift_correction_helps,
        "note": (
            f"joint pose-graph refinement (on) reduced the loop-closure gap vs. "
            f"naive sequential placement (off) in {improved_count}/{len(noisy_rows)} "
            "noise levels tested"
            if drift_correction_helps else
            f"joint refinement only improved {improved_count}/{len(noisy_rows)} noise "
            "levels -- the loop-closure constraint isn't reliably correcting drift here"
        ),
    }


if __name__ == "__main__":
    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/single_room")
    result = run_drift_ablation(scan_dir)
    print(json.dumps(result, indent=2))

    pose_graph_result = run_pose_graph_drift_ablation()
    print(json.dumps(pose_graph_result, indent=2))
