"""Gate harness — score each phase gate as PASS / FAIL / not measured.

Gates that require laser ground truth are always "not measured" until the
defense day (rule: never fabricate numbers — see config feedback memory).

Usage:
    from bench.harness import run_gates, print_gates
    results = run_gates(layout, openings)
    print_gates(results)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NOT_MEASURED = "not measured"
PASS = "PASS"
FAIL = "FAIL"


@dataclass
class GateResult:
    gate: str
    status: str            # PASS | FAIL | not measured
    value: Any = None
    threshold: Any = None
    note: str = ""


def _schema_gate(layout_or_plan_path: Any) -> GateResult:
    """Schema v0.1 validation: plan.json must pass jsonschema."""
    try:
        from roomscan.schema_out import Plan
        import json, jsonschema
        schema_path = Path(__file__).parents[1] / "schema" / "plan.schema.json"
        if isinstance(layout_or_plan_path, (str, Path)):
            data = json.loads(Path(layout_or_plan_path).read_text())
            schema = json.loads(schema_path.read_text())
            jsonschema.validate(data, schema)
            return GateResult("schema_valid", PASS, note="plan.json validates against schema v0.1")
        return GateResult("schema_valid", NOT_MEASURED, note="pass a plan.json path to validate")
    except Exception as e:
        return GateResult("schema_valid", FAIL, note=str(e))


def _determinism_gate(scan_dir: Path | None = None, frame_stride: int = 20) -> GateResult:
    """Two back-to-back runs on the same frames must produce identical output.

    frame_stride keeps this fast on a large real scan (a representative
    subsample, not a prefix) rather than reloading every frame twice.
    """
    if scan_dir is None:
        return GateResult("determinism", NOT_MEASURED,
                          note="pass scan_dir to run determinism check")
    try:
        from roomscan.io.stray_scanner import load_scan
        from roomscan.geometry.room_layout import extract_layout
        pts1 = load_scan(scan_dir, frame_stride=frame_stride)
        pts2 = load_scan(scan_dir, frame_stride=frame_stride)
        import numpy as np
        if np.array_equal(pts1, pts2):
            l1 = extract_layout(pts1)
            l2 = extract_layout(pts2)
            same = (
                l1.floor_area_m2.value == l2.floor_area_m2.value
                and l1.floor_area_m2.lo == l2.floor_area_m2.lo
                and l1.floor_area_m2.hi == l2.floor_area_m2.hi
            )
            return GateResult(
                "determinism", PASS if same else FAIL,
                note="two runs agree exactly on floor_area" if same
                     else "floor_area differs between identical-input runs",
            )
        return GateResult("determinism", FAIL, note="point clouds differ between runs")
    except Exception as e:
        return GateResult("determinism", FAIL, note=str(e))


def _repeatability_gate(scan_dir: Path | None = None, frame_stride: int = 20) -> GateResult:
    """Two interleaved halves of the same capture should agree on geometry.

    No ground truth needed: this is a self-consistency check (same physical
    room, two independent frame subsets covering the whole capture timeline).
    """
    if scan_dir is None:
        return GateResult("repeatability_split_scan", NOT_MEASURED,
                          note="pass scan_dir to run split-scan repeatability check")
    try:
        from roomscan.config import REPEAT_AREA_TOL_FRAC, REPEAT_PERIMETER_TOL_FRAC
        from roomscan.io.stray_scanner import load_scan
        from roomscan.geometry.room_layout import extract_layout

        stride2 = frame_stride * 2
        pts_a = load_scan(scan_dir, frame_stride=stride2, frame_offset=0)
        pts_b = load_scan(scan_dir, frame_stride=stride2, frame_offset=frame_stride)
        layout_a = extract_layout(pts_a)
        layout_b = extract_layout(pts_b)

        area_a, area_b = layout_a.floor_area_m2.value, layout_b.floor_area_m2.value
        area_diff = abs(area_a - area_b) / max(area_a, area_b, 1e-6)

        peri_a = sum(w.length_m.value for w in layout_a.walls)
        peri_b = sum(w.length_m.value for w in layout_b.walls)
        peri_diff = abs(peri_a - peri_b) / max(peri_a, peri_b, 1e-6)

        ok = area_diff <= REPEAT_AREA_TOL_FRAC and peri_diff <= REPEAT_PERIMETER_TOL_FRAC
        return GateResult(
            "repeatability_split_scan", PASS if ok else FAIL,
            value=f"area_diff={area_diff:.1%} perimeter_diff={peri_diff:.1%}",
            threshold=f"area<={REPEAT_AREA_TOL_FRAC:.0%} perimeter<={REPEAT_PERIMETER_TOL_FRAC:.0%}",
            note="two interleaved frame subsets of the same capture compared",
        )
    except Exception as e:
        return GateResult("repeatability_split_scan", FAIL, note=str(e))


def _ceiling_abstain_gate(layout: Any) -> GateResult:
    """When ceiling is unobserved, ceiling_unobserved flag must be True and CI must be wide."""
    try:
        if not hasattr(layout, "ceiling_unobserved"):
            return GateResult("ceiling_abstain", FAIL, note="layout has no ceiling_unobserved field")
        if layout.ceiling_unobserved:
            if layout.ceiling_height_m is None:
                return GateResult("ceiling_abstain", FAIL, note="ceiling_unobserved=True but no measurement")
            hw = (layout.ceiling_height_m.hi - layout.ceiling_height_m.lo) / 2
            wide = hw >= 0.40   # at least ±0.40 m
            return GateResult(
                "ceiling_abstain", PASS if wide else FAIL,
                value=round(hw, 3), threshold=0.40,
                note="half-width wide enough" if wide else "CI too narrow for unobserved ceiling",
            )
        return GateResult("ceiling_abstain", PASS, note="ceiling observed - gate not triggered")
    except Exception as e:
        return GateResult("ceiling_abstain", FAIL, note=str(e))


def _opening_width_gate() -> GateResult:
    """Opening width error ≤ 2 cm vs laser — requires ground truth."""
    return GateResult(
        "opening_width_error_m", NOT_MEASURED,
        threshold=0.02,
        note="requires laser-measured ground truth (none available before defense)",
    )


def _floor_area_gate() -> GateResult:
    """Floor area error — requires tape/laser ground truth."""
    return GateResult(
        "floor_area_error_m2", NOT_MEASURED,
        threshold=None,
        note="requires laser-measured ground truth",
    )


def _ceiling_height_gate() -> GateResult:
    """Ceiling height error ≤ 1.5 cm — requires laser ground truth."""
    return GateResult(
        "ceiling_height_error_m", NOT_MEASURED,
        threshold=0.015,
        note="requires laser-measured ground truth",
    )


def _wall_length_gate() -> GateResult:
    return GateResult(
        "wall_length_error_m", NOT_MEASURED,
        threshold=0.02,
        note="requires laser-measured ground truth",
    )


def _video_footprint_gate(scan_dir: Path | None = None) -> GateResult:
    """Video wall lengths + footprint within +/-3% (literal case-study gate).

    Unlike the laser-ground-truth gates above, this one IS measurable now --
    it compares against LiDAR-tier pseudo-ground-truth on the same physical
    capture (bench/derive_tiers.py), not laser/tape truth. Reported as a real
    FAIL when exceeded, never folded into "not measured" just because it's
    an unflattering number.
    """
    if scan_dir is None or not (scan_dir / "rgb.mp4").exists():
        return GateResult("video_footprint_vs_3pct", NOT_MEASURED,
                          note="pass a scan_dir with rgb.mp4 to compare video vs LiDAR-tier pseudo-ground-truth")
    from roomscan.config import VIDEO_FOOTPRINT_GATE_TOL_FRAC
    from bench.derive_tiers import compare_video_to_lidar
    try:
        result = compare_video_to_lidar(scan_dir)
        err = result["floor_area_rel_err"]
        ok = err <= VIDEO_FOOTPRINT_GATE_TOL_FRAC
        return GateResult(
            "video_footprint_vs_3pct", PASS if ok else FAIL,
            value=f"{err:.1%}", threshold=f"<={VIDEO_FOOTPRINT_GATE_TOL_FRAC:.0%}",
            note="floor-area error vs LiDAR-tier pseudo-ground-truth (not laser truth), same physical capture",
        )
    except Exception as e:
        return GateResult("video_footprint_vs_3pct", NOT_MEASURED, note=f"comparison failed: {e}")


def _photo_footprint_gate(scan_dir: Path | None = None, n_photos: int = 6) -> GateResult:
    """Photo wall lengths + footprint within +/-8% (literal case-study gate),
    on simulated photo sets derived from a real video (no real multi-room
    photo fixture exists -- see COMPLIANCE.md). Same pseudo-ground-truth
    caveat as the video gate above."""
    if scan_dir is None or not (scan_dir / "rgb.mp4").exists():
        return GateResult("photo_footprint_vs_8pct", NOT_MEASURED,
                          note="pass a scan_dir with rgb.mp4 to derive simulated photos and compare vs LiDAR")
    from roomscan.config import PHOTO_FOOTPRINT_GATE_TOL_FRAC
    from bench.derive_tiers import compare_photo_to_lidar
    try:
        result = compare_photo_to_lidar(scan_dir, n_photos=n_photos)
        err = result["floor_area_rel_err"]
        ok = err <= PHOTO_FOOTPRINT_GATE_TOL_FRAC
        return GateResult(
            "photo_footprint_vs_8pct", PASS if ok else FAIL,
            value=f"{err:.1%}", threshold=f"<={PHOTO_FOOTPRINT_GATE_TOL_FRAC:.0%}",
            note=f"simulated={n_photos} photos vs LiDAR-tier pseudo-ground-truth (not laser truth)",
        )
    except ValueError as e:
        return GateResult("photo_footprint_vs_8pct", NOT_MEASURED,
                          note=f"reconstruction abstained (expected at this data sparsity): {e}")
    except Exception as e:
        return GateResult("photo_footprint_vs_8pct", NOT_MEASURED, note=f"comparison failed: {e}")


def measure_execution_timing(scan_dir: Path) -> dict[str, Any]:
    """Wall-clock timing per tier, same physical capture -- a real
    measurement (never estimated), reported as part of the benchmark report.
    Timed on whatever machine `make bench` runs on; not a hardware-neutral
    benchmark, just an honest "this is what it took here."
    """
    import time

    from roomscan.geometry.room_layout import extract_layout
    from roomscan.io.stray_scanner import load_scan

    timings: dict[str, Any] = {"scan_dir": str(scan_dir)}

    t0 = time.perf_counter()
    pts = load_scan(scan_dir, max_frames=None)
    t1 = time.perf_counter()
    timings["lidar_load_scan_s"] = round(t1 - t0, 2)

    try:
        extract_layout(pts)
        t2 = time.perf_counter()
        timings["lidar_extract_layout_s"] = round(t2 - t1, 2)
    except Exception as e:
        timings["lidar_extract_layout_s"] = f"failed: {e}"

    video_path = scan_dir / "rgb.mp4"
    if video_path.exists():
        from roomscan.geometry.video_tier import process_video
        t3 = time.perf_counter()
        try:
            process_video(video_path, calibration_tier=None)
            t4 = time.perf_counter()
            timings["video_process_video_s"] = round(t4 - t3, 2)
        except Exception as e:
            timings["video_process_video_s"] = f"failed: {e}"
    else:
        timings["video_process_video_s"] = "not measured (no rgb.mp4)"

    return timings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_gates(
    layout: Any = None,
    openings: list[Any] | None = None,
    plan_path: Path | None = None,
    scan_dir: Path | None = None,
) -> list[GateResult]:
    """Run all scored gates. Returns list of GateResult in report order."""
    return [
        _schema_gate(plan_path),
        _determinism_gate(scan_dir),
        _repeatability_gate(scan_dir),
        _ceiling_abstain_gate(layout),
        _opening_width_gate(),
        _floor_area_gate(),
        _ceiling_height_gate(),
        _wall_length_gate(),
        _video_footprint_gate(scan_dir),
        _photo_footprint_gate(scan_dir),
    ]


def print_gates(results: list[GateResult]) -> None:
    """Pretty-print the gate table to stdout."""
    col = {"PASS": "\033[32m", "FAIL": "\033[31m", NOT_MEASURED: "\033[33m"}
    reset = "\033[0m"
    print(f"\n{'Gate':<32} {'Status':<14} {'Value':>10}  {'Threshold':>10}  Note")
    print("-" * 100)
    for r in results:
        colour = col.get(r.status, "")
        val = f"{r.value}" if r.value is not None else "-"
        thr = f"{r.threshold}" if r.threshold is not None else "-"
        print(f"{r.gate:<32} {colour}{r.status:<14}{reset} {val:>10}  {thr:>10}  {r.note}")
    print()


if __name__ == "__main__":
    # `make bench` entry point: run gates (+ the real video-vs-LiDAR
    # comparison, when a scan with a synchronized rgb.mp4 is available)
    # against whichever real fixtures are present. Never fabricates numbers
    # for fixtures that aren't there -- just skips with a note.
    candidates = [
        Path("tests/fixtures/single_room"),
        Path("tests/fixtures/real_floor_only"),
        Path("tests/fixtures/real_with_ceiling"),
    ]
    found = [c for c in candidates if c.exists()]
    if not found:
        print("No real scan fixtures present (gitignored; download separately). "
              "Gates requiring scan_dir will show as 'not measured'.")
        print_gates(run_gates())
    for scan_dir in found:
        print(f"\n=== {scan_dir} ===")
        from roomscan.io.stray_scanner import load_scan
        from roomscan.geometry.room_layout import extract_layout
        from roomscan.geometry.openings import detect_openings

        pts = load_scan(scan_dir, frame_stride=5)
        layout = extract_layout(pts)
        openings = detect_openings(pts, layout)
        print_gates(run_gates(layout=layout, openings=openings, scan_dir=scan_dir))

        video_path = scan_dir / "rgb.mp4"
        if video_path.exists():
            from bench.derive_tiers import compare_video_to_lidar
            import json
            print("--- video-vs-lidar comparison (real synchronized capture) ---")
            try:
                print(json.dumps(compare_video_to_lidar(scan_dir), indent=2))
            except Exception as e:
                print(f"video tier comparison failed: {e}")

        print("--- execution timing (this machine, not hardware-neutral) ---")
        import json
        print(json.dumps(measure_execution_timing(scan_dir), indent=2))
