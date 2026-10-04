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
