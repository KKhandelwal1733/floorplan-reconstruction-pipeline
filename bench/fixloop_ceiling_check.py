"""Reproducible before/after evidence for the Phase 11 fix loop (see
DECLARATION.md): does find_floor_ceiling correctly accept the real
`real_with_ceiling` ceiling candidate while still rejecting a wall-top ring?

Run: python -m bench.fixloop_ceiling_check
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from roomscan.geometry.planes import (
    _ceiling_fill_ratio,
    _ceiling_interior_frac,
    detect_gravity,
    find_floor_ceiling,
)


def _synthetic_wall_top_cloud() -> np.ndarray:
    """A floor-only room (no real ceiling) whose wall tops form a horizontal
    ring -- the false positive this check must keep rejecting."""
    rng = np.random.default_rng(5)
    floor = rng.uniform(-3, 3, (10_000, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, 10_000).astype(np.float32)
    walls = []
    for xval in [-3.0, 3.0]:
        w = rng.uniform(-3, 3, (2500, 3)).astype(np.float32)
        w[:, 0] = xval + rng.normal(0, 0.01, 2500).astype(np.float32)
        w[:, 2] = rng.uniform(0, 2.5, 2500).astype(np.float32)
        walls.append(w)
    for yval in [-3.0, 3.0]:
        w = rng.uniform(-3, 3, (2500, 3)).astype(np.float32)
        w[:, 1] = yval + rng.normal(0, 0.01, 2500).astype(np.float32)
        w[:, 2] = rng.uniform(0, 2.5, 2500).astype(np.float32)
        walls.append(w)
    return np.concatenate([floor] + walls)


def _case_result(pts: np.ndarray, label: str) -> dict[str, Any]:
    gravity = detect_gravity(pts)
    floor, ceiling = find_floor_ceiling(pts, gravity=gravity)
    result: dict[str, Any] = {"case": label, "ceiling_detected": ceiling is not None}
    if ceiling is not None and floor is not None:
        floor_h = float(np.median(pts[floor[2]] @ gravity))
        ceil_h = float(np.median(pts[ceiling[2]] @ gravity))
        ceil_pts = pts[ceiling[2]]
        result.update({
            "implied_room_height_m": round(abs(ceil_h - floor_h), 3),
            "fill_ratio": round(_ceiling_fill_ratio(ceil_pts, gravity), 3),
            "interior_frac": round(_ceiling_interior_frac(ceil_pts, gravity), 3),
        })
    return result


def run_check(fixtures_dir: Path = Path("tests/fixtures")) -> dict[str, Any]:
    results = [_case_result(_synthetic_wall_top_cloud(), "synthetic_wall_top (must reject)")]

    for name, expect_ceiling in [
        ("real_floor_only", False),
        ("real_with_ceiling", True),
    ]:
        scan_dir = fixtures_dir / name
        if not scan_dir.exists():
            results.append({"case": name, "skipped": "fixture not present (gitignored)"})
            continue
        from roomscan.io.stray_scanner import load_scan
        pts = load_scan(scan_dir, frame_stride=5)
        r = _case_result(pts, f"{name} (expect ceiling_detected={expect_ceiling})")
        r["pass"] = r["ceiling_detected"] == expect_ceiling
        results.append(r)

    return {"results": results}


if __name__ == "__main__":
    print(json.dumps(run_check(), indent=2))
