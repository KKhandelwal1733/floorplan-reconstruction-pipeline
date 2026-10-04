"""Reproduction bundle (case-study realignment): `make repro` / `make live`.

This project calls no external model/API (see docs/model_registry.md --
every stage is classical geometry/CV/statistics, run entirely offline), so
there is no learned "model output" to cache in the sense the brief
describes for a VLM/API damage classifier. What cache/ stores instead is
this pipeline's OWN deterministic output for a given (input, tier) --
computing it is the expensive part (loading a multi-million-point LiDAR
scan, running SfM), and the project's existing determinism hard rule
(config.SEED, sorted file iteration, no wall-clock dependence -- see
tests/test_determinism.py) is exactly what makes a cached replay valid:
it's guaranteed bit-identical to a fresh run, not merely "close enough."

make live:  always recomputes from raw inputs, cache/ untouched.
make repro: replays from cache/ if a prior run exists, else computes once
            and populates it -- then (either way) runs live TWICE more and
            checks plan.json is byte-identical across all of it. That
            comparison, not the cache file itself, is the actual evidence.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

CACHE_ROOT = Path("cache")


def _cache_key(input_path: Path, tier: str) -> str:
    raw = f"{Path(input_path).resolve()}|{tier}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


_OUTPUT_FILES = ("plan.json", "plan.svg", "report.md")


def run_live(input_path: Path, tier: str, out_dir: Path) -> None:
    """Full live path: always recomputes from raw inputs, cache/ untouched."""
    from roomscan.cli import main as cli_main

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cli_main(["run", str(input_path), "--tier", tier, "--out", str(out_dir)])


def run_repro(
    input_path: Path, tier: str, out_dir: Path, cache_root: Path = CACHE_ROOT,
) -> bool:
    """Replay from cache_root/ if a prior run exists for this (input, tier);
    otherwise compute once live and populate the cache.

    Returns True if this call replayed from cache (didn't recompute),
    False if it had to compute.
    """
    cached_dir = Path(cache_root) / _cache_key(input_path, tier)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (cached_dir / "plan.json").exists():
        for name in _OUTPUT_FILES:
            src = cached_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
        return True

    run_live(input_path, tier, out_dir)
    cached_dir.mkdir(parents=True, exist_ok=True)
    for name in _OUTPUT_FILES:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, cached_dir / name)
    return False


def _plan_hash(out_dir: Path) -> str:
    text = (Path(out_dir) / "plan.json").read_text(encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def verify_reproducible(
    input_path: Path,
    tier: str,
    work_dir: Path = Path("out/_repro_check"),
    cache_root: Path = CACHE_ROOT,
) -> dict:
    """The actual evidence: populate/replay cache_root/, then run live twice
    more (bypassing the cache both times) and check every plan.json
    produced is byte-identical -- the determinism claim this bundle exists
    to support, not just "the cache returned something."
    """
    work_dir = Path(work_dir)
    cache_dir = work_dir / "cache_run"
    live_a = work_dir / "live_a"
    live_b = work_dir / "live_b"

    replayed = run_repro(input_path, tier, cache_dir, cache_root=cache_root)
    run_live(input_path, tier, live_a)
    run_live(input_path, tier, live_b)

    hashes = {
        "cache_run": _plan_hash(cache_dir),
        "live_a": _plan_hash(live_a),
        "live_b": _plan_hash(live_b),
    }
    identical = len(set(hashes.values())) == 1
    return {
        "input_path": str(input_path),
        "tier": tier,
        "cache_replayed_first_call": replayed,
        "plan_json_hashes": hashes,
        "identical": identical,
        "note": (
            "cache replay and two independent live runs all produced byte-"
            "identical plan.json -- reproduction verified"
            if identical else
            "plan.json differed between runs -- determinism is broken, "
            "investigate before trusting any cached or repeated output"
        ),
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "repro"
    input_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tests/fixtures/single_room")
    tier = sys.argv[3] if len(sys.argv) > 3 else "lidar"

    if mode == "live":
        run_live(input_path, tier, Path("out"))
        print(f"[repro] live run complete -> out/ (cache bypassed; plan hash {_plan_hash(Path('out'))})")
    else:
        result = verify_reproducible(input_path, tier)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["identical"] else 1)
