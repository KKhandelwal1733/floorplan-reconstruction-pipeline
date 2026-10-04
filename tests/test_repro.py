"""Reproduction bundle tests (case-study realignment): bench/repro.py.

The cache-mechanics tests stub out run_live (the real pipeline is slow and
already covered elsewhere) to check hit/miss behaviour in isolation. One
slow, real-fixture test exercises the actual determinism claim end to end.
"""
from pathlib import Path

import pytest

import bench.repro as repro_mod
from bench.repro import _cache_key, run_repro, verify_reproducible

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_ROOM = FIXTURES / "single_room"


def test_cache_key_differs_by_tier():
    key_lidar = _cache_key(Path("some/input"), "lidar")
    key_video = _cache_key(Path("some/input"), "video")
    assert key_lidar != key_video


def test_cache_key_stable_for_same_input_and_tier():
    assert _cache_key(Path("some/input"), "lidar") == _cache_key(Path("some/input"), "lidar")


def _stub_run_live(calls: list):
    def _run(input_path, tier, out_dir):
        calls.append((str(input_path), tier))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plan.json").write_text('{"ok": true}')
        (out_dir / "plan.svg").write_text("<svg></svg>")
        (out_dir / "report.md").write_text("# fake report\n")
    return _run


def test_run_repro_second_call_hits_cache_and_skips_recompute(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(repro_mod, "run_live", _stub_run_live(calls))

    cache_root = tmp_path / "cache"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    replayed_first = run_repro(Path("fake_input"), "lidar", out1, cache_root=cache_root)
    replayed_second = run_repro(Path("fake_input"), "lidar", out2, cache_root=cache_root)

    assert replayed_first is False     # cache miss -> had to compute
    assert replayed_second is True      # cache hit -> replayed, no recompute
    assert len(calls) == 1              # run_live only invoked once total
    assert (out1 / "plan.json").read_text() == (out2 / "plan.json").read_text()


def test_run_repro_different_tier_is_a_separate_cache_entry(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(repro_mod, "run_live", _stub_run_live(calls))

    cache_root = tmp_path / "cache"
    run_repro(Path("fake_input"), "lidar", tmp_path / "out_lidar", cache_root=cache_root)
    run_repro(Path("fake_input"), "video", tmp_path / "out_video", cache_root=cache_root)

    assert len(calls) == 2   # distinct tiers never share a cache entry


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="single_room fixture not present")
def test_verify_reproducible_real_fixture(tmp_path):
    # explicit cache_root: never touch the project's real cache/ from a test
    result = verify_reproducible(SINGLE_ROOM, "lidar", work_dir=tmp_path / "repro_check",
                                  cache_root=tmp_path / "cache")
    assert result["identical"] is True
    assert len(set(result["plan_json_hashes"].values())) == 1
