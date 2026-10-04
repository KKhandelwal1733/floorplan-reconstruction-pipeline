"""Phase 10: damage detection, rules, and scope tests.

No labelled damage dataset exists to validate detection accuracy against
(see COMPLIANCE.md / config.py's DAMAGE_* comment block) -- this is a
crude, advisory-only colour heuristic, not a classifier. These tests check
behavioural correctness (detects an obvious synthetic stain, doesn't
false-positive on a uniform frame, never crashes, produces schema-valid
output) rather than real-world detection accuracy, which has not been
demonstrated and isn't claimed.
"""
import numpy as np
import pytest

from roomscan.damage.detector import detect_damage_candidates, detect_damage_candidates_multi
from roomscan.damage.pipeline import detect_damage_for_room
from roomscan.damage.rules import classify_damage, flags_for_damage, scope_for_damage
from roomscan.geometry.room_layout import extract_layout


def _room_cloud():
    rng = np.random.default_rng(7)
    floor = rng.uniform(0, 4, (6000, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, 6000).astype(np.float32)
    wall = rng.uniform(0, 4, (2000, 3)).astype(np.float32)
    wall[:, 0] = 4.0
    wall[:, 2] = rng.uniform(0, 2.5, 2000).astype(np.float32)
    return extract_layout(np.concatenate([floor, wall]))


def _uniform_frame(value=200, shape=(400, 600, 3)):
    return np.full(shape, value, dtype=np.uint8)


def _frame_with_patch(y0, y1, x0, x1, bg=200, patch=40, shape=(400, 600, 3)):
    frame = _uniform_frame(bg, shape)
    frame[y0:y1, x0:x1] = patch
    return frame


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def test_uniform_frame_no_candidates():
    assert detect_damage_candidates(_uniform_frame()) == []


def test_patch_frame_detects_one_candidate():
    frame = _frame_with_patch(250, 350, 100, 300)  # bottom-half dark patch
    candidates = detect_damage_candidates(frame)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.area_frac > 0
    assert c.mean_value < 0.5


def test_tiny_patch_filtered_out_as_too_small():
    frame = _frame_with_patch(0, 2, 0, 2)  # 2x2 px speck
    assert detect_damage_candidates(frame) == []


def test_detect_multi_tags_frame_index():
    frames = [_uniform_frame(), _frame_with_patch(250, 350, 100, 300)]
    candidates = detect_damage_candidates_multi(frames)
    assert len(candidates) == 1
    assert candidates[0].frame_index == 1


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def test_classify_damage_floor_zone():
    from roomscan.damage.detector import DamageCandidate
    c = DamageCandidate(0, bbox_px=(0, 320, 50, 50), area_frac=0.01, mean_value=0.2, mean_saturation=0.1)
    assert classify_damage(c, frame_height=400) == "possible_water_staining"


def test_classify_damage_ceiling_zone():
    from roomscan.damage.detector import DamageCandidate
    c = DamageCandidate(0, bbox_px=(0, 10, 50, 50), area_frac=0.01, mean_value=0.2, mean_saturation=0.1)
    assert classify_damage(c, frame_height=400) == "possible_ceiling_leak"


def test_classify_damage_middle_zone():
    from roomscan.damage.detector import DamageCandidate
    c = DamageCandidate(0, bbox_px=(0, 190, 50, 50), area_frac=0.01, mean_value=0.2, mean_saturation=0.1)
    assert classify_damage(c, frame_height=400) == "discoloration"


def test_large_floor_stain_triggers_flag():
    from roomscan.damage.detector import DamageCandidate
    c = DamageCandidate(0, bbox_px=(0, 320, 300, 50), area_frac=0.10, mean_value=0.3, mean_saturation=0.1)
    flags = flags_for_damage(c, "possible_water_staining")
    assert any(f.rule_id == "R001_large_floor_stain" for f in flags)


def test_very_dark_region_triggers_flag():
    from roomscan.damage.detector import DamageCandidate
    c = DamageCandidate(0, bbox_px=(0, 190, 10, 10), area_frac=0.001, mean_value=0.05, mean_saturation=0.1)
    flags = flags_for_damage(c, "discoloration")
    assert any(f.rule_id == "R002_very_dark_region" for f in flags)


def test_scope_for_damage_no_fabricated_cost():
    items = scope_for_damage("possible_water_staining", area_m2=1.5, surface_id="wall_0")
    assert len(items) == 1
    assert items[0].qty == 1.5
    assert items[0].unit == "m2"
    assert "cost" not in items[0].item.lower() and "$" not in items[0].item


# ---------------------------------------------------------------------------
# Pipeline (integration)
# ---------------------------------------------------------------------------

def test_pipeline_empty_frames_gives_empty_results():
    layout = _room_cloud()
    damages, scope = detect_damage_for_room([], layout)
    assert damages == []
    assert scope == []


def test_pipeline_detects_synthetic_stain():
    layout = _room_cloud()
    frame = _frame_with_patch(250, 350, 100, 300)
    damages, scope = detect_damage_for_room([frame], layout)
    assert len(damages) == 1
    assert damages[0].class_ == "possible_water_staining"
    assert damages[0].area_m2.value > 0
    assert damages[0].area_m2.lo < damages[0].area_m2.value < damages[0].area_m2.hi
    assert len(scope) == 1


def test_pipeline_never_crashes_with_no_walls():
    """A degenerate layout (no walls) must not crash the damage pipeline --
    surface_id falls back to 'unknown'."""
    layout = _room_cloud()
    layout.walls = []
    frame = _frame_with_patch(250, 350, 100, 300)
    damages, _ = detect_damage_for_room([frame], layout)
    assert all(d.surface_id == "unknown" for d in damages)


@pytest.mark.parametrize("n_frames", [1, 3])
def test_pipeline_runs_on_multiple_frames(n_frames):
    layout = _room_cloud()
    frames = [_frame_with_patch(250, 350, 100, 300) for _ in range(n_frames)]
    damages, scope = detect_damage_for_room(frames, layout)
    assert len(damages) == n_frames
    assert len(scope) == n_frames
