"""Phase 10: plan.json builder tests (schema round-trip).

CLAUDE.md documents plan.json as a per-capture output; before Phase 10
nothing in the CLI ever wrote one. These check the builder produces
schema-valid output in both the success and total-abstention cases.
"""
import json

import numpy as np
import pytest

from roomscan.geometry.room_layout import extract_layout
from roomscan.plan_builder import (
    build_abstained_plan,
    build_single_room_plan,
    write_abstained_plan,
    write_single_room_plan,
)
from roomscan.schema_out import validate_plan_dict


def _room_layout():
    rng = np.random.default_rng(11)
    floor = rng.uniform(0, 4, (6000, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, 6000).astype(np.float32)
    wall = rng.uniform(0, 4, (2000, 3)).astype(np.float32)
    wall[:, 0] = 4.0
    wall[:, 2] = rng.uniform(0, 2.5, 2000).astype(np.float32)
    return extract_layout(np.concatenate([floor, wall]))


def test_build_single_room_plan_schema_valid():
    layout = _room_layout()
    plan = build_single_room_plan("lidar", layout)
    data = plan.model_dump(mode="json")
    for room in data["rooms"]:
        for d in room.get("damage", []):
            d["class"] = d.pop("class_")
    validate_plan_dict(data)


def test_write_single_room_plan_roundtrip(tmp_path):
    layout = _room_layout()
    out = tmp_path / "plan.json"
    write_single_room_plan("lidar", layout, out)
    data = json.loads(out.read_text())
    validate_plan_dict(data)
    assert data["capture"]["tier"] == "lidar"
    assert len(data["rooms"]) == 1
    assert data["rooms"][0]["floor_area_m2"]["value"] == pytest.approx(layout.floor_area_m2.value)


def test_build_abstained_plan_schema_valid():
    plan = build_abstained_plan("video", "SfM reconstruction produced only 3 points")
    data = plan.model_dump(mode="json")
    validate_plan_dict(data)
    assert data["capture"]["status"] == "abstained"
    assert data["rooms"] == []


def test_write_abstained_plan_roundtrip(tmp_path):
    out = tmp_path / "plan.json"
    write_abstained_plan("photo", "no rooms reconstructed", out)
    data = json.loads(out.read_text())
    validate_plan_dict(data)
    assert data["capture"]["tier"] == "photo"
    assert data["capture"]["status"] == "abstained"


def test_build_single_room_plan_degraded_status_when_ceiling_unobserved():
    layout = _room_layout()
    layout.ceiling_unobserved = True
    plan = build_single_room_plan("lidar", layout)
    assert plan.capture.status == "degraded"
