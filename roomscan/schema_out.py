"""Pydantic models for plan.json (schema v0.1) + validation against schema/plan.schema.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import jsonschema
from pydantic import BaseModel

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "plan.schema.json"


class Measurement(BaseModel):
    value: float
    lo: float
    hi: float
    confidence_level: float = 0.9


class Pose(BaseModel):
    x: float
    y: float
    yaw: float


class Wall(BaseModel):
    id: str
    length_m: Measurement


class Opening(BaseModel):
    id: str
    type: Literal["door", "window"]
    wall: str
    width_m: Measurement
    confidence: float


class DamageFlag(BaseModel):
    rule_id: str
    text: str
    evidence: dict = {}


class Damage(BaseModel):
    id: str
    surface_id: str
    class_: str
    area_m2: Measurement
    extent: dict = {}
    flags: list[DamageFlag] = []

    model_config = {"populate_by_name": True}


class ScopeItem(BaseModel):
    item: str
    surface_id: str
    qty: float
    unit: str


class Room(BaseModel):
    id: str
    polygon: list[tuple[float, float]]
    pose: Pose
    walls: list[Wall]
    ceiling_height_m: Measurement
    floor_area_m2: Measurement
    openings: list[Opening] = []
    damage: list[Damage] = []
    scope: list[ScopeItem] = []


class Adjacency(BaseModel):
    a: str
    b: str
    via: str
    confidence: float


class Capture(BaseModel):
    tier: Literal["lidar", "video", "photo"]
    status: Literal["ok", "degraded", "abstained"]
    warnings: list[str] = []


class PropertyBlock(BaseModel):
    rooms: list[str]
    adjacency: list[Adjacency] = []
    footprint_m2: Measurement


class Plan(BaseModel):
    schema_version: Literal["0.1"] = "0.1"
    capture: Capture
    property: PropertyBlock
    rooms: list[Room]


_schema_cache: dict | None = None


def _load_schema() -> dict:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = json.loads(SCHEMA_PATH.read_text())
    return _schema_cache


def validate_plan_dict(plan: dict) -> None:
    jsonschema.validate(plan, _load_schema())


def write_plan(plan: Plan, out_path: Path) -> None:
    data = plan.model_dump(mode="json", by_alias=False)
    # Damage.class_ -> "class" in JSON per schema
    for room in data.get("rooms", []):
        for d in room.get("damage", []):
            d["class"] = d.pop("class_")
    validate_plan_dict(data)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
