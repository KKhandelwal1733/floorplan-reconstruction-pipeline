"""Load conformal calibration factors written by bench/calibrate.py.

Only returns a factor when that tier's target coverage was actually
achievable with the available calibration points (see conformal.py) --
otherwise returns None, so callers fall back to their existing hand-set
empirical floor rather than silently applying an undersized adjustment.
"""
from __future__ import annotations

import json
from pathlib import Path

FACTORS_PATH = Path(__file__).parent / "factors.json"


def load_calibration_factor(tier: str, path: Path = FACTORS_PATH) -> float | None:
    """Return the conformal CI-widening factor for `tier`, or None if no
    calibration file exists yet, or that tier's target coverage wasn't
    achievable with the calibration points available when it was written.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tier_result = data.get("tiers", {}).get(tier)
        if tier_result is None or not tier_result.get("achievable"):
            return None
        return float(tier_result["factor"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def apply_conformal_widening(lo: float, hi: float, value: float, factor: float) -> tuple[float, float]:
    """Multiply a CI's half-width by a conformal factor, keeping it centred
    on `value`."""
    half_width = (hi - lo) / 2 * factor
    return value - half_width, value + half_width
