"""Conformal calibration of tier confidence intervals (Phase 9).

Standard split-conformal prediction: given a calibration set of (predicted
value, predicted CI, true value) triples, compute how many CI half-widths
off each prediction actually was (its "nonconformity score"), then take the
appropriate finite-sample quantile of those scores as a multiplicative
widening factor. Applied to new predictions, this gives a marginal coverage
guarantee -- ASSUMING the calibration set and future captures are drawn
from the same distribution ("exchangeable"), which is itself an assumption
worth stating, not a proof.

The finite-sample quantile formula below honestly reports when a target
coverage is mathematically unreachable with too few calibration points,
rather than silently returning an undersized factor (see
min_calibration_points_needed). With only a handful of real fixtures
available (see bench/calibrate.py), this is the expected outcome at the
90% confidence level this project otherwise uses -- the existing hand-set
empirical floors (config.VIDEO_EMPIRICAL_MIN_REL_HW / PHOTO_EMPIRICAL_MIN_REL_HW)
remain the active mechanism until enough real captures accumulate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationPoint:
    tier: str
    metric: str            # e.g. "floor_area_m2", "ceiling_height_m"
    predicted_value: float
    predicted_lo: float
    predicted_hi: float
    true_value: float      # pseudo-ground-truth (e.g. LiDAR-tier geometry)

    @property
    def half_width(self) -> float:
        return (self.predicted_hi - self.predicted_lo) / 2

    @property
    def nonconformity_score(self) -> float:
        """|true - predicted| normalized by the predicted CI half-width.

        Scale-invariant, so scores from different metrics (area vs height)
        can be pooled into one calibration set per tier -- a score of 1.0
        means the true value landed exactly on the original CI boundary.
        """
        hw = self.half_width
        if hw <= 0:
            return float("inf")
        return abs(self.true_value - self.predicted_value) / hw


def min_calibration_points_needed(target_coverage: float) -> int:
    """Smallest n for which a finite split-conformal quantile exists at all.

    Split-conformal needs ceil((n+1)*target_coverage) <= n; below that, no
    finite factor computed from n points can guarantee target_coverage on
    new data, regardless of how widely the existing points are spread.
    """
    n = 1
    while int(np.ceil((n + 1) * target_coverage)) > n:
        n += 1
        if n > 100_000:
            raise RuntimeError("unreachable")  # pragma: no cover
    return n


def conformal_quantile(scores: np.ndarray, target_coverage: float) -> float:
    """Finite-sample split-conformal quantile of nonconformity scores.

    Returns the smallest q such that, treating an (n+1)-th "always exceeds"
    score as present, at least ceil((n+1)*target_coverage) of the n+1
    scores are <= q. Returns float('inf') when that requires a point beyond
    the observed sample (see min_calibration_points_needed).
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * target_coverage))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


@dataclass
class TierCalibration:
    tier: str
    n_points: int
    target_coverage: float
    factor: float             # multiplicative CI-widening factor; inf if unreachable
    min_points_needed: int
    achievable: bool          # whether target_coverage is reachable with n_points


def calibrate_tier(
    points: list[CalibrationPoint],
    tier: str,
    target_coverage: float = 0.9,
) -> TierCalibration:
    """Pool all calibration points for one tier (across metrics) and compute
    its conformal widening factor at target_coverage."""
    tier_points = [p for p in points if p.tier == tier]
    scores = np.array([p.nonconformity_score for p in tier_points])
    factor = conformal_quantile(scores, target_coverage) if len(scores) else float("inf")
    min_needed = min_calibration_points_needed(target_coverage)
    return TierCalibration(
        tier=tier,
        n_points=len(tier_points),
        target_coverage=target_coverage,
        factor=factor,
        min_points_needed=min_needed,
        achievable=bool(np.isfinite(factor)),
    )
