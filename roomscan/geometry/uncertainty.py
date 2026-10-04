"""Derive Measurement (value, lo, hi, confidence_level) from RANSAC inliers."""
from __future__ import annotations

import numpy as np

from roomscan.schema_out import Measurement

_Z90 = 1.6449  # z-score for 90 % CI


def measurement_from_inliers(
    values: np.ndarray,
    confidence_level: float = 0.9,
) -> Measurement:
    """Scalar Measurement from a set of inlier observations.

    Uses the sample mean as the point estimate and a normal-approximation
    CI: mean ± z * (std / sqrt(n)).  Falls back to std-only bounds when n < 2.
    """
    n = len(values)
    mu = float(values.mean())
    if n < 2:
        return Measurement(value=mu, lo=mu, hi=mu,
                           confidence_level=confidence_level)
    sigma = float(values.std(ddof=1))
    z = _Z90 if confidence_level == 0.9 else float(
        np.percentile(np.abs(np.random.default_rng(0).standard_normal(100_000)),
                      confidence_level * 100)
    )
    half = z * sigma / np.sqrt(n)
    return Measurement(value=mu, lo=mu - half, hi=mu + half,
                       confidence_level=confidence_level)
