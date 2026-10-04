"""Phase 9: conformal calibration algorithm tests.

These check the split-conformal math itself against hand-worked cases, not
real data (that's bench/calibrate.py + test_calibrate.py). The finite-sample
quantile formula is the standard split-conformal construction; get this
right and the real-data numbers follow.
"""
import numpy as np
import pytest

from roomscan.calibration.conformal import (
    CalibrationPoint,
    calibrate_tier,
    conformal_quantile,
    min_calibration_points_needed,
)


def test_min_points_needed_90pct():
    # ceil((n+1)*0.9) <= n first holds at n=9: ceil(10*0.9)=9<=9
    assert min_calibration_points_needed(0.9) == 9


def test_min_points_needed_50pct():
    # ceil((n+1)*0.5) <= n first holds at n=1: ceil(2*0.5)=1<=1
    assert min_calibration_points_needed(0.5) == 1


def test_conformal_quantile_empty():
    assert conformal_quantile(np.array([]), 0.9) == float("inf")


def test_conformal_quantile_unreachable_with_too_few_points():
    # n=3 < min_calibration_points_needed(0.9)=9 -> must be unreachable
    scores = np.array([0.1, 0.2, 0.3])
    assert conformal_quantile(scores, 0.9) == float("inf")


def test_conformal_quantile_reachable_with_enough_points():
    # n=9 is exactly enough for 90% target: rank = ceil(10*0.9) = 9 = n
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q = conformal_quantile(scores, 0.9)
    assert q == pytest.approx(0.9)  # the largest (9th) score


def test_conformal_quantile_matches_hand_worked_example():
    # n=4, target=0.5 -> rank = ceil(5*0.5) = 3 -> 3rd smallest score
    scores = np.array([4.0, 1.0, 3.0, 2.0])
    q = conformal_quantile(scores, 0.5)
    assert q == pytest.approx(3.0)


def test_calibration_point_nonconformity_score():
    p = CalibrationPoint(
        tier="video", metric="floor_area_m2",
        predicted_value=10.0, predicted_lo=8.0, predicted_hi=12.0,
        true_value=14.0,
    )
    # half_width = 2.0, |14-10| = 4.0 -> score = 2.0
    assert p.half_width == pytest.approx(2.0)
    assert p.nonconformity_score == pytest.approx(2.0)


def test_calibration_point_zero_half_width_gives_inf_score():
    p = CalibrationPoint(
        tier="video", metric="floor_area_m2",
        predicted_value=10.0, predicted_lo=10.0, predicted_hi=10.0,
        true_value=11.0,
    )
    assert p.nonconformity_score == float("inf")


def test_calibrate_tier_pools_across_metrics():
    points = [
        CalibrationPoint("video", "floor_area_m2", 10.0, 9.0, 11.0, 10.5),   # score 0.5
        CalibrationPoint("video", "ceiling_height_m", 2.0, 1.8, 2.2, 2.3),   # score 1.5
        CalibrationPoint("photo", "floor_area_m2", 10.0, 9.0, 11.0, 20.0),   # different tier
    ]
    cal = calibrate_tier(points, "video", target_coverage=0.5)
    assert cal.n_points == 2
    assert cal.tier == "video"
    # n=2, target=0.5 -> rank=ceil(3*0.5)=2 -> 2nd smallest of [0.5, 1.5] = 1.5
    assert cal.achievable
    assert cal.factor == pytest.approx(1.5)


def test_calibrate_tier_no_points_for_tier():
    cal = calibrate_tier([], "video", target_coverage=0.9)
    assert cal.n_points == 0
    assert not cal.achievable
    assert cal.factor == float("inf")
