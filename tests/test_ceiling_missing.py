"""Phase 5: ceiling-missing abstain / widen tests.

Verifies that when the ceiling plane is not detected:
  - ceiling_unobserved is True
  - ceiling_height_m is not None (we still emit an estimate)
  - The CI half-width is ≥ CEIL_UNOBSERVED_HALF_WIDTH_M (wide interval)
  - A capture_warning is emitted

Also verifies that a scan WITH a ceiling returns ceiling_unobserved=False and
a narrower CI.
"""
import numpy as np
import pytest
from pathlib import Path

from roomscan.geometry.room_layout import extract_layout
from roomscan.config import CEIL_UNOBSERVED_HALF_WIDTH_M

FIXTURES = Path(__file__).parent / "fixtures"
FLOOR_ONLY = FIXTURES / "single_scan_floor_only"
WITH_CEILING = FIXTURES / "single_scan_with_ceiling"


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _floor_only_cloud(n: int = 10_000) -> np.ndarray:
    """Box room cloud with NO ceiling points."""
    rng = np.random.default_rng(5)
    floor = rng.uniform(-3, 3, (n, 3)).astype(np.float32)
    floor[:, 2] = rng.normal(0, 0.005, n).astype(np.float32)

    # Walls carry their own small surface noise (0.01 m, larger than the floor's
    # 0.005 m) so they're never *exactly* flat — an exactly-constant coordinate
    # would look flatter than the noisy floor to the gravity heuristic and get
    # mis-picked as "up", which never happens with real (noisy) sensor data.
    walls: list[np.ndarray] = []
    for xval in [-3.0, 3.0]:
        w = rng.uniform(-3, 3, (n // 4, 3)).astype(np.float32)
        w[:, 0] = xval + rng.normal(0, 0.01, n // 4).astype(np.float32)
        w[:, 2] = rng.uniform(0, 2.5, n // 4).astype(np.float32)
        walls.append(w)
    for yval in [-3.0, 3.0]:
        w = rng.uniform(-3, 3, (n // 4, 3)).astype(np.float32)
        w[:, 1] = yval + rng.normal(0, 0.01, n // 4).astype(np.float32)
        w[:, 2] = rng.uniform(0, 2.5, n // 4).astype(np.float32)
        walls.append(w)

    return np.concatenate([floor] + walls)


def _room_with_ceiling_cloud(n: int = 10_000, ceiling_h: float = 2.6) -> np.ndarray:
    """Box room cloud WITH ceiling points."""
    base = _floor_only_cloud(n)
    rng = np.random.default_rng(6)
    ceil_pts = rng.uniform(-3, 3, (n // 2, 3)).astype(np.float32)
    ceil_pts[:, 2] = rng.normal(ceiling_h, 0.005, n // 2).astype(np.float32)
    return np.concatenate([base, ceil_pts])


# ---------------------------------------------------------------------------
# Synthetic tests (always run)
# ---------------------------------------------------------------------------

def test_ceiling_unobserved_flag_synthetic():
    pts = _floor_only_cloud()
    layout = extract_layout(pts)
    assert layout.ceiling_unobserved, "floor-only cloud should set ceiling_unobserved=True"


def test_ceiling_unobserved_emits_estimate():
    pts = _floor_only_cloud()
    layout = extract_layout(pts)
    assert layout.ceiling_height_m is not None, \
        "must emit a ceiling_height_m even when ceiling not detected"


def test_ceiling_unobserved_wide_ci():
    pts = _floor_only_cloud()
    layout = extract_layout(pts)
    m = layout.ceiling_height_m
    half_width = (m.hi - m.lo) / 2
    assert half_width >= CEIL_UNOBSERVED_HALF_WIDTH_M, (
        f"CI half-width {half_width:.3f} m < required {CEIL_UNOBSERVED_HALF_WIDTH_M} m"
    )


def test_ceiling_unobserved_warning():
    pts = _floor_only_cloud()
    layout = extract_layout(pts)
    assert layout.capture_warnings, "should emit at least one warning when ceiling missing"
    assert any("ceiling" in w.lower() for w in layout.capture_warnings)


def test_ceiling_observed_flag_synthetic():
    pts = _room_with_ceiling_cloud()
    layout = extract_layout(pts)
    assert not layout.ceiling_unobserved, \
        "room with ceiling should have ceiling_unobserved=False"


def test_ceiling_observed_narrower_ci():
    pts = _room_with_ceiling_cloud()
    layout = extract_layout(pts)
    if layout.ceiling_height_m is None:
        pytest.skip("ceiling not detected in synthetic cloud")
    half_width = (layout.ceiling_height_m.hi - layout.ceiling_height_m.lo) / 2
    assert half_width < CEIL_UNOBSERVED_HALF_WIDTH_M, (
        f"observed ceiling CI half-width {half_width:.3f} m should be narrower than "
        f"{CEIL_UNOBSERVED_HALF_WIDTH_M} m"
    )


# ---------------------------------------------------------------------------
# Fixture-based tests (Stray Scanner format, skipped when absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FLOOR_ONLY.exists(), reason="single_scan_floor_only fixture not present")
def test_fixture_floor_only_unobserved():
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(FLOOR_ONLY, max_frames=50)
    layout = extract_layout(pts)
    assert layout.ceiling_unobserved
    assert layout.ceiling_height_m is not None
    hw = (layout.ceiling_height_m.hi - layout.ceiling_height_m.lo) / 2
    assert hw >= CEIL_UNOBSERVED_HALF_WIDTH_M


@pytest.mark.skipif(not WITH_CEILING.exists(), reason="single_scan_with_ceiling fixture not present")
def test_fixture_with_ceiling_observed():
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(WITH_CEILING, max_frames=50)
    layout = extract_layout(pts)
    assert not layout.ceiling_unobserved


# ---------------------------------------------------------------------------
# Real Stray Scanner captures (large, gitignored; see .gitignore).
#
# frame_stride (not max_frames) is used deliberately: Stray Scanner frames are
# chronological, so a prefix subset can see only part of the room (e.g. the
# floor-walking portion before the user ever tilts up). A stride spans the
# whole capture timeline and was verified to match the full-scan (stride=1)
# result for real_floor_only: both show no ceiling plane, floor_area ~83-93
# sq m. Full stride=1 loads on real_with_ceiling were not practical in this
# environment (I/O-bound on ~19k individual PNG reads, not CPU/memory).
# ---------------------------------------------------------------------------

REAL_FLOOR_ONLY = FIXTURES / "real_floor_only"
REAL_WITH_CEILING = FIXTURES / "real_with_ceiling"


@pytest.mark.skipif(not REAL_FLOOR_ONLY.exists(), reason="real_floor_only fixture not present")
def test_real_floor_only_ceiling_unobserved():
    """Confirmed against the full 5251-frame scan (not just this stride=5
    subsample): no horizontal plane above the floor ever reaches the ceiling
    fill-ratio bar. floor_area ~83.7 sq m on the full scan."""
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(REAL_FLOOR_ONLY, frame_stride=5)
    layout = extract_layout(pts)
    assert layout.ceiling_unobserved
    assert layout.floor_area_m2.value > 10  # sanity: a real room, not a fragment
    hw = (layout.ceiling_height_m.hi - layout.ceiling_height_m.lo) / 2
    assert hw >= CEIL_UNOBSERVED_HALF_WIDTH_M


@pytest.mark.skipif(not REAL_WITH_CEILING.exists(), reason="real_with_ceiling fixture not present")
def test_real_with_ceiling_detected_after_fix_loop():
    """Phase 11 fix loop (see DECLARATION.md): the highest horizontal plane
    candidate here (room-height estimate ~2.4 m, physically plausible) only
    reached a 2-D fill ratio of ~0.23 - too close to the ~0.28-0.29 fill
    ratio measured on a synthetic wall-top false positive to trust alone.
    interior_frac (what fraction of inliers fall in the shrunk-margin
    interior of the candidate's own bbox) separates the two cases by a wide
    margin instead (0.0 for any wall-top ring vs 0.6-0.73 here) -- but
    testing against real_floor_only (which has no real ceiling) surfaced a
    third failure mode: a large interior surface that isn't a wall-top ring
    (plausibly furniture) can have interior_frac nearly identical to a
    genuine ceiling's. Implied room height (1.21 m there vs 2.39 m here)
    is what actually tells them apart, so acceptance now requires a
    plausible height AND (fill_ratio OR interior_frac)."""
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(REAL_WITH_CEILING, frame_stride=5)
    layout = extract_layout(pts)
    assert not layout.ceiling_unobserved
    assert layout.floor_area_m2.value > 10
    assert 1.8 <= layout.ceiling_height_m.value <= 6.0
    # a real (non-abstained) measurement should be far narrower than the
    # abstain-path's wide CI
    hw = (layout.ceiling_height_m.hi - layout.ceiling_height_m.lo) / 2
    assert hw < CEIL_UNOBSERVED_HALF_WIDTH_M
