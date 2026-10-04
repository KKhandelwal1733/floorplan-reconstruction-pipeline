# roomscan: Technical Report

*Applied AI case study — floor plan reconstruction from handheld iPhone captures*

## 1. Executive Summary

`roomscan` reconstructs dimensioned floor plans, with confidence intervals,
damage flags, and remediation scope, from one of three iPhone capture tiers:
LiDAR (Stray Scanner), a single handheld video, or 2-8 stills per room. It was
built across 12 phases, each committed separately, with a consistent
methodology: implement the real mechanism, test it against real captured data
as soon as any existed, and report what was actually found — including
several cases where the first implementation didn't hold up and had to be
revised. That discipline is the throughline of this report.

**Headline honest finding**: the LiDAR tier is solid and well-tested. The
video and photo tiers, built with a deliberately lightweight (no bundle
adjustment, CPU-only) monocular structure-from-motion approach, measured
**65-99%** and **~94%** error respectively against LiDAR pseudo-ground-truth —
nowhere near the aspirational ±3%/±8% gates in the original brief. For one
real capture, the video tier's reported interval doesn't just have a large
error, it fails to cover the true value at all (Section 6). This is
disclosed prominently (COMPLIANCE.md, README.md) rather than hidden, and
every confidence interval in those tiers is widened to reflect it.

## 2. Architecture

Three capture tiers converge on one shared geometry core
(`roomscan/geometry/room_layout.py::extract_layout`): given *any* fused 3-D
point cloud, it detects gravity direction, fits floor/ceiling planes via
RANSAC, extracts a 2-D floor polygon and wall segments, and attaches a
confidence interval to every measurement. The LiDAR tier feeds it a dense
point cloud directly from depth data; the video and photo tiers first
reconstruct a *sparse* point cloud via classical monocular SfM (ORB/SIFT
features, essential-matrix pose recovery, DLT triangulation — see
`roomscan/geometry/sfm.py`) and recover metric scale via an ensemble of
physical-size priors (`scale_ensemble.py`) before handing off to the same
`extract_layout`.

A quality gate (floor point count, point density) widens CIs further when
coverage looks partial. A split-conformal calibration layer
(`roomscan/calibration/`) sits on top, ready to apply a statistically
justified additional widening factor once enough real calibration data
exists — currently a documented no-op (Section 6).

Damage detection (`roomscan/damage/`) is a separate, simple pipeline: HSV
colour-blob detection on RGB frames, rule-based classification, and
scope-item generation — explicitly not a trained classifier (no labelled
damage data exists for this project).

## 3. Methodology and Key Findings, Phase by Phase

**Phases 1-3** (scaffold, LiDAR loader, layout extraction) established the
core geometry pipeline and schema (v0.1, `schema/plan.schema.json`) against
the first real Stray Scanner capture.

**Phases 4-5** (openings, ceiling-missing abstain) added door/window
detection via occupancy-grid analysis and a graceful fallback (wide-CI
estimate from the point cloud's own upper bound) when no ceiling plane is
detected. A real bug was found and fixed here: the convex hull of a noisy
floor point cloud produces dozens of micro-segments per physical wall, not
one clean segment — fixed with angle-based vertex merging
(`_simplify_polygon`), which also made the SVG/schema wall output
meaningfully cleaner.

**Phase 6** (quality gate, determinism, repeatability) found and fixed a real
determinism bug: `roomscan/geometry/planes.py` used a module-level shared
random number generator, so results silently depended on how many prior
random draws had happened earlier in the *same process* — a genuine
violation of the project's determinism rule, not just test flakiness. Fixed
by seeding fresh per call. A split-scan repeatability gate (two interleaved
frame subsets of the same capture, compared with no ground truth needed) now
passes against real data.

**Phase 7** (video tier) is where the honest-accuracy discipline really
kicks in. Built with OpenCV (chosen over COLMAP specifically to avoid a
large/GPU dependency — a deliberate trade-off, not an oversight), the first
working version measured **65-91% error** against real LiDAR ground truth for
the same room. Two real bugs were fixed along the way (outlier contamination
in triangulated points from near-zero-parallax matches; a metric-tuned RANSAC
threshold wrongly applied to an unscaled point cloud), but the residual error
remained large and, notably, *unstable* — a small change to one scale-prior
constant shifted results in an unexpected, non-monotonic direction, a sign
that pairwise-chained reconstruction without bundle adjustment has real,
not-easily-tuned-away error sources. Rather than keep chasing a tighter
number against one example (overfitting risk), the confidence interval was
floored at this empirically observed error rate and documented as such.

**Phase 8** (photo tier, multi-room stitcher, drift ablation) found that
consecutive-frame matching (reusing the video tier's strategy) found *zero*
usable pairs for realistic photo counts — stills taken seconds apart during a
walkthrough share far less visual overlap than video's narrow-baseline
consecutive frames. Fixed with an all-pairs best-match strategy. A second,
subtler bug surfaced during that fix: a well-matched pair can still be
numerically degenerate to triangulate (near-zero real camera baseline
collapses every point to the same spot) — now detected and skipped in favour
of the next-best candidate. Even after both fixes, reconstruction succeeded
in only one of four simulated trials at realistic photo counts, with ~94%
error when it did succeed. This is reported as expected, correct behavior for
this method at this data sparsity (abstain over confident garbage), not a
remaining bug. The multi-room stitcher is deliberately schematic only — no
real multi-room fixture exists to validate true cross-room geometric
stitching against, so it only flags likely adjacency via best-effort feature
matching and always falls back to a simple grid layout. The drift ablation
(reconstructing the same real video at several frame counts) found, somewhat
surprisingly, that error did *not* grow with chain length — consistent with
other noise sources (scale priors, degenerate triangulation) dominating over
pose-chaining drift for this method.

**Phase 9** (conformal calibration) implemented standard split-conformal
prediction, correctly validated against hand-worked examples. Run against the
6 real calibration points available (4 video, 2 photo, after some
reconstruction abstentions), it honestly reports **"not achievable" at the
project's 90% target coverage** — split-conformal mathematically needs at
least 9 calibration points for any finite 90%-coverage factor regardless of
how spread out they are. A real circularity bug was caught and fixed before
it could bite: the functions that *gather* calibration data must skip
applying any previously-computed calibration factor, or recalibrating from
already-calibrated data would compound incorrectly.

**Phase 10** (damage module, plan.json) added colour-heuristic damage
detection and closed a real, previously undocumented gap: `plan.json` was
documented as a standard output since Phase 1, but nothing in the CLI had
ever written one. Tested against a real, non-damaged room, the damage
heuristic flagged 23 candidates across 15 frames — an expected high
false-positive rate for a heuristic with no labelled training/validation
data, documented rather than hidden.

**Phase 11** (fix loop) is a dedicated, fully-documented
declare→fix→before/after cycle on a specific, previously-identified issue:
the ceiling-detection heuristic rejected a genuine ceiling in real test data
because its single discriminating signal (2-D fill ratio) was too close
between a real patchy ceiling (0.23) and a synthetic wall-top false positive
(0.28-0.29) to trust. The first fix (an "interior concentration" signal)
worked for that case but, when tested against a *different* real fixture,
surfaced a third failure mode nobody had specifically tested for: a large
interior surface that isn't a wall-top ring (plausibly furniture) can have
nearly identical interior-concentration to a genuine ceiling. The real
discriminator turned out to be implied room height (1.21 m for the false
positive vs 2.39 m for the genuine ceiling) — physically implausible vs
plausible. The final fix requires a plausible height *and* either of the
original two signals. Full before/after evidence, including the mid-course
correction, is in `DECLARATION.md` and reproducible via
`bench/fixloop_ceiling_check.py`.

## 4. Tier Accuracy Results

| Metric | LiDAR | Video | Photo |
|---|---|---|---|
| Method | Dense RANSAC geometry | Monocular SfM + scale ensemble | Same, best-pair strategy |
| Validated against | — (no laser ground truth exists) | Real LiDAR pseudo-ground-truth, n=2 real captures with synchronized video (a 3rd, `real_with_ceiling`, honestly abstains rather than comparing) | Simulated-from-real stills, n=4 trials |
| Floor-area error | Not measured (no independent ground truth) | 65-99% (`single_room` 91.2%, `real_floor_only` 99.2%, both freshly regenerated and confirmed reproducible across reruns) | ~94% (1 success of 4 trials) |
| Typical outcome | Confident measurement | Confident but wide-CI measurement | Frequent honest abstention |
| CI mechanism | RANSAC inlier residuals + quality gate | Empirical floor (`VIDEO_EMPIRICAL_MIN_REL_HW=0.85`) | Empirical floor (`PHOTO_EMPIRICAL_MIN_REL_HW=0.90`) |

## 5. Error Budget

Every `Measurement` carries a confidence interval, but the project does not
(yet) decompose that interval into an additive budget across independent
noise sources — each CI is a **max/floor over whichever single signal is
most conservative** for that measurement, not a quadrature sum. The table
below names every contributor actually present in the code and whether it's
modeled:

| Source | Where | Modeled in CI? |
|---|---|---|
| Plane-fit / sensor noise (RANSAC inlier spread) | `roomscan/geometry/uncertainty.py::measurement_from_inliers` (all tiers) | Yes — the base mechanism for every measurement |
| View coverage / point density | Quality gate (`QUALITY_MIN_FLOOR_INLIERS`, `QUALITY_MIN_PTS_PER_M2`) | Yes — widens CI by `QUALITY_CI_WIDEN_FACTOR` when triggered |
| Capture-quality issues (low light, mirror/glass/wet-look glare) | `roomscan/geometry/capture_quality.py` | Yes — widens CI by the same factor when flagged |
| Scale-recovery disagreement (video/photo) | `scale_ensemble.py::combine_estimates`, ensemble spread | Yes, but see Section 6 — this signal has been observed to *underestimate* real error on its own |
| Systematic monocular-SfM bias (no bundle adjustment) | video/photo tiers | Only indirectly, via the hand-set empirical floor (`VIDEO_/PHOTO_EMPIRICAL_MIN_REL_HW`) — not decomposed from the sources above |
| Multi-room pose-graph placement error | `roomscan/geometry/pose_graph.py` | Not at all — a placed room's pose carries no CI of its own; `has_overlap` rejects implausible placements but doesn't bound plausible ones. The synthetic drift ablation (`bench/ablate.py`) demonstrates the correction *mechanism* works, not a measured real-world placement error |
| Device/sensor calibration error (camera intrinsics, LiDAR depth bias) | Not modeled anywhere | No — would require laser ground truth to even estimate, which doesn't exist for this project (Section 7) |

A fully decomposed, additive error budget (`total_variance = sensor² +
coverage² + scale² + ...`, each term independently calibrated) is the
natural next step, but requires per-source ground truth this project
doesn't have — the same blocking constraint as conformal calibration
(Section 8). What's delivered here is the literal map above: every named
source, and an honest accounting of which ones the current CI mechanism
actually responds to.

## 6. Bias vs. Variance Diagnosis

**Variance** (repeat-measurement spread, no ground truth needed) is directly
measurable via the split-scan repeatability gate (two interleaved frame
subsets of the same physical capture). Run fresh against all three real
fixtures for this report (`python -m bench.harness`):

| Fixture | Floor-area spread | Wall-perimeter spread | Gate (≤5%/≤5%) |
|---|---|---|---|
| `real_floor_only` | 0.9% | 0.5% | PASS |
| `real_with_ceiling` | 1.3% | 0.6% | PASS |
| `single_room` | 6.0% | 1.8% | **FAIL** (area) |

This is a genuine, previously-unreported finding, surfaced while assembling
this section: `single_room`'s own split-scan repeatability had never
actually been checked against the project's gate before (the passing test,
`tests/test_repeatability.py`, runs against `real_floor_only`, not
`single_room`). At the same methodology (two interleaved halves,
`frame_stride=20`), `single_room` — the largest and most geometrically
complex of the three real captures — fails the 5% area threshold. Variance
is evidently capture-dependent, not a single constant the project can claim
once and reuse; this is now also logged under Known Limitations (Section
10) rather than only here.

**Bias** (systematic offset from the true room) cannot be measured directly
— no laser/tape ground truth exists (Section 7), and even the LiDAR-tier
"pseudo-ground-truth" used for cross-tier comparison is itself an assumed
±1-2cm, not independently verified. What *can* be examined is whether the
video tier's own reported uncertainty (ensemble spread) is consistent with
its observed error against that pseudo-ground-truth — and the answer
differs by capture, which matters more than either single number:

- **`single_room`** (the case that originally motivated
  `VIDEO_EMPIRICAL_MIN_REL_HW`): freshly regenerated for this report
  (`bench.derive_tiers.compare_video_to_lidar`, confirmed byte-for-byte
  identical across two back-to-back reruns, so this is a stable number, not
  run-to-run noise) gives **91.2%** floor-area error (3.26 vs 36.85 m²) —
  the exact figure has drifted a little from the "4.4 vs 36.6 m²" originally
  cited in `config.py`'s comment as the pipeline evolved across later
  phases, but lands in the same severely-wrong territory that comment
  describes. Only 2 of the 3 scale priors fired this run (room-diagonal
  0.263, camera-height 0.457; the ceiling-height prior didn't find a
  ceiling plane in this sparse reconstruction), so their spread is modest —
  yet the actual error is enormous. Scale priors that *agree with each
  other* (low apparent variance) while the point estimate is still ~90%
  wrong is the textbook signature of **bias dominating variance**: the
  error is shared and systematic (every prior operates on the same
  underlying, un-bundle-adjusted reconstruction), not estimator-to-
  estimator noise a wider ensemble would average out. The video tier's
  ceiling-height comparison for this same capture is similarly bad (43.8%
  error, 3.29m reconstructed vs 1.85m true) — not previously reported for
  this fixture, included here for completeness; its CI is already wide
  enough (-0.51 to 7.08m) not to present it as a confident number.
- **`real_floor_only`** (freshly measured for this report): the three scale
  priors disagree sharply with *each other* (room-diagonal 0.297,
  ceiling-height 0.579, camera-height 0.167 — a ~3.5x spread), correctly
  driving the ensemble's own uncertainty up (~69% relative half-width,
  close to the 85% hard floor) before the observed error (99.2%, 0.81 vs
  97.3 m²) is even compared. Here the **variance signal is doing real
  work** — the method is visibly uncertain about itself, not silently
  confident while wrong. But stated precisely: the reported interval
  (0.07–1.54 m²) does not merely undershoot the true value's magnitude, it
  **fails to cover it at all** — 97.3 m² is nowhere near that range, by two
  orders of magnitude. No relative-CI widening anchored to a point estimate
  this wrong could ever restore coverage; combined with only 5,889
  reconstructed sparse points for a ~97 m² room, this points to a
  *coverage* failure (too little of the room triangulated at all)
  compounding the scale bias, not scale bias alone.

**Conclusion:** bias and variance both contribute, and which one dominates
is capture-dependent — this project does not have enough real examples (n=2
captures with a synchronized video) to generalize further, and says so
rather than extrapolating from either single case. The practical
implication is unchanged from Section 3's Phase 7 finding: tightening the
ensemble's variance estimate alone would not fix `single_room`'s case
(bias-dominated), while `real_floor_only`'s case needs better reconstruction
*coverage*, not just a wider scale CI. A proper bundle-adjusted
reconstruction (Section 11.2) is the lever that addresses both.

## 7. Benchmark Data Situation

No tape or laser ground truth exists for this project — the user's own iPhone
LiDAR/video/photo captures are the only real data available. Rather than
fabricate accuracy numbers against an assumed ground truth, every reported
figure above uses **LiDAR-tier geometry on the same physical capture** as
pseudo-ground-truth (clearly labelled as such everywhere it's used — see
`bench/derive_tiers.py`), and gates that would need real laser ground truth
(opening width, wall length, floor area absolute error vs a tape measure) are
explicitly reported as `not measured` with the reason
(`bench/harness.py::run_gates`), never estimated.

## 8. Conformal Calibration Status

Implemented and algorithmically validated, but **not yet statistically
active**: with only 6 real calibration points (4 video, 2 photo) against a
mathematical requirement of at least 9 for any finite 90%-coverage factor,
`bench/calibrate.py` correctly reports both tiers as "not achievable." The
hand-set empirical CI floors remain the actual operative safety mechanism.
This is infrastructure ready to activate automatically (`roomscan/
calibration/factors.py` loads and applies a factor the moment one becomes
achievable) as more real captures accumulate — not a deferred feature, a
correctly-gated one.

## 9. Determinism, Reproducibility, and Testing Discipline

Every stage is seeded from a single `config.SEED`, and the Phase 6 RNG bug
(Section 3) is the project's clearest illustration of why this matters in
practice: a shared, module-level random number generator made results depend
on execution history rather than the seed alone — a subtle violation of the
determinism rule that a naive "it passed my test" check would not have
caught, since any single test run in isolation looked fine. It was only
caught by deliberately re-running the same test file in combination with
others in a different order. The fix (seed fresh inside every function that
needs randomness, never at module scope) is now the standing convention
across the codebase.

More broadly, the project's testing discipline follows one consistent rule:
*verify against real data before trusting a heuristic's own confidence, and
when real data reveals a gap, widen or abstain rather than force a tighter
number to match one example.* Every numeric claim in this report was
produced by actually running the pipeline (`bench/derive_tiers.py`,
`bench/ablate.py`, `bench/calibrate.py`, `bench/fixloop_ceiling_check.py`,
`bench/repro.py` are all directly runnable and regenerate their own
numbers), not estimated by inspection. 102 tests (confirmed passing on the
most recent full run) cover unit-level correctness,
structural guarantees (never crash, always produce valid schema output,
even in total-abstention cases), and integration against both real and
clearly-labelled-simulated data, with real-data tests skipping cleanly
rather than failing when the (gitignored, multi-hundred-megabyte) real
fixtures aren't present on a given machine.

## 10. Known Limitations

- Video and photo tier accuracy is poor and should be treated as a rough
  approximation, not a measurement, until validated against more real data.
  For `real_floor_only` specifically, the reported confidence interval
  doesn't just have a large relative error — it fails to **cover the true
  value at all** (0.07-1.54 m² reported vs. 97.3 m² true, see Section 6).
  No relative-CI widening can fix this; it needs a coverage/completeness
  check the pipeline doesn't currently have (too few points triangulated to
  trust the reconstruction's shape at all, not just its scale).
- Multi-room stitching (`roomscan/geometry/pose_graph.py`) now solves a real
  door-to-door pose graph per room, but no real multi-room fixture (any
  tier) exists to validate the result against ground truth — treat it as
  internally self-consistent, not externally verified. Rooms without a door
  match still fall back to the schematic grid.
- Damage detection is a heuristic prompt for human review, not a diagnosis;
  expect a high false-positive rate.
- Conformal calibration cannot yet provide a statistically justified
  guarantee at this project's target confidence level.
- Split-scan repeatability is capture-dependent, not a constant the project
  can claim once: `single_room` fails the project's own 5% floor-area
  repeatability threshold (6.0% measured — see Section 6), the first time
  this specific fixture has been checked against that gate. The passing
  repeatability claim in Section 3 (Phase 6) refers to `real_floor_only` and
  `real_with_ceiling`, not this fixture.

## 11. Recommendations for Future Work

1. **More real captures, prioritized over more code.** Nearly every
   limitation above is a data problem, not an algorithm problem — the fix
   loop in Phase 11 specifically demonstrated that a fix validated against
   too few real cases can look correct and still be wrong.
2. If video/photo accuracy must improve materially, the real lever is a
   proper multi-view bundle-adjusted reconstruction (e.g. COLMAP), which was
   explicitly deferred this round to stay within the no-GPU/no-large-
   dependency constraint — not a limitation of effort, a stated trade-off.
3. A small labelled damage dataset (even informally labelled) would let the
   damage heuristic be measured for precision/recall instead of just
   observed qualitatively.
4. A real multi-room fixture (any tier) would let the pose-graph stitch be
   validated against ground truth instead of only its own internal
   consistency (does the synthetic ablation's finding hold up on a real
   capture?).
5. Investigate why `single_room` fails split-scan repeatability (Section 6)
   while the other two real fixtures pass comfortably — is it room size,
   opening count, or scan-path coverage that's driving the difference? A
   single additional real capture can't answer this; several would start to.
6. Add a minimum-reconstructed-point / coverage check to the video tier,
   not just a minimum-count threshold to attempt scale estimation at all
   (`VIDEO_MIN_RECONSTRUCTED_PTS`). `real_floor_only`'s case (Section 6)
   shows a relative-CI floor cannot rescue coverage when the point estimate
   itself is wrong by two orders of magnitude — the reported interval
   should instead widen to "not measured" or abstain outright below some
   points-per-m²-of-assumed-room-size ratio, the same philosophy already
   applied to the LiDAR tier's quality gate.
