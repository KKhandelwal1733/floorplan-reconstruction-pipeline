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
**65-91%** and **~94%** error respectively against LiDAR pseudo-ground-truth —
nowhere near the aspirational ±3%/±8% gates in the original brief. This is
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
| Validated against | — (no laser ground truth exists) | Real LiDAR pseudo-ground-truth, n=1 real capture | Simulated-from-real stills, n=4 trials |
| Floor-area error | Not measured (no independent ground truth) | 65-91% | ~94% (1 success of 4 trials) |
| Typical outcome | Confident measurement | Confident but wide-CI measurement | Frequent honest abstention |
| CI mechanism | RANSAC inlier residuals + quality gate | Empirical floor (`VIDEO_EMPIRICAL_MIN_REL_HW=0.85`) | Empirical floor (`PHOTO_EMPIRICAL_MIN_REL_HW=0.90`) |

## 5. Benchmark Data Situation

No tape or laser ground truth exists for this project — the user's own iPhone
LiDAR/video/photo captures are the only real data available. Rather than
fabricate accuracy numbers against an assumed ground truth, every reported
figure above uses **LiDAR-tier geometry on the same physical capture** as
pseudo-ground-truth (clearly labelled as such everywhere it's used — see
`bench/derive_tiers.py`), and gates that would need real laser ground truth
(opening width, wall length, floor area absolute error vs a tape measure) are
explicitly reported as `not measured` with the reason
(`bench/harness.py::run_gates`), never estimated.

## 6. Conformal Calibration Status

Implemented and algorithmically validated, but **not yet statistically
active**: with only 6 real calibration points (4 video, 2 photo) against a
mathematical requirement of at least 9 for any finite 90%-coverage factor,
`bench/calibrate.py` correctly reports both tiers as "not achievable." The
hand-set empirical CI floors remain the actual operative safety mechanism.
This is infrastructure ready to activate automatically (`roomscan/
calibration/factors.py` loads and applies a factor the moment one becomes
achievable) as more real captures accumulate — not a deferred feature, a
correctly-gated one.

## 7. Determinism, Reproducibility, and Testing Discipline

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
`bench/ablate.py`, `bench/calibrate.py`, `bench/fixloop_ceiling_check.py` are
all directly runnable and regenerate their own numbers), not estimated by
inspection. 84 tests (as of Phase 11, all passing) cover unit-level correctness,
structural guarantees (never crash, always produce valid schema output,
even in total-abstention cases), and integration against both real and
clearly-labelled-simulated data, with real-data tests skipping cleanly
rather than failing when the (gitignored, multi-hundred-megabyte) real
fixtures aren't present on a given machine.

## 8. Known Limitations

- Video and photo tier accuracy is poor and should be treated as a rough
  approximation, not a measurement, until validated against more real data.
- Multi-room stitching does not determine true relative room position or
  orientation — schematic grid layout only.
- Damage detection is a heuristic prompt for human review, not a diagnosis;
  expect a high false-positive rate.
- Conformal calibration cannot yet provide a statistically justified
  guarantee at this project's target confidence level.
- `report.md` (documented as a standard per-capture CLI output alongside
  `plan.json`/`plan.svg`) is not yet wired up — a known, scoped gap, separate
  from this report.

## 9. Recommendations for Future Work

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
4. Wire up `report.md` to close the one remaining documented-but-missing CLI
   output.
