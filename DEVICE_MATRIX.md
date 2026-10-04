# Device Matrix

Which tier runs on which hardware, and the accuracy each tier *honestly* delivers —
taken from the benchmark actually regenerated this session (`bench/derive_tiers.py`,
`COMPLIANCE.md`), not aspirational gate targets.

| Tier | Minimum hardware | Why | Status |
|---|---|---|---|
| LiDAR | iPhone/iPad **Pro-class** (12 Pro or later Pro/Pro Max, or LiDAR-equipped iPad Pro) | Requires a physical LiDAR sensor for depth + Stray Scanner's confidence/pose export | Runs; most mature and tested tier |
| Video | Any iPhone (practically, iPhone 15 or newer for adequate camera/IMU quality, per the case study's own device floor) | Only needs a standard camera; no depth sensor required | Runs |
| Photo | Any iPhone (iPhone 15 or newer) | Only needs standard stills; no depth sensor, no continuous capture required | Runs |

## Honest accuracy per tier

| Tier | What was measured | Result | Source |
|---|---|---|---|
| LiDAR | No independent ground truth exists (no laser/tape data — see `COMPLIANCE_MATRIX.md`) | **Not measured.** RANSAC-derived confidence intervals are self-consistent (split-scan repeatability gate passes against real data) but not validated against an external truth | `bench/harness.py::_repeatability_gate`, `tests/test_repeatability.py` |
| Video | Floor area & ceiling height vs. LiDAR-tier pseudo-ground-truth, same physical room, two real videos | **65-99% error** (not 3%); one capture's reported interval fails to cover the true value at all, not just a large relative error (see `docs/TECHNICAL_REPORT.md` §6). Confidence intervals are widened to reflect this (`VIDEO_EMPIRICAL_MIN_REL_HW=0.85` floor) | `bench/derive_tiers.py::compare_video_to_lidar`, `COMPLIANCE.md` |
| Photo | Same comparison, simulated photo sets (stills derived from the real video, since no real multi-room photo fixture exists) | Reconstruction succeeded in **1 of 4** simulated trials at realistic photo counts (6-8 photos); when it succeeds, **~94% error**. The other 3 trials abstained outright (insufficient shared visual structure) rather than reporting a number | `bench/derive_tiers.py::compare_photo_to_lidar`, `COMPLIANCE.md` |

## What this means in practice

- **LiDAR is the only tier with any validated self-consistency**, and even that isn't
  checked against an independent (laser/tape) truth yet.
- **Video and photo tiers should be treated as rough approximations, not
  measurements**, until validated against real multi-tier, ground-truthed captures.
  This is a direct, disclosed consequence of using a lightweight, CPU-only, no-GPU
  monocular SfM approach (ORB/SIFT + essential matrix + triangulation) instead of a
  heavier reconstruction engine — a deliberate trade-off per the project's hard rule
  to avoid large/GPU dependencies without asking first (see `DISCLOSURES.md`).
- No fiducial marker (printed reference card) calibration is implemented, which the
  case study's own research notes identify as "what makes the tight gates reachable"
  for monocular tiers. This is a known, scoped gap (see `COMPLIANCE_MATRIX.md`).

## Multi-room support

All three tiers accept a property folder of per-room subfolders (lidar: each a Stray
Scanner export; video: each containing one video file; photo: each containing 2-8
stills) and stitch them via a door-to-door pose graph (`roomscan/geometry/pose_graph.py`):
adjacent rooms whose detected doors agree in width are placed so the doors coincide,
with Manhattan-snapped rotation; rooms with no door match fall back to a schematic
non-overlapping grid. No real multi-room fixture (any tier) exists to validate this
against ground truth — see `COMPLIANCE_MATRIX.md` for exact status and `COMPLIANCE.md`
for the honest limitations.
