# Compliance & Limitations

## Scope
This tool reconstructs dimensioned floor plans from iPhone LiDAR, video, and photo captures.
Outputs are **estimates**, not certified surveys.

## Limitations
- LiDAR tier: ±2 cm typical; degrades in low-reflectance or specular surfaces.
- Video tier: lightweight monocular SfM (no bundle adjustment) + a scale
  ensemble of physical-size priors, chosen over a heavier dependency
  (e.g. COLMAP) per the hard rule to avoid large/GPU dependencies without
  asking first. On the one real video tested against LiDAR-tier pseudo-
  ground-truth for the same room (see `bench/derive_tiers.py`), floor area
  and ceiling height errors were 65-91% (see project `config.py`
  `VIDEO_EMPIRICAL_MIN_REL_HW`), not a small percentage — intervals are
  widened accordingly, but accuracy should be treated as unvalidated and
  this tier used with caution until more real comparisons exist.
- Photo tier: 2-8 unordered stills per room, no depth/poses. Reuses the
  video tier's monocular SfM core, but matches ALL photo pairs (not just
  consecutive) and reconstructs from only the single best-matching pair,
  since stills taken seconds apart during a walkthrough often share far
  less visual overlap than a video's consecutive frames. In testing with
  simulated photo sets (derived from the real video, since no real
  multi-room photo fixture exists), reconstruction frequently abstains
  outright (insufficient shared structure to find a floor plane at all) at
  realistic counts (6-8 photos), and succeeded in only one of several
  simulated trials (n=12), with ~94% floor-area error against LiDAR
  pseudo-ground-truth. This is the most honest and least validated of the
  three tiers; treat any output as a rough approximation and expect
  frequent abstention rather than a number, never a crash either way.
- Multi-room stitching is schematic only: each room's shape/area is
  independently reconstructed, but this tool does not determine true
  relative room position or orientation (no real multi-room fixture exists
  to validate cross-room feature matching against). Rooms are laid out on
  a simple non-overlapping grid for visualization, not a geometric stitch.
- Conformal calibration (`bench/calibrate.py`, `roomscan/calibration/`): a
  standard split-conformal procedure that would, given enough real
  video/photo-vs-LiDAR comparison points, compute a statistically justified
  CI-widening factor per tier. With only 3 real fixtures (6 total
  calibration points: n=4 video, n=2 photo after some abstentions), this
  honestly reports "not achievable" at the project's 90% target coverage —
  split-conformal needs at least 9 calibration points for any finite
  90%-coverage factor at all, regardless of how spread out they are. The
  existing hand-set empirical floors (`VIDEO_/PHOTO_EMPIRICAL_MIN_REL_HW`)
  remain the actual active mechanism; rerun `make calibrate` as more real
  captures accumulate.
- Damage classification is advisory. Human review required before remediation decisions.

## Not a Substitute For
Professional structural assessment, licensed surveying, or building-code inspection.

## Data Handling
Scan data and derived plans may contain personally identifiable location information.
Handle in accordance with applicable privacy regulations (GDPR, CCPA, etc.).

## Model Provenance
All ML models used are documented in `docs/model_registry.md` (added in phase 12).

## Version
Schema v0.1 — outputs are subject to breaking changes before v1.0.
