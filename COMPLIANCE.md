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
- Multi-room stitching (`roomscan/geometry/pose_graph.py`, `multi_room.py`):
  each room is still independently reconstructed (own local frame, own
  scale), but adjacent rooms whose detected doors agree in width are now
  placed by solving a real door-to-door pose graph (Manhattan-snapped
  rotation + robust least-squares translation, `scipy.optimize.least_squares`)
  so that corresponding doors coincide up to a wall-thickness gap -- applied
  uniformly across all three tiers. Rooms with no door correspondence (or
  whose solved placement would overlap another room implausibly -- an AABB
  proxy check, not full polygon intersection) fall back to the old
  non-overlapping grid, which still makes no spatial claim beyond "a
  different room." No real multi-room fixture (any tier) exists to validate
  either path against ground truth -- a pose-graph placement is internally
  self-consistent (doors really do coincide), not externally verified.
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
- Damage classification (`roomscan/damage/`) is a crude colour heuristic,
  NOT a trained classifier -- no labelled damage imagery exists for this
  project to train or validate one against. It flags dark, desaturated
  blobs in RGB frames that deviate from each frame's own median brightness
  (HSV thresholding + connected components), classifies them by vertical
  position in-frame (floor-adjacent / ceiling-adjacent / neither) as a
  crude proxy for likely damage type, and applies simple, explainable
  if/then rules to produce flags and scope line items -- never a cost
  estimate (no real pricing data exists to ground one in). Tested against
  a real (non-damaged) room, it flagged 23 candidates across 15 frames: an
  expected high false-positive rate (shadows, dark furniture, ordinary wall
  texture all trigger it), not a bug. surface_id and area_m2 are also rough
  proxies (round-robin wall assignment, pixel-area scaled by the room's
  average wall area), not a true per-wall 3-D projection -- this module
  exists to prompt human review, not to diagnose. Human review required
  before any remediation decision.

## Benchmark Data Access

No iPhone, tape measure, or laser measurer is currently available to this project.
This blocks, until device access exists: the case-study-specified benchmark set
(multi-room capture, staged-damage room, same rooms at all three tiers, genuine
repeat captures), the head-to-head comparison against an incumbent app, and any
laser/tape ground truth. See `COMPLIANCE_MATRIX.md` for exactly which requirements
this affects and which remain open regardless. Nothing in this section is worked
around by assumption or estimation — every affected gate is marked `not measured`
with this reason, per the project's hard rule against fabricating numbers.

## Not a Substitute For
Professional structural assessment, licensed surveying, or building-code inspection.

## Data Handling
Scan data and derived plans may contain personally identifiable location information.
Handle in accordance with applicable privacy regulations (GDPR, CCPA, etc.).

## Model Provenance
See `docs/model_registry.md`: no trained ML models are used anywhere in this
pipeline. Every stage is classical geometry, computer vision, or statistics.

## Process Evidence
This project's fix loop (ceiling-detection false negatives on real patchy
ceiling data) is documented end to end in `DECLARATION.md`, including a
mid-course correction the real-data testing surfaced and a reproducible
before/after check (`bench/fixloop_ceiling_check.py`).

## Version
Schema v0.1 — outputs are subject to breaking changes before v1.0.
