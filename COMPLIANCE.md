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
- Photo tier: ±10 cm typical; no depth sensor, inference only.
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
