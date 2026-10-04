# Compliance & Limitations

## Scope
This tool reconstructs dimensioned floor plans from iPhone LiDAR, video, and photo captures.
Outputs are **estimates**, not certified surveys.

## Limitations
- LiDAR tier: ±2 cm typical; degrades in low-reflectance or specular surfaces.
- Video tier: ±5 cm typical; scale derived from monocular depth ensemble.
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
