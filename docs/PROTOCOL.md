# Capture Protocol

Practical field instructions for operating `roomscan` at a live defense, one per
tier. Read COMPLIANCE.md first for what each tier's output actually means.

## LiDAR tier (Stray Scanner) — use this one if at all possible

Accuracy and reliability are both far better validated here than the other two
tiers (see COMPLIANCE.md). This should be the default choice whenever an
iPhone/iPad with LiDAR is available.

1. Install the Stray Scanner app (App Store) on a LiDAR-equipped device.
2. Start a new scan. Walk the full perimeter of the room at a normal pace,
   holding the phone roughly chest-height, tilted slightly down toward the
   floor-wall junction.
3. **Tilt up to capture the ceiling explicitly** — a brief upward sweep (phone
   pointed at the ceiling) partway through the walk, not just forward-facing
   frames. The pipeline can recover without this (Phase 5's ceiling-missing
   abstain/widen path), but a measured ceiling height is far more useful than a
   wide-CI estimate, and Phase 11's fix loop showed that even a genuine but
   *patchy* ceiling sweep is usually enough to be detected now — a full,
   careful sweep isn't required, but a decent one helps.
4. Walk slowly past any doors/windows, facing them directly for a few seconds
   each, so enough points land on the opening itself and its surrounding wall.
5. Stop the scan. Export via Stray Scanner's share function (produces the
   `depth/`, `confidence/`, `rgb.mp4`, `odometry.csv`, `camera_matrix.csv`
   folder structure this pipeline expects).
6. Run: `python -m roomscan run <exported_folder> --tier lidar --out <output_dir>`
   (for a multi-room property, put one exported-scan folder per room inside a
   parent folder and point `<exported_folder>` at the parent instead — rooms
   are stitched via door-to-door pose-graph matching, see COMPLIANCE.md)

## Video tier — only if no LiDAR device is available

Documented accuracy is poor (65-91% error on the one real example tested — see
COMPLIANCE.md). Use this tier's output as a rough sanity check, not a
measurement, and say so if presenting it live.

1. Record a single continuous handheld video walking the room's perimeter,
   similar motion to the LiDAR tier's walk (steady pace, not jerky).
   Phone in landscape orientation. 20-40 seconds is enough; longer doesn't
   reliably help (see `bench/ablate.py`'s drift-ablation finding — error did
   not grow with more sampled frames in testing, so there's no strong reason
   to over-capture).
2. Avoid pointing at blank, low-texture surfaces (plain walls, ceilings) for
   extended periods — the reconstruction needs visual features to track.
3. Run: `python -m roomscan run <video_file.mp4> --tier video --out <output_dir>`
   (for a multi-room property, put one video per room in a named subfolder
   inside a parent folder and point at the parent instead — same pose-graph
   stitch as the lidar tier)

## Photo tier — last resort only

Documented accuracy is the worst of the three tiers and reconstruction
frequently abstains outright at realistic photo counts (see COMPLIANCE.md /
`feedback_photo_tier_accuracy` finding: 1 success in 4 simulated trials). Only
use this if neither LiDAR nor video capture was possible.

1. Create one subfolder per room inside a property folder.
2. In each room, take 2-8 still photos from **varied, overlapping** vantage
   points — not all facing the same direction. A photo set that only shows
   one wall will not reconstruct; deliberately include some photos that
   overlap in what they show (e.g. two photos both showing the same corner
   from different angles), since the pipeline needs to match shared features
   between at least one photo pair.
3. Run: `python -m roomscan run <property_folder> --tier photo --out <output_dir>`

## Reading the output

Every run produces `plan.svg` (visual) and `plan.json` (schema v0.1,
machine-readable — see `schema/plan.schema.json`) in the output directory.
`capture.status` in plan.json is one of:

- `ok`: full-confidence reconstruction.
- `degraded`: reconstructed, but with widened confidence intervals (sparse
  coverage, unobserved ceiling, etc.) — check `capture.warnings`.
- `abstained`: reconstruction failed entirely; `rooms` is empty and
  `capture.warnings` explains why. This is a valid, intentional outcome
  (never a crash) when the input data was insufficient — see COMPLIANCE.md.

Damage entries in `plan.json` (`rooms[].damage`, `rooms[].scope`) are advisory
only — see COMPLIANCE.md's damage-classification section before presenting
them as findings.
