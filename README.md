# roomscan

Turns handheld iPhone captures (LiDAR, video, or photos) into dimensioned,
stitched floor plans with confidence intervals, damage flags, and scope line
items. Built as an applied-AI case study, scored live against a laser
measurer.

See `docs/PROTOCOL.md` for field capture instructions, `DEVICE_MATRIX.md` for
which tier runs on which hardware and its honest accuracy, `COMPLIANCE.md` for
what the output actually means and its documented limitations,
`COMPLIANCE_MATRIX.md` for the literal requirement-by-requirement status against
the case study brief, `DISCLOSURES.md` for dependencies/data/AI-assistance
provenance, and `docs/TECHNICAL_REPORT.md` for the full project writeup.

## Quickstart

```bash
make install   # pip install -e ".[dev]"
make test      # run the full test suite
make demo      # run the pipeline on the bundled sample scan
make bench     # gate table + real video-vs-LiDAR comparison, against whatever
               # real fixtures are present
make ablate    # drift ablation (does chain length affect video-tier error?)
make calibrate # conformal calibration across tiers
```

## One command, three tiers

```bash
python -m roomscan.cli <input_path> --out <dir> [--tier lidar|video|photo]
```

Tier is auto-detected from the input shape if `--tier` is omitted: a
`.mp4`/`.mov` file → video, a directory containing `depth/` → lidar, any
other directory → photo (property folder of per-room subfolders).

Every tier also accepts a **property folder** of per-room subfolders (lidar:
each a Stray Scanner export; video: each containing one video file) when
`--tier` is passed explicitly — auto-detect only recognizes the single-room
shapes above. Rooms are stitched via a door-to-door pose graph
(`roomscan/geometry/pose_graph.py`) when their detected doors agree in
width, falling back to a schematic grid otherwise — see COMPLIANCE.md.

Every run writes `plan.json` (schema v0.1, see `schema/plan.schema.json`) and
`plan.svg` to the output directory — never crashes, and always produces valid
output, even when reconstruction has to abstain (see COMPLIANCE.md).

## Tier comparison (honest numbers, not aspirational ones)

| Tier | Input | Validated accuracy | Notes |
|---|---|---|---|
| LiDAR | Stray Scanner export | ±2 cm (assumed; no laser ground truth exists to confirm — see Benchmark Data Situation below) | Most mature, most tested tier |
| Video | Single handheld `.mp4` | **65-91% error** on the one real video tested against LiDAR pseudo-ground-truth | Lightweight monocular SfM, no bundle adjustment — a deliberate CPU-only, no-GPU-dependency choice (see DISCLOSURES.md) |
| Photo | 2-8 stills per room | **~94% error** when it succeeds; frequently abstains outright at realistic photo counts | Least validated; treat any output as a rough approximation |

Full detail, including *why* these numbers are what they are (and the real
bugs found and fixed while measuring them), is in COMPLIANCE.md and
`docs/TECHNICAL_REPORT.md`.

## Architecture

```
roomscan/
  cli.py                    # entry point, tier auto-detect, per-tier dispatch
  config.py                 # ALL tunable thresholds (no magic numbers elsewhere)
  plan_builder.py            # RoomLayout/PropertyLayout -> schema v0.1 plan.json
  schema_out.py              # Pydantic models + jsonschema validation
  io/
    stray_scanner.py          # LiDAR tier loader (depth/confidence/rgb/poses)
    video_loader.py            # time-based frame sampling from .mp4
    photo_loader.py            # per-room photo folder loader
  geometry/
    planes.py                  # gravity detection, RANSAC floor/ceiling
    room_layout.py              # floor polygon, area, walls, quality gate
    openings.py                 # door/window detection (occupancy-grid)
    sfm.py                      # monocular SfM core (video + photo tiers)
    scale_ensemble.py           # metric-scale recovery for monocular SfM
    video_tier.py / photo_tier.py / multi_room.py
  damage/
    detector.py / rules.py / pipeline.py   # colour-heuristic damage flagging
  calibration/
    conformal.py / factors.py              # split-conformal CI calibration
  render/svg_plan.py           # dimensioned SVG rendering

bench/           # harness.py (gates), derive_tiers.py, ablate.py, calibrate.py,
                 # fixloop_ceiling_check.py -- all runnable, all regenerate
                 # real numbers, never fabricated
schema/plan.schema.json   # JSON Schema for plan.json, v0.1
tests/           # pytest suite; real-data tests skip cleanly when gitignored
                 # fixtures aren't present locally
```

## Benchmark data situation

No tape/laser ground truth exists for this project (see DISCLOSURES.md). All
accuracy numbers above use **LiDAR-tier geometry on the same physical capture**
as pseudo-ground-truth (not laser-measured truth) — see `bench/derive_tiers.py`.
Where a claim would need data this project doesn't have, it's marked
`not measured` with the reason (see `bench/harness.py`'s gate table), never
estimated.

## Process evidence

The required fix-loop deliverable (declare → fix → before/after, committed as
separate steps) is `DECLARATION.md` at the repo root, with reproducible
before/after evidence in `bench/fixloop_ceiling_check.py`.
