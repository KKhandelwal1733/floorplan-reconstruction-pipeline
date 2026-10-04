Applied AI Case Study, Room-Scan Pipeline

> v2 changes: audited against the case-study PDF (added device matrix, bias-vs-variance diagnosis, error budget, reproduction bundle with cache replay, timing) and revised using prior-art research (section 14). Items marked **[verify]** are claims from web sources that must be confirmed on real data before being stated in the report.

Work in the phases in section 12. **Commit after every phase** (process evidence is scored). Never claim a number you did not regenerate with `make bench`.

---

## 0. Role and goal

You are building a Python pipeline that turns handheld iPhone captures into a dimensioned, stitched, whole-property floor plan with confidence intervals, damage regions, concealed-damage flags, and scope line items. It must run **three input tiers** through one shared core and emit the same JSON contract from each:

| Tier | Input | Has depth? | Has poses? |
|---|---|---|---|
| `lidar` | Stray Scanner export folder | yes | yes |
| `video` | one handheld `.mp4`/`.mov` | no | no |
| `photo` | a folder of per-room subfolders, 2-8 stills each | no | no |

It will be run **cold on an unseen space at a live defense**, scored against a laser measurer. Priorities: (1) never crash, (2) always output something, (3) honest, calibrated intervals, (4) abstain/widen rather than emit confident garbage.

## 1. Hard requirements (from the brief)

- One command per capture: `python -m roomscan run <input_path> --tier {lidar,video,photo,auto} --out <dir>`
- Outputs per capture: `plan.json` (to our schema), `plan.svg` (rendered floor plan), `report.md`.
- Per room: walls (length), ceiling height, floor area, openings (doors/windows with width). **Every measurement carries a confidence interval** (`value`, `lo`, `hi`, `confidence_level=0.9`).
- Multi-room: stitched whole-property plan with correct adjacency, no room overlaps, every room placed + dimensioned, **from every tier including photo**.
- Damage: per-surface regions with class and metric extent; concealed-damage flags with the rule ID that fired; scope line items keyed to surface IDs.
- Gates: opening widths <=2 cm on >=85% of openings (missed and phantom both count as misses); ceiling height <=1.5 cm, repeat spread <=1 cm; repeatability within 1 cm or 0.5%/wall; drift ablation (on/off footprint); photo-tier wall lengths and footprint within +/-8% with calibrated intervals; video +/-3%.
- Determinism: fixed seeds, sorted file iteration, no wall-clock dependence in outputs.
- Runs offline at the defense. Weights fetched by `scripts/fetch_weights.sh` and disclosed in `DISCLOSURES.md`.
- Must handle mirrors, glass, wet-look surfaces, low light: detect, widen intervals or abstain, document in known failure modes.
- **Device matrix (Part 1, mandatory):** `DEVICE_MATRIX.md` states which tier runs on which hardware (LiDAR tier: iPhone Pro-class only; video and photo: any iPhone 15 or newer) and the accuracy each tier *honestly* delivers, taken from the benchmark, not aspirations.
- **Ceiling-height diagnosis (Part 2 gate):** the report must state whether our ceiling result is *repeatable-but-biased* or *unrepeatable* (compute mean error = bias and spread across captures = variance separately; both can fail).
- **Reproduction bundle:** cached model outputs are allowed only if the cache replays deterministically AND the live path also runs (`make repro` = replay from cache, `make live` = full live path). Every reported number regenerates from raw inputs.
- **Benchmark report** includes: gates at all three tiers, repeatability table, head-to-head table, and **timing** per tier/stage.
- **Technical report (max 6 pages)** must contain: architecture, tier design + device matrix, drift handling, **error budget**, calibration analysis, fix-loop story, known failure modes.

## 2. Input format facts (verified on `single_room.zip`)

Stray Scanner export, one folder per scan (e.g. `c00a170fe1/`):

```
camera_matrix.csv   3x3 intrinsics for the RGB image (fx=fy~1599.7, cx~955.5, cy~717.8)  -> RGB is 1920x1440
rgb.mp4             1920x1440 video, ~60 fps
depth/000000.png    uint16 PNG, 256x192, values in MILLIMETRES
confidence/000000.png uint8, 256x192, values {0,1,2} (0=low, 2=high)
odometry.csv        header: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
                    1715 rows, one per depth frame; pose of camera in world (ARKit frame)
imu.csv             accelerometer/gyro
```

Rules:
- Depth intrinsics = RGB intrinsics scaled by (256/1920, 192/1440): fx_d ~ 213.3.
- Frame index in odometry maps to depth/confidence file number. RGB video frame = same index (verify by timestamp; count of mp4 frames may differ, so match by timestamp or frame count, and assert).
- **Pose convention is NOT settled. [verify]** Two sources disagree: native ARKit uses camera +X right, +Y up, +Z backward (world +Y is gravity-up), while an unofficial Stray Scanner data-model page describes the camera frame as z forward, x toward the bottom of the phone, y toward the left edge in portrait. Do not assume either. **Write a unit test** that tries the candidate camera-axis conversions (and camera-to-world vs world-to-camera), fuses 100 frames each way, and selects the one that yields a horizontal floor plane and mutually perpendicular walls with the smallest residual. Record the chosen convention in `config.py` with the test result.
- Official docs describe depth as `.npy`; your zip contains `.png` (uint16, mm). The loader must accept both.
- Quaternion order in `odometry.csv` is (qx, qy, qz, qw). Convert carefully to a rotation matrix (scipy `Rotation.from_quat` uses xyzw).
- Drop pixels with confidence < 2 (configurable), depth == 0, or depth > 5 m.
- Other provided zips: `single_scan_floor_only` and `single_scan_with_ceiling`. Same format. They are the ceiling-missing vs ceiling-present test (see section 8).

## 3. Repository layout

```
roomscan/
  README.md                 # fresh-machine run in <15 min, one command per capture
  DISCLOSURES.md            # every pretrained model/dataset/API used
  COMPLIANCE.md             # requirement -> file path -> artifact -> status
  Makefile                  # make setup | demo | bench | ablate | fixloop-before | fixloop-after
  pyproject.toml
  schema/plan.schema.json   # versioned (v0.1); we did not receive the published schema, say so
  protocol/CAPTURE_PROTOCOL.md   # one page, non-engineer, Stray Scanner + native camera
                                 # must state: what to install, how to walk, how long, what to avoid,
                                 # how to hand files to the pipeline (AirDrop/USB -> folder layout)
  DEVICE_MATRIX.md          # tier x hardware x honest accuracy (from bench, with/without fiducial)
  cache/                    # deterministic model-output cache for `make repro`; `make live` bypasses it
  scripts/fetch_weights.sh
  roomscan/
    cli.py
    config.py               # all thresholds in one place, seeded
    io/
      stray_scanner.py      # loader -> Frames(rgb, depth_m, conf, pose_c2w, K)
      video.py              # frame sampling by sharpness + parallax
      photos.py             # folder-per-room loader, EXIF focal
    tiers/
      lidar.py              # fuse depth+poses -> point cloud (metric)
      video_tier.py         # feed-forward SfM + metric scale ensemble
      photo_tier.py         # same engine, sparse views, wider intervals
      scale.py              # scale estimators + ensemble + disagreement -> interval
    geometry/
      planes.py             # seeded RANSAC / region growing, Manhattan alignment
      room_layout.py        # walls, floor/ceiling planes, polygon, area, height
      openings.py           # doors/windows from plane holes + depth discontinuity + VLM
      uncertainty.py        # bootstrap + propagated sensor noise -> intervals
    stitch/
      pose_graph.py         # rooms as nodes, door constraints as edges (scipy least squares)
      drift.py              # on/off modes for the ablation
      photo_adjacency.py    # cross-folder matching, door-to-door constraints
      layout_solver.py      # Manhattan rotations, wall thickness gap, no-overlap
    quality/
      gate.py               # blur, parallax, exposure, coverage, mirror/glass flags
      calibration.py        # coverage, PIT, interval scaling (conformal)
    damage/
      segment.py            # VLM/segmenter masks -> backproject to plane -> metric extent
      rules.yaml            # rule_id, conditions, flag text
      rules.py              # rule engine, returns rule_id that fired
      scope.py              # templated line items keyed to surface_id
    render/svg_plan.py
    schema_out.py           # pydantic models -> plan.json, validated against schema
  bench/
    data/                   # raw scans + ground_truth.yaml (tape/laser)
    harness.py              # per-gate metrics, repeatability, calibration, timing
    head_to_head.py         # our error vs the incumbent app, per dimension
    derive_tiers.py         # LiDAR scan -> simulated video & photo tiers (disclosed)
  fixloop/
    DECLARATION.md          # committed BEFORE the fix
    before/ after/ DIFF.md
  tests/
  report/technical_report.md  # max 6 pages: architecture, tier design + device matrix, drift, error budget,
                              # calibration analysis, fix-loop story, known failure modes
```

## 4. Tier designs

### 4.1 LiDAR tier (build first, it is the reference)

1. Load frames, filter by confidence and range, subsample to ~every 3rd frame (deterministic).
2. Backproject to world with pose, voxel-downsample (2 cm), keep per-point sensor sigma (grows with range, ~0.5% of depth + 1 cm floor).
3. Gravity-align from pose (+Y up), find floor and ceiling as the dominant horizontal planes by histogram of Y, then refine with plane fit. **Ceiling height = distance between fitted planes** with bootstrap interval over frame subsets.
4. Walls: vertical plane extraction (seeded RANSAC, iterate removal), Manhattan-snap normals to the dominant yaw (allow non-orthogonal walls when the residual is high; do not force it).
5. Room polygon: intersect adjacent wall lines, order by angle, area by shoelace; propagate intervals via Monte Carlo over plane-fit covariance.
6. Openings: on each wall plane, rasterize points onto a 2D (u,v) grid; openings are holes/recesses in the occupancy and depth-offset map. Doors reach the floor, windows do not. Report width interval. **Detection is scored**, so tune recall and false positives separately, and log both.
   Optional second opinion: SpatialLM (3D LLM, predicts walls/doors/windows from point clouds, including from monocular-video reconstructions) can propose openings, but it needs CUDA + TorchSparse and carries non-commercial/Llama-family licences **[verify]**. Use it only as a cross-check against the geometric detector; the 2 cm gate has to be met by geometry (edge-to-edge hole measurement on the fitted wall plane), not by a learned box.
7. If the ceiling is not observed (floor-only scan): do **not** output a confident height. Fall back to the camera-pose upper bound plus a wide interval, flag `ceiling_unobserved`.

### 4.2 Video tier

1. Sample ~40-80 frames by variance-of-Laplacian sharpness + minimum parallax (deterministic).
2. **Primary engine: MapAnything** (Meta/CMU, feed-forward). It takes N images with *optional* intrinsics, poses or depth and directly regresses a metric-scale reconstruction plus a `metric_scaling_factor`; an Apache-2.0 checkpoint (`facebook/map-anything-apache`) exists **[verify licence on the model card]**. Because it accepts optional geometric inputs, the **same model can serve all three tiers**: photos/video frames only (tiers 2 and 3), or frames + LiDAR depth + poses (tier 1, as a cross-check against our own fusion). Fallbacks if no GPU or if it fails: VGGT-class reconstruction, then COLMAP/OpenSfM (CPU) with up-to-scale output and the scale ensemble below.
3. **Metric scale = ensemble** (`tiers/scale.py`): (a) metric monocular depth (Depth Pro / MoGe-2 / UniDepth) scale-aligned to the reconstruction, (b) EXIF/intrinsics prior, (c) weak structural priors (door height with a wide region-aware prior, not hardcoded), (d) optional fiducial: an A4 AprilTag/printed card in the protocol; if detected, scale interval shrinks sharply. Combine as a weighted log-scale estimate; **interval width = f(disagreement between estimators)**.
**Honesty about scale:** published benchmarks show metric monocular depth errors of several percent on many datasets and far worse on some, so a +/-3% video gate cannot be assumed from learned scale alone. The fiducial (printed reference card/AprilTag) is what makes the tight gates reachable; report video/photo accuracy separately *with* and *without* the fiducial in `DEVICE_MATRIX.md`. Cross-check MapAnything's scale against MoGe-2 and fiducial; their disagreement sets the interval.
4. From here, the same plane extraction -> layout as the LiDAR tier, with the scale uncertainty folded into every length (lengths scale by one global factor, so errors are correlated; keep that correlation when computing areas).

### 4.3 Photo tier

- Same engine as video with 2-8 views per room and no temporal ordering. Intervals are wider by construction (fewer views -> larger plane covariance + scale uncertainty). Quality gate abstains per-surface when coverage is too thin (for example, a wall seen in no photo gets no length, flagged `not_observed`).
- **Multi-room stitch from per-room folders** (mandatory, single-room-only fails the gate): see 5.2.

### 4.4 `--tier auto`

Detect by input: Stray Scanner layout -> lidar; single video file -> video; directory of image subfolders -> photo. Wrong guess must still produce output.

## 5. Stitching and drift

### 5.1 LiDAR / video multi-room

- Rooms are nodes with a pose (x, y, yaw). Door/connector openings shared between consecutive rooms create relative-pose edges; loop closure via the connector when the walk returns.
- Solve with robust nonlinear least squares (`scipy.optimize.least_squares`, Huber), with priors: Manhattan yaw snapping, shared-wall coplanarity, wall thickness (~10-25 cm) as a gap prior.
- **Ablation (required):** `drift.py` exposes `mode in {off, on}`. `off` = poses used as-is (this is the automatic fail baseline, so we only run it for the comparison). Harness writes stitched footprint area and per-wall error for both into `bench/results/drift_ablation.json` + a figure.
- Do not trust device poses blindly. Prior work (PriMo) uses Stray Scanner/ARKit poses only as *priors* that get refined, and a developer-forum report claims an ARKit world-tracking drift regression on LiDAR iPhones under iOS 26.4+ (single report, unverified **[verify]**). Treat both as reasons to run the correction by default and to include a long-walk scan in the ablation if one exists.

### 5.2 Photo-tier whole-property stitch

Precedent: Zillow's SALVe (ECCV 2022) infers doors/windows/openings per panorama, *hypothesises* pairwise room adjacency or overlap, verifies each hypothesis, initialises a pose graph, optimises it with GTSAM, then stitches the room layouts. Adopt the same hypothesise-verify-optimise structure. Note the difference: they use 360-degree panoramas and learned verifiers; we have 2-8 ordinary stills per room, so the verifier is geometric (door-width agreement, wall-thickness gap, no overlap) plus the doorway-photo matching below. This adaptation is ours and unproven; benchmark it honestly.

1. Per-room layout from its own folder (section 4.3).
2. Adjacency inference: protocol asks for **one photo from each doorway looking into the neighbouring room**. Match across folders by feature retrieval (DINOv2 global descriptors -> top-k, then local matching with LightGlue/LoFTR + geometric verification). A match between room A's doorway photo and room B's photos yields A-door <-> B-door correspondence.
3. Solve placement: door-to-door constraints (opposing door midpoints coincide, walls antiparallel with a wall-thickness gap), Manhattan-aligned rotations (discrete 90-degree search), non-overlap penalty. Small combinatorial search over room orderings and rotations; pick min-energy, deterministic tie-break.
4. Fallback: optional `rooms.txt` (`kitchen - hall`) to supply adjacency. If matching fails with no hint, output rooms in a labelled grid layout with `adjacency_confidence: low` and wide interval. Never overlap rooms.

## 6. Uncertainty and calibration

- Every measurement: `{value, lo, hi}` at 90%.
- Sources: sensor noise, plane-fit covariance, view coverage, scale ensemble disagreement, quality-gate multipliers.
- **Conformal calibration** (`quality/calibration.py`): on the benchmark, compute residual / predicted-sigma per tier, fit a per-tier inflation factor so empirical coverage ~90%. Report coverage per tier and per measurement type; ship the factors in `config.py`. Guard against overfitting: leave-one-scan-out.
- Confident garbage cap: if the quality gate fails (blur, too dark, too few views, mirror-dominated), widen heavily or return `status: "abstained"` with a reason. Track an abstention rate in the benchmark.

## 7. Damage module

1. Candidate regions: VLM/open-vocab segmenter (SAM2 + CLIP/VLM classifier; a local model is preferred for the live run, an API is allowed with disclosure and a cached/offline fallback). Classes: water stain, mould, crack, peeling/bubbling paint, hole/impact, efflorescence.
2. Back-project the mask onto its wall/ceiling/floor plane (LiDAR: use the actual depth; video/photo: use the fitted plane) -> metric area and extents with intervals.
3. Rule engine (`rules.yaml`): each rule has `id`, conditions over (damage class, surface type, adjacency such as the wall behind a wet room, size, height above floor), output flag. Example IDs: `R-CEIL-STAIN-WETROOM`, `R-PAINT-BUBBLE-MOISTURE`, `R-CRACK-DIAGONAL-STRUCT`. The emitted JSON includes `rule_id` and `evidence`.
4. Scope items: templated per (class, surface, area) with unit quantities derived from the measured extents, keyed by `surface_id`.

## 8. Benchmark plan (honest about what we have)

We have **no iPhone, no tape/laser data, no multi-tier same-room captures**. Therefore:

- `bench/derive_tiers.py`: from each Stray Scanner scan, derive **simulated** video (RGB only, depth and poses discarded) and **simulated** photo (N in 2..8 sharpest, well-spread frames, discarded depth and poses). Label these `simulated=true` everywhere.
- **Pseudo-ground-truth:** LiDAR-tier geometry on the full scan. State plainly that this has ~1-2 cm error and is not laser truth. If anyone can hand-measure even 3 walls + 1 door + ceiling height of any room with a tape, add it to `ground_truth.yaml` (this is the highest-value hour of work available).
- Scans: `single_room` (full), `single_scan_floor_only` and `single_scan_with_ceiling` (ceiling-unobserved vs observed: the abstention test). For the multi-room and repeatability gates, run: (a) split a long scan into two halves as pseudo-repeat captures, labelled as such; (b) synthesize a multi-room case by combining separate scans as separate rooms for the stitcher (labelled synthetic). Do **not** present these as the required real captures; list the gap in the report.
- Harness outputs per tier: gate table (pass/fail with numbers), repeatability table, calibration coverage, timing, abstention rate.
- Head-to-head (Part 3): choose ONE incumbent (Polycam or magicplan), record **app name and version**, scan the same 2 rooms, submit its export, and produce one table: our error vs theirs per shared dimension, against tape/laser truth. Gate: beat or tie on >= 70% of shared dimensions. Check what the free tier actually exports **[verify]**; if it only displays dimensions, screenshot them and say so. Calibrate expectations: Polycam claims about +/-1/2 inch (~1.3 cm) on standard interiors, one review puts iPhone-LiDAR apps at roughly 1-3 cm, and an older independent test of magicplan's LiDAR mode saw errors around 15-17 cm in complex rooms. A tie within the measurement noise is the realistic target. If no iPhone is available, mark this part *not completed*. Do not invent numbers.

## 9. Fix loop (25%: plan for it from the start)

1. Run `make bench` early. The worst gate will be visible by mid-day.
2. Write `fixloop/DECLARATION.md` (one page: worst gate + failing number, root-cause hypothesis + evidence, the fix, predicted number) and **commit it before touching the fix**.
3. `make fixloop-before` writes `fixloop/before/` from a pinned commit/tag; implement the fix; `make fixloop-after` writes `fixloop/after/`; `DIFF.md` is generated from `git diff`.
4. Prefer fixing systematic bias or undercoverage (for example plane-fit bias, interval inflation, opening-edge thresholds) over fixing detection recall; they move more reliably.

## 10. Output JSON (schema v0.1, sketch)

```json
{
  "schema_version": "0.1",
  "capture": {"tier": "lidar|video|photo", "status": "ok|degraded|abstained", "warnings": []},
  "property": {"rooms": [ ... ], "adjacency": [{"a": "r1", "b": "r2", "via": "o3", "confidence": 0.9}],
               "footprint_m2": {"value": 0, "lo": 0, "hi": 0}},
  "rooms": [{"id": "r1", "polygon": [[0,0]], "pose": {"x":0,"y":0,"yaw":0},
             "walls": [{"id":"w1","length_m":{"value":0,"lo":0,"hi":0}}],
             "ceiling_height_m": {"value":0,"lo":0,"hi":0},
             "floor_area_m2": {"value":0,"lo":0,"hi":0},
             "openings": [{"id":"o1","type":"door|window","wall":"w1","width_m":{...},"confidence":0.0}],
             "damage": [{"id":"d1","surface_id":"w1","class":"water_stain","area_m2":{...},"extent":{...},
                         "flags":[{"rule_id":"R-...","text":"..."}]}],
             "scope": [{"item":"...","surface_id":"w1","qty":0,"unit":"m2"}]}]
}
```

Validate with `jsonschema` in a test and at runtime.

## 11. Defense-day robustness checklist

- `python -m roomscan run` works with zero network after `make setup`.
- Each tier completes in a bounded time (target: lidar < 2 min, video < 5 min, photo < 3 min on the target machine); add a `--fast` mode.
- Top-level try/except per stage: a failing stage degrades that output with a warning, it never kills the run.
- Handles portrait/landscape video, HEIC photos (convert with pillow-heif), missing EXIF, different iPhone models.
- Logs a one-screen summary to the terminal so the live run is legible.
- Tests: loader sanity, gravity/pose test, deterministic rerun (hash of `plan.json` identical across two runs), schema validation.

## 12. Phases (commit after each; do not skip ahead)

1. **Scaffold:** repo, pyproject, Makefile, `COMPLIANCE.md` skeleton, schema v0.1, CLI stub. *Commit.*
2. **Stray Scanner loader + pose test**, point-cloud fusion, saved PLY for visual check. *Commit.*
3. **LiDAR layout:** floor/ceiling/walls/area/height with intervals; SVG render for `single_room`. *Commit.*
4. **Openings** + harness with whatever ground truth exists. *Commit.*
5. **Ceiling-missing behaviour** on `single_scan_floor_only` (abstain/widen). *Commit.*
6. **Quality gate + determinism test + repeatability (split-scan)**. *Commit.*
7. **Video tier** (reconstruct + scale ensemble) and `derive_tiers.py`. *Commit.*
8. **Photo tier + multi-room photo stitcher** + pose-graph + drift ablation. *Commit.*
9. **Calibration** (conformal) across tiers. *Commit.*
10. **Damage module** + rules + scope. *Commit.*
11. **Fix loop:** declaration -> fix -> before/after. *Commit each step separately.*
12. **README, protocol page, DISCLOSURES, COMPLIANCE, 6-page technical report.** *Commit.*

**Realistic cut line for a same-day deadline:** phases 1-6 and 12 are the floor (LiDAR tier + harness + honest report). Add phase 11 (fix loop) next because it is 25% of the score. Then phase 8's photo stitch and drift ablation (both are hard gates). Video/photo single-room via MapAnything comes before damage. If the walk-in uses a non-Pro iPhone, only the video/photo tiers run, so do not let them rot.

**If time runs out, cut in this order:** damage sophistication, SVG polish, extra hard-case captures. **Never cut:** all three tiers running, photo multi-room stitch, drift ablation on/off, the fix loop with before/after, the reproduction bundle, honest labelling of simulated or missing data.

## 13. Rules for you, the coding agent

- Ask before adding any dependency that needs more than ~500 MB or a GPU; offer a CPU fallback.
- Put every threshold in `config.py`; no magic numbers inside functions.
- Never fabricate benchmark numbers, ground truth, or a competitor result. If something could not be measured, write `not measured` and say why.
- Prefer simple, inspectable geometry (RANSAC, least squares) over opaque models wherever it meets the gate.
- After each phase, print what is done, what is stubbed, and the next risk.

## 14. Prior art reviewed (what we borrowed, what is unverified)

| Source | What it shows | How we use it |
|---|---|---|
| MapAnything (Meta/CMU, arXiv 2509.13414) | Feed-forward metric 3D reconstruction from images with optional intrinsics/poses/depth; Apache-2.0 variant exists | Single engine for all tiers (4.2); verify licence + CPU/GPU feasibility |
| SALVe (Zillow, ECCV 2022) | Door/window/opening cues -> adjacency hypotheses -> verify -> pose graph (GTSAM) -> stitched layouts, from sparse panoramas | Structure of photo-tier stitcher (5.2) |
| SpatialLM (ManyCore) | LLM over point clouds predicting walls/doors/windows; works on monocular-video reconstructions | Optional cross-check for openings only (4.1) |
| MoGe-2 / Depth Pro / UniDepthV2 | Monocular metric geometry; errors of several percent or worse depending on dataset | Scale ensemble members and the reason for the fiducial (4.2) |
| PriMo (GitHub) | Stray Scanner exports used as ARKit prior poses/intrinsics + PromptDA depth prompting, refined rather than trusted | Justifies pose-as-prior and the drift ablation (5.1) |
| VGGT-Geo / VGP-Nav (papers) | Learned reconstruction plus metric anchors/ground-plane scale recovery to fix scale drift | Idea: anchor scale on floor plane + ceiling-floor distance priors |
| Polycam / magicplan / RoomScan reviews | Vendor claims ~1.3 cm; reviews 1-3 cm; one older test ~15 cm in complex rooms | Sets realistic head-to-head expectations (section 8) |

