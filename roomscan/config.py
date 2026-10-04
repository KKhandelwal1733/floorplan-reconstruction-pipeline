"""All tunable thresholds live here (rule in section 13 of the brief) -- no magic numbers in functions."""

SEED = 0

# --- Depth loader filters ---
DEPTH_MIN_M = 0.10      # drop pixels shallower than this
DEPTH_MAX_M = 5.00      # drop pixels deeper than this
CONFIDENCE_MIN = 2      # drop pixels with Stray Scanner confidence < this (0=low, 2=high)

# --- RANSAC plane fitting ---
RANSAC_THRESH_M = 0.05  # max point-to-plane distance to count as inlier
RANSAC_ITER = 500       # number of RANSAC iterations
HORIZ_TOL_RAD = 0.30    # max angle from gravity for a plane to be "horizontal" (radians)
GRAVITY_SAMPLE = 8_000  # points used by detect_gravity

# --- Ceiling-missing fallback (Phase 5) ---
CEIL_UNOBSERVED_MARGIN_M = 0.30   # add this above the highest observed point
CEIL_UNOBSERVED_HALF_WIDTH_M = 0.50  # half-width of the wide CI when ceiling not seen
CEIL_MIN_FILL_RATIO = 0.35        # min 2D fill ratio of ceiling inliers (rejects wall tops)
# Fix loop (Phase 11, see DECLARATION.md): fill ratio alone measured 0.23 for a real
# patchy ceiling vs 0.28-0.29 for a synthetic wall-top ring -- too close to trust.
# interior_frac (what fraction of inliers fall in the shrunk-margin interior of the
# candidate plane's own bbox) separates them by a wide margin instead: 0.0 for any
# wall-top ring (by construction, it can never have interior points) vs 0.6-0.73 for
# a real ceiling. A ceiling is accepted if EITHER signal passes, not just fill ratio.
CEIL_MIN_INTERIOR_FRAC = 0.30
CEIL_INTERIOR_MARGIN = 0.20        # shrink margin (each side) defining "interior"
# Testing interior_frac against real_floor_only (which has no real ceiling) surfaced
# a THIRD failure mode beyond the original two-case check: a large interior surface
# that isn't a wall-top ring (e.g. furniture/a tabletop) can have interior_frac
# nearly identical to a genuine ceiling's (0.729 vs 0.719) -- interior_frac alone
# can't tell them apart. What does: the implied room height (candidate height minus
# floor height) was 1.21 m for the furniture-like false positive vs 2.39 m for the
# genuine ceiling -- physically implausible vs plausible. Required as a gate,
# generous enough to admit short (attic/crawlspace) and tall (vaulted/commercial)
# real ceilings without over-fitting to residential norms.
CEIL_MIN_PLAUSIBLE_HEIGHT_M = 1.8
CEIL_MAX_PLAUSIBLE_HEIGHT_M = 6.0

# --- Wall extraction ---
WALL_MERGE_ANGLE_DEG = 8.0  # merge convex-hull vertices turning less than this (collapses noise)

# --- Openings detection (Phase 4) ---
# 10 cm keeps each cell's expected point count statistically robust even on
# a modest point cloud; a door/window is still located to within one cell.
OPENING_CELL_M = 0.10          # occupancy grid cell size (10 cm)
OPENING_MAX_DIST_M = 0.10      # include points within this distance of wall plane
OPENING_MIN_W_M = 0.40         # minimum opening width to report
OPENING_MAX_W_M = 3.00         # maximum opening width to report
OPENING_DOOR_GAP_M = 0.15      # opening bottom must be within this of floor to be a door
OPENING_MIN_H_M = 0.50         # minimum opening height to report
OPENING_OCCUPANCY_THRESH = 0.25  # fraction of cells occupied to count a column as "filled"
OPENING_MIN_WALL_PTS = 30       # skip wall if fewer nearby points (insufficient coverage)

# --- Quality gate (Phase 6) ---
QUALITY_MIN_FLOOR_INLIERS = 20_000  # below this, floor coverage is suspect (partial scan)
QUALITY_MIN_PTS_PER_M2 = 500         # point density floor for a confident area estimate
QUALITY_CI_WIDEN_FACTOR = 2.0        # multiply CI half-widths when either check fails

# --- Repeatability / split-scan (Phase 6) ---
REPEAT_AREA_TOL_FRAC = 0.05       # split-scan floor-area relative tolerance
REPEAT_PERIMETER_TOL_FRAC = 0.05  # split-scan wall-perimeter relative tolerance

# --- Video tier (Phase 7): monocular SfM + scale ensemble ---
VIDEO_FRAME_INTERVAL_S = 0.5      # seconds between sampled frames for SfM
VIDEO_MAX_FRAMES = 40             # cap on sampled frames (keeps SfM tractable)
VIDEO_ASSUMED_FOV_DEG = 65.0      # assumed horizontal FOV (true focal length unknown)
VIDEO_ORB_FEATURES = 3000         # ORB features extracted per frame
VIDEO_MATCH_RATIO = 0.75          # Lowe's ratio test threshold for feature matching
VIDEO_MIN_MATCHES = 30            # minimum good matches to accept a frame pair
VIDEO_RANSAC_PROB = 0.999         # essential-matrix RANSAC confidence
VIDEO_RANSAC_THRESH_PX = 1.0      # essential-matrix RANSAC reprojection threshold (px)
VIDEO_MAX_POINT_DIST = 100.0      # drop triangulated points farther than this (unscaled units)
VIDEO_OUTLIER_MEDIAN_MULT = 2.0   # drop points farther than this x the median distance from the median point
VIDEO_MIN_RECONSTRUCTED_PTS = 50  # minimum sparse points to attempt scale recovery
# Scale-ensemble priors (independent heuristics; their spread -> CI width)
VIDEO_ASSUMED_CEILING_M = 2.4         # typical residential ceiling height
VIDEO_ASSUMED_DIAGONAL_M = 7.5        # typical room's full 3-D bbox diagonal (floor diagonal
                                       # + height combined via Pythagoras, not just floor footprint)
VIDEO_ASSUMED_CAMERA_HEIGHT_M = 1.3   # typical handheld phone height above floor
VIDEO_DEFAULT_SCALE_REL_HW = 0.30     # fallback relative CI half-width if ensemble has 1 estimate
VIDEO_DRIFT_ABLATION_FRAME_COUNTS = [5, 10, 20, 40]  # chain lengths tested by bench/ablate.py

# --- Photo tier (Phase 8): 2-8 unordered stills per room, no depth/poses ---
PHOTO_MIN_PHOTOS = 2
PHOTO_MAX_PHOTOS = 8
# No real multi-room photo fixture exists to calibrate this against (unlike
# the video tier's single real example). Set conservatively at/above the
# video tier's empirical floor, since fewer, unordered stills give the same
# reconstruction method strictly less information to work with -- revisit
# once real photo-tier data is available.
PHOTO_EMPIRICAL_MIN_REL_HW = 0.90
# A single two-view (best_pair) reconstruction inherently has far fewer
# triangulated points than the video tier's many-pair accumulation -- don't
# reuse that tier's point-count floor.
PHOTO_MIN_RECONSTRUCTED_PTS = 5
# extract_layout's floor/ceiling RANSAC defaults to min_inliers=200 (tuned for
# dense LiDAR/video clouds); a sparse best_pair reconstruction needs this much
# lower just to attempt plane detection at all. Confidence at this point
# count is inherently low -- PHOTO_EMPIRICAL_MIN_REL_HW above is what keeps
# that honest, not this threshold.
PHOTO_MIN_PLANE_INLIERS = 5
# Multi-room stitcher (roomscan/geometry/multi_room.py): best-effort cross-
# room matching, falling back to grid placement when it doesn't find enough
# shared structure between two rooms' photos (e.g. a shared doorway view).
MULTI_ROOM_MIN_MATCHES = 30
MULTI_ROOM_GRID_GAP_M = 1.0   # gap between rooms placed on the fallback grid

# --- Pose-graph multi-room stitching (case-study realignment) ---
# Door-to-door correspondence: two openings (one per room) are considered the
# same physical door if their measured widths agree within this fraction.
POSE_GRAPH_DOOR_WIDTH_TOL_FRAC = 0.35
POSE_GRAPH_WALL_THICKNESS_M = 0.15   # gap prior between two rooms' shared wall faces
POSE_GRAPH_HUBER_DELTA = 0.05        # scipy least_squares robust-loss scale (metres)
POSE_GRAPH_MIN_OVERLAP_FRAC = 0.05   # reject a placement if AABB overlap exceeds this
                                      # fraction of the smaller room's own area
VIDEO_PLANE_THRESH_FRAC = 0.02        # plane-RANSAC threshold as a fraction of the cloud's bbox diagonal
                                       # (the unscaled cloud has no metric units yet, so a fixed-metre
                                       # threshold like RANSAC_THRESH_M is meaningless here)
# This single-frame-chaining SfM (no bundle adjustment) measured ~65-88% error
# against real LiDAR ground truth on the one real video tested (single_room:
# floor area 4.4 vs 36.6 sq m). The scale ensemble's own spread doesn't
# reliably predict error that large, so CIs are floored at this relative
# half-width regardless of ensemble agreement. Calibrated on n=1 example --
# revisit with more real video/LiDAR comparison pairs (see derive_tiers.py).
VIDEO_EMPIRICAL_MIN_REL_HW = 0.85

# --- Damage detection (Phase 10): crude colour heuristic, advisory only ---
# No trained classifier exists here (would need labelled damage imagery this
# project has none of) -- this flags dark, desaturated blobs that deviate
# from a frame's own median brightness. It WILL also flag shadows, dark
# furniture, and normal wall texture. COMPLIANCE.md already states damage
# classification is advisory, human review required -- this module produces
# exactly that: a prompt for review, not a diagnosis.
DAMAGE_VALUE_DROP_THRESH = 0.25     # min brightness drop (relative to frame median) to flag a pixel
DAMAGE_SATURATION_THRESH = 0.35     # max saturation to flag a pixel (stains are usually desaturated)
DAMAGE_MIN_BLOB_AREA_FRAC = 0.005   # ignore specks smaller than this fraction of frame area
DAMAGE_MAX_BLOB_AREA_FRAC = 0.40    # ignore blobs this large (more likely whole-frame lighting/shadow)
DAMAGE_FLOOR_ZONE_FRAC = 0.30       # bottom fraction of frame height counted as "floor-adjacent"
DAMAGE_CEILING_ZONE_FRAC = 0.20     # top fraction of frame height counted as "ceiling-adjacent"
DAMAGE_LARGE_STAIN_AREA_FRAC = 0.05  # floor-adjacent blob above this frame-fraction triggers R001
DAMAGE_VERY_DARK_VALUE = 0.15        # mean V below this triggers the mould-risk flag R002
# Area-estimate CI: converting a 2-D pixel blob to a 3-D m^2 without a real
# wall projection is crude -- wide, asymmetric bounds (quick easy
# underestimate, i.e. missed extent behind furniture, more likely than
# overestimate) rather than a false-precision number.
DAMAGE_AREA_CI_LOW_MULT = 0.3
DAMAGE_AREA_CI_HIGH_MULT = 2.5
