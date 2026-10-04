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
