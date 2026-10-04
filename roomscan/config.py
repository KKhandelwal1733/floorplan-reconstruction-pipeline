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
