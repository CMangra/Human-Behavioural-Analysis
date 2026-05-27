# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

# --- BASE DATASET CONSTANTS ---
CAMERAS = ['blu79CF', 'grn43E3', 'red707B', 'ylw79D0']

# --- STEP 2: PEDESTRIAN FILTERING & TURN MATH ---
MIN_VISIBILITY_RATIO = 0.6
TURN_CUMULATIVE_DEG = 35    # Lowered slightly because heavy smoothing compresses the angles
TURN_NOISE_FLOOR_DEG = 1.0  # With smoothed X/Y, the baseline will drop close to 0
TURN_PEAK_PROMINENCE = 1.5  # Require a distinct spike over the smoothed baseline
XY_SMOOTHING_WINDOW = 15    # Look across ~1.5 seconds of movement to kill footstep wobble
TURN_DEBOUNCE_FRAMES = 30   # Minimum frames (3 secs) between distinct turns to prevent double-counting
