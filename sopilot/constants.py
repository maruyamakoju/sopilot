"""Single source of truth for default values and thresholds."""

# --- Score weights (used by schemas, scoring, sopilot_service) ---
DEFAULT_WEIGHTS = {
    "w_miss": 0.40,
    "w_swap": 0.25,
    "w_dev": 0.25,
    "w_time": 0.10,
}

# --- Deviation policy defaults ---
DEFAULT_DEVIATION_POLICY = {
    "missing_step": "critical",
    "step_deviation": "quality",
    "order_swap": "quality",
    "over_time": "efficiency",
}

# --- Video quality thresholds ---
BRIGHTNESS_LOW = 20.0
BRIGHTNESS_HIGH = 235.0
BLUR_SHARPNESS_THRESHOLD = 40.0

# --- Scoring collapse detection ratios ---
COLLAPSE_UNIQUE_VS_GOLD_RATIO = 0.45
COLLAPSE_UNIQUE_VS_EXPECTED_RATIO = 0.35
COLLAPSE_COST_FACTOR = 0.8

# --- Over-time cap for penalty calculation ---
OVER_TIME_CAP = 1.5

# --- Dataset progress targets ---
DATASET_TARGET_GOLD = 20
DATASET_TARGET_TRAINEE = 50
