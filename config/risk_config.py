# """
# Risk severity, weights, and thresholds (BALANCED VERSION)
# """
# from typing import Dict


# # ─────────────────────────────────────────────────────────────
# # SEVERITY (re-balanced by category)
# # ─────────────────────────────────────────────────────────────

# SEVERITY: Dict[str, float] = {

#     # ── Dynamic agents (highest priority)
#     "person": 3.0,
#     "cyclist": 3.0,
#     "motorcyclist": 3.0,
#     "wheelchair": 2.5,

#     # ── Vehicles (high but slightly controlled)
#     "car": 2.8,
#     "truck": 3.0,
#     "bus": 3.0,
#     "bicycle": 2.0,
#     "motorcycle": 2.3,
#     "scooter": 2.0,

#     # ── Blocking obstacles (moderate)
#     "barrier": 2.0,
#     "construction zone": 2.2,

#     # ── Path hazards (context-based)
#     "pothole": 1.8,
#     "uneven surface": 1.2,
#     "puddle": 0.5,

#     # ── Small obstacles (LOW unless grouped)
#     "traffic cone": 1.2,
#     "bollard": 1.2,
#     "fire hydrant": 1.0,

#     # ── Structural / background (very low)
#     "tree": 0.5,
#     "pole": 0.5,
#     "fence": 0.6,
#     "railing": 0.5,
#     "bench": 0.4,
# }

# DEFAULT_SEVERITY = 0.5


# # ─────────────────────────────────────────────────────────────
# # DISTANCE WEIGHT (more realistic)
# # ─────────────────────────────────────────────────────────────

# DIST_WEIGHT: Dict[str, float] = {
#     "near": 1.0,
#     "mid": 0.6,
#     "far": 0.2,   # important: don’t ignore future obstacles
# }


# # ─────────────────────────────────────────────────────────────
# # RISK THRESHOLDS (aligned with SUM logic)
# # ─────────────────────────────────────────────────────────────

# CAUTION_URGENT = 7.0
# CAUTION_HIGH   = 4.5
# CAUTION_MEDIUM = 2.5
# CAUTION_LOW    = 0.8


# # ─────────────────────────────────────────────────────────────
# # OPTIONAL: CATEGORY GROUPING (use later if needed)
# # ─────────────────────────────────────────────────────────────

# DYNAMIC_CLASSES = {
#     "person", "cyclist", "motorcyclist", "wheelchair",
#     "car", "truck", "bus", "bicycle", "motorcycle", "scooter"
# }

# STATIC_CLASSES = {
#     "traffic cone", "barrier", "bollard", "pole", "tree",
#     "fence", "railing", "bench", "fire hydrant"
# }

# SURFACE_HAZARDS = {
#     "pothole", "uneven surface", "construction zone"
# }


"""
Risk severity, weights, and thresholds — DYNAMIC VERSION

Key fixes:
- Direction-specific risk (obstacle right ≠ affect left)
- Dynamic cone cluster scaling
- Better caution thresholds
"""
from typing import Dict

# ─────────────────────────────────────────────
# SEVERITY (re-balanced)
# ─────────────────────────────────────────────

SEVERITY: Dict[str, float] = {
    # Dynamic agents (highest priority)
    "person":       3.0,
    "pedestrian":   3.0,
    "cyclist":      3.0,
    "motorcyclist": 3.0,
    "wheelchair":   2.5,
    "stroller":     2.5,

    # Vehicles
    "car":        2.8,
    "truck":      3.0,
    "bus":        3.0,
    "bicycle":    2.0,
    "motorcycle": 2.3,
    "scooter":    2.0,

    # Blocking obstacles
    "barrier":           2.0,
    "construction zone": 2.2,
    "bollard":           1.5,

    # Path hazards
    "pothole":        1.8,
    "uneven surface": 1.2,
    "puddle":         0.5,

    # Small obstacles (LOW unless grouped — dynamic boost applied at runtime)
    "traffic cone":  1.0,   # base; boosted dynamically per cluster
    "fire hydrant":  1.0,
    "pole":          0.6,
    "fence":         0.6,
    "railing":       0.5,
    "bench":         0.5,
    "tree":          0.5,
}

DEFAULT_SEVERITY = 0.5


# ─────────────────────────────────────────────
# DISTANCE WEIGHT
# ─────────────────────────────────────────────

DIST_WEIGHT: Dict[str, float] = {
    "near": 1.0,
    "mid":  0.6,
    "far":  0.15,   # far objects matter less now
}


# ─────────────────────────────────────────────
# RISK THRESHOLDS (tuned for SUM logic)
# ─────────────────────────────────────────────

CAUTION_URGENT = 6.5
CAUTION_HIGH   = 4.0
CAUTION_MEDIUM = 2.0
CAUTION_LOW    = 0.7


# ─────────────────────────────────────────────
# DYNAMIC CLUSTER SCALING
# ─────────────────────────────────────────────

def get_cluster_boost(label: str, count: int) -> float:
    """
    Dynamic severity boost for clustered objects.
    A cluster of 3 cones is more dangerous than 1 car sometimes.
    """
    if label == "person":
        if count >= 5:   return 2.2
        if count >= 3:   return 1.6
        if count >= 2:   return 1.2
        return 1.0

    if label == "traffic cone":
        if count >= 5:   return 2.5   # construction zone level
        if count >= 3:   return 1.8
        if count >= 2:   return 1.3
        return 1.0

    if label in ("car", "bicycle", "motorcycle", "scooter"):
        if count >= 3:   return 1.5
        if count >= 2:   return 1.2
        return 1.0

    return 1.0


# ─────────────────────────────────────────────
# DIRECTION-SPECIFIC RISK WEIGHTS
# ─────────────────────────────────────────────

# How much does an obstacle in one direction affect others?
# Format: {obstacle_dir: {affected_dir: weight}}
DIRECTION_SPILLOVER = {
    "center": {"left": 0.3, "center": 1.0, "right": 0.3},
    "left":   {"left": 1.0, "center": 0.2, "right": 0.0},
    "right":  {"left": 0.0, "center": 0.2, "right": 1.0},
}


# ─────────────────────────────────────────────
# CATEGORY GROUPINGS (for future use)
# ─────────────────────────────────────────────

DYNAMIC_CLASSES = {
    "person", "pedestrian", "cyclist", "motorcyclist", "wheelchair",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter"
}

STATIC_CLASSES = {
    "traffic cone", "barrier", "bollard", "pole", "tree",
    "fence", "railing", "bench", "fire hydrant"
}

SURFACE_HAZARDS = {
    "pothole", "uneven surface", "construction zone"
}
