"""
Risk severity, weights, and thresholds (BALANCED VERSION)
"""
from typing import Dict


# ─────────────────────────────────────────────────────────────
# SEVERITY (re-balanced by category)
# ─────────────────────────────────────────────────────────────

SEVERITY: Dict[str, float] = {

    # ── Dynamic agents (highest priority)
    "person": 3.0,
    "cyclist": 3.0,
    "motorcyclist": 3.0,
    "wheelchair": 2.5,

    # ── Vehicles (high but slightly controlled)
    "car": 2.8,
    "truck": 3.0,
    "bus": 3.0,
    "bicycle": 2.0,
    "motorcycle": 2.3,
    "scooter": 2.0,

    # ── Blocking obstacles (moderate)
    "barrier": 2.0,
    "construction zone": 2.2,

    # ── Path hazards (context-based)
    "pothole": 1.8,
    "uneven surface": 1.2,
    "puddle": 0.5,

    # ── Small obstacles (LOW unless grouped)
    "traffic cone": 1.2,
    "bollard": 1.2,
    "fire hydrant": 1.0,

    # ── Structural / background (very low)
    "tree": 0.5,
    "pole": 0.5,
    "fence": 0.6,
    "railing": 0.5,
    "bench": 0.4,
}

DEFAULT_SEVERITY = 0.5


# ─────────────────────────────────────────────────────────────
# DISTANCE WEIGHT (more realistic)
# ─────────────────────────────────────────────────────────────

DIST_WEIGHT: Dict[str, float] = {
    "near": 1.0,
    "mid": 0.6,
    "far": 0.2,   # important: don’t ignore future obstacles
}


# ─────────────────────────────────────────────────────────────
# RISK THRESHOLDS (aligned with SUM logic)
# ─────────────────────────────────────────────────────────────

CAUTION_URGENT = 7.0
CAUTION_HIGH   = 4.5
CAUTION_MEDIUM = 2.5
CAUTION_LOW    = 0.8


# ─────────────────────────────────────────────────────────────
# OPTIONAL: CATEGORY GROUPING (use later if needed)
# ─────────────────────────────────────────────────────────────

DYNAMIC_CLASSES = {
    "person", "cyclist", "motorcyclist", "wheelchair",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter"
}

STATIC_CLASSES = {
    "traffic cone", "barrier", "bollard", "pole", "tree",
    "fence", "railing", "bench", "fire hydrant"
}

SURFACE_HAZARDS = {
    "pothole", "uneven surface", "construction zone"
}