# """
# Object role taxonomy — FIXED & BALANCED
# """

# # ─────────────────────────────────────────────
# # SURFACES
# # ─────────────────────────────────────────────

# PEDESTRIAN_SURFACES = {
#     "sidewalk", "crosswalk", "footpath",
#     "tactile paving", "curb cut",
# }

# VEHICLE_SURFACES = {"road", "street", "asphalt"}

# NON_WALKABLE_SURFACES = {"grass", "gravel", "dirt", "plant"}

# SAM_CLASSES = PEDESTRIAN_SURFACES | VEHICLE_SURFACES | NON_WALKABLE_SURFACES


# # ─────────────────────────────────────────────
# # ACCESSIBILITY
# # ─────────────────────────────────────────────

# ACCESSIBILITY_FEATURES = {
#     "curb cut", "tactile paving", "crosswalk",
# }

# ACCESSIBILITY_HAZARDS = {
#     "pothole", "puddle",
#     "construction zone", "uneven surface",
# }


# # ─────────────────────────────────────────────
# # OBJECT ROLES (FIXED)
# # ─────────────────────────────────────────────

# OBJECT_ROLES: dict[str, str] = {

#     # Walkable
#     "sidewalk": "walkable",
#     "crosswalk": "walkable",
#     "footpath": "walkable",
#     "tactile paving": "walkable",
#     "curb cut": "walkable",

#     # Semi-walkable
#     "road": "semi_walkable",
#     "street": "semi_walkable",
#     "asphalt": "semi_walkable",

#     # Non-walkable
#     "grass": "non_walkable",
#     "dirt": "non_walkable",
#     "plant": "non_walkable",

#     # Dynamic (MOST IMPORTANT)
#     "person": "dynamic_hazard",
#     "cyclist": "dynamic_hazard",
#     "motorcyclist": "dynamic_hazard",
#     "wheelchair": "dynamic_hazard",

#     # Vehicles
#     "car": "hazard",
#     "truck": "hazard",
#     "bus": "hazard",
#     "bicycle": "hazard",
#     "motorcycle": "hazard",
#     "scooter": "hazard",

#     # Fixed obstacles 
#     "traffic cone": "obstacle",  
#     "barrier": "obstacle",
#     "bollard": "obstacle",
#     "pole": "obstacle",
#     "fence": "obstacle",
#     "railing": "obstacle",
#     "bench": "obstacle",
#     "fire hydrant": "obstacle",
#     "tree": "obstacle",

#     # Accessibility (NOT HARD BLOCK)
#     "pothole": "accessibility_hazard",
#     "puddle": "accessibility_hazard",
#     "construction zone": "accessibility_hazard",
#     "uneven surface": "accessibility_hazard",

#     # Context
#     "traffic light": "landmark",
#     "stop sign": "landmark",
#     "traffic sign": "landmark",
#     "building": "context",
# }


# # ─────────────────────────────────────────────
# # BLOCKING LOGIC (FIXED)
# # ─────────────────────────────────────────────

# HARD_BLOCK_ROLES = {"obstacle", "hazard"}  

# SOFT_BLOCK_ROLES = {"dynamic_hazard"}

# OBSTACLE_ROLES_SET = HARD_BLOCK_ROLES | SOFT_BLOCK_ROLES


# # ─────────────────────────────────────────────
# # RELEVANCE TIERS (FIXED)
# # ─────────────────────────────────────────────

# NAV_CRITICAL = {
#     "person", "cyclist", "motorcyclist", "wheelchair",
#     "car", "truck", "bus", "motorcycle", "scooter",
# }

# NAV_CONDITIONAL = {
#     "tree", "pole", "fence", "railing", "bench", "fire hydrant",
#     "traffic cone"
# }

# CONTEXT_ONLY = {
#     "grass", "gravel", "dirt", "plant",
#     "building", "traffic light", "traffic sign", "stop sign",
#     "curb cut", "tactile paving",
# }

# DYNAMIC_LABELS = {
#     "person", "pedestrian", "cyclist", "motorcyclist",
#     "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
# }

# SURFACE_QUALITY = {
#     "sidewalk": "smooth",
#     "crosswalk": "smooth",
#     "footpath": "rough",
#     "tactile paving": "textured",
#     "curb cut": "smooth",
#     "road": "smooth",
#     "asphalt": "smooth",
#     "gravel": "rough",
#     "dirt": "rough",
#     "grass": "soft",
#     "pothole": "damaged",
#     "puddle": "wet",
#     "construction zone": "uneven",
#     "uneven surface": "uneven",
# }


# # ─────────────────────────────────────────────
# # OUTPUT CATEGORY (FIXED CORE LOGIC)
# # ─────────────────────────────────────────────

# def get_output_category(
#     label: str,
#     role: str,
#     on_path: bool,
#     distance: str,
#     in_obstacle_roles: bool,
# ) -> str:

#     # Accessibility hazards → only if relevant
#     if label in ACCESSIBILITY_HAZARDS:
#         if on_path and distance in ("near", "mid"):
#             return "obstacle"
#         return "silent"

#     # Accessibility features
#     if label in ACCESSIBILITY_FEATURES:
#         return "surface"

#     # Context ignore
#     if label in CONTEXT_ONLY:
#         return "silent"

#     # Conditional objects (trees, cones, etc.)
#     if label in NAV_CONDITIONAL:
#         if on_path and distance == "near":
#             return "obstacle"
#         return "silent"

#     # Surfaces
#     if role in ("walkable", "semi_walkable"):
#         return "surface"

#     # Main obstacle logic (FIXED)
#     if in_obstacle_roles:
#         if on_path and distance in ("near", "mid"):
#             return "obstacle"
#         return "silent"

#     return "silent"


"""
Object role taxonomy — FIXED & BALANCED

Key fixes:
- Sidewalk/road disambiguation via context scoring
- Proper blocking logic with size awareness
- Better output category gating
"""

# ─────────────────────────────────────────────
# SURFACES
# ─────────────────────────────────────────────

PEDESTRIAN_SURFACES = {
    "sidewalk", "crosswalk", "footpath",
    "tactile paving", "curb cut",
}

VEHICLE_SURFACES = {"road", "street", "asphalt"}

NON_WALKABLE_SURFACES = {"grass", "gravel", "dirt", "plant"}

SAM_CLASSES = PEDESTRIAN_SURFACES | VEHICLE_SURFACES | NON_WALKABLE_SURFACES

# FIXED: Surface priority for disambiguation — when both sidewalk+road detected,
# sidewalk wins for pedestrian navigation purposes
SURFACE_PRIORITY = {
    "sidewalk":      10,   # highest pedestrian priority
    "crosswalk":     9,
    "footpath":      9,
    "tactile paving": 8,
    "curb cut":      8,
    "road":          3,    # lower — vehicle surface
    "street":        3,
    "asphalt":       2,
    "grass":         1,
    "gravel":        1,
    "dirt":          1,
}

# ─────────────────────────────────────────────
# ACCESSIBILITY
# ─────────────────────────────────────────────

ACCESSIBILITY_FEATURES = {
    "curb cut", "tactile paving", "crosswalk",
}

ACCESSIBILITY_HAZARDS = {
    "pothole", "puddle",
    "construction zone", "uneven surface",
}

# ─────────────────────────────────────────────
# OBJECT ROLES (FIXED)
# ─────────────────────────────────────────────

OBJECT_ROLES: dict[str, str] = {
    # Walkable
    "sidewalk":       "walkable",
    "crosswalk":      "walkable",
    "footpath":       "walkable",
    "tactile paving": "walkable",
    "curb cut":       "walkable",

    # Semi-walkable
    "road":      "semi_walkable",
    "street":    "semi_walkable",
    "asphalt":   "semi_walkable",

    # Non-walkable
    "grass":  "non_walkable",
    "dirt":   "non_walkable",
    "plant":  "non_walkable",
    "gravel": "non_walkable",

    # Dynamic (MOST IMPORTANT)
    "person":       "dynamic_hazard",
    "pedestrian":   "dynamic_hazard",
    "cyclist":      "dynamic_hazard",
    "motorcyclist": "dynamic_hazard",
    "wheelchair":   "dynamic_hazard",
    "stroller":     "dynamic_hazard",

    # Vehicles
    "car":        "hazard",
    "truck":      "hazard",
    "bus":        "hazard",
    "bicycle":    "hazard",
    "motorcycle": "hazard",
    "scooter":    "hazard",

    # Fixed obstacles
    "traffic cone":  "obstacle",
    "barrier":       "obstacle",
    "bollard":       "obstacle",
    "pole":          "obstacle",
    "fence":         "obstacle",
    "railing":       "obstacle",
    "bench":         "obstacle",
    "fire hydrant":  "obstacle",
    "tree":          "obstacle",

    # Accessibility (NOT HARD BLOCK)
    "pothole":           "accessibility_hazard",
    "puddle":            "accessibility_hazard",
    "construction zone": "accessibility_hazard",
    "uneven surface":    "accessibility_hazard",

    # Context
    "traffic light": "landmark",
    "stop sign":     "landmark",
    "traffic sign":  "landmark",
    "building":      "context",
}

# ─────────────────────────────────────────────
# BLOCKING LOGIC
# ─────────────────────────────────────────────

HARD_BLOCK_ROLES = {"obstacle", "hazard"}
SOFT_BLOCK_ROLES = {"dynamic_hazard"}
OBSTACLE_ROLES_SET = HARD_BLOCK_ROLES | SOFT_BLOCK_ROLES

# FIXED: Size-aware blocking thresholds (fraction of image area)
# Small objects need to be closer/more numerous to be considered blocking
SIZE_WEIGHT_THRESHOLDS = {
    "large":  0.05,    # > 5% of image → full weight
    "medium": 0.015,   # 1.5–5%        → 0.6x weight
    "small":  0.005,   # 0.5–1.5%      → 0.3x weight
    "tiny":   0.0,     # < 0.5%        → 0.1x weight
}

def get_size_weight(area_ratio: float) -> float:
    """Return a 0.1–1.0 weight based on object size relative to image."""
    if area_ratio >= SIZE_WEIGHT_THRESHOLDS["large"]:
        return 1.0
    elif area_ratio >= SIZE_WEIGHT_THRESHOLDS["medium"]:
        return 0.6
    elif area_ratio >= SIZE_WEIGHT_THRESHOLDS["small"]:
        return 0.3
    else:
        return 0.1

# ─────────────────────────────────────────────
# RELEVANCE TIERS
# ─────────────────────────────────────────────

NAV_CRITICAL = {
    "person", "pedestrian", "cyclist", "motorcyclist", "wheelchair", "stroller",
    "car", "truck", "bus", "motorcycle", "scooter",
}

NAV_CONDITIONAL = {
    "tree", "pole", "fence", "railing", "bench", "fire hydrant",
    "traffic cone", "bollard", "barrier",
}

CONTEXT_ONLY = {
    "grass", "gravel", "dirt", "plant",
    "building", "traffic light", "traffic sign", "stop sign",
    "curb cut", "tactile paving",
}

DYNAMIC_LABELS = {
    "person", "pedestrian", "cyclist", "motorcyclist",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
}

SURFACE_QUALITY = {
    "sidewalk":          "smooth",
    "crosswalk":         "smooth",
    "footpath":          "rough",
    "tactile paving":    "textured",
    "curb cut":          "smooth",
    "road":              "smooth",
    "asphalt":           "smooth",
    "gravel":            "rough",
    "dirt":              "rough",
    "grass":             "soft",
    "pothole":           "damaged",
    "puddle":            "wet",
    "construction zone": "uneven",
    "uneven surface":    "uneven",
}

# ─────────────────────────────────────────────
# OUTPUT CATEGORY
# ─────────────────────────────────────────────

def get_output_category(
    label: str,
    role: str,
    on_path: bool,
    distance: str,
    in_obstacle_roles: bool,
) -> str:
    """Determine how an object should appear in navigation output."""

    # Accessibility hazards — only if on path and close
    if label in ACCESSIBILITY_HAZARDS:
        if on_path and distance in ("near", "mid"):
            return "obstacle"
        return "silent"

    # Accessibility features
    if label in ACCESSIBILITY_FEATURES:
        return "surface"

    # Context: always silent
    if label in CONTEXT_ONLY:
        return "silent"

    # Conditional objects: only if on path AND near
    if label in NAV_CONDITIONAL:
        if on_path and distance == "near":
            return "obstacle"
        return "silent"

    # Surfaces
    if role in ("walkable", "semi_walkable"):
        return "surface"

    # Main obstacle logic
    if in_obstacle_roles:
        if on_path and distance in ("near", "mid"):
            return "obstacle"
        return "silent"

    return "silent"