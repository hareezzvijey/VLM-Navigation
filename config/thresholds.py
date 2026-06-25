# """Confidence thresholds and filter parameters"""
# import os
# from .paths import WEIGHTS_DIR

# # Box thresholds - will be adjusted by CPU fallback at runtime
# BOX_THRESHOLD_DEFAULT = 0.30
# THRESHOLD_LADDER = [0.30, 0.25, 0.20]

# # Per-class confidence thresholds (GPU baseline; auto-scaled for CPU)
# PER_CLASS_THRESHOLDS: dict[str, float] = {
#     # Dynamic agents
#     "person": 0.40,
#     "pedestrian": 0.38,
#     "cyclist": 0.38,
#     "motorcyclist": 0.38,
#     "wheelchair": 0.35,
#     "stroller": 0.35,
#     # Vehicles
#     "car": 0.45,
#     "truck": 0.45,
#     "bus": 0.45,
#     "bicycle": 0.40,
#     "motorcycle": 0.40,
#     "scooter": 0.40,
#     # Landmarks
#     "traffic light": 0.42,
#     "stop sign": 0.42,
#     "traffic sign": 0.38,
#     # Fixed obstacles
#     "pole": 0.35,
#     "bollard": 0.35,
#     "barrier": 0.40,
#     "bench": 0.40,
#     "traffic cone": 0.38,
#     # Surfaces
#     "road": 0.42,
#     "sidewalk": 0.35,
#     "crosswalk": 0.35,
#     "grass": 0.35,
#     "gravel": 0.35,
#     # Context
#     "building": 0.50,
#     "tree": 0.45,
#     # Accessibility features
#     "curb cut": 0.32,
#     "tactile paving": 0.32,
#     # Accessibility hazards
#     "pothole": 0.30,
#     "puddle": 0.30,
#     "construction zone": 0.35,
#     "uneven surface": 0.30,
# }
# DEFAULT_THRESHOLD = 0.42

# # Area filter
# MIN_ABSOLUTE_PIXELS = 400
# MAX_RELATIVE_AREA = 0.85
# MAX_ASPECT_RATIO = 12.0

# # NMS
# SOFT_NMS_SIGMA = 0.5
# SOFT_NMS_SCORE_GATE = 0.20

# # Model weights paths
# DINO_CONFIG = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
# DINO_CKPT = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
# SAM_CKPT = os.path.join(WEIGHTS_DIR, "sam_vit_l_0b3195.pth")

"""
Confidence thresholds and filter parameters — FIXED & BALANCED
"""
import os
from .paths import WEIGHTS_DIR

# Box thresholds
BOX_THRESHOLD_DEFAULT = 0.28
THRESHOLD_LADDER = [0.28, 0.22, 0.18]

# Per-class confidence thresholds (GPU baseline; auto-scaled for CPU)
# FIXED: Lowered person/pedestrian to prevent aggressive filtering
PER_CLASS_THRESHOLDS: dict[str, float] = {
    # Dynamic agents — CRITICAL, keep LOW to avoid missing them
    "person":        0.32,   # was 0.40 — too aggressive
    "pedestrian":    0.30,
    "cyclist":       0.32,
    "motorcyclist":  0.32,
    "wheelchair":    0.28,
    "stroller":      0.28,
    # Vehicles
    "car":           0.38,   # was 0.45
    "truck":         0.38,
    "bus":           0.38,
    "bicycle":       0.32,
    "motorcycle":    0.32,
    "scooter":       0.32,
    # Landmarks
    "traffic light": 0.38,
    "stop sign":     0.38,
    "traffic sign":  0.32,
    # Fixed obstacles
    "pole":          0.28,
    "bollard":       0.28,
    "barrier":       0.32,
    "bench":         0.32,
    "traffic cone":  0.30,
    # Surfaces — use LOWER thresholds so sidewalk is NOT replaced by road
    "road":          0.35,
    "sidewalk":      0.28,   # CRITICAL FIX: was too high, sidewalk got dropped
    "crosswalk":     0.28,
    "grass":         0.28,
    "gravel":        0.28,
    # Context
    "building":      0.45,
    "tree":          0.38,
    # Accessibility
    "curb cut":      0.25,
    "tactile paving": 0.25,
    # Hazards
    "pothole":       0.25,
    "puddle":        0.25,
    "construction zone": 0.28,
    "uneven surface": 0.25,
}
DEFAULT_THRESHOLD = 0.35

# Area filter — RELAXED to avoid dropping people in distance
MIN_ABSOLUTE_PIXELS = 200       # was 400 — too aggressive
MAX_RELATIVE_AREA = 0.90        # was 0.85
MAX_ASPECT_RATIO = 14.0         # was 12.0

# NMS
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_SCORE_GATE = 0.15      # was 0.20 — dropping too many

# Model weights paths
DINO_CONFIG = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
DINO_CKPT   = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
SAM_CKPT    = os.path.join(WEIGHTS_DIR, "sam_vit_l_0b3195.pth")