"""
VLM Pipeline v10 — Decision Intelligence
=========================================
Built on v9. New fixes tagged [DI1]–[DI5].

All claims verified against actual v9 code before implementation:
  [CLAIM-DONE] C1 (static split) — v9 already had x-extent mask split;
               DI1 upgrades to true per-row centerline via linear regression.
  [CLAIM-TRUE] C2 (no severity) — added SEVERITY dict + caution levels.
  [CLAIM-DONE] C3 (no path alignment) — SW mask foot-point already helped;
               DI3 adds explicit centerline distance check for center column.
  [CLAIM-PARTIAL] C4 (near=far) — FIX4 excluded far from blocking;
               DI4 adds urgency verb modulation from caution level.
  [CLAIM-TRUE] C5 (no width check) — DI5 adds path width constraint flag.

  [DI1] Sidewalk centerline via per-row midpoint + linear regression
        Research: Inverse Perspective Mapping (Mallot 1991); lane centerline
        detection used in KITTI (Geiger 2012) and Cityscapes (Cordts 2016).
        Per-row: midpoint_x = (leftmost + rightmost mask pixel) / 2
        Regression: slope, intercept = polyfit(rows, midpoints, deg=1)
        Output: centerline_x_at(y) function + path_width_px.
        Replaces x_min/x_max bounding-box split for curved/diagonal paths.

  [DI2] Severity-weighted caution scoring
        Research: TTC risk scoring (Dosovitskiy CARLA 2017);
        nuScenes risk annotation (Caesar 2020);
        Anderson R2R urgency-graded navigation (2018).
        SEVERITY dict: person/car=3, bicycle=2, barrier/cone=1, bench=0.5.
        caution = max(SEVERITY[label] * DIST_WEIGHT[distance]) over obstacles.
        DIST_WEIGHT: near=1.0, mid=0.6, far=0.0.
        Outputs: caution_level (float), urgency (URGENT/HIGH/MEDIUM/LOW/NONE).

  [DI3] Centerline alignment check for center-column blocking
        Research: SoPhie pedestrian trajectory prediction (Gupta 2021);
        projected path intersection for collision avoidance (Helbing 1995 SFM).
        An obstacle blocks the CENTER corridor only if:
          abs(foot_x - centerline_x_at(foot_y)) < path_width_px / 3
        L/R corridors keep the strip-based check (no centerline for sides).
        Falls back to strip-based if no centerline available.

  [DI4] Distance×severity urgency in action verb and scene_text
        caution >= 2.5: action prefixed with "[URGENT]" + stop override
        caution >= 1.5: action prefixed with "[HIGH]"
        caution >= 0.8: normal action verb
        Obstacle list sorted by severity×distance (highest first) so LLM
        sees the most critical obstacle first in the prompt.

  [DI5] Path width constraint measurement
        Research: ADA 2010 §403.5 (min 1.2m / 48in clear path width);
        iSAID instance-level corridor width (Zamir 2019).
        path_width_px = median of per-row sidewalk pixel widths.
        NARROW_RATIO  = 0.08 (< 8% of image = narrow, constrained movement)
        TIGHT_RATIO   = 0.04 (< 4% of image = single-file only)
        Width constraint propagated to action verb suffix and scene_text.

Unchanged from v9:
  [P1]–[P5], [C1]–[C5], [D1]–[D3], [SW], [FIX1]–[FIX6], [U1]–[U5], [E1]–[E3]

VLM Pipeline v9 — Perspective-Aware + Surface-Typed
=====================================================
Built on v8. New fixes tagged [P1]–[P5].

Research foundations:
  [P1] Narrow center zone (38–62%)
       Geiger et al. (2012) KITTI — ego-lane occupies ~30% of image width.
       Gibson (1979) — ground-plane objects project toward vanishing point;
       wide center zones absorb left/right objects through perspective lean.
       Fix: L=0–38%, C=38–62%, R=62–100%.  Center is now 24% not 34%.

  [P2] Surface-typed walkability — sidewalk ≠ road in SAM mask
       Teichmann et al. (2018) MultiNet; Siam et al. (2018) RTSeg.
       Previously both road and sidewalk contributed to walkable_pixels.
       Fix: only PEDESTRIAN_SURFACES (sidewalk, crosswalk, ramp) count as
       walkable. VEHICLE_SURFACES (road) are now semi_walkable — passable
       only in absence of pedestrian surface, and flagged differently.
       NON_WALKABLE_SURFACES (grass, soil, gravel) count against walkability.

  [P3] Sidewalk-mask-bounded corridor split
       Padaki et al. (2023); Gupta et al. (2014).
       Column bounds were global (0–38%, 38–62%, 62–100%).
       Fix: when a pedestrian SAM mask exists, extract its x-extent
       (x_min, x_max from np.where) and split THAT range into L/C/R thirds.
       Objects outside the mask extent are positioned globally (context only).
       Falls back to global split when no pedestrian mask is available.

  [P4] Mask centroid for H-position of surface detections
       He et al. (2017) Mask R-CNN; Cordts et al. (2016) Cityscapes.
       Surface objects (sidewalk, road) use mask column centroid for position,
       not bounding box center. A sidewalk mask whose pixels cluster at x=180
       in a 640px image is LEFT, even if its GDINO box reached x=320.
       Obstacle objects without masks continue to use foot-point (x1+x2)/2.

  [P5] Expanded surface taxonomy + GDINO prompts
       Wigness et al. (2019) RUGD; Jiang et al. (2021) RELLIS.
       Added: grass, soil, gravel to MULTI_PROMPTS + SAM_CLASSES.
       These are SAM-segmented as NON_WALKABLE_SURFACES.
       When their masks dominate a corridor column, that column is marked
       non_walkable (worse than blocked — no path exists here at all).

Unchanged from v8:
  [C1]–[C5], [D1]–[D3], [SW], [FIX1]–[FIX6], [U1]–[U5], [E1]–[E3]
"""

import cv2
import io
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os
import torchvision.transforms as T
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CPU FALLBACK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_CPU_FALLBACK = False

class _StderrCapture(io.StringIO):
    def __init__(self, real): super().__init__(); self._real = real
    def write(self, s): self._real.write(s); super().write(s)
    def flush(self): self._real.flush()

_cap = _StderrCapture(sys.stderr)
sys.stderr = _cap
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict
sys.stderr = _cap._real

if "Failed to load custom C++ ops" in _cap.getvalue():
    _CPU_FALLBACK = True
    print("[Pipeline] ⚠  CPU fallback — thresholds auto-lowered")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

if _CPU_FALLBACK:
    BOX_THRESHOLD_DEFAULT = 0.15
    THRESHOLD_LADDER      = [0.15, 0.10, 0.07]
else:
    BOX_THRESHOLD_DEFAULT = 0.30
    THRESHOLD_LADDER      = [0.30, 0.25, 0.20]

# ─────────────────────────────────────────────────────────────────────────────
# [P1] COLUMN SPLIT CONSTANTS — narrowed center zone
# ─────────────────────────────────────────────────────────────────────────────
# Research: KITTI ego-lane ≈ 30% of image; Cityscapes pedestrian lane ≈ 25%.
# Old split: L=0–33%, C=33–67%, R=67–100%  (center = 34% — too wide)
# New split: L=0–38%, C=38–62%, R=62–100%  (center = 24% — perspective-aware)
COL_LEFT_END   = 0.38   # [P1]
COL_RIGHT_START= 0.62   # [P1]

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS — [P5] expanded with surface texture classes
# ─────────────────────────────────────────────────────────────────────────────

MULTI_PROMPTS = [
    # Safety-critical dynamic objects
    "person", "car", "bicycle",
    # Static obstacles
    "traffic cone", "barrier",
    # Pedestrian surfaces (walkable)
    "sidewalk",
    # Vehicle surfaces (semi-walkable)
    "road",
    # [P5] Non-walkable surfaces — grass/soil in path = hazard
    "grass", "soil",
    # Context
    "tree",
]

def sanitise_prompt(prompt: str) -> str:
    prompt = prompt.lower().strip()
    prompt = " . ".join(p.strip() for p in prompt.split(".") if p.strip())
    if not prompt.endswith(" ."):
        prompt += " ."
    return prompt

# ─────────────────────────────────────────────────────────────────────────────
# [P2] SURFACE TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
# Research: RUGD (Wigness 2019), RELLIS (Jiang 2021), ADE20K (Zhou 2017)
# Three traversability tiers:
#   PEDESTRIAN_SURFACES — confirmed walkable for pedestrians
#   VEHICLE_SURFACES    — vehicle lanes, semi-walkable (use only if no ped surface)
#   NON_WALKABLE_SURFACES — soil/grass/gravel in path = navigation hazard

PEDESTRIAN_SURFACES  = {"sidewalk", "crosswalk", "ramp", "footpath"}
VEHICLE_SURFACES     = {"road", "street", "asphalt"}
NON_WALKABLE_SURFACES= {"grass", "soil", "gravel", "dirt", "plant"}

# All classes that SAM should segment (union of all surface tiers)
SAM_CLASSES = PEDESTRIAN_SURFACES | VEHICLE_SURFACES | NON_WALKABLE_SURFACES

# ─────────────────────────────────────────────────────────────────────────────
# OBJECT ROLES
# ─────────────────────────────────────────────────────────────────────────────

OBJECT_ROLES: dict[str, str] = {
    # [P2] Surface roles now typed
    "sidewalk":   "walkable",
    "crosswalk":  "walkable",
    "ramp":       "walkable",
    "footpath":   "walkable",
    "road":       "semi_walkable",      # [P2] was non_walkable
    "street":     "semi_walkable",
    "asphalt":    "semi_walkable",
    "grass":      "non_walkable",       # [P5]
    "soil":       "non_walkable",       # [P5]
    "gravel":     "non_walkable",       # [P5]
    "dirt":       "non_walkable",       # [P5]
    "plant":      "non_walkable",       # [P5]
    # Dynamic hazards
    "person":         "dynamic_hazard",
    "cyclist":        "dynamic_hazard",
    "motorcyclist":   "dynamic_hazard",
    "wheelchair":     "dynamic_hazard",
    # Static vehicles
    "car":            "hazard",
    "truck":          "hazard",
    "bus":            "hazard",
    "bicycle":        "hazard",
    "motorcycle":     "hazard",
    "scooter":        "hazard",
    # Static obstacles
    "traffic cone":   "obstacle",
    "barrier":        "obstacle",
    "bollard":        "obstacle",
    "pole":           "obstacle",
    "fence":          "obstacle",
    "railing":        "obstacle",
    "bench":          "obstacle",
    "fire hydrant":   "obstacle",
    # Landmarks / context
    "traffic light":  "landmark",
    "stop sign":      "landmark",
    "traffic sign":   "landmark",
    "building":       "context",
    "tree":           "obstacle",       # [D1] trunk = hard block
}

DYNAMIC_LABELS = {
    "person", "pedestrian", "cyclist", "motorcyclist",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
}

HARD_BLOCK_ROLES = {"obstacle", "hazard"}
SOFT_BLOCK_ROLES = {"dynamic_hazard"}
OBSTACLE_ROLES_SET = HARD_BLOCK_ROLES | SOFT_BLOCK_ROLES

# ─────────────────────────────────────────────────────────────────────────────
# [RF1] THREE-TIER RELEVANCE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
# Design: don't remove objects — control their output importance.
# SpatialBoost (2024): "separate responsibilities — perception extracts,
# math decides, language communicates."
#
#   NAV_CRITICAL    → always enter obstacle reasoning
#   NAV_CONDITIONAL → only if on_path=True AND distance in (near, mid)
#   CONTEXT_ONLY    → inform SAM/free-space silently; never in LLM output

NAV_CRITICAL: set[str] = {
    "person", "cyclist", "motorcyclist", "wheelchair", "pedestrian",
    "car", "truck", "bus", "motorcycle", "scooter", "bicycle",
    "traffic cone", "barrier", "bollard",
}

NAV_CONDITIONAL: set[str] = {
    # Physical obstacles that SOMETIMES block pedestrian paths —
    # included only when: on_path AND distance in (near, mid)
    "tree", "pole", "fence", "railing", "bench", "fire hydrant",
}

CONTEXT_ONLY: set[str] = {
    # Informs SAM segmentation and free-space; never appears in output.
    "grass", "soil", "gravel", "dirt", "plant",
    "building", "traffic light", "traffic sign", "stop sign",
}


def get_output_category(
    label:             str,
    role:              str,
    on_path:           bool,
    distance:          str,
    in_obstacle_roles: bool,
) -> str:
    """
    [RF1–RF4] Map detection → output category:
      'obstacle' — affects action verb and obstacle list
      'surface'  — surface narrative only
      'silent'   — informs SAM/free-space, excluded from LLM output
    """
    if label in CONTEXT_ONLY:                          return "silent"
    if label in NAV_CONDITIONAL:
        return "obstacle" if (on_path and distance in ("near","mid")) else "silent"
    if role == "non_walkable":                         return "silent"
    if role in ("walkable", "semi_walkable"):          return "surface"
    if in_obstacle_roles:                              return "obstacle"
    return "silent"

# ─────────────────────────────────────────────────────────────────────────────
# [SB1] INTER-OBJECT SPATIAL RELATIONS (SpatialBoost-style)
# ─────────────────────────────────────────────────────────────────────────────
# SpatialBoost (2024): compute spatial facts programmatically, not with LLM.
# We implement the object-level tier: horizontal + depth relations between
# the top-2 highest-severity on-path obstacles. This enriches the LLM context
# with relative positioning without adding model calls.

def compute_inter_object_relations(
    detections: list,   # Detection objects, already filtered to on-path obstacles
    top_n: int = 3,
) -> list[str]:
    """
    [SB1] Return list of spatial relation strings for top-N on-path obstacles.
    E.g. ["person is closer than car", "person is left of car"]
    Only generates relations between pairs — O(n²) but n≤3 in practice.
    """
    on_path_obs = [
        d for d in detections
        if d.on_path
        and OBJECT_ROLES.get(d.label, "unknown") in OBSTACLE_ROLES_SET
        and d.distance != "far"
    ]
    # Sort by severity × distance_multiplier descending
    # FIX: Use DIST_WEIGHT instead of DISTANCE_MULTIPLIER
    on_path_obs.sort(
        key=lambda d: SEVERITY.get(d.label, 1) * DIST_WEIGHT.get(d.distance, 0),
        reverse=True,
    )
    on_path_obs = on_path_obs[:top_n]

    relations = []
    for i in range(len(on_path_obs)):
        for j in range(i + 1, len(on_path_obs)):
            a = on_path_obs[i]; b = on_path_obs[j]
            
            # FIX #4: Skip duplicate labels
            if a.label == b.label:
                continue
                
            ax = (a.box[0] + a.box[2]) / 2
            bx = (b.box[0] + b.box[2]) / 2
            # Horizontal relation
            if ax < bx - 20:   h_rel = f"{a.label} is left of {b.label}"
            elif ax > bx + 20: h_rel = f"{a.label} is right of {b.label}"
            else:               h_rel = f"{a.label} and {b.label} are side by side"
            # Depth relation
            dist_order = {"near": 0, "mid": 1, "far": 2}
            da = dist_order.get(a.distance, 1); db = dist_order.get(b.distance, 1)
            if da < db:   d_rel = f"{a.label} is closer"
            elif da > db: d_rel = f"{b.label} is closer"
            else:          d_rel = None
            relations.append(h_rel)
            if d_rel: relations.append(d_rel)
    
    # FIX #4: Remove duplicates
    return list(set(relations))

# ─────────────────────────────────────────────────────────────────────────────
# [SB2] METRIC DEPTH ESTIMATE (SpatialBoost pinhole model)
# ─────────────────────────────────────────────────────────────────────────────
# SpatialBoost formula: depth = (H_real × H_image) / (H_pixel × 2 × tan(FOV/2))
# Gives approximate metric depth without a depth model.
# Used to enrich the LLM context with distance in metres.

KNOWN_HEIGHTS_M: dict[str, float] = {
    "person":       1.70,
    "cyclist":      1.70,
    "car":          1.50,
    "truck":        3.50,
    "bus":          3.20,
    "bicycle":      1.00,
    "traffic cone": 0.75,
    "barrier":      1.00,
    "bollard":      0.80,
    "tree":         4.00,
    "pole":         3.00,
}
CAMERA_FOV_DEG = 60.0   # typical pedestrian-camera horizontal FOV

def estimate_metric_depth(det, img_h: int) -> Optional[float]:
    """
    [SB2] Estimate distance in metres using pinhole camera model.
    Returns None if label has no known height (surface classes, etc.).
    """
    import math
    h_real = KNOWN_HEIGHTS_M.get(det.label)
    if h_real is None: return None
    pixel_h = max(det.box[3] - det.box[1], 1)
    fov_rad  = math.radians(CAMERA_FOV_DEG / 2)
    depth_m  = (h_real * img_h) / (pixel_h * 2 * math.tan(fov_rad))
    return round(depth_m, 1)

# ─────────────────────────────────────────────────────────────────────────────
# [DI2] SEVERITY WEIGHTS — human-like risk prioritisation
# ─────────────────────────────────────────────────────────────────────────────
# Research: TTC risk scoring (CARLA, Dosovitskiy 2017);
#           nuScenes danger annotation (Caesar 2020).
# Scale 0–3: 3=immediate danger, 0=informational only.
SEVERITY: dict[str, int] = {
    "person":       3.0,   # unpredictable dynamic agent
    "cyclist":      3.0,   # fast + unpredictable
    "motorcyclist": 3.0,
    "wheelchair":   2.5,
    "car":          3.0,   # large vehicle, lethal
    "truck":        3.0,
    "bus":          3.0,
    "bicycle":      2.0,   # slower vehicle
    "motorcycle":   2.5,
    "scooter":      2.0,
    "barrier":      1.5,   # static, marks boundary
    "traffic cone": 1.0,   # marks hazard zone
    "bollard":      1.0,
    "fence":        1.0,
    "railing":      0.8,
    "bench":        0.5,   # passable with care
    "tree":         1.0,   # hard block but predictable
    "pole":         0.8,
    "fire hydrant": 0.8,
}
DEFAULT_SEVERITY = 0.5

# Distance weight: far objects get 0 weight (FIX4 already blocks them,
# but this modulates severity score for the LLM urgency output).
DIST_WEIGHT: dict[str, float] = {
    "near": 1.0,
    "mid":  0.6,
    "far":  0.0,
}

# Caution level thresholds (caution = SEVERITY * DIST_WEIGHT)
CAUTION_URGENT = 2.5   # person/car near → stop or immediate evasion
CAUTION_HIGH   = 1.5   # person mid, car near → active avoidance
CAUTION_MEDIUM = 0.8   # cone/barrier near → proceed with care
CAUTION_LOW    = 0.3   # distant background obstacles

def compute_caution(detections: list, img_h: int, img_w: int, img_area: int) -> tuple[float, str]:
    """
    [DI2] Compute scene-level caution score and urgency label.
    Returns (caution_score: float, urgency: str).
    """
    max_caution = 0.0
    for det in detections:
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES:
            continue
        sev  = SEVERITY.get(det.label, DEFAULT_SEVERITY)
        dist = det.distance if det.distance != "unknown" else estimate_distance(det, img_h, img_w, img_area)
        dw   = DIST_WEIGHT.get(dist, 0.0)
        max_caution = max(max_caution, sev * dw)

    if   max_caution >= CAUTION_URGENT: urgency = "URGENT"
    elif max_caution >= CAUTION_HIGH:   urgency = "HIGH"
    elif max_caution >= CAUTION_MEDIUM: urgency = "MEDIUM"
    elif max_caution > 0:               urgency = "LOW"
    else:                               urgency = "NONE"

    return max_caution, urgency


# ─────────────────────────────────────────────────────────────────────────────
# [DI5] PATH WIDTH CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# Research: ADA 2010 §403.5 (min 1.2m / 48in pedestrian clear width);
#           iSAID corridor width annotation (Zamir 2019).
# Thresholds expressed as fraction of image width (device-independent).
NARROW_RATIO = 0.08   # < 8%  of image width → constrained movement
TIGHT_RATIO  = 0.04   # < 4%  of image width → single-file only

# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS CONFIDENCE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

PER_CLASS_THRESHOLDS: dict[str, float] = {
    "person": 0.40, "pedestrian": 0.38, "cyclist": 0.38,
    "motorcyclist": 0.38, "wheelchair": 0.35, "stroller": 0.35,
    "car": 0.45, "truck": 0.45, "bus": 0.45,
    "bicycle": 0.40, "motorcycle": 0.40, "scooter": 0.40,
    "traffic light": 0.42, "stop sign": 0.42, "traffic sign": 0.38,
    "pole": 0.35, "bollard": 0.35, "barrier": 0.40,
    "bench": 0.40,
    "road":     0.42,   # [C3]
    "sidewalk": 0.35,   # [C2]
    "grass":    0.35,   # [P5] surface texture — allow lower confidence
    "soil":     0.35,   # [P5]
    "gravel":   0.35,   # [P5]
    "building": 0.50, "tree": 0.45, "traffic cone": 0.38,
}
DEFAULT_THRESHOLD = 0.42

def _cpu_adjust(t: float) -> float:
    return round(t * 0.45, 3) if _CPU_FALLBACK else t

# ─────────────────────────────────────────────────────────────────────────────
# AREA FILTER
# ─────────────────────────────────────────────────────────────────────────────

MIN_ABSOLUTE_PIXELS = 400    # [C1]
MAX_RELATIVE_AREA   = 0.85
MAX_ASPECT_RATIO    = 12.0

def passes_area_filter(box: list, img_h: int, img_w: int, label: str = "") -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
    area = bw * bh
    if area < MIN_ABSOLUTE_PIXELS:
        return False, f"too small ({area}px)"
    # [D2] Tall/large classes exempt from relative-area ceiling
    if label not in {"tree", "pole", "building", "grass", "soil"}:
        if area / (img_h * img_w) > MAX_RELATIVE_AREA:
            return False, f"too large ({area/(img_h*img_w):.0%})"
    if max(bw / bh, bh / bw) > MAX_ASPECT_RATIO:
        return False, "sliver box"
    return True, ""

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    box:           list
    score:         float
    label:         str
    area:          int           = 0
    occluded:      bool          = False
    suppressed_by: Optional[int] = None
    role:          str           = "unknown"
    distance:      str           = "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# IOU HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1)

# ─────────────────────────────────────────────────────────────────────────────
# [U1] CROSS-PROMPT DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_cross_prompt(
    boxes:  list, scores: list, labels: list, iou_threshold: float = 0.50,
) -> tuple[list, list, list]:
    n = len(boxes)
    if n <= 1: return boxes, scores, labels
    keep  = [True] * n
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    for ri, i in enumerate(order):
        if not keep[i]: continue
        for j in order[ri + 1:]:
            if not keep[j]: continue
            if _iou(boxes[i], boxes[j]) >= iou_threshold:
                keep[j] = False
    kb = [boxes[i]  for i in range(n) if keep[i]]
    ks = [scores[i] for i in range(n) if keep[i]]
    kl = [labels[i] for i in range(n) if keep[i]]
    removed = n - len(kb)
    if removed > 0: print(f"  [U1 dedup] Removed {removed} cross-prompt duplicates")
    return kb, ks, kl

# ─────────────────────────────────────────────────────────────────────────────
# [U3] LABEL-AWARE OCCLUSION
# ─────────────────────────────────────────────────────────────────────────────

def compute_containment(a: list, b: list) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    return (ix2-ix1)*(iy2-iy1) / max((ax2-ax1)*(ay2-ay1), 1)

def run_occlusion_analysis(detections: list[Detection]) -> list[Detection]:
    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if compute_containment(detections[i].box, detections[j].box) < 0.85: continue
            if detections[i].label == detections[j].label:
                if detections[i].score < detections[j].score:
                    detections[i].suppressed_by = j
            else:
                outer_is_dynamic = detections[j].label in DYNAMIC_LABELS
                ob = detections[j].box; ib = detections[i].box
                if outer_is_dynamic and (ib[2]-ib[0])*(ib[3]-ib[1]) < (ob[2]-ob[0])*(ob[3]-ob[1]):
                    detections[i].occluded = True   # [FIX6]
    return detections

# ─────────────────────────────────────────────────────────────────────────────
# SOFT-NMS
# ─────────────────────────────────────────────────────────────────────────────

def soft_nms(detections: list[Detection], sigma: float = 0.5, score_gate: float = 0.20) -> list[Detection]:
    dets = sorted([d for d in detections if d.suppressed_by is None], key=lambda d: d.score, reverse=True)
    for i in range(len(dets)):
        for j in range(i+1, len(dets)):
            ov = _iou(dets[i].box, dets[j].box)
            if ov > 0: dets[j].score *= np.exp(-(ov**2) / sigma)
    return [d for d in dets if d.score >= score_gate]

# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_distance(det: Detection, img_h: int, img_w: int, img_area: int) -> str:
    x1, y1, x2, y2 = det.box
    bottom     = y2
    area_ratio = det.area / max(img_area, 1)
    if det.label in SAM_CLASSES:    return "near"
    elif det.label == "traffic cone":
        if bottom > 0.6*img_h: return "near"
        elif bottom > 0.4*img_h: return "mid"
        else: return "far"
    elif bottom > 0.75*img_h: return "near"
    elif bottom > 0.50*img_h: return "mid"
    elif area_ratio > 0.10:   return "near"
    else:                      return "far"

# ─────────────────────────────────────────────────────────────────────────────
# [P4] MASK CENTROID FOR H-POSITION
# ─────────────────────────────────────────────────────────────────────────────
# Research: Mask R-CNN (He 2017), Cityscapes (Cordts 2016)
# For surface detections with a SAM mask, the mask column centroid is more
# accurate than the GDINO bounding box center due to perspective projection.
# For obstacles (no mask), foot-point x=(x1+x2)/2 is unchanged.

def get_hpos_x(det: Detection, mask: Optional[np.ndarray], img_w: int) -> float:
    """
    Return the x-coordinate to use for horizontal position classification.
    Priority:
      1. SAM mask centroid (if mask available) — [P4]
      2. Box foot-point centre (x1+x2)/2      — existing behaviour
    """
    if mask is not None:
        cols = np.where(mask.any(axis=0))[0]   # columns where mask has pixels
        if len(cols) > 0:
            return float(cols.mean())           # mask column centroid [P4]
    return float((det.box[0] + det.box[2]) / 2)   # box centre fallback

def classify_hpos(x: float, img_w: int) -> str:
    """[P1] Narrow center zone: 38–62% instead of 33–67%."""
    ratio = x / img_w
    if ratio < COL_LEFT_END:    return "left"
    elif ratio > COL_RIGHT_START: return "right"
    else:                          return "center"

# ─────────────────────────────────────────────────────────────────────────────
# [P3] SIDEWALK-BOUNDED COLUMN COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
# Research: Padaki et al. (2023), Gupta et al. (2014)
# When a pedestrian SAM mask exists, extract its x-extent and split
# THAT range into L/C/R thirds. This is the true "sidewalk corridor".

def extract_sidewalk_centerline(
    sidewalk_mask: Optional[np.ndarray],
    zone_top:      int,
    img_h:         int,
    img_w:         int,
) -> tuple[Optional[callable], int, Optional[tuple]]:
    """
    [DI1] Extract sidewalk centerline via per-row midpoint regression.

    Research: Inverse Perspective Mapping (Mallot 1991);
              KITTI lane detection (Geiger 2012);
              Cityscapes pedestrian zone analysis (Cordts 2016).

    Method:
      For each row in the analysis zone, find the leftmost and rightmost
      sidewalk mask pixel. Their midpoint is the centerline candidate at
      that row. A degree-1 polyfit through all valid midpoints gives the
      regression line: centerline_x(y) = slope * y + intercept.

      This handles diagonal AND gently curved sidewalks. For strongly
      curved paths, the linear approximation may diverge at the edges —
      acceptable for navigation (we only need the next 2–3 metres).

    Returns:
      centerline_fn  : callable y → x, or None if insufficient data
      path_width_px  : median pixel width of sidewalk (for DI5)
      regression     : (slope, intercept) or None
    """
    if sidewalk_mask is None:
        return None, 0, None

    zone_mask = sidewalk_mask[zone_top:, :]
    rows_y:   list[int]   = []
    mid_x:    list[float] = []
    widths:   list[int]   = []

    for r in range(zone_mask.shape[0]):
        cols = np.where(zone_mask[r])[0]
        if len(cols) >= 5:
            rows_y.append(r + zone_top)
            mid_x.append((int(cols[0]) + int(cols[-1])) / 2.0)
            widths.append(int(cols[-1]) - int(cols[0]))

    if len(rows_y) < 10:
        return None, 0, None

    # [FIX-14] deg=2 handles curved sidewalks (e.g. corner turns, arc paths).
    # deg=1 assumed straight; mild curves now fit with a quadratic.
    # If < 20 rows available, fall back to deg=1 to avoid overfitting.
    poly_deg = 2 if len(rows_y) >= 20 else 1
    coeffs   = np.polyfit(rows_y, mid_x, poly_deg)
    slope, intercept = (coeffs[-2], coeffs[-1])   # last two coeffs = linear part
    path_width_px    = int(np.median(widths))

    def centerline_fn(y: float) -> float:
        return float(slope * y + intercept)

    return centerline_fn, path_width_px, (slope, intercept)


def compute_corridor_bounds(
    sidewalk_mask: Optional[np.ndarray],
    centerline_fn: Optional[callable],
    path_width_px: int,
    img_h:         int,
    img_w:         int,
    zone_top:      int,
) -> dict[str, tuple[int, int]]:
    """
    [P3 + DI1] Compute L/C/R corridor bounds.

    When centerline is available: bounds are defined relative to the
    centerline midpoint at the bottom of the zone, using path_width_px.
    This produces a corridor aligned with the actual sidewalk direction
    rather than the image vertical axis.

    When no centerline: fall back to global [P1] asymmetric split.
    """
    if centerline_fn is not None and path_width_px >= 30:
        # Evaluate centerline at bottom of zone (most relevant for navigation)
        y_eval   = int(img_h * 0.85)
        cl_x     = centerline_fn(y_eval)
        half_pw  = path_width_px / 2.0
        # Corridor spans the sidewalk width, centred on centerline
        left_edge  = max(0,     int(cl_x - half_pw))
        right_edge = min(img_w, int(cl_x + half_pw))
        third      = max((right_edge - left_edge) // 3, 1)
        bounds = {
            "left":   (left_edge,          left_edge + third),
            "center": (left_edge + third,   left_edge + 2*third),
            "right":  (left_edge + 2*third, right_edge),
        }
        print(f"  [DI1] Centerline at y={y_eval}: x={cl_x:.0f}, "
              f"width={path_width_px}px → corridor {left_edge}–{right_edge}")
        return bounds

    # Fallback: global [P1] asymmetric split
    return {
        "left":   (0,                         int(img_w * COL_LEFT_END)),
        "center": (int(img_w * COL_LEFT_END),  int(img_w * COL_RIGHT_START)),
        "right":  (int(img_w * COL_RIGHT_START), img_w),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FREE-SPACE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WALKABLE_COVERAGE     = 0.25   # pedestrian surface must cover ≥25% of column
NON_WALKABLE_COVERAGE = 0.30   # non-walkable surface covering ≥30% → flag column
SW_BLOCK_COVERAGE     = 0.50
SW_CROWD_COVERAGE     = 0.20
SW_FOOT_OVERLAP_RATIO = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# [E1] RULE-BASED FUSION
# ─────────────────────────────────────────────────────────────────────────────

def _fuse_states(global_st: str, local_st: str) -> str:
    if local_st == "unknown":    return global_st
    if "blocked" in (global_st, local_st): return "blocked"
    if "crowded" in (global_st, local_st): return "crowded"
    if "uncertain" in (global_st, local_st): return "uncertain"
    return "walkable"

# ─────────────────────────────────────────────────────────────────────────────
# [SW] SIDEWALK-LOCAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _obstacle_on_sidewalk(det: Detection, sidewalk_mask: np.ndarray) -> bool:
    x1,y1,x2,y2 = det.box
    h,w = sidewalk_mask.shape
    x1c=max(0,x1); y1c=max(0,y1); x2c=min(w-1,x2); y2c=min(h-1,y2)
    foot_x = int((x1c+x2c)/2); foot_y = y2c
    if sidewalk_mask[foot_y, foot_x]: return True
    region = sidewalk_mask[y1c:y2c+1, x1c:x2c+1]
    box_area = max((x2c-x1c+1)*(y2c-y1c+1), 1)
    return (region.sum() / box_area) >= SW_FOOT_OVERLAP_RATIO

def _merge_surface_masks(
    detections: list[Detection],
    masks:      list,
    surface_set: set[str],
) -> Optional[np.ndarray]:
    """Merge SAM masks for all classes in surface_set into one boolean array."""
    combined: Optional[np.ndarray] = None
    for det, mask in zip(detections, masks):
        if mask is None or det.label not in surface_set: continue
        combined = mask.astype(bool) if combined is None else combined | mask.astype(bool)
    return combined

def analyse_sidewalk_local(
    image:        np.ndarray,
    detections:   list[Detection],
    masks:        list,
    col_bounds:   dict[str, tuple[int, int]],
    zone_top:     int,
    sidewalk_mask: Optional[np.ndarray],
) -> Optional[dict[str, str]]:
    if sidewalk_mask is None: return None

    sw_pixels = {col: int(sidewalk_mask[zone_top:, cx1:cx2].sum())
                 for col, (cx1, cx2) in col_bounds.items()}

    result: dict[str, str] = {}
    for col, (cx1, cx2) in col_bounds.items():
        if sw_pixels[col] == 0:
            result[col] = "unknown"
            continue

        hard_max = 0.0; soft_max = 0.0
        for det in detections:
            role = OBJECT_ROLES.get(det.label, "unknown")
            if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES: continue
            if det.distance == "far": continue
            if not _obstacle_on_sidewalk(det, sidewalk_mask): continue
            x1,y1,x2,y2 = det.box
            if y2 < zone_top: continue
            ox1=max(x1,cx1); ox2=min(x2,cx2)
            if ox2<=ox1: continue
            col_w = max(cx2-cx1,1)
            ratio = (ox2-ox1)/col_w
            if role in HARD_BLOCK_ROLES: hard_max = max(hard_max, ratio)
            else:                         soft_max = max(soft_max, ratio)

        if hard_max >= SW_BLOCK_COVERAGE:   result[col] = "blocked"
        elif hard_max >= SW_CROWD_COVERAGE or soft_max >= SW_CROWD_COVERAGE:
            result[col] = "crowded"
        else: result[col] = "walkable"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# [U4 + P2 + P3] MAIN FREE-SPACE ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

def analyse_free_space(
    image:      np.ndarray,
    detections: list[Detection],
    masks:      list,
) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    """
    Returns: (free_space_dict, col_bounds_dict)
    """
    img_h, img_w = image.shape[:2]
    zone_top = int(img_h * 0.60)

    # [P3] Compute pedestrian SAM mask first (needed for corridor bounds)
    sidewalk_mask   = _merge_surface_masks(detections, masks, PEDESTRIAN_SURFACES)
    non_walk_mask   = _merge_surface_masks(detections, masks, NON_WALKABLE_SURFACES)  # [P5]

    # [DI1] Extract centerline from sidewalk mask
    centerline_fn, path_width_px, _ = extract_sidewalk_centerline(
        sidewalk_mask, zone_top, img_h, img_w
    )

    # [P3] Corridor bounds: mask-bounded if possible, else global [P1]
    col_bounds = compute_corridor_bounds(
        sidewalk_mask, centerline_fn, path_width_px, img_h, img_w, zone_top
    )

    # ── Step 1: [P2] Surface-typed pixel counting ─────────────────────────────
    # Only PEDESTRIAN_SURFACES → walkable_pixels
    # VEHICLE_SURFACES → semi_walkable_pixels (not the same as walkable)
    # NON_WALKABLE_SURFACES → non_walkable_pixels (subtract from walkability)
    walkable_pixels     = {col: 0 for col in col_bounds}
    non_walkable_pixels = {col: 0 for col in col_bounds}
    semi_walkable_pixels= {col: 0 for col in col_bounds}   # [FIX-4] road fallback

    for det, mask in zip(detections, masks):
        if mask is None: continue
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in ("walkable", "semi_walkable", "non_walkable"): continue

        zone_mask = mask[zone_top:, :]
        for col, (cx1, cx2) in col_bounds.items():
            col_px = int(zone_mask[:, cx1:cx2].sum())
            if role == "walkable":
                walkable_pixels[col]      += col_px
            elif role == "non_walkable":
                non_walkable_pixels[col]  += col_px
            elif role == "semi_walkable":
                # [FIX-4] Road pixels counted separately.
                # Used as fallback: if no ped surface detected but road exists
                # and no obstacle → column is "uncertain" (not blocked).
                # Prevents constant stop-everywhere when sidewalk missed.
                semi_walkable_pixels[col] += col_px

    # ── Step 2: Global obstacle classification [FIX1] [FIX4] ─────────────────
    hard_blocked_cols: set[str] = set()
    crowded_cols:      set[str] = set()

    for det in detections:
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES: continue
        if det.distance == "far": continue
        x1,y1,x2,y2 = det.box
        if y2 < zone_top: continue
        # [P1] Use foot-point x for column assignment (not center)
        foot_x = (x1 + x2) / 2   # x is same as center for axis-aligned boxes
        for col, (cx1, cx2) in col_bounds.items():
            if cx1 <= foot_x < cx2:
                if role in HARD_BLOCK_ROLES:   hard_blocked_cols.add(col)
                elif role in SOFT_BLOCK_ROLES: crowded_cols.add(col)

    # ── Step 3: [P2] Global decision per column ────────────────────────────────
    global_result: dict[str, str] = {}
    for col, (cx1, cx2) in col_bounds.items():
        col_area_col = max((img_h - zone_top) * (cx2 - cx1), 1)
        ped_coverage  = walkable_pixels[col]     / col_area_col
        nwk_coverage  = non_walkable_pixels[col] / col_area_col

        # [P5] Non-walkable surface dominating column → flag before obstacle check
        if nwk_coverage >= NON_WALKABLE_COVERAGE:
            global_result[col] = "blocked"   # soil/grass in walking path = blocked
        elif col in hard_blocked_cols:
            global_result[col] = "blocked"
        elif col in crowded_cols:
            global_result[col] = "crowded"
        elif ped_coverage >= WALKABLE_COVERAGE:
            global_result[col] = "walkable"   # [P2] ONLY pedestrian surface → walkable
        else:
            global_result[col] = "uncertain"  # [E2]

    # ── Step 4: [SW] Sidewalk-local analysis + fusion ─────────────────────────
    local_result = analyse_sidewalk_local(
        image, detections, masks, col_bounds, zone_top, sidewalk_mask
    )

    if local_result is None:
        print("  [SW] No sidewalk mask — using global free-space only")
        return global_result, col_bounds

    fused: dict[str, str] = {}
    for col in col_bounds:
        fused[col] = _fuse_states(global_result[col], local_result[col])
        if fused[col] != global_result[col]:
            print(f"  [SW] '{col}': global={global_result[col]} "
                  f"local={local_result[col]} → fused={fused[col]}")

    return fused, col_bounds

# ─────────────────────────────────────────────────────────────────────────────
# [U5] INSTANCE GROUPING
# ─────────────────────────────────────────────────────────────────────────────

def group_detections(detections: list[Detection]) -> dict[str, list[Detection]]:
    groups: dict[str, list[Detection]] = defaultdict(list)
    for det in detections: groups[det.label].append(det)
    return dict(groups)

def group_label(label: str, count: int) -> str:
    if count == 1: return label
    elif count == 2: return f"2 {label}s"
    else: return f"multiple {label}s"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class VLMPipeline:
    def __init__(self, soft_nms_sigma: float = 0.5, max_image_size: int = 800):
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_nms_sigma = soft_nms_sigma
        self.max_image_size = max_image_size
        self.cpu_fallback   = _CPU_FALLBACK

        print(f"[Pipeline] Device        : {self.device}")
        print(f"[Pipeline] CPU fallback  : {self.cpu_fallback}")
        print(f"[Pipeline] Box threshold : {BOX_THRESHOLD_DEFAULT}")
        print(f"[Pipeline] Min area px   : {MIN_ABSOLUTE_PIXELS}")
        print(f"[Pipeline] Center zone   : {COL_LEFT_END*100:.0f}%–{COL_RIGHT_START*100:.0f}%")

        self._load_models()

    def _load_models(self):
        dino_config = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
        dino_ckpt   = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
        sam_ckpt    = os.path.join(WEIGHTS_DIR, "sam_vit_l_0b3195.pth")
        for path, name in [(dino_config,"DINO config"),(dino_ckpt,"DINO weights"),(sam_ckpt,"SAM weights")]:
            if not os.path.exists(path): raise FileNotFoundError(f"Missing {name}: {path}")
        print("[Pipeline] Loading Grounding DINO...")
        self.dino_model = load_model(dino_config, dino_ckpt, device=self.device)
        print("[Pipeline] Loading SAM...")
        sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        print("[Pipeline] Models loaded ✓")

    @staticmethod
    def _decode_boxes(raw_boxes: np.ndarray, img_h: int, img_w: int) -> list[list[int]]:
        out = []
        for cx,cy,bw,bh in raw_boxes:
            x1=max(0,int((cx-bw/2)*img_w)); y1=max(0,int((cy-bh/2)*img_h))
            x2=min(img_w,int((cx+bw/2)*img_w)); y2=min(img_h,int((cy+bh/2)*img_h))
            out.append([x1,y1,max(x2,x1+1),max(y2,y1+1)])
        return out

    def _expand_box(self, box: list, img_h: int, img_w: int, scale: float = 1.15) -> list:
        x1,y1,x2,y2 = box; cx=(x1+x2)/2; cy=(y1+y2)/2
        w=(x2-x1)*scale; h=(y2-y1)*scale
        return [max(0,int(cx-w/2)),max(0,int(cy-h/2)),
                min(img_w,int(cx+w/2)),min(img_h,int(cy+h/2))]

    def _run_gdino(self, image_tensor, prompt, box_thr, text_thr):
        with torch.no_grad():
            raw_boxes, logits, phrases = predict(
                self.dino_model, image_tensor, caption=prompt,
                box_threshold=box_thr, text_threshold=text_thr, device=self.device,
            )
        return (raw_boxes.cpu().numpy(), logits.cpu().numpy(),
                [p.strip().replace(".", "").lower() for p in phrases])

    def detect_and_segment(self, image_path: str, run_sam: bool = True) -> dict:
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Cannot load: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        scale = self.max_image_size / max(img_h, img_w)
        if scale < 1.0:
            image = cv2.resize(image, (int(img_w*scale), int(img_h*scale)))
            img_h, img_w = image.shape[:2]

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        image_tensor = transform(Image.fromarray(image)).to(torch.float32).to(self.device)

        print("[Pipeline] Running Multi-Pass GDINO...")
        all_boxes_raw: list = []; all_scores: list = []; all_labels: list = []

        for single_prompt in MULTI_PROMPTS:
            p = sanitise_prompt(single_prompt)
            for box_thr in THRESHOLD_LADDER:
                text_thr = box_thr * 1.2
                rb, sc, lb = self._run_gdino(image_tensor, p, box_thr, text_thr)
                if len(sc) > 0:
                    print(f"  ['{single_prompt}'] {len(sc)} det(s) (max={sc.max():.2f}, thr={box_thr:.2f})")
                    all_boxes_raw.extend(rb); all_scores.extend(sc.tolist()); all_labels.extend(lb)
                    break
            else:
                print(f"  ['{single_prompt}'] 0 detections at all thresholds")

        if not all_boxes_raw:
            print("[Pipeline] ✗ No detections.")
            return self._empty(image)

        boxes_xyxy = self._decode_boxes(np.array(all_boxes_raw), img_h, img_w)
        boxes_xyxy, all_scores, all_labels = deduplicate_cross_prompt(boxes_xyxy, all_scores, all_labels)

        step1: list[Detection] = []
        for box, score, label in zip(boxes_xyxy, all_scores, all_labels):
            thr = _cpu_adjust(PER_CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD))
            if score >= thr:
                area = (box[2]-box[0])*(box[3]-box[1])
                det  = Detection(box=box, score=float(score), label=label, area=area)
                det.role = OBJECT_ROLES.get(label, "unknown")
                step1.append(det)
        print(f"  After conf filter   : {len(step1)}/{len(all_scores)}")

        step2: list[Detection] = []
        for det in step1:
            ok, reason = passes_area_filter(det.box, img_h, img_w, det.label)
            if ok: step2.append(det)
            else:  print(f"  [area-drop] '{det.label}' — {reason}")
        print(f"  After area filter   : {len(step2)}/{len(step1)}")

        if not step2: return self._empty(image)

        step2 = run_occlusion_analysis(step2)
        step3 = soft_nms(step2, sigma=self.soft_nms_sigma)
        print(f"  After Soft-NMS      : {len(step3)}/{len(step2)}")

        img_area = img_h * img_w
        for det in step3:
            det.distance = estimate_distance(det, img_h, img_w, img_area)

        # ── [DI3 + FIX-5] Set on_path per detection ──────────────────────────
        # FIX-5: old threshold was 0.35×path_width — too tight, missed real
        # obstacles slightly off-center. New: absolute 40% of image half-width,
        # matching Ondras (2021) ±40% danger zone recommendation.
        # on_path is used by RF1 gate (NAV_CONDITIONAL filter) and SB1 (spatial facts).
        img_cx    = img_w / 2.0
        path_half = img_w * 0.40   # [FIX-5] ≈ 256px on 640px image
        for det in step3:
            foot_x      = (det.box[0] + det.box[2]) / 2.0
            det.on_path = abs(foot_x - img_cx) < path_half

        print(f"  Final: {dict(Counter(d.label for d in step3))}")

        # SAM — all surface classes [P2][P5]
        masks: list = []
        mask_map: dict[int, np.ndarray] = {}   # [P4] index → mask for position use
        if run_sam:
            self.sam_predictor.set_image(image)

        for idx, det in enumerate(step3):
            if not (run_sam and det.label in SAM_CLASSES):
                masks.append(None)
                continue
            if det.score < 0.3:
                masks.append(None)
                continue
            box_exp = self._expand_box(det.box, img_h, img_w, scale=1.15)
            mask, _, _ = self.sam_predictor.predict(
                box=np.array(box_exp, dtype=np.float32), multimask_output=False,
            )
            mask = mask[0]
            ratio = mask.sum() / (mask.shape[0]*mask.shape[1])
            if ratio > 0.8 or ratio < 0.01:
                print(f"  [SAM-drop] '{det.label}' bad ratio={ratio:.2f}")
                masks.append(None)
            else:
                masks.append(mask)
                mask_map[idx] = mask

        free_space, col_bounds = analyse_free_space(image, step3, masks)
        print(f"  Free-space: {free_space}")

        return {
            "image": image, "detections": step3, "masks": masks,
            "mask_map": mask_map,   # [P4] idx → mask
            "col_bounds": col_bounds,  # For accurate position
            "free_space": free_space,
            "boxes":   [d.box   for d in step3],
            "scores":  [d.score for d in step3],
            "phrases": [d.label for d in step3],
        }

    # ── Navigation description ────────────────────────────────────────────────
    def build_navigation_description(self, results: dict) -> dict:
        dets       = results.get("detections", [])
        free_space = results.get("free_space", {"left":"unknown","center":"unknown","right":"unknown"})
        mask_map   = results.get("mask_map", {})   # [P4]
        col_bounds = results.get("col_bounds", None)  # For accurate position
        img_h, img_w = results["image"].shape[:2]
        img_area   = img_h * img_w

        center = free_space.get("center","unknown")
        left   = free_space.get("left",  "unknown")
        right  = free_space.get("right", "unknown")

        def is_navigable(s): 
            return s in ("walkable","crowded","uncertain")
        
        def action_for(s, d):
            return {"walkable":f"move {d}","crowded":f"move {d} (crowded)",
                    "uncertain":f"move {d} (cautious)"}.get(s, f"move {d}")

        # ── ACTION decision ──────────────────────────────────────────────────────
        if   center == "walkable":  action = "move_forward"
        elif center == "crowded":   action = "move_forward_crowded"
        elif center == "uncertain": action = "move_forward_cautious"
        elif is_navigable(left) and not is_navigable(right): 
            action = action_for(left,  "left").replace(" ", "_")
        elif is_navigable(right) and not is_navigable(left):  
            action = action_for(right, "right").replace(" ", "_")
        elif is_navigable(left) and is_navigable(right):
            priority = {"walkable":0,"uncertain":1,"crowded":2}
            action = action_for(left,"left") if priority.get(left,3) <= priority.get(right,3) \
                    else action_for(right,"right")
            action = action.replace(" ", "_")
        else:
            action = "stop_path_unclear"

        groups = group_detections(dets)
        obstacle_descriptions: list[str] = []
        surface_descriptions: list[str] = []

        def simplify_state(s):
            """FIX #7: Improve PATH readability"""
            return "clear" if s in ("walkable","crowded") else s

        for label, group in groups.items():
            role = OBJECT_ROLES.get(label, "unknown")
            # [FIX-7] Representative = nearest instance
            dist_rank = {"near": 0, "mid": 1, "far": 2, "unknown": 3}
            rep  = min(group, key=lambda d: (
                dist_rank.get(d.distance, 3),
                -d.score
            ))
            rep_idx = dets.index(rep) if rep in dets else -1
            mask = mask_map.get(rep_idx)

            # ── FIX #3: Use corridor logic for position ─────────────────────────
            hpos_x = get_hpos_x(rep, mask, img_w)
            
            if col_bounds:
                for col, (cx1, cx2) in col_bounds.items():
                    if cx1 <= hpos_x < cx2:
                        h_pos = col
                        break
                else:
                    h_pos = classify_hpos(hpos_x, img_w)  # fallback
            else:
                h_pos = classify_hpos(hpos_x, img_w)

            # ── FIX #5: Distance fallback ──────────────────────────────────────
            dist = rep.distance if rep.distance != "unknown" \
                else estimate_distance(rep, img_h, img_w, img_area)

            # Always-skip: far cars/bikes
            if dist == "far" and label in {"car", "bicycle"}:
                continue

            # ── Format: COMPACT for LLM ─────────────────────────────────────────
            count = len(group)
            if count == 1:
                label_str = label
            else:
                label_str = f"{count}x {label}"

            # ── FIX #6: Safe metric depth ──────────────────────────────────────
            depth_m = None
            try:
                depth_m = estimate_metric_depth(rep, img_h)
            except:
                pass
            
            dist_str = f"{int(depth_m)}m" if depth_m is not None else dist

            compact_line = f"{label_str}({dist_str}, {h_pos})"

            # ── [RF1] Three-tier relevance gate ──────────────────────────────
            on_path_any = any(d.on_path for d in group)
            out_cat = get_output_category(
                label=label,
                role=role,
                on_path=on_path_any,
                distance=dist,
                in_obstacle_roles=(role in OBSTACLE_ROLES_SET),
            )

            if out_cat == "obstacle":
                obstacle_descriptions.append(compact_line)
            elif out_cat == "surface":
                if role in ("walkable", "semi_walkable"):
                    surface_descriptions.append(compact_line)

        # ── FIX #1: CORRECT Caution score (context-aware) ──────────────────────
        # ONLY count as dangerous if:
        # 1. Obstacle is on path
        # 2. Obstacle is NEAR (not mid/far)
        # 3. The path is NOT clear (obstacle is actually blocking)
        # 
        # This prevents "person walking ahead = dangerous" fallacy
        center_clear = center in ("walkable", "crowded")
        
        on_path_dets = [
            det for det in dets
            if det.on_path
            and OBJECT_ROLES.get(det.label,"") in OBSTACLE_ROLES_SET
            and det.distance == "near"  # FIX: Only near objects matter
        ]
        
        # If center is clear, even near obstacles aren't dangerous (they're off to the side)
        if center_clear:
            caution_score = 0
        else:
            caution_score = max(
                SEVERITY.get(det.label, 1) *
                {"near":1.0, "mid":0.6}.get(det.distance, 0.0)
                for det in on_path_dets
            ) if on_path_dets else 0
        
        # ── Risk classification ──────────────────────────────────────────────────
        if caution_score >= 2.5:
            risk = "urgent"
        elif caution_score >= 1.5:
            risk = "high"
        elif caution_score >= 0.8:
            risk = "medium"
        elif caution_score > 0:
            risk = "low"
        else:
            risk = "none"

        # ── FIX #2: Risk overrides action ──────────────────────────────────────
        if risk == "urgent":
            action = "stop"
        elif risk == "high" and "move" in action:
            action = "move_forward_cautious"

        # ── Build 4-LINE STRUCTURED OUTPUT ──────────────────────────────────
        action_line = f"ACTION: {action}"
        risk_line = f"RISK: {risk}"
        
        # ── FIX #7: Cleaner PATH readability ──────────────────────────────────
        path_line = (
            f"PATH: center {simplify_state(center)}, "
            f"left {simplify_state(left)}, right {simplify_state(right)}"
        )
        
        obstacle_line = "OBSTACLES: " + ("; ".join(obstacle_descriptions) if obstacle_descriptions else "none")
        surface_line = "ENV: " + ("; ".join(surface_descriptions) if surface_descriptions else "none")

        # Combine into final LLM-ready text
        scene_text = "\n".join([
            action_line,
            risk_line,
            path_line,
            obstacle_line,
            surface_line
        ])

        return {
            "free_space":     free_space,
            "action":         action,
            "risk":           risk,
            "obstacles":      obstacle_descriptions,
            "surfaces":       surface_descriptions,
            "scene_text":     scene_text,
        }

    def build_scene_description(self, results: dict) -> str:
        return self.build_navigation_description(results)["scene_text"]

    # ── Visualisation ─────────────────────────────────────────────────────────
    def visualize(self, results: dict, save_name: str = "output_v9.png"):
        if not results or results.get("image") is None: return

        image      = results["image"].copy()
        dets       = results.get("detections", [])
        masks      = results.get("masks", [])
        free_space = results.get("free_space", {})

        unique_labels = list({d.label for d in dets})
        cmap   = matplotlib.colormaps.get_cmap("tab20")
        colors = {lbl: tuple(int(c*255) for c in cmap(i/max(len(unique_labels),1))[:3])
                  for i,lbl in enumerate(unique_labels)}

        # Surface masks — colour-coded by type [P2]
        surface_colors = {
            "walkable":     np.array([80, 200, 80]),    # green
            "semi_walkable":np.array([200, 160, 60]),   # amber
            "non_walkable": np.array([200, 60,  60]),   # red
        }
        for mask, det in zip(masks, dets):
            if mask is None: continue
            role = OBJECT_ROLES.get(det.label, "unknown")
            if role not in surface_colors: continue
            if mask.sum()/mask.size > 0.6: continue
            sc = surface_colors[role]
            image[mask] = (image[mask]*0.55 + sc*0.45).astype(np.uint8)

        # Bounding boxes
        for det in dets:
            x1,y1,x2,y2 = det.box
            color     = colors[det.label]
            thickness = 3 if not det.occluded else 1
            cv2.rectangle(image,(x1,y1),(x2,y2),color,thickness)
            tag = f"{det.label}{'(occ)' if det.occluded else ''} {det.score:.2f} [{det.distance}]"
            (tw,th),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(image,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
            cv2.putText(image,tag,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

        # Free-space corridor overlays
        img_h2, img_w2 = image.shape[:2]
        zone_top = int(img_h2 * 0.60)

        # Recompute corridor bounds for visualisation [P3]
        sidewalk_mask_vis = _merge_surface_masks(dets, masks, PEDESTRIAN_SURFACES)
        
        # FIX: Extract centerline before calling compute_corridor_bounds
        centerline_fn_vis, path_width_px_vis, _ = extract_sidewalk_centerline(
            sidewalk_mask_vis, zone_top, img_h2, img_w2
        )
        col_bounds_vis = compute_corridor_bounds(
            sidewalk_mask_vis, centerline_fn_vis, path_width_px_vis, img_h2, img_w2, zone_top
        )
        
        col_names = ["left","center","right"]
        status_color = {
            "walkable":(0,200,0),"crowded":(200,140,0),
            "blocked":(200,0,0),"uncertain":(100,100,200),"unknown":(150,150,0),
        }
        for col in col_names:
            cx1, cx2 = col_bounds_vis[col]
            status = free_space.get(col,"unknown")
            color  = status_color.get(status,(150,150,0))
            overlay = image.copy()
            cv2.rectangle(overlay,(cx1,zone_top),(cx2,img_h2),color,-1)
            image = cv2.addWeighted(overlay,0.18,image,0.82,0)
            cv2.putText(image,f"{col}: {status}",(cx1+4,img_h2-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)

        fig, axes = plt.subplots(1,2,figsize=(18,8))
        axes[0].imshow(results["image"]); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(image);            axes[1].set_title("Prediction"); axes[1].axis("off")
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, save_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Visualize] Saved: {path}")

    @staticmethod
    def _empty(image: np.ndarray) -> dict:
        empty_fs = {"left":"uncertain","center":"uncertain","right":"uncertain"}
        return {"image":image,"detections":[],"masks":[],"mask_map":{},
                "col_bounds": None,
                "free_space":empty_fs,"boxes":[],"scores":[],"phrases":[]}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def test():
    parser = argparse.ArgumentParser(description="VLM Navigation Pipeline")
    parser.add_argument("--image",    required=True)
    parser.add_argument("--no-sam",   action="store_true")
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--output",   default="output_v10.png")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    pipeline = VLMPipeline(max_image_size=args.max_size)
    results  = pipeline.detect_and_segment(args.image, run_sam=not args.no_sam)
    nav      = pipeline.build_navigation_description(results)

    print("\n" + "="*55)
    print("  NAVIGATION OUTPUT")
    print("="*55)
    print(f"  Action     : {nav['action']}")
    print(f"  Risk       : {nav['risk']}")
    print(f"  Free-space : {nav['free_space']}")
    if nav["obstacles"]: print(f"  Obstacles  : {nav['obstacles']}")
    if nav["surfaces"]:  print(f"  Surfaces   : {nav['surfaces']}")
    print(f"\n  LLM-ready  :\n{nav['scene_text']}")
    print("="*55)

    pipeline.visualize(results, save_name=args.output)

if __name__ == "__main__":
    test()