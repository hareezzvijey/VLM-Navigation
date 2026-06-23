"""
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

def compute_corridor_bounds(
    sidewalk_mask: Optional[np.ndarray],
    img_h: int,
    img_w: int,
    zone_top: int,
) -> dict[str, tuple[int, int]]:
    """
    [P3] If sidewalk mask available, split its x-extent into L/C/R.
    Otherwise fall back to global [P1] split.
    """
    if sidewalk_mask is not None:
        # Consider only the analysis zone (bottom 40%)
        zone_mask = sidewalk_mask[zone_top:, :]
        cols_with_mask = np.where(zone_mask.any(axis=0))[0]
        if len(cols_with_mask) >= 30:   # need enough pixels to be meaningful
            x_min = int(cols_with_mask.min())
            x_max = int(cols_with_mask.max())
            width = x_max - x_min
            if width >= 60:             # corridor must be at least 60px wide
                third = width // 3
                bounds = {
                    "left":   (x_min,          x_min + third),
                    "center": (x_min + third,   x_min + 2*third),
                    "right":  (x_min + 2*third, x_max),
                }
                print(f"  [P3] Sidewalk corridor: x={x_min}–{x_max} "
                      f"→ L={bounds['left']}, C={bounds['center']}, R={bounds['right']}")
                return bounds

    # Fallback: global [P1] asymmetric split
    return {
        "left":   (0,                    int(img_w * COL_LEFT_END)),
        "center": (int(img_w * COL_LEFT_END), int(img_w * COL_RIGHT_START)),
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
) -> dict[str, str]:
    img_h, img_w = image.shape[:2]
    zone_top = int(img_h * 0.60)

    # [P3] Compute pedestrian SAM mask first (needed for corridor bounds)
    sidewalk_mask   = _merge_surface_masks(detections, masks, PEDESTRIAN_SURFACES)
    non_walk_mask   = _merge_surface_masks(detections, masks, NON_WALKABLE_SURFACES)  # [P5]

    # [P3] Corridor bounds: mask-bounded if possible, else global [P1]
    col_bounds = compute_corridor_bounds(sidewalk_mask, img_h, img_w, zone_top)
    col_area   = (img_h - zone_top) * max(
        max(cx2-cx1 for cx1,cx2 in col_bounds.values()), 1
    )

    # ── Step 1: [P2] Surface-typed pixel counting ─────────────────────────────
    # Only PEDESTRIAN_SURFACES → walkable_pixels
    # VEHICLE_SURFACES → semi_walkable_pixels (not the same as walkable)
    # NON_WALKABLE_SURFACES → non_walkable_pixels (subtract from walkability)
    walkable_pixels    = {col: 0 for col in col_bounds}
    non_walkable_pixels= {col: 0 for col in col_bounds}

    for det, mask in zip(detections, masks):
        if mask is None: continue
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in ("walkable", "semi_walkable", "non_walkable"): continue

        zone_mask = mask[zone_top:, :]
        for col, (cx1, cx2) in col_bounds.items():
            col_px = int(zone_mask[:, cx1:cx2].sum())
            if role == "walkable":      walkable_pixels[col]     += col_px
            elif role == "non_walkable": non_walkable_pixels[col] += col_px
            # semi_walkable (road): not counted in either — it's a separate signal

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
        return global_result

    fused: dict[str, str] = {}
    for col in col_bounds:
        fused[col] = _fuse_states(global_result[col], local_result[col])
        if fused[col] != global_result[col]:
            print(f"  [SW] '{col}': global={global_result[col]} "
                  f"local={local_result[col]} → fused={fused[col]}")

    return fused

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

        free_space = analyse_free_space(image, step3, masks)
        print(f"  Free-space: {free_space}")

        return {
            "image": image, "detections": step3, "masks": masks,
            "mask_map": mask_map,   # [P4] idx → mask
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
        img_h, img_w = results["image"].shape[:2]
        img_area   = img_h * img_w

        center = free_space.get("center","unknown")
        left   = free_space.get("left",  "unknown")
        right  = free_space.get("right", "unknown")

        def is_navigable(s): return s in ("walkable","crowded","uncertain")
        def action_for(s, d):
            return {"walkable":f"move {d}","crowded":f"move {d} (crowded)",
                    "uncertain":f"move {d} (cautious)"}.get(s, f"move {d}")

        if   center == "walkable":  action = "move forward"
        elif center == "crowded":   action = "move forward (crowded)"
        elif center == "uncertain": action = "move forward (cautious)"
        elif is_navigable(left) and not is_navigable(right): action = action_for(left,  "left")
        elif is_navigable(right)and not is_navigable(left):  action = action_for(right, "right")
        elif is_navigable(left) and is_navigable(right):
            priority = {"walkable":0,"uncertain":1,"crowded":2}
            action = action_for(left,"left") if priority.get(left,3)<=priority.get(right,3) \
                     else action_for(right,"right")
        else:
            action = "stop — path unclear"

        groups = group_detections(dets)
        obstacle_lines: list[str] = []
        surface_lines:  list[str] = []

        for label, group in groups.items():
            role = OBJECT_ROLES.get(label, "unknown")
            rep  = max(group, key=lambda d: d.score)
            rep_idx = dets.index(rep) if rep in dets else -1
            mask = mask_map.get(rep_idx)   # [P4] get mask if available

            # [P4] Use mask centroid for H-position; fallback to box center
            hpos_x = get_hpos_x(rep, mask, img_w)
            h_pos  = classify_hpos(hpos_x, img_w)   # [P1] narrow center

            dist = rep.distance if rep.distance != "unknown" \
                   else estimate_distance(rep, img_h, img_w, img_area)

            if dist == "far" and label in {"car","bicycle"}: continue

            gl   = group_label(label, len(group))
            occ  = " [occluded]" if rep.occluded else ""
            line = f"{gl} at {h_pos} ({dist}){occ}"

            if role in ("walkable","semi_walkable","non_walkable"):
                surface_lines.append(line)
            elif role in ("obstacle","hazard","dynamic_hazard"):
                obstacle_lines.append(line)

        parts = []
        cap_action = action.capitalize()
        parts.append(f"{cap_action}. Obstacles: {'; '.join(obstacle_lines)}."
                     if obstacle_lines else f"{cap_action}. Path appears clear.")
        if surface_lines:
            parts.append("Surfaces: " + "; ".join(surface_lines) + ".")
        fs_summary = ", ".join(f"{col} {st}" for col,st in free_space.items() if st != "unknown")
        if fs_summary:
            parts.append(f"Walkability: {fs_summary}.")

        return {
            "free_space":  free_space,
            "action":      action,
            "obstacles":   obstacle_lines,
            "surfaces":    surface_lines,
            "scene_text":  " ".join(parts),
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
        col_bounds_vis = compute_corridor_bounds(sidewalk_mask_vis, img_h2, img_w2, zone_top)
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
                "free_space":empty_fs,"boxes":[],"scores":[],"phrases":[]}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def test():
    parser = argparse.ArgumentParser(description="VLM Navigation Pipeline")
    parser.add_argument("--image",    required=True)
    parser.add_argument("--no-sam",   action="store_true")
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--output",   default="output_v9.png")
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
    print(f"  Free-space : {nav['free_space']}")
    if nav["obstacles"]: print(f"  Obstacles  : {nav['obstacles']}")
    if nav["surfaces"]:  print(f"  Surfaces   : {nav['surfaces']}")
    print(f"\n  LLM-ready  :\n  {nav['scene_text']}")
    print("="*55)

    pipeline.visualize(results, save_name=args.output)

if __name__ == "__main__":
    test()