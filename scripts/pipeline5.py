"""
VLM Pipeline v7 — Tree-Aware + Fusion-Safe
============================================
Built on v6. New fixes tagged [D1]–[D3]. Two doc suggestions rejected.

  [D1] tree role: "context" → "obstacle"
       Tree trunks are hard physical objects on sidewalks. Treating them
       as context meant they never entered blocking logic. Now they do,
       via the same HARD_BLOCK_ROLES path as bollards and barriers.
       Also added "tree" to MULTI_PROMPTS so GDINO actually looks for them.

  [D2] Area filter: exempt tall/large classes from relative-area ceiling.
       Trees, poles, and buildings often produce boxes covering >85% of
       image height. The original MAX_RELATIVE_AREA check was dropping
       them ([area-drop] 'tree' — too large (92%)).
       Fix: skip the relative-area ceiling for these classes only.
       Min-size (900px) and aspect-ratio (12:1) guards still apply —
       so garbage boxes are still rejected.
       "pole" and "building" also exempted for the same reason.

  [D3] Fusion safety override in _fuse_states():
       The weighted blend could downgrade a real detection:
         global=crowded(1), local=walkable(0) → 0.55*0+0.45*1=0.45 → walkable ❌
       Fix: before doing any math, hard overrides run first:
         - global "blocked"  → always return "blocked"
         - global "crowded" + local "walkable" → return "crowded"
       Weighted blend only runs when local is same severity or worse —
       i.e. when local is adding signal, not cancelling it.

  [DOC-SKIP-D4] Weight swap to 0.45 local / 0.55 global — REJECTED.
       The doc argues global should dominate for safety. But the entire
       point of sidewalk-local analysis is filtering out road-lane objects
       that global incorrectly counts. Making global dominate again defeats
       the v6 upgrade. Fix D3 already handles the safety case correctly.

  [DOC-SKIP-D5] unknown → not walkable fallback — REJECTED.
       On Indian roads GDINO frequently misses sidewalks entirely.
       Reverting FIX2 would cause constant unnecessary stops. The
       sidewalk-local layer (SW) already handles no-mask cases by returning
       None and falling through to global without modification.

  [SW] Sidewalk-mask-filtered free-space analysis.

  WHAT THE DOC PROPOSED:
    Crop the bounding rectangle of the sidewalk mask → re-run detection
    inside that crop. This is wrong: the crop's rectangle includes off-
    sidewalk pixels (road, grass, building edge), and re-running GDINO
    costs a second full model pass per image.

  WHAT IS ACTUALLY IMPLEMENTED:
    Zero extra model calls. We reuse the SAM mask that already exists.

    For obstacle relevance, two conditions must BOTH be true:
      (a) The obstacle box foot-point (bottom-centre pixel) falls inside
          the sidewalk SAM mask, OR the obstacle box overlaps the mask
          by ≥ SW_FOOT_OVERLAP_RATIO of the obstacle's own area.
      (b) The obstacle is near/mid distance (already enforced by FIX4).

    This means a car sitting in the road lane — even if its bounding
    box centre falls in the "center" column strip — is NOT counted as
    blocking the sidewalk corridor, because its foot-point is on the
    road mask, not the sidewalk mask.

    Coverage-based column state (replaces binary blocked/crowded):
      obstacle_pixels / sidewalk_pixels_in_column
        ≥ SW_BLOCK_COVERAGE  → "blocked"
        ≥ SW_CROWD_COVERAGE  → "crowded"
        <  SW_CROWD_COVERAGE → column stays walkable even with objects present

    The global column state (from analyse_free_space) is fused with the
    sidewalk-local state using a weighted blend:
      final = 0.55 * sidewalk_local + 0.45 * global
    giving the sidewalk-filtered view more weight without discarding the
    global awareness that catches objects outside any sidewalk mask.

    Sidewalk analysis is bypassed when no SAM sidewalk mask exists,
    falling back to the global result unchanged.

Additional fixes tagged [FIX1]–[FIX6]:

  [FIX1] Dynamic vs static obstacle classification in free-space analysis.
         dynamic_hazard (persons etc.) → "crowded", not "blocked".
         Only static obstacles/hazards mark a corridor as "blocked".

  [FIX2] unknown → walkable fallback: if a column has no SAM mask AND no
         obstacle box, treat it as "walkable" instead of "unknown".
         Prevents unnecessary stopping when detection simply missed the surface.

  [FIX3] Relaxed stop logic: choose the best available corridor instead of
         stopping whenever center is not walkable. Stop only when all
         corridors are blocked with no walkable/crowded option.

  [FIX4] Distance-aware corridor blocking: far objects (dist == "far") do
         not contribute to blocking a corridor even if their centre x falls
         in it. Near/mid objects still block normally.

  [FIX5] Crowded state propagates to action verb: "move forward (crowded)"
         distinguishes a clear path from a navigable-but-busy one.

  [FIX6] Occlusion analysis guard: containment check was using raw area
         comparison without checking that the outer box is actually larger
         in both dimensions. Added explicit outer-area > inner-area guard.

WHAT THE RECOMMENDATION DOC GOT RIGHT:
  ✅ Free-space estimation needed (U4)
  ✅ Occlusion logic too aggressive (U3 — but their fix was incomplete)
  ✅ Instance grouping useful (U5)
  ✅ Navigation-style description (U5)
  ✅ Class roles / OBJECT_ROLES (U5)

WHAT THE RECOMMENDATION DOC GOT WRONG:
  ❌ [DOC-SKIP] Synonym expansion (sidewalk+pavement+footpath)
     → Multi-prompt already runs 1 pass per class. Adding 4 synonyms
       = 28 GDINO passes on CPU = ~3 minutes per image. Not done.
  ❌ [DOC-SKIP] Priority filter "already in v4" — doc listed it as new.
  ❌ [DOC-SKIP] Free-space via box overlap on hardcoded strip — replaced
     with mask-pixel analysis which handles angled/partial paths.
  ❌ [DOC-SKIP] Occlusion fix "add area check" — incomplete. Real fix
     is label-awareness: surfaces never occlude, only dynamic objects do.

REAL UPGRADES NOT IN DOC:
  [U1] Cross-prompt deduplication — same object detected by multiple
       prompts (cyclist caught by both 'person' and 'bicycle' prompts).
  [U2] expand_box() boundary clamping — was producing out-of-image
       coords that crashed SAM or produced garbage masks.
  [U3] Label-aware occlusion — geometry-only was wrong for surface labels.
  [U4] Mask-based free-space (3-column, bottom 40%) — replaces naive strip.
  [U5] Navigation description: roles + grouping + action verbs + dict output.
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
from dataclasses import dataclass, field
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
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# [DOC-SKIP] NOT expanded to synonyms — cost: 28 GDINO passes on CPU (~3 min).
# One canonical noun per class is the correct design for multi-prompt pipelines.
MULTI_PROMPTS = [
    "person", "car", "bicycle", "traffic cone", "barrier", "road", "sidewalk", "tree"
]

def sanitise_prompt(prompt: str) -> str:
    prompt = prompt.lower().strip()
    prompt = " . ".join(p.strip() for p in prompt.split(".") if p.strip())
    if not prompt.endswith(" ."):
        prompt += " ."
    return prompt

# ─────────────────────────────────────────────────────────────────────────────
# [U5] OBJECT ROLES — object → navigation meaning
# ─────────────────────────────────────────────────────────────────────────────
# These roles drive the action verbs in build_navigation_description().
# Three role families:
#   WALKABLE       — surfaces you can step on
#   OBSTACLE       — static things to avoid (hard block)
#   HAZARD         — static vehicles (hard block)
#   DYNAMIC_HAZARD — moving things (make corridor "crowded", not "blocked")

OBJECT_ROLES: dict[str, str] = {
    # Walkable surfaces
    "road":       "non_walkable",   # road = car lane, not for pedestrians
    "sidewalk":   "walkable",
    "crosswalk":  "walkable",
    "ramp":       "walkable",
    # Dynamic hazards — can move; make path crowded, not blocked [FIX1]
    "person":         "dynamic_hazard",
    "cyclist":        "dynamic_hazard",
    "motorcyclist":   "dynamic_hazard",
    "wheelchair":     "dynamic_hazard",
    # Static vehicles — parked/moving; treat as hard block
    "car":            "hazard",
    "truck":          "hazard",
    "bus":            "hazard",
    "bicycle":        "hazard",
    "motorcycle":     "hazard",
    "scooter":        "hazard",
    # Static obstacles — hard block
    "traffic cone":   "obstacle",
    "barrier":        "obstacle",
    "bollard":        "obstacle",
    "pole":           "obstacle",
    "fence":          "obstacle",
    "railing":        "obstacle",
    "bench":          "obstacle",
    "fire hydrant":   "obstacle",
    "traffic light":  "landmark",
    "stop sign":      "landmark",
    "traffic sign":   "landmark",
    "building":       "context",
    "tree":           "obstacle",   # trunk = hard physical block on sidewalk [D1]
}

# Only SAM-segment these (surface masks for free-space analysis)
SAM_CLASSES = {"road", "sidewalk"}

# Dynamic labels — only these can occlude other objects [U3]
DYNAMIC_LABELS = {
    "person", "pedestrian", "cyclist", "motorcyclist",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
}

# [FIX1] Roles that produce a HARD BLOCK (static, won't move aside)
HARD_BLOCK_ROLES = {"obstacle", "hazard"}

# [FIX1] Roles that produce a SOFT/CROWDED state (dynamic, may move)
SOFT_BLOCK_ROLES = {"dynamic_hazard"}

# Per-class confidence thresholds (GPU baseline; scaled for CPU by _cpu_adjust)
PER_CLASS_THRESHOLDS: dict[str, float] = {
    "person": 0.40, "pedestrian": 0.38, "cyclist": 0.38,
    "motorcyclist": 0.38, "wheelchair": 0.35, "stroller": 0.35,
    "car": 0.45, "truck": 0.45, "bus": 0.45,
    "bicycle": 0.40, "motorcycle": 0.40, "scooter": 0.40,
    "traffic light": 0.42, "stop sign": 0.42, "traffic sign": 0.38,
    "pole": 0.35, "bollard": 0.35, "barrier": 0.40,
    "bench": 0.40, "road": 0.50, "sidewalk": 0.48,
    "building": 0.50, "tree": 0.45, "traffic cone": 0.38,
}
DEFAULT_THRESHOLD = 0.42

def _cpu_adjust(t: float) -> float:
    return round(t * 0.45, 3) if _CPU_FALLBACK else t

# ─────────────────────────────────────────────────────────────────────────────
# AREA FILTER
# ─────────────────────────────────────────────────────────────────────────────

MIN_ABSOLUTE_PIXELS = 900
MAX_RELATIVE_AREA   = 0.85
MAX_ASPECT_RATIO    = 12.0

def passes_area_filter(box: list, img_h: int, img_w: int, label: str = "") -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
    area = bw * bh
    if area < MIN_ABSOLUTE_PIXELS:
        return False, f"too small ({area}px)"
    # [D2] Trees / poles span tall vertical space — skip relative-area ceiling
    # but still enforce minimum size and aspect ratio so garbage boxes are dropped.
    if label not in {"tree", "pole", "building"}:
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
    role:          str           = "unknown"  # [U5] filled after init
    distance:      str           = "unknown"  # [FIX4] filled after init

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
    boxes:  list[list[int]],
    scores: list[float],
    labels: list[str],
    iou_threshold: float = 0.50,
) -> tuple[list, list, list]:
    """
    Remove duplicate detections that arose from different prompt passes
    targeting the same physical object.
    """
    n = len(boxes)
    if n <= 1:
        return boxes, scores, labels

    keep = [True] * n
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    for rank_i, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[rank_i + 1:]:
            if not keep[j]:
                continue
            if _iou(boxes[i], boxes[j]) >= iou_threshold:
                keep[j] = False

    kept_boxes  = [boxes[i]  for i in range(n) if keep[i]]
    kept_scores = [scores[i] for i in range(n) if keep[i]]
    kept_labels = [labels[i] for i in range(n) if keep[i]]

    removed = n - len(kept_boxes)
    if removed > 0:
        print(f"  [U1 dedup] Removed {removed} cross-prompt duplicates")

    return kept_boxes, kept_scores, kept_labels

# ─────────────────────────────────────────────────────────────────────────────
# [U3] LABEL-AWARE OCCLUSION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_containment(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((ax2-ax1) * (ay2-ay1), 1)

def run_occlusion_analysis(detections: list[Detection]) -> list[Detection]:
    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j: continue

            containment = compute_containment(detections[i].box, detections[j].box)
            if containment < 0.85:
                continue

            same_label = detections[i].label == detections[j].label

            if same_label:
                if detections[i].score < detections[j].score:
                    detections[i].suppressed_by = j

            else:
                # [U3] Only mark occluded if OUTER box is a dynamic object
                outer_is_dynamic = detections[j].label in DYNAMIC_LABELS
                # [FIX6] Explicit outer-area > inner-area guard (not just area attribute)
                outer_box = detections[j].box
                inner_box = detections[i].box
                outer_area = (outer_box[2]-outer_box[0]) * (outer_box[3]-outer_box[1])
                inner_area = (inner_box[2]-inner_box[0]) * (inner_box[3]-inner_box[1])
                inner_is_smaller = inner_area < outer_area

                if outer_is_dynamic and inner_is_smaller:
                    detections[i].occluded = True

    return detections

# ─────────────────────────────────────────────────────────────────────────────
# SOFT-NMS
# ─────────────────────────────────────────────────────────────────────────────

def soft_nms(
    detections: list[Detection],
    sigma: float = 0.5,
    score_gate: float = 0.20,
) -> list[Detection]:
    dets = sorted(
        [d for d in detections if d.suppressed_by is None],
        key=lambda d: d.score, reverse=True,
    )
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            overlap = _iou(dets[i].box, dets[j].box)
            if overlap > 0:
                dets[j].score *= np.exp(-(overlap ** 2) / sigma)
    return [d for d in dets if d.score >= score_gate]

# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE ESTIMATION (extracted as standalone so FIX4 can use it early)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_distance(det: Detection, img_h: int, img_w: int, img_area: int) -> str:
    """
    Return "near" | "mid" | "far" for a detection.
    Surface classes are always "near" (they extend toward the camera).
    """
    x1, y1, x2, y2 = det.box
    bottom     = y2
    area_ratio = det.area / max(img_area, 1)

    if det.label in SAM_CLASSES:
        return "near"
    elif det.label == "traffic cone":
        if bottom > 0.6 * img_h:   return "near"
        elif bottom > 0.4 * img_h: return "mid"
        else:                       return "far"
    elif bottom > 0.75 * img_h:    return "near"
    elif bottom > 0.50 * img_h:    return "mid"
    elif area_ratio > 0.10:        return "near"
    else:                           return "far"

# ─────────────────────────────────────────────────────────────────────────────
# [U4 + FIX1 + FIX2 + FIX3 + FIX4] MASK-BASED FREE-SPACE ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
# Column states (extended from v5):
#   "walkable" — SAM surface mask present and no hard-block obstacle
#   "crowded"  — dynamic hazard present (person etc.) but path is passable [FIX1]
#   "blocked"  — static obstacle or vehicle in the corridor
#   "unknown"  — no mask, no obstacle → treated as walkable by fallback [FIX2]
#
# [FIX4] Far objects are excluded from blocking columns entirely.

WALKABLE_COVERAGE = 0.25   # 25% of column must be masked as walkable surface

# ─────────────────────────────────────────────────────────────────────────────
# [SW] SIDEWALK-MASK OBSTACLE FILTER + COVERAGE SCORING
# ─────────────────────────────────────────────────────────────────────────────
# Thresholds for coverage-based column state inside the sidewalk mask.
SW_BLOCK_COVERAGE      = 0.50   # obstacle pixels / sidewalk pixels → "blocked"
SW_CROWD_COVERAGE      = 0.20   # obstacle pixels / sidewalk pixels → "crowded"
SW_FOOT_OVERLAP_RATIO  = 0.15   # min fraction of obstacle box that must overlap
                                 # the sidewalk mask to count as on-sidewalk
# Fusion weights: sidewalk-local vs global column state
SW_LOCAL_WEIGHT  = 0.55
SW_GLOBAL_WEIGHT = 0.45

# Numeric scores no longer needed — rule-based fusion replaces weighted math [E1]

def _fuse_states(global_st: str, local_st: str) -> str:
    """
    [E1] Rule-based fusion of global and sidewalk-local column states.

    Replaces the weighted-average approach from v6/v7.
    Weighted math had rounding artifacts and was an approximation of
    these rules anyway. Pure rule-based is deterministic and correct:

      blocked wins:   if either layer says blocked → blocked
      crowded wins:   if either layer says crowded → crowded
      uncertain:      if local is unknown (no sidewalk pixels in column)
                      defer entirely to global
      walkable:       both layers agree path is clear

    "uncertain" state (new in v8) is propagated from global when there
    is no positive surface evidence — see [E2] in analyse_free_space.
    """
    if local_st == "unknown":
        return global_st   # no local data — defer to global entirely

    if global_st == "blocked" or local_st == "blocked":
        return "blocked"

    if global_st == "crowded" or local_st == "crowded":
        return "crowded"

    if global_st == "uncertain" or local_st == "uncertain":
        return "uncertain"

    return "walkable"


def _obstacle_on_sidewalk(
    det:           Detection,
    sidewalk_mask: np.ndarray,   # full-image boolean mask
) -> bool:
    """
    Return True if this detection is meaningfully ON the sidewalk surface.

    Two-part test (either passes → True):
      1. Foot-point test: the pixel at the bottom-centre of the box is
         inside the sidewalk mask.  Works well for upright objects.
      2. Overlap test: the fraction of the obstacle box that lies inside
         the sidewalk mask exceeds SW_FOOT_OVERLAP_RATIO.  Catches wide
         objects (e.g. benches, barriers) whose foot-point might straddle
         the mask edge.
    """
    x1, y1, x2, y2 = det.box
    img_h, img_w   = sidewalk_mask.shape

    # Clamp to image bounds
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(img_w - 1, x2); y2c = min(img_h - 1, y2)

    # Test 1: foot-point (bottom-centre)
    foot_x = int((x1c + x2c) / 2)
    foot_y = y2c
    if sidewalk_mask[foot_y, foot_x]:
        return True

    # Test 2: box overlap fraction
    box_mask_region  = sidewalk_mask[y1c:y2c+1, x1c:x2c+1]
    box_area         = max((x2c - x1c + 1) * (y2c - y1c + 1), 1)
    overlap_fraction = box_mask_region.sum() / box_area
    return overlap_fraction >= SW_FOOT_OVERLAP_RATIO


def _merge_sidewalk_masks(
    detections: list[Detection],
    masks:      list,
    img_h:      int,
    img_w:      int,
) -> Optional[np.ndarray]:
    """
    Merge all SAM sidewalk masks into a single boolean array.
    Returns None if no sidewalk mask is available.
    """
    combined: Optional[np.ndarray] = None
    for det, mask in zip(detections, masks):
        if mask is None or det.label != "sidewalk":
            continue
        if combined is None:
            combined = mask.astype(bool)
        else:
            combined |= mask.astype(bool)
    return combined


def analyse_sidewalk_local(
    image:      np.ndarray,
    detections: list[Detection],
    masks:      list,
    col_bounds: dict[str, tuple[int, int]],
    zone_top:   int,
) -> Optional[dict[str, str]]:
    """
    [SW] Sidewalk-local column analysis.

    For each column, counts only those obstacle pixels whose detection
    box is confirmed to be ON the sidewalk mask (foot-point or overlap).
    Coverage ratio determines the column state:
        ≥ SW_BLOCK_COVERAGE  → "blocked"
        ≥ SW_CROWD_COVERAGE  → "crowded"
        < SW_CROWD_COVERAGE  → "walkable"

    Returns None if no sidewalk SAM mask exists (caller falls back to
    global result).
    """
    img_h, img_w = image.shape[:2]
    sidewalk_mask = _merge_sidewalk_masks(detections, masks, img_h, img_w)

    if sidewalk_mask is None:
        return None   # no sidewalk mask — skip local analysis

    # Per-column sidewalk pixel counts (analysis zone only)
    sw_pixels: dict[str, int] = {}
    for col, (cx1, cx2) in col_bounds.items():
        zone_sw = sidewalk_mask[zone_top:, cx1:cx2]
        sw_pixels[col] = int(zone_sw.sum())

    # Build obstacle masks per column, filtered to on-sidewalk objects only
    # We rasterise obstacle boxes that pass the on-sidewalk test.
    result: dict[str, str] = {}
    for col, (cx1, cx2) in col_bounds.items():
        sw_col_px = sw_pixels[col]

        if sw_col_px == 0:
            # No sidewalk in this column — local analysis has nothing to say
            result[col] = "unknown"
            continue

        # [E3] Width-ratio blocking: measure how much of the COLUMN WIDTH the
        # object spans, not raw pixel area.  A traffic cone in the center of a
        # 300px-wide column that is only 30px wide gets ratio 0.10 → not blocked.
        # A barrier spanning 200px of that column gets ratio 0.67 → blocked.
        hard_max_ratio = 0.0
        soft_max_ratio = 0.0

        for det in detections:
            role = OBJECT_ROLES.get(det.label, "unknown")
            if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES:
                continue
            if det.distance == "far":
                continue

            # [SW] Only count if the object is actually on the sidewalk
            if not _obstacle_on_sidewalk(det, sidewalk_mask):
                continue

            x1, y1, x2, y2 = det.box
            if y2 < zone_top:
                continue

            # Intersection width with this column strip
            ox1 = max(x1, cx1); ox2 = min(x2, cx2)
            if ox2 <= ox1:
                continue
            oy1 = max(y1, zone_top); oy2 = y2
            if oy2 <= oy1:
                continue

            col_width   = max(cx2 - cx1, 1)
            obj_w_ratio = (ox2 - ox1) / col_width   # [E3] width fraction

            if role in HARD_BLOCK_ROLES:
                hard_max_ratio = max(hard_max_ratio, obj_w_ratio)
            elif role in SOFT_BLOCK_ROLES:
                soft_max_ratio = max(soft_max_ratio, obj_w_ratio)

        # Thresholds for width-ratio (reuse SW_BLOCK/CROWD_COVERAGE names;
        # values unchanged: 0.50 = half column blocked, 0.20 = crowded)
        if hard_max_ratio >= SW_BLOCK_COVERAGE:
            result[col] = "blocked"
        elif hard_max_ratio >= SW_CROWD_COVERAGE or soft_max_ratio >= SW_CROWD_COVERAGE:
            result[col] = "crowded"
        else:
            result[col] = "walkable"

    return result


def analyse_free_space(
    image:      np.ndarray,
    detections: list[Detection],
    masks:      list,
) -> dict[str, str]:
    """
    Analyse walkability of left / center / right navigation corridors.

    Returns:
        {"left": "walkable"|"crowded"|"blocked"|"unknown",
         "center": ..., "right": ...}
    """
    img_h, img_w = image.shape[:2]

    # Analysis zone: bottom 40% of image
    zone_top = int(img_h * 0.60)

    col_bounds = {
        "left":   (0,                 int(img_w * 0.33)),
        "center": (int(img_w * 0.33), int(img_w * 0.67)),
        "right":  (int(img_w * 0.67), img_w),
    }
    col_area = (img_h - zone_top) * (img_w // 3)

    # ── Step 1: walkable pixel coverage per column from SAM masks ────────────
    walkable_pixels = {col: 0 for col in col_bounds}
    for det, mask in zip(detections, masks):
        if mask is None or det.label not in SAM_CLASSES:
            continue
        zone_mask = mask[zone_top:, :]
        for col, (cx1, cx2) in col_bounds.items():
            col_mask = zone_mask[:, cx1:cx2]
            walkable_pixels[col] += int(col_mask.sum())

    # ── Step 2: global obstacle classification per column ────────────────────
    # [FIX1] Separate hard-block (static) from soft-block (dynamic)
    # [FIX4] Far objects are excluded from blocking
    hard_blocked_cols: set[str] = set()
    crowded_cols:      set[str] = set()

    for det in detections:
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES:
            continue
        if det.distance == "far":
            continue
        x1, y1, x2, y2 = det.box
        if y2 < zone_top:
            continue
        obj_cx = (x1 + x2) / 2
        for col, (cx1, cx2) in col_bounds.items():
            if cx1 <= obj_cx < cx2:
                if role in HARD_BLOCK_ROLES:
                    hard_blocked_cols.add(col)
                elif role in SOFT_BLOCK_ROLES:
                    crowded_cols.add(col)

    # ── Step 3: global decision per column ───────────────────────────────────
    global_result: dict[str, str] = {}
    for col in col_bounds:
        coverage          = walkable_pixels[col] / max(col_area, 1)
        surface_confirmed = coverage >= WALKABLE_COVERAGE

        if col in hard_blocked_cols:
            global_result[col] = "blocked"
        elif col in crowded_cols:
            global_result[col] = "crowded"
        elif surface_confirmed:
            global_result[col] = "walkable"
        else:
            # [E2] No obstacle, no confirmed surface → "uncertain" (not walkable,
            # not blocked). Lets navigation proceed cautiously rather than stopping
            # or assuming clear. Better than FIX2's optimistic "walkable" when
            # GDINO simply missed the sidewalk, but safer than "blocked".
            global_result[col] = "uncertain"

    # ── Step 4: [SW] Sidewalk-local analysis + fusion ────────────────────────
    local_result = analyse_sidewalk_local(
        image, detections, masks, col_bounds, zone_top
    )

    if local_result is None:
        # No sidewalk mask available — global result is the final answer
        print("  [SW] No sidewalk mask — using global free-space only")
        return global_result

    # Fuse per-column: sidewalk-local result weighted more than global
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
    for det in detections:
        groups[det.label].append(det)
    return dict(groups)

def group_label(label: str, count: int) -> str:
    if count == 1:
        return label
    elif count == 2:
        return f"2 {label}s"
    else:
        return f"multiple {label}s"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class VLMPipeline:
    def __init__(
        self,
        soft_nms_sigma: float = 0.5,
        max_image_size: int   = 800,
    ):
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_nms_sigma = soft_nms_sigma
        self.max_image_size = max_image_size
        self.cpu_fallback   = _CPU_FALLBACK

        print(f"[Pipeline] Device        : {self.device}")
        print(f"[Pipeline] CPU fallback  : {self.cpu_fallback}")
        print(f"[Pipeline] Box threshold : {BOX_THRESHOLD_DEFAULT}")

        self._load_models()

    def _load_models(self):
        dino_config = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
        dino_ckpt   = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
        sam_ckpt    = os.path.join(WEIGHTS_DIR, "sam_vit_l_0b3195.pth")
        for path, name in [(dino_config,"DINO config"),(dino_ckpt,"DINO weights"),(sam_ckpt,"SAM weights")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name}: {path}")
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
        for cx, cy, bw, bh in raw_boxes:
            x1 = max(0,     int((cx - bw / 2) * img_w))
            y1 = max(0,     int((cy - bh / 2) * img_h))
            x2 = min(img_w, int((cx + bw / 2) * img_w))
            y2 = min(img_h, int((cy + bh / 2) * img_h))
            out.append([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)])
        return out

    def _expand_box(
        self,
        box: list[int],
        img_h: int,
        img_w: int,
        scale: float = 1.15,
    ) -> list[int]:
        """[U2] Expand box by scale around centre, CLAMPED to image boundaries."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        w  = (x2 - x1) * scale; h = (y2 - y1) * scale
        return [
            max(0,     int(cx - w / 2)),
            max(0,     int(cy - h / 2)),
            min(img_w, int(cx + w / 2)),
            min(img_h, int(cy + h / 2)),
        ]

    def _run_gdino(self, image_tensor, prompt, box_thr, text_thr):
        with torch.no_grad():
            raw_boxes, logits, phrases = predict(
                self.dino_model, image_tensor,
                caption=prompt,
                box_threshold=box_thr,
                text_threshold=text_thr,
                device=self.device,
            )
        return (
            raw_boxes.cpu().numpy(),
            logits.cpu().numpy(),
            [p.strip().replace(".", "").lower() for p in phrases],
        )

    # ── Main detection + segmentation ────────────────────────────────────────
    def detect_and_segment(
        self,
        image_path: str,
        run_sam: bool = True,
    ) -> dict:

        # Load & resize
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        scale = self.max_image_size / max(img_h, img_w)
        if scale < 1.0:
            image = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))
            img_h, img_w = image.shape[:2]

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(Image.fromarray(image)).to(torch.float32).to(self.device)

        # ── Multi-prompt GDINO ────────────────────────────────────────────────
        print("[Pipeline] Running Multi-Pass GDINO...")
        all_boxes_raw: list  = []
        all_scores:    list  = []
        all_labels:    list  = []

        for single_prompt in MULTI_PROMPTS:
            p = sanitise_prompt(single_prompt)
            for box_thr in THRESHOLD_LADDER:
                text_thr = box_thr * 1.2
                rb, sc, lb = self._run_gdino(image_tensor, p, box_thr, text_thr)
                if len(sc) > 0:
                    print(f"  ['{single_prompt}'] {len(sc)} det(s) "
                          f"(max={sc.max():.2f}, thr={box_thr:.2f})")
                    all_boxes_raw.extend(rb)
                    all_scores.extend(sc.tolist())
                    all_labels.extend(lb)
                    break
            else:
                print(f"  ['{single_prompt}'] 0 detections at all thresholds")

        if not all_boxes_raw:
            print("[Pipeline] ✗ No detections.")
            return self._empty(image)

        boxes_xyxy = self._decode_boxes(np.array(all_boxes_raw), img_h, img_w)
        scores     = all_scores
        labels     = all_labels

        # ── [U1] Cross-prompt deduplication ──────────────────────────────────
        boxes_xyxy, scores, labels = deduplicate_cross_prompt(boxes_xyxy, scores, labels)

        # ── Per-class confidence filter ───────────────────────────────────────
        step1: list[Detection] = []
        for box, score, label in zip(boxes_xyxy, scores, labels):
            thr = _cpu_adjust(PER_CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD))
            if score >= thr:
                area = (box[2] - box[0]) * (box[3] - box[1])
                det  = Detection(box=box, score=float(score), label=label, area=area)
                det.role = OBJECT_ROLES.get(label, "unknown")   # [U5]
                step1.append(det)
        print(f"  After conf filter   : {len(step1)}/{len(scores)}")

        # ── Area filter ───────────────────────────────────────────────────────
        step2: list[Detection] = []
        for det in step1:
            ok, reason = passes_area_filter(det.box, img_h, img_w, det.label)
            if ok:
                step2.append(det)
            else:
                print(f"  [area-drop] '{det.label}' — {reason}")
        print(f"  After area filter   : {len(step2)}/{len(step1)}")

        if not step2:
            return self._empty(image)

        # ── [U3] Label-aware occlusion + Soft-NMS ─────────────────────────────
        step2 = run_occlusion_analysis(step2)
        step3 = soft_nms(step2, sigma=self.soft_nms_sigma)
        print(f"  After Soft-NMS      : {len(step3)}/{len(step2)}")

        # ── [FIX4] Attach distance to each detection (needed before free-space)
        img_area = img_h * img_w
        for det in step3:
            det.distance = estimate_distance(det, img_h, img_w, img_area)

        print(f"  Final: {dict(Counter(d.label for d in step3))}")

        # ── SAM (surface classes only, with [U2] clamped expand_box) ─────────
        masks: list = []
        if run_sam:
            self.sam_predictor.set_image(image)

        for det in step3:
            if not (run_sam and det.label in SAM_CLASSES):
                masks.append(None)
                continue
            if det.score < 0.3:
                masks.append(None)
                continue

            box_exp = self._expand_box(det.box, img_h, img_w, scale=1.15)
            mask, _, _ = self.sam_predictor.predict(
                box=np.array(box_exp, dtype=np.float32),
                multimask_output=False,
            )
            mask  = mask[0]
            ratio = mask.sum() / (mask.shape[0] * mask.shape[1])
            if ratio > 0.8 or ratio < 0.01:
                print(f"  [SAM-drop] '{det.label}' bad ratio={ratio:.2f}")
                masks.append(None)
            else:
                masks.append(mask)

        # ── [U4 + FIX1 + FIX2 + FIX4] Free-space analysis ───────────────────
        free_space = analyse_free_space(image, step3, masks)
        print(f"  Free-space: {free_space}")

        return {
            "image":       image,
            "detections":  step3,
            "masks":       masks,
            "free_space":  free_space,
            # Legacy flat keys
            "boxes":   [d.box   for d in step3],
            "scores":  [d.score for d in step3],
            "phrases": [d.label for d in step3],
        }

    # ── [U5 + FIX3 + FIX5] Navigation description ────────────────────────────
    def build_navigation_description(self, results: dict) -> dict:
        """
        Returns a structured navigation dict suitable for direct LLM injection.

        Column states: "walkable" | "crowded" | "blocked" | "unknown"

        Action hierarchy [FIX3]:
          1. Forward is walkable             → "move forward"
          2. Forward is crowded              → "move forward (crowded)" [FIX5]
          3. Forward blocked, left walkable  → "move left"
          4. Forward blocked, right walkable → "move right"
          5. Forward blocked, left crowded   → "move left (crowded)"
          6. Forward blocked, right crowded  → "move right (crowded)"
          7. All corridors blocked           → "stop — path unclear"
        """
        dets       = results.get("detections", [])
        free_space = results.get("free_space", {"left":"unknown","center":"unknown","right":"unknown"})
        img_h, img_w = results["image"].shape[:2]
        img_area   = img_h * img_w

        center = free_space.get("center", "unknown")
        left   = free_space.get("left",   "unknown")
        right  = free_space.get("right",  "unknown")

        # ── [FIX3 + E2] Relaxed action logic — prefer any navigable corridor ──
        # "walkable" and "crowded" are both navigable; "blocked" is not.
        # "uncertain" is cautiously navigable — proceed but with reduced speed. [E2]
        def is_navigable(state: str) -> bool:
            return state in ("walkable", "crowded", "uncertain")

        def action_for(state: str, direction: str) -> str:
            if state == "walkable":
                return f"move {direction}"
            elif state == "crowded":
                return f"move {direction} (crowded)"
            elif state == "uncertain":
                return f"move {direction} (cautious)"   # [E2]
            else:
                return f"move {direction}"

        if center == "walkable":
            action = "move forward"
        elif center == "crowded":
            action = "move forward (crowded)"
        elif center == "uncertain":
            action = "move forward (cautious)"          # [E2]
        elif is_navigable(left) and not is_navigable(right):
            action = action_for(left, "left")
        elif is_navigable(right) and not is_navigable(left):
            action = action_for(right, "right")
        elif is_navigable(left) and is_navigable(right):
            # Both sides open — prefer walkable > uncertain > crowded, else left
            priority = {"walkable": 0, "uncertain": 1, "crowded": 2}
            action = action_for(left, "left") if priority.get(left, 3) <= priority.get(right, 3) else action_for(right, "right")
        else:
            action = "stop — path unclear"

        # ── Per-detection descriptions with roles + grouping ──────────────────
        groups = group_detections(dets)

        obstacle_lines: list[str] = []
        surface_lines:  list[str] = []

        for label, group in groups.items():
            role = OBJECT_ROLES.get(label, "unknown")

            rep = max(group, key=lambda d: d.score)
            x1, y1, x2, y2 = rep.box
            cx = (x1 + x2) / 2

            h_pos = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
            dist  = rep.distance if rep.distance != "unknown" else estimate_distance(rep, img_h, img_w, img_area)

            # Skip far hazards — they don't affect the immediate path [FIX4]
            if dist == "far" and label in {"car", "bicycle"}:
                continue

            count = len(group)
            gl    = group_label(label, count)
            occ   = " [occluded]" if rep.occluded else ""
            line  = f"{gl} at {h_pos} ({dist}){occ}"

            if role in ("walkable", "non_walkable"):
                surface_lines.append(line)
            elif role in ("obstacle", "hazard", "dynamic_hazard"):
                obstacle_lines.append(line)

        # ── Compose scene_text ────────────────────────────────────────────────
        parts      = []
        cap_action = action.capitalize()

        if obstacle_lines:
            avoid_str = "; ".join(obstacle_lines)
            parts.append(f"{cap_action}. Obstacles: {avoid_str}.")
        else:
            parts.append(f"{cap_action}. Path appears clear.")

        if surface_lines:
            parts.append("Surfaces: " + "; ".join(surface_lines) + ".")

        fs_summary = ", ".join(
            f"{col} {st}" for col, st in free_space.items() if st != "unknown"
        )
        if fs_summary:
            parts.append(f"Walkability: {fs_summary}.")

        scene_text = " ".join(parts)

        return {
            "free_space":  free_space,
            "action":      action,
            "obstacles":   obstacle_lines,
            "surfaces":    surface_lines,
            "scene_text":  scene_text,
        }

    def build_scene_description(self, results: dict) -> str:
        nav = self.build_navigation_description(results)
        return nav["scene_text"]

    # ── Visualisation ─────────────────────────────────────────────────────────
    def visualize(self, results: dict, save_name: str = "output_v5.png"):
        if not results or results.get("image") is None:
            return

        image      = results["image"].copy()
        dets       = results.get("detections", [])
        masks      = results.get("masks", [])
        free_space = results.get("free_space", {})

        unique_labels = list({d.label for d in dets})
        cmap   = matplotlib.colormaps.get_cmap("tab20")
        colors = {
            lbl: tuple(int(c * 255) for c in cmap(i / max(len(unique_labels), 1))[:3])
            for i, lbl in enumerate(unique_labels)
        }

        # Draw surface masks
        for mask, det in zip(masks, dets):
            if mask is None or det.label not in SAM_CLASSES:
                continue
            if mask.sum() / mask.size > 0.6:
                continue
            color = np.array(colors[det.label])
            image[mask] = (image[mask] * 0.6 + color * 0.4).astype(np.uint8)

        # Draw bounding boxes
        for det in dets:
            x1, y1, x2, y2 = det.box
            color     = colors[det.label]
            thickness = 3 if not det.occluded else 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            tag = f"{det.label}{'(occ)' if det.occluded else ''} {det.score:.2f} [{det.distance}]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(image, tag, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # [U4] Draw free-space corridor overlays — now with crowded state colour
        img_h, img_w = image.shape[:2]
        zone_top = int(img_h * 0.60)
        col_x = [0, int(img_w*0.33), int(img_w*0.67), img_w]
        col_names = ["left", "center", "right"]
        status_color = {
            "walkable":  (0, 200, 0),
            "crowded":   (200, 140, 0),   # orange
            "blocked":   (200, 0,   0),
            "uncertain": (100, 100, 200), # blue-grey — proceed with caution [E2]
            "unknown":   (150, 150, 0),
        }
        for i, col in enumerate(col_names):
            status = free_space.get(col, "unknown")
            color  = status_color.get(status, (150, 150, 0))
            overlay = image.copy()
            cv2.rectangle(overlay, (col_x[i], zone_top), (col_x[i+1], img_h), color, -1)
            image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
            cv2.putText(image, f"{col}: {status}",
                        (col_x[i] + 4, img_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes[0].imshow(results["image"]); axes[0].set_title("Original");     axes[0].axis("off")
        axes[1].imshow(image);            axes[1].set_title("Navigation v8 (rule-fusion + uncertain)"); axes[1].axis("off")
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, save_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Visualize] Saved: {path}")

    @staticmethod
    def _empty(image: np.ndarray) -> dict:
        # [E2] No detections at all → "uncertain" per column, not "walkable"
        # The scene is unread, not confirmed clear.
        empty_fs = {"left": "uncertain", "center": "uncertain", "right": "uncertain"}
        return {"image":image,"detections":[],"masks":[],"free_space":empty_fs,
                "boxes":[],"scores":[],"phrases":[]}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def test():
    parser = argparse.ArgumentParser(description="VLM Navigation Pipeline v7 — Tree-Aware + Fusion-Safe")
    parser.add_argument("--image",    required=True, help="Path to input image")
    parser.add_argument("--no-sam",   action="store_true", help="Skip SAM segmentation")
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--output",   default="output_v7.png")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    pipeline = VLMPipeline(max_image_size=args.max_size)
    results  = pipeline.detect_and_segment(args.image, run_sam=not args.no_sam)
    nav      = pipeline.build_navigation_description(results)

    print("\n" + "=" * 55)
    print("  NAVIGATION OUTPUT")
    print("=" * 55)
    print(f"  Action     : {nav['action']}")
    print(f"  Free-space : {nav['free_space']}")
    if nav["obstacles"]:
        print(f"  Obstacles  : {nav['obstacles']}")
    if nav["surfaces"]:
        print(f"  Surfaces   : {nav['surfaces']}")
    print(f"\n  LLM-ready  :\n  {nav['scene_text']}")
    print("=" * 55)

    pipeline.visualize(results, save_name=args.output)


if __name__ == "__main__":
    test()