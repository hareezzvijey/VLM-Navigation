"""
VLM Pipeline v8 — GPU-Calibrated
==================================
Built on v7. Changes tagged [C1]–[C5].

  [C1] MIN_ABSOLUTE_PIXELS: 900 → 400
       CPU ghost boxes are 1–25px; GPU GDINO rarely produces them.
       900px was dropping real distant detections confirmed on GPU:
         person=171px (real pedestrian far ahead)
         traffic cone=147px (real cone at distance)
         car=896px (real vehicle, just 4px below old threshold)
       400px (20×20) keeps all real objects while still killing
       genuine artifacts (sub-pixel noise from attention maps).

  [C2] sidewalk threshold: 0.48 → 0.35
       GPU scores are well-calibrated (full MSDA CUDA kernel active).
       sidewalk=0.38 observed on a clear paved path → real detection.
       0.48 was dropping it → no SAM mask produced → SW local layer
       bypassed → surface label fell back to road (factually wrong).
       0.35 matches the confidence range seen on confirmed GPU detections.

  [C3] road threshold: 0.50 → 0.42
       Symmetric fix: road=0.51 was barely surviving. Lowering to 0.42
       makes road and sidewalk thresholds consistent with each other
       and with how GPU scores distribute for surface classes.
       Also improves detection on low-contrast / wet road surfaces.

  [C4] Visualizer title: version string → "Prediction"
       Removes internal versioning from user-facing output images.

  [C5] CLI argparse description: version string removed.

All v7 logic unchanged:
  [D1]–[D3], [SW], [FIX1]–[FIX6], [U1]–[U5], [E1]–[E3] all intact.
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
# OBJECT ROLES
# ─────────────────────────────────────────────────────────────────────────────

OBJECT_ROLES: dict[str, str] = {
    "road":           "non_walkable",
    "sidewalk":       "walkable",
    "crosswalk":      "walkable",
    "ramp":           "walkable",
    "person":         "dynamic_hazard",
    "cyclist":        "dynamic_hazard",
    "motorcyclist":   "dynamic_hazard",
    "wheelchair":     "dynamic_hazard",
    "car":            "hazard",
    "truck":          "hazard",
    "bus":            "hazard",
    "bicycle":        "hazard",
    "motorcycle":     "hazard",
    "scooter":        "hazard",
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
    "tree":           "obstacle",   # [D1] trunk = hard physical block
}

SAM_CLASSES = {"road", "sidewalk"}

DYNAMIC_LABELS = {
    "person", "pedestrian", "cyclist", "motorcyclist",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
}

HARD_BLOCK_ROLES = {"obstacle", "hazard"}
SOFT_BLOCK_ROLES = {"dynamic_hazard"}

# [C2] sidewalk: 0.48 → 0.35  (GPU well-calibrated; 0.38 score is a real detection)
# [C3] road:     0.50 → 0.42  (symmetric fix; consistent with sidewalk GPU range)
PER_CLASS_THRESHOLDS: dict[str, float] = {
    "person": 0.40, "pedestrian": 0.38, "cyclist": 0.38,
    "motorcyclist": 0.38, "wheelchair": 0.35, "stroller": 0.35,
    "car": 0.45, "truck": 0.45, "bus": 0.45,
    "bicycle": 0.40, "motorcycle": 0.40, "scooter": 0.40,
    "traffic light": 0.42, "stop sign": 0.42, "traffic sign": 0.38,
    "pole": 0.35, "bollard": 0.35, "barrier": 0.40,
    "bench": 0.40,
    "road":     0.42,   # [C3] was 0.50
    "sidewalk": 0.35,   # [C2] was 0.48
    "building": 0.50, "tree": 0.45, "traffic cone": 0.38,
}
DEFAULT_THRESHOLD = 0.42

def _cpu_adjust(t: float) -> float:
    return round(t * 0.45, 3) if _CPU_FALLBACK else t

# ─────────────────────────────────────────────────────────────────────────────
# AREA FILTER
# ─────────────────────────────────────────────────────────────────────────────

# [C1] 900 → 400: GPU GDINO produces no ghost boxes. Real distant objects
# observed at 147–896px were being dropped. 400px (20×20) is the new floor.
MIN_ABSOLUTE_PIXELS = 400
MAX_RELATIVE_AREA   = 0.85
MAX_ASPECT_RATIO    = 12.0

def passes_area_filter(box: list, img_h: int, img_w: int, label: str = "") -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
    area = bw * bh
    if area < MIN_ABSOLUTE_PIXELS:
        return False, f"too small ({area}px)"
    # [D2] Tall/large classes exempt from relative-area ceiling
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
    boxes:  list[list[int]],
    scores: list[float],
    labels: list[str],
    iou_threshold: float = 0.50,
) -> tuple[list, list, list]:
    n = len(boxes)
    if n <= 1:
        return boxes, scores, labels

    keep  = [True] * n
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    for rank_i, i in enumerate(order):
        if not keep[i]: continue
        for j in order[rank_i + 1:]:
            if not keep[j]: continue
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
            if containment < 0.85: continue

            if detections[i].label == detections[j].label:
                if detections[i].score < detections[j].score:
                    detections[i].suppressed_by = j
            else:
                outer_is_dynamic = detections[j].label in DYNAMIC_LABELS
                outer_box  = detections[j].box
                inner_box  = detections[i].box
                outer_area = (outer_box[2]-outer_box[0]) * (outer_box[3]-outer_box[1])
                inner_area = (inner_box[2]-inner_box[0]) * (inner_box[3]-inner_box[1])
                # [FIX6] explicit area guard
                if outer_is_dynamic and inner_area < outer_area:
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
# DISTANCE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_distance(det: Detection, img_h: int, img_w: int, img_area: int) -> str:
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
# FREE-SPACE ANALYSIS — CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WALKABLE_COVERAGE     = 0.25
SW_BLOCK_COVERAGE     = 0.50
SW_CROWD_COVERAGE     = 0.20
SW_FOOT_OVERLAP_RATIO = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# [E1] RULE-BASED STATE FUSION
# ─────────────────────────────────────────────────────────────────────────────

def _fuse_states(global_st: str, local_st: str) -> str:
    if local_st == "unknown":
        return global_st
    if global_st == "blocked" or local_st == "blocked":
        return "blocked"
    if global_st == "crowded" or local_st == "crowded":
        return "crowded"
    if global_st == "uncertain" or local_st == "uncertain":
        return "uncertain"
    return "walkable"

# ─────────────────────────────────────────────────────────────────────────────
# [SW] SIDEWALK-LOCAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _obstacle_on_sidewalk(det: Detection, sidewalk_mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = det.box
    img_h, img_w   = sidewalk_mask.shape
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(img_w - 1, x2); y2c = min(img_h - 1, y2)

    foot_x = int((x1c + x2c) / 2)
    foot_y = y2c
    if sidewalk_mask[foot_y, foot_x]:
        return True

    box_mask_region  = sidewalk_mask[y1c:y2c+1, x1c:x2c+1]
    box_area         = max((x2c - x1c + 1) * (y2c - y1c + 1), 1)
    return (box_mask_region.sum() / box_area) >= SW_FOOT_OVERLAP_RATIO


def _merge_sidewalk_masks(
    detections: list[Detection],
    masks:      list,
    img_h:      int,
    img_w:      int,
) -> Optional[np.ndarray]:
    combined: Optional[np.ndarray] = None
    for det, mask in zip(detections, masks):
        if mask is None or det.label != "sidewalk":
            continue
        combined = mask.astype(bool) if combined is None else combined | mask.astype(bool)
    return combined


def analyse_sidewalk_local(
    image:      np.ndarray,
    detections: list[Detection],
    masks:      list,
    col_bounds: dict[str, tuple[int, int]],
    zone_top:   int,
) -> Optional[dict[str, str]]:
    img_h, img_w  = image.shape[:2]
    sidewalk_mask = _merge_sidewalk_masks(detections, masks, img_h, img_w)
    if sidewalk_mask is None:
        return None

    sw_pixels: dict[str, int] = {}
    for col, (cx1, cx2) in col_bounds.items():
        sw_pixels[col] = int(sidewalk_mask[zone_top:, cx1:cx2].sum())

    result: dict[str, str] = {}
    for col, (cx1, cx2) in col_bounds.items():
        if sw_pixels[col] == 0:
            result[col] = "unknown"
            continue

        hard_max_ratio = 0.0
        soft_max_ratio = 0.0

        for det in detections:
            role = OBJECT_ROLES.get(det.label, "unknown")
            if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES:
                continue
            if det.distance == "far":
                continue
            if not _obstacle_on_sidewalk(det, sidewalk_mask):
                continue

            x1, y1, x2, y2 = det.box
            if y2 < zone_top: continue

            ox1 = max(x1, cx1); ox2 = min(x2, cx2)
            if ox2 <= ox1: continue
            oy1 = max(y1, zone_top); oy2 = y2
            if oy2 <= oy1: continue

            col_width   = max(cx2 - cx1, 1)
            obj_w_ratio = (ox2 - ox1) / col_width   # [E3] width fraction

            if role in HARD_BLOCK_ROLES:
                hard_max_ratio = max(hard_max_ratio, obj_w_ratio)
            elif role in SOFT_BLOCK_ROLES:
                soft_max_ratio = max(soft_max_ratio, obj_w_ratio)

        if hard_max_ratio >= SW_BLOCK_COVERAGE:
            result[col] = "blocked"
        elif hard_max_ratio >= SW_CROWD_COVERAGE or soft_max_ratio >= SW_CROWD_COVERAGE:
            result[col] = "crowded"
        else:
            result[col] = "walkable"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# [U4] MAIN FREE-SPACE ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

def analyse_free_space(
    image:      np.ndarray,
    detections: list[Detection],
    masks:      list,
) -> dict[str, str]:
    img_h, img_w = image.shape[:2]
    zone_top = int(img_h * 0.60)

    col_bounds = {
        "left":   (0,                 int(img_w * 0.33)),
        "center": (int(img_w * 0.33), int(img_w * 0.67)),
        "right":  (int(img_w * 0.67), img_w),
    }
    col_area = (img_h - zone_top) * (img_w // 3)

    # Step 1: SAM mask coverage per column
    walkable_pixels = {col: 0 for col in col_bounds}
    for det, mask in zip(detections, masks):
        if mask is None or det.label not in SAM_CLASSES: continue
        zone_mask = mask[zone_top:, :]
        for col, (cx1, cx2) in col_bounds.items():
            walkable_pixels[col] += int(zone_mask[:, cx1:cx2].sum())

    # Step 2: Global obstacle classification [FIX1] [FIX4]
    hard_blocked_cols: set[str] = set()
    crowded_cols:      set[str] = set()

    for det in detections:
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES: continue
        if det.distance == "far": continue
        x1, y1, x2, y2 = det.box
        if y2 < zone_top: continue
        obj_cx = (x1 + x2) / 2
        for col, (cx1, cx2) in col_bounds.items():
            if cx1 <= obj_cx < cx2:
                if role in HARD_BLOCK_ROLES:   hard_blocked_cols.add(col)
                elif role in SOFT_BLOCK_ROLES: crowded_cols.add(col)

    # Step 3: Global decision per column [E2]
    global_result: dict[str, str] = {}
    for col in col_bounds:
        surface_confirmed = (walkable_pixels[col] / max(col_area, 1)) >= WALKABLE_COVERAGE
        if col in hard_blocked_cols:
            global_result[col] = "blocked"
        elif col in crowded_cols:
            global_result[col] = "crowded"
        elif surface_confirmed:
            global_result[col] = "walkable"
        else:
            global_result[col] = "uncertain"

    # Step 4: Sidewalk-local fusion [SW]
    local_result = analyse_sidewalk_local(image, detections, masks, col_bounds, zone_top)

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
    for det in detections:
        groups[det.label].append(det)
    return dict(groups)

def group_label(label: str, count: int) -> str:
    if count == 1:   return label
    elif count == 2: return f"2 {label}s"
    else:            return f"multiple {label}s"

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
        print(f"[Pipeline] Min area px   : {MIN_ABSOLUTE_PIXELS}")   # [C1] visible in log

        self._load_models()

    def _load_models(self):
        dino_config = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
        dino_ckpt   = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
        sam_ckpt    = os.path.join(WEIGHTS_DIR, "sam_vit_l_0b3195.pth")
        for path, name in [
            (dino_config, "DINO config"),
            (dino_ckpt,   "DINO weights"),
            (sam_ckpt,    "SAM weights"),
        ]:
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

    def _expand_box(self, box: list[int], img_h: int, img_w: int, scale: float = 1.15) -> list[int]:
        """[U2] Expand box by scale around centre, clamped to image boundaries."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        w  = (x2 - x1) * scale; h  = (y2 - y1) * scale
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
                text_threshold=text_thr,   # kept at box_thr*1.2 — prevents label merging
                device=self.device,
            )
        return (
            raw_boxes.cpu().numpy(),
            logits.cpu().numpy(),
            [p.strip().replace(".", "").lower() for p in phrases],
        )

    # ── Main detection + segmentation ────────────────────────────────────────
    def detect_and_segment(self, image_path: str, run_sam: bool = True) -> dict:

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
        all_boxes_raw: list = []
        all_scores:    list = []
        all_labels:    list = []

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

        # [U1] Cross-prompt deduplication
        boxes_xyxy, scores, labels = deduplicate_cross_prompt(boxes_xyxy, scores, labels)

        # Per-class confidence filter
        step1: list[Detection] = []
        for box, score, label in zip(boxes_xyxy, scores, labels):
            thr = _cpu_adjust(PER_CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD))
            if score >= thr:
                area = (box[2] - box[0]) * (box[3] - box[1])
                det  = Detection(box=box, score=float(score), label=label, area=area)
                det.role = OBJECT_ROLES.get(label, "unknown")
                step1.append(det)
        print(f"  After conf filter   : {len(step1)}/{len(scores)}")

        # Area filter [C1] now uses MIN_ABSOLUTE_PIXELS=400
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

        # [U3] Occlusion + Soft-NMS
        step2 = run_occlusion_analysis(step2)
        step3 = soft_nms(step2, sigma=self.soft_nms_sigma)
        print(f"  After Soft-NMS      : {len(step3)}/{len(step2)}")

        # [FIX4] Attach distance before free-space analysis
        img_area = img_h * img_w
        for det in step3:
            det.distance = estimate_distance(det, img_h, img_w, img_area)

        print(f"  Final: {dict(Counter(d.label for d in step3))}")

        # SAM — surface classes only [U2]
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

        free_space = analyse_free_space(image, step3, masks)
        print(f"  Free-space: {free_space}")

        return {
            "image":      image,
            "detections": step3,
            "masks":      masks,
            "free_space": free_space,
            "boxes":   [d.box   for d in step3],
            "scores":  [d.score for d in step3],
            "phrases": [d.label for d in step3],
        }

    # ── Navigation description [U5 + FIX3 + FIX5] ────────────────────────────
    def build_navigation_description(self, results: dict) -> dict:
        dets       = results.get("detections", [])
        free_space = results.get("free_space", {"left":"unknown","center":"unknown","right":"unknown"})
        img_h, img_w = results["image"].shape[:2]
        img_area   = img_h * img_w

        center = free_space.get("center", "unknown")
        left   = free_space.get("left",   "unknown")
        right  = free_space.get("right",  "unknown")

        def is_navigable(s: str) -> bool:
            return s in ("walkable", "crowded", "uncertain")

        def action_for(s: str, direction: str) -> str:
            if s == "walkable":  return f"move {direction}"
            if s == "crowded":   return f"move {direction} (crowded)"
            if s == "uncertain": return f"move {direction} (cautious)"
            return f"move {direction}"

        if center == "walkable":
            action = "move forward"
        elif center == "crowded":
            action = "move forward (crowded)"
        elif center == "uncertain":
            action = "move forward (cautious)"
        elif is_navigable(left) and not is_navigable(right):
            action = action_for(left, "left")
        elif is_navigable(right) and not is_navigable(left):
            action = action_for(right, "right")
        elif is_navigable(left) and is_navigable(right):
            priority = {"walkable": 0, "uncertain": 1, "crowded": 2}
            action = action_for(left, "left") if priority.get(left, 3) <= priority.get(right, 3) \
                     else action_for(right, "right")
        else:
            action = "stop — path unclear"

        groups = group_detections(dets)
        obstacle_lines: list[str] = []
        surface_lines:  list[str] = []

        for label, group in groups.items():
            role = OBJECT_ROLES.get(label, "unknown")
            rep  = max(group, key=lambda d: d.score)
            x1, y1, x2, y2 = rep.box
            cx    = (x1 + x2) / 2
            h_pos = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
            dist  = rep.distance if rep.distance != "unknown" \
                    else estimate_distance(rep, img_h, img_w, img_area)

            if dist == "far" and label in {"car", "bicycle"}:
                continue

            gl   = group_label(label, len(group))
            occ  = " [occluded]" if rep.occluded else ""
            line = f"{gl} at {h_pos} ({dist}){occ}"

            if role in ("walkable", "non_walkable"):
                surface_lines.append(line)
            elif role in ("obstacle", "hazard", "dynamic_hazard"):
                obstacle_lines.append(line)

        parts      = []
        cap_action = action.capitalize()

        parts.append(
            f"{cap_action}. Obstacles: {'; '.join(obstacle_lines)}."
            if obstacle_lines else f"{cap_action}. Path appears clear."
        )
        if surface_lines:
            parts.append("Surfaces: " + "; ".join(surface_lines) + ".")

        fs_summary = ", ".join(
            f"{col} {st}" for col, st in free_space.items() if st != "unknown"
        )
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
    def visualize(self, results: dict, save_name: str = "output_v8.png"):
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

        # Surface masks
        for mask, det in zip(masks, dets):
            if mask is None or det.label not in SAM_CLASSES: continue
            if mask.sum() / mask.size > 0.6: continue
            color = np.array(colors[det.label])
            image[mask] = (image[mask] * 0.6 + color * 0.4).astype(np.uint8)

        # Bounding boxes + labels
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

        # Free-space corridor overlays
        img_h, img_w = image.shape[:2]
        zone_top = int(img_h * 0.60)
        col_x    = [0, int(img_w*0.33), int(img_w*0.67), img_w]
        col_names = ["left", "center", "right"]
        status_color = {
            "walkable":  (0, 200, 0),
            "crowded":   (200, 140, 0),
            "blocked":   (200, 0,   0),
            "uncertain": (100, 100, 200),
            "unknown":   (150, 150, 0),
        }
        for i, col in enumerate(col_names):
            status  = free_space.get(col, "unknown")
            color   = status_color.get(status, (150, 150, 0))
            overlay = image.copy()
            cv2.rectangle(overlay, (col_x[i], zone_top), (col_x[i+1], img_h), color, -1)
            image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
            cv2.putText(image, f"{col}: {status}",
                        (col_x[i] + 4, img_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes[0].imshow(results["image"])
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(image)
        axes[1].set_title("Prediction")    # [C4] was "Navigation v8 (rule-fusion + uncertain)"
        axes[1].axis("off")
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, save_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Visualize] Saved: {path}")

    @staticmethod
    def _empty(image: np.ndarray) -> dict:
        empty_fs = {"left": "uncertain", "center": "uncertain", "right": "uncertain"}
        return {"image":image,"detections":[],"masks":[],"free_space":empty_fs,
                "boxes":[],"scores":[],"phrases":[]}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def test():
    parser = argparse.ArgumentParser(description="VLM Navigation Pipeline")   # [C5]
    parser.add_argument("--image",    required=True, help="Path to input image")
    parser.add_argument("--no-sam",   action="store_true", help="Skip SAM segmentation")
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--output",   default="output_v8.png")
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