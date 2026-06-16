"""
VLM Pipeline v5 — Navigation-Grade
====================================
Built on v4. Changes are tagged [U1]–[U5] and [DOC-SKIP].

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
    "person", "car", "bicycle", "traffic cone", "barrier", "road", "sidewalk"
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
#   WALKABLE   — surfaces you can step on
#   OBSTACLE   — static things to avoid
#   HAZARD     — dynamic things that may move into your path

OBJECT_ROLES: dict[str, str] = {
    # Walkable surfaces
    "road":       "non_walkable",   # road = car lane, not for pedestrians
    "sidewalk":   "walkable",
    "crosswalk":  "walkable",
    "ramp":       "walkable",
    # Dynamic hazards (can move)
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
    # Static obstacles
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
    "tree":           "context",
}

# Only SAM-segment these (surface masks for free-space analysis)
SAM_CLASSES = {"road", "sidewalk"}

# Dynamic labels — only these can occlude other objects [U3]
DYNAMIC_LABELS = {
    "person", "pedestrian", "cyclist", "motorcyclist",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
}

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

def passes_area_filter(box: list, img_h: int, img_w: int) -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
    area = bw * bh
    if area < MIN_ABSOLUTE_PIXELS:
        return False, f"too small ({area}px)"
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
# Problem: 'person' prompt and 'bicycle' prompt both detect the same cyclist.
# Two boxes for the same object → wrong count, double occlusion flags, noisy output.
#
# Solution: before per-class confidence filtering, do a global IoU pass.
# Any pair with IoU > 0.5 → keep the one with higher score.
# This runs on raw (box, score, label) tuples, before Detection objects are built.

def deduplicate_cross_prompt(
    boxes:  list[list[int]],
    scores: list[float],
    labels: list[str],
    iou_threshold: float = 0.50,
) -> tuple[list, list, list]:
    """
    Remove duplicate detections that arose from different prompt passes
    targeting the same physical object.

    Returns filtered (boxes, scores, labels).
    """
    n = len(boxes)
    if n <= 1:
        return boxes, scores, labels

    keep = [True] * n

    # Sort by score descending so we always keep the higher-confidence detection
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    for rank_i, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[rank_i + 1:]:
            if not keep[j]:
                continue
            if _iou(boxes[i], boxes[j]) >= iou_threshold:
                keep[j] = False   # suppress lower-score duplicate

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
# v4 problem: road box inside building box → road.occluded = True (wrong).
# The fix in the doc (add area check) is incomplete — it still marks
# static surfaces as occluded by other static surfaces.
#
# Correct logic:
#   occluded=True ONLY when:
#     (a) smaller box is ≥85% inside a larger box AND
#     (b) the OUTER (larger) box belongs to a DYNAMIC object AND
#     (c) the inner object is NOT the same category as the outer

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
                # Genuine duplicate — suppress weaker one
                if detections[i].score < detections[j].score:
                    detections[i].suppressed_by = j

            else:
                # [U3] Only mark occluded if OUTER box is a dynamic object
                # AND inner object is smaller (area check)
                outer_is_dynamic = detections[j].label in DYNAMIC_LABELS
                inner_is_smaller = detections[i].area < detections[j].area

                if outer_is_dynamic and inner_is_smaller:
                    detections[i].occluded = True
                # [U3] Static outer boxes (road, building, sky) never occlude anything

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
# [U4] MASK-BASED FREE-SPACE ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
# Doc version: box overlap on hardcoded 30–70% horizontal strip.
# Fails on: angled sidewalks, partial paths, Indian roads, stairs.
#
# v5 version: use actual SAM mask pixels.
# Analyse bottom 40% of image, divided into 3 columns (left/center/right).
# Per column:
#   WALKABLE  → sidewalk/road SAM mask covers ≥ WALKABLE_COVERAGE of column
#   BLOCKED   → any obstacle box overlaps column AND no walkable mask
#   UNKNOWN   → not enough information
#
# Returns dict: {"left": status, "center": status, "right": status}
# where status ∈ {"walkable", "blocked", "unknown"}

WALKABLE_COVERAGE = 0.25   # 25% of column must be masked as walkable surface

def analyse_free_space(
    image:      np.ndarray,
    detections: list[Detection],
    masks:      list,           # parallel to detections; None if no SAM mask
) -> dict[str, str]:
    """
    Analyse walkability of left / center / right navigation corridors.

    Returns:
        {"left": "walkable"|"blocked"|"unknown",
         "center": ..., "right": ...}
    """
    img_h, img_w = image.shape[:2]

    # Analysis zone: bottom 40% of image (where near-ground objects are)
    zone_top = int(img_h * 0.60)

    # 3 columns
    col_bounds = {
        "left":   (0,              int(img_w * 0.33)),
        "center": (int(img_w * 0.33), int(img_w * 0.67)),
        "right":  (int(img_w * 0.67), img_w),
    }

    col_area = (img_h - zone_top) * (img_w // 3)   # approx column pixel count

    # Step 1: compute walkable pixel coverage per column from SAM masks
    walkable_pixels = {col: 0 for col in col_bounds}
    for det, mask in zip(detections, masks):
        if mask is None or det.label not in SAM_CLASSES:
            continue
        # Crop mask to analysis zone
        zone_mask = mask[zone_top:, :]
        for col, (cx1, cx2) in col_bounds.items():
            col_mask = zone_mask[:, cx1:cx2]
            walkable_pixels[col] += int(col_mask.sum())

    # Step 2: check obstacle box overlap per column
    obstacle_roles = {"obstacle", "hazard", "dynamic_hazard"}
    blocked_cols: set[str] = set()

    for det in detections:
        if OBJECT_ROLES.get(det.label, "unknown") not in obstacle_roles:
            continue
        x1, y1, x2, y2 = det.box
        if y2 < zone_top:   # object entirely above analysis zone
            continue
        obj_cx = (x1 + x2) / 2
        for col, (cx1, cx2) in col_bounds.items():
            if cx1 <= obj_cx < cx2:
                blocked_cols.add(col)

    # Step 3: decision per column
    result: dict[str, str] = {}
    for col, (cx1, cx2) in col_bounds.items():
        coverage = walkable_pixels[col] / max(col_area, 1)
        if coverage >= WALKABLE_COVERAGE:
            # Walkable surface confirmed — check if an obstacle is on top
            result[col] = "blocked" if col in blocked_cols else "walkable"
        elif col in blocked_cols:
            result[col] = "blocked"
        else:
            result[col] = "unknown"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# [U5] INSTANCE GROUPING
# ─────────────────────────────────────────────────────────────────────────────
# Problem: 5 traffic cones → 5 separate description lines.
# Solution: group by label, describe collectively when count > 2.

def group_detections(detections: list[Detection]) -> dict[str, list[Detection]]:
    """Group detections by label. Returns {label: [Detection, ...]}."""
    groups: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        groups[det.label].append(det)
    return dict(groups)

def group_label(label: str, count: int) -> str:
    """
    Return a natural-language count descriptor.
    group_label('traffic cone', 4) → 'multiple traffic cones'
    group_label('person', 1)       → 'person'
    group_label('car', 2)          → '2 cars'
    """
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
        """
        [U2] Expand box by scale around centre, CLAMPED to image boundaries.
        v4 had no clamping → negative coords → SAM crash / bad masks.
        """
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
                text_thr = box_thr * 1.2   # > box_thr prevents label merging
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

        # Decode raw normalised boxes → pixel xyxy
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
                det.role = OBJECT_ROLES.get(label, "unknown")   # [U5] attach role
                step1.append(det)
        print(f"  After conf filter   : {len(step1)}/{len(scores)}")

        # ── Area filter ───────────────────────────────────────────────────────
        step2: list[Detection] = []
        for det in step1:
            ok, reason = passes_area_filter(det.box, img_h, img_w)
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
        print(f"  Final: {dict(Counter(d.label for d in step3))}")

        # ── SAM (surface classes only, with [U2] clamped expand_box) ─────────
        masks: list = []
        if run_sam:
            self.sam_predictor.set_image(image)

        for det in step3:
            if not (run_sam and det.label in SAM_CLASSES):
                masks.append(None)
                continue
            min_score = 0.3
            if det.score < min_score:
                masks.append(None)
                continue

            # [U2] expand_box now receives img_h, img_w for clamping
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

        # ── [U4] Free-space analysis ──────────────────────────────────────────
        free_space = analyse_free_space(image, step3, masks)
        print(f"  Free-space: {free_space}")

        return {
            "image":       image,
            "detections":  step3,
            "masks":       masks,
            "free_space":  free_space,        # [U4]
            # Legacy flat keys
            "boxes":   [d.box   for d in step3],
            "scores":  [d.score for d in step3],
            "phrases": [d.label for d in step3],
        }

    # ── [U5] Navigation description ──────────────────────────────────────────
    def build_navigation_description(self, results: dict) -> dict:
        """
        Returns a structured navigation dict suitable for direct LLM injection:

        {
          "free_space":   {"left": "walkable", "center": "blocked", "right": "unknown"},
          "action":       "move left",        ← primary navigation instruction
          "obstacles":    ["person at center (near)", "traffic cone at right (mid)"],
          "surfaces":     ["sidewalk at left (near)"],
          "scene_text":   "Move left. Person directly ahead. ..."   ← LLM-ready string
        }
        """
        dets       = results.get("detections", [])
        free_space = results.get("free_space", {"left":"unknown","center":"unknown","right":"unknown"})
        img_h, img_w = results["image"].shape[:2]
        img_area   = img_h * img_w

        # ── Primary action from free-space ────────────────────────────────────
        center_status = free_space.get("center", "unknown")
        left_status   = free_space.get("left",   "unknown")
        right_status  = free_space.get("right",  "unknown")

        if center_status == "walkable":
            action = "move forward"
        elif left_status == "walkable" and right_status != "walkable":
            action = "move left"
        elif right_status == "walkable" and left_status != "walkable":
            action = "move right"
        elif left_status == "walkable" and right_status == "walkable":
            action = "move forward or detour available"
        else:
            action = "stop — path unclear"

        # ── Per-detection descriptions with roles + grouping ──────────────────
        groups = group_detections(dets)

        obstacle_lines: list[str] = []
        surface_lines:  list[str] = []

        for label, group in groups.items():
            role = OBJECT_ROLES.get(label, "unknown")

            # Representative detection = highest confidence in group
            rep = max(group, key=lambda d: d.score)
            x1, y1, x2, y2 = rep.box
            cx = (x1 + x2) / 2

            h_pos = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
            bottom     = y2
            area_ratio = rep.area / img_area

            # Distance (same logic as v4, unchanged — it was correct)
            if label in SAM_CLASSES:
                dist = "near"
            elif label == "traffic cone":
                dist = "near" if bottom > 0.6*img_h else ("mid" if bottom > 0.4*img_h else "far")
            elif bottom > 0.75 * img_h:    dist = "near"
            elif bottom > 0.50 * img_h:    dist = "mid"
            elif area_ratio > 0.10:        dist = "near"
            else:                           dist = "far"

            # Skip far hazards (unchanged from v4)
            if dist == "far" and label in {"car", "bicycle"}:
                continue

            count    = len(group)
            gl       = group_label(label, count)     # [U5] grouping
            occ      = " [occluded]" if rep.occluded else ""
            line     = f"{gl} at {h_pos} ({dist}){occ}"

            if role in ("walkable", "non_walkable"):
                surface_lines.append(line)
            elif role in ("obstacle", "hazard", "dynamic_hazard"):
                obstacle_lines.append(line)
            # landmarks and context are silently skipped from navigation output

        # ── Compose scene_text for LLM ────────────────────────────────────────
        parts = []
        cap_action = action.capitalize()

        if obstacle_lines:
            avoid_str = "; ".join(obstacle_lines)
            parts.append(f"{cap_action}. Obstacles: {avoid_str}.")
        else:
            parts.append(f"{cap_action}. Path appears clear.")

        if surface_lines:
            parts.append("Surfaces: " + "; ".join(surface_lines) + ".")

        # Free-space summary
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

    # Preserve v4 build_scene_description as a fallback plain-text method
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
            tag = f"{det.label}{'(occ)' if det.occluded else ''} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(image, tag, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # [U4] Draw free-space corridor overlays (bottom 40% zone)
        img_h, img_w = image.shape[:2]
        zone_top = int(img_h * 0.60)
        col_x = [0, int(img_w*0.33), int(img_w*0.67), img_w]
        col_names = ["left", "center", "right"]
        status_color = {
            "walkable": (0, 200, 0),
            "blocked":  (200, 0, 0),
            "unknown":  (150, 150, 0),
        }
        for i, col in enumerate(col_names):
            status = free_space.get(col, "unknown")
            color  = status_color[status]
            # Semi-transparent fill on the zone strip
            overlay = image.copy()
            cv2.rectangle(overlay, (col_x[i], zone_top), (col_x[i+1], img_h), color, -1)
            image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
            # Status label at bottom
            cv2.putText(image, f"{col}: {status}",
                        (col_x[i] + 4, img_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes[0].imshow(results["image"]); axes[0].set_title("Original");     axes[0].axis("off")
        axes[1].imshow(image);            axes[1].set_title("Navigation v5"); axes[1].axis("off")
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, save_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Visualize] Saved: {path}")

    @staticmethod
    def _empty(image: np.ndarray) -> dict:
        empty_fs = {"left": "unknown", "center": "unknown", "right": "unknown"}
        return {"image":image,"detections":[],"masks":[],"free_space":empty_fs,
                "boxes":[],"scores":[],"phrases":[]}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def test():
    parser = argparse.ArgumentParser(description="VLM Navigation Pipeline v5")
    parser.add_argument("--image",    required=True, help="Path to input image")
    parser.add_argument("--no-sam",   action="store_true", help="Skip SAM segmentation")
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--output",   default="output_v5.png")
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