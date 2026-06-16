"""
VLM Pipeline v4 — Clean, Production-Ready
==========================================
Changes from your working version (based on full audit):

  BUGS FIXED
  [B1] Dangling variable: 'box = expand_box(det.box)' was outside the SAM
       loop, using the last value of 'det' from a different loop. Removed.
       The SAM loop already recomputes expand_box(det.box) per detection.

  [B2] text_thr = box_thr * 0.8  →  changed to box_thr * 1.2
       text_thr < box_thr means multiple BERT tokens match per box,
       producing merged labels like 'barrier fence railing stairs'.
       text_thr > box_thr forces only the single best-matching token.

  IMPORTS CLEANED
  [I1] 'from PIL import Image' appeared 11 times — reduced to 1.
  [I2] 'from torch import det' removed — shadowed the 'det' loop variable
       used in every for-loop in detect_and_segment() and visualize().
  [I3] 'from math import dist' removed — shadowed the 'dist' distance
       variable computed in build_scene_description().
  [I4] 'from torchvision.ops import boxes' removed — shadowed the local
       'boxes' list variable. Never used anywhere in the code.
  [I5] 'import warnings' removed — never called, not needed.
  [I6] 'from dataclasses import dataclass' and 'from typing import Optional'
       were each imported twice — reduced to one each.

  CODE QUALITY
  [Q1] MAX_SIZE moved from inside detect_and_segment() to __init__ param.
       Caller can now override: VLMPipeline(max_image_size=1024)
  [Q2] prompt=None dead parameter removed from detect_and_segment().
       Method always uses MULTI_PROMPTS; the parameter was misleading.
  [Q3] plt.cm.get_cmap() replaced with matplotlib.colormaps[] —
       the old API is removed in Matplotlib 3.11.
  [Q4] test() now uses argparse — no hardcoded absolute paths.
  [Q5] _load_models() called inside __init__ (was a separate manual call
       that could be forgotten).
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
import urllib.request
import os
import torchvision.transforms as T
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CPU FALLBACK DETECTION  (must happen before GDINO import)
# ─────────────────────────────────────────────────────────────────────────────

_CPU_FALLBACK = False

class _StderrCapture(io.StringIO):
    """Tee: write to both real stderr and our buffer simultaneously."""
    def __init__(self, real_stderr):
        super().__init__()
        self._real = real_stderr
    def write(self, s):
        self._real.write(s)
        super().write(s)
    def flush(self):
        self._real.flush()

_capture = _StderrCapture(sys.stderr)
sys.stderr = _capture

from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict

sys.stderr = _capture._real
if "Failed to load custom C++ ops" in _capture.getvalue():
    _CPU_FALLBACK = True
    print("[Pipeline] ⚠  CPU fallback detected — thresholds auto-lowered")

try:
    from torchvision.ops import nms as hard_nms
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD PROFILES
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
    "person", "car", "bicycle", "traffic cone", "barrier", "road", "sidewalk"
]

def sanitise_prompt(prompt: str) -> str:
    """Lowercase, normalise spaces around dots, guarantee trailing ' .'"""
    prompt = prompt.lower().strip()
    prompt = " . ".join(p.strip() for p in prompt.split(".") if p.strip())
    if not prompt.endswith(" ."):
        prompt += " ."
    return prompt

# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Only SAM-segment these classes (surface masking for navigation)
SAM_CLASSES = {"road", "sidewalk"}

# Per-class confidence thresholds (GPU baseline; auto-scaled for CPU)
PER_CLASS_THRESHOLDS: dict[str, float] = {
    "person": 0.40, "pedestrian": 0.38, "cyclist": 0.38,
    "motorcyclist": 0.38, "wheelchair": 0.35, "stroller": 0.35,
    "car": 0.45, "truck": 0.45, "bus": 0.45,
    "bicycle": 0.40, "motorcycle": 0.40, "scooter": 0.40,
    "traffic light": 0.42, "stop sign": 0.42, "traffic sign": 0.38,
    "pole": 0.35, "bollard": 0.35, "barrier": 0.40,
    "bench": 0.40, "road": 0.50, "sidewalk": 0.48,
    "building": 0.50, "tree": 0.45,
}
DEFAULT_THRESHOLD = 0.42

def _cpu_adjust(threshold: float) -> float:
    """Scale GPU threshold down proportionally for CPU fallback mode."""
    return round(threshold * 0.45, 3) if _CPU_FALLBACK else threshold

# ─────────────────────────────────────────────────────────────────────────────
# AREA FILTER
# ─────────────────────────────────────────────────────────────────────────────

MIN_ABSOLUTE_PIXELS = 900     # 30×30 px minimum — kills artifact ghosts
MAX_RELATIVE_AREA   = 0.85   # box cannot cover > 85% of image
MAX_ASPECT_RATIO    = 12.0   # kills horizontal sliver boxes

def passes_area_filter(box: list, img_h: int, img_w: int) -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
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
    box:          list
    score:        float
    label:        str
    area:         int            = 0
    occluded:     bool           = False
    suppressed_by: Optional[int] = None

# ─────────────────────────────────────────────────────────────────────────────
# OCCLUSION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_containment(a: list, b: list) -> float:
    """Fraction of box-A's area that lies inside box-B (0.0 – 1.0)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((ax2 - ax1) * (ay2 - ay1), 1)

def run_occlusion_analysis(detections: list[Detection]) -> list[Detection]:
    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = compute_containment(detections[i].box, detections[j].box)
            if c >= 0.85:
                if detections[i].label == detections[j].label:
                    if detections[i].score < detections[j].score:
                        detections[i].suppressed_by = j
                else:
                    detections[i].occluded = True
    return detections

# ─────────────────────────────────────────────────────────────────────────────
# SOFT-NMS
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1)

def soft_nms(
    detections: list[Detection],
    sigma: float = 0.5,
    score_gate: float = 0.20,
) -> list[Detection]:
    """Gaussian Soft-NMS: decays scores instead of hard-killing overlapping boxes."""
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
# MAIN PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class VLMPipeline:
    def __init__(
        self,
        soft_nms_sigma: float = 0.5,
        max_image_size: int   = 800,   # [Q1] was hardcoded inside detect()
    ):
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_nms_sigma = soft_nms_sigma
        self.max_image_size = max_image_size
        self.cpu_fallback   = _CPU_FALLBACK

        print(f"[Pipeline] Device         : {self.device}")
        print(f"[Pipeline] CPU fallback   : {self.cpu_fallback}")
        print(f"[Pipeline] Box threshold  : {BOX_THRESHOLD_DEFAULT}")
        print(f"[Pipeline] Max image size : {self.max_image_size}")

        self._load_models()   # [Q5] always called — can't forget it

    # ── Model loading ─────────────────────────────────────────────────────────
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

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _decode_boxes(raw_boxes: np.ndarray, img_h: int, img_w: int) -> list[list[int]]:
        """Convert GDINO normalised cxcywh → absolute pixel xyxy, clamped."""
        out = []
        for cx, cy, bw, bh in raw_boxes:
            x1 = max(0,     int((cx - bw / 2) * img_w))
            y1 = max(0,     int((cy - bh / 2) * img_h))
            x2 = min(img_w, int((cx + bw / 2) * img_w))
            y2 = min(img_h, int((cy + bh / 2) * img_h))
            out.append([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)])
        return out

    @staticmethod
    def _expand_box(box: list[int], scale: float = 1.15) -> list[int]:
        """Expand a bounding box by `scale` around its centre."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = (x2 - x1) * scale
        h  = (y2 - y1) * scale
        return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]

    def _run_gdino(
        self,
        image_tensor: torch.Tensor,
        prompt: str,
        box_thr: float,
        text_thr: float,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        with torch.no_grad():
            raw_boxes, logits, phrases = predict(
                self.dino_model, image_tensor,
                caption=prompt,
                box_threshold=box_thr,
                text_threshold=text_thr,    # [B2] now > box_thr → no label merging
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
        # [Q2] prompt= parameter removed — always uses MULTI_PROMPTS global
    ) -> dict:

        # Load & resize
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        scale = self.max_image_size / max(img_h, img_w)   # [Q1] uses self.max_image_size
        if scale < 1.0:
            image = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))
            img_h, img_w = image.shape[:2]

        # Build image tensor (float32 enforced)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(Image.fromarray(image)).to(torch.float32).to(self.device)

        # ── Multi-prompt GDINO with threshold ladder ──────────────────────
        print("[Pipeline] Running Multi-Pass GDINO...")
        all_boxes: list  = []
        all_scores: list = []
        all_labels: list = []

        for single_prompt in MULTI_PROMPTS:
            p = sanitise_prompt(single_prompt)
            for box_thr in THRESHOLD_LADDER:
                text_thr = box_thr * 1.2   # [B2] FIXED: was 0.8 → caused label merging
                rb, sc, lb = self._run_gdino(image_tensor, p, box_thr, text_thr)
                if len(sc) > 0:
                    print(f"  ['{single_prompt}'] {len(sc)} det(s) "
                          f"(max={sc.max():.2f}, thr={box_thr:.2f})")
                    all_boxes.extend(rb)
                    all_scores.extend(sc)
                    all_labels.extend(lb)
                    break
            else:
                print(f"  ['{single_prompt}'] 0 detections at all thresholds")

        if not all_boxes:
            print("[Pipeline] ✗ No detections across all prompts.")
            return self._empty(image)

        raw_boxes  = np.array(all_boxes)
        scores     = np.array(all_scores)
        labels     = all_labels
        boxes_xyxy = self._decode_boxes(raw_boxes, img_h, img_w)

        # ── Per-class confidence filter ───────────────────────────────────
        step1: list[Detection] = []
        for box, score, label in zip(boxes_xyxy, scores, labels):
            thr = _cpu_adjust(PER_CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD))
            if score >= thr:
                area = (box[2] - box[0]) * (box[3] - box[1])
                step1.append(Detection(box=box, score=float(score), label=label, area=area))
        print(f"  After confidence filter : {len(step1)}/{len(scores)}")

        # ── Area filter ───────────────────────────────────────────────────
        step2: list[Detection] = []
        for det in step1:
            ok, reason = passes_area_filter(det.box, img_h, img_w)
            if ok:
                step2.append(det)
            else:
                print(f"  [area-drop] '{det.label}' — {reason}")
        print(f"  After area filter       : {len(step2)}/{len(step1)}")

        if not step2:
            return self._empty(image)

        # ── Occlusion + Soft-NMS ──────────────────────────────────────────
        step2 = run_occlusion_analysis(step2)
        step3 = soft_nms(step2, sigma=self.soft_nms_sigma)
        print(f"  After Soft-NMS          : {len(step3)}/{len(step2)}")
        print(f"  Final: {dict(Counter(d.label for d in step3))}")

        # ── SAM segmentation (surface classes only) ───────────────────────
        # [B1] FIXED: dangling 'box = expand_box(det.box)' removed.
        #      It was placed after the confidence-filter loop (not the SAM loop)
        #      so 'det' was the last element from that filter, not from step3.
        #      The SAM loop below already calls _expand_box(det.box) per detection.
        masks: list = []
        if run_sam:
            self.sam_predictor.set_image(image)

        for det in step3:
            if not (run_sam and det.label in SAM_CLASSES):
                masks.append(None)
                continue

            # Score gate per surface type
            min_score = 0.3 if det.label in SAM_CLASSES else 0.5
            if det.score < min_score:
                masks.append(None)
                continue

            box_expanded = self._expand_box(det.box)   # [B1] correct — inside loop
            mask, _, _ = self.sam_predictor.predict(
                box=np.array(box_expanded, dtype=np.float32),
                multimask_output=False,
            )
            mask = mask[0]
            ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

            if ratio > 0.8 or ratio < 0.01:
                print(f"  [SAM-drop] '{det.label}' bad mask ratio={ratio:.2f}")
                masks.append(None)
            else:
                masks.append(mask)

        return {
            "image":      image,
            "detections": step3,
            "masks":      masks,
            # Legacy flat keys for backward compatibility
            "boxes":   [d.box   for d in step3],
            "scores":  [d.score for d in step3],
            "phrases": [d.label for d in step3],
        }

    # ── Scene description ─────────────────────────────────────────────────────
    def build_scene_description(self, results: dict) -> str:
        dets = results.get("detections", [])
        if not dets:
            return "Scene: No objects detected."

        img_h, img_w = results["image"].shape[:2]
        img_area = img_h * img_w

        # Keep highest-confidence detection per label
        best: dict[str, Detection] = {}
        for det in dets:
            if det.label not in best or det.score > best[det.label].score:
                best[det.label] = det

        lines = []
        for det in best.values():
            x1, y1, x2, y2 = det.box
            cx = (x1 + x2) / 2

            # Horizontal position (3-zone)
            if cx < img_w / 3:
                h_pos = "left"
            elif cx > 2 * img_w / 3:
                h_pos = "right"
            else:
                h_pos = "center"

            # Distance estimation
            bottom     = y2
            area_ratio = det.area / img_area

            if det.label in SAM_CLASSES:
                # Surfaces are always navigation-relevant regardless of size
                dist = "near"
            elif det.label == "traffic cone":
                if bottom > 0.6 * img_h:   dist = "near"
                elif bottom > 0.4 * img_h: dist = "mid"
                else:                       dist = "far"
            elif bottom > 0.75 * img_h:    dist = "near"
            elif bottom > 0.50 * img_h:    dist = "mid"
            elif area_ratio > 0.10:        dist = "near"  # large object, any position
            else:                           dist = "far"

            # Skip far cars/bikes — not immediately relevant for navigation
            if dist == "far" and det.label in {"car", "bicycle"}:
                continue

            occ = " [occluded]" if det.occluded else ""
            lines.append(f"{det.label} at {h_pos} ({dist}){occ}")

        return "Scene: " + " | ".join(lines) if lines else "Scene: No important objects nearby."

    # ── Visualisation ─────────────────────────────────────────────────────────
    def visualize(self, results: dict, save_name: str = "output_v4.png"):
        if not results or results.get("image") is None:
            return

        image = results["image"].copy()
        dets  = results.get("detections", [])
        masks = results.get("masks", [])

        unique_labels = list({d.label for d in dets})

        # [Q3] FIXED: deprecated plt.cm.get_cmap → matplotlib.colormaps[]
        cmap = matplotlib.colormaps.get_cmap("tab20")
        colors: dict[str, tuple] = {
            lbl: tuple(int(c * 255) for c in cmap(i / max(len(unique_labels), 1))[:3])
            for i, lbl in enumerate(unique_labels)
        }

        # Draw surface masks (road / sidewalk only)
        for mask, det in zip(masks, dets):
            if mask is None or det.label not in SAM_CLASSES:
                continue
            if mask.sum() / mask.size > 0.6:   # skip near-full-frame masks
                continue
            color = np.array(colors[det.label])
            image[mask] = (image[mask] * 0.6 + color * 0.4).astype(np.uint8)

        # Draw bounding boxes + labels
        for det in dets:
            x1, y1, x2, y2 = det.box
            color = colors[det.label]
            thickness = 3 if not det.occluded else 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            tag = f"{det.label}{'(occ)' if det.occluded else ''} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes[0].imshow(results["image"]); axes[0].set_title("Original");     axes[0].axis("off")
        axes[1].imshow(image);            axes[1].set_title("Detections v4"); axes[1].axis("off")
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, save_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Visualize] Saved: {path}")

    @staticmethod
    def _empty(image: np.ndarray) -> dict:
        return {"image": image, "detections": [], "masks": [],
                "boxes": [], "scores": [], "phrases": []}


# ─────────────────────────────────────────────────────────────────────────────
# CLI  [Q4] argparse instead of hardcoded paths
# ─────────────────────────────────────────────────────────────────────────────

def test():
    parser = argparse.ArgumentParser(description="VLM Pipeline v4")
    parser.add_argument(
        "--image", required=True,
        help="Path to input image  e.g. --image C:/Users/.../image_6.jpg"
    )
    parser.add_argument(
        "--no-sam", action="store_true",
        help="Skip SAM segmentation (faster)"
    )
    parser.add_argument(
        "--max-size", type=int, default=800,
        help="Resize longest edge to this many pixels (default: 800)"
    )
    parser.add_argument(
        "--output", default="output_v4.png",
        help="Output filename in outputs/ directory"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    pipeline = VLMPipeline(max_image_size=args.max_size)
    results  = pipeline.detect_and_segment(args.image, run_sam=not args.no_sam)
    desc     = pipeline.build_scene_description(results)

    print(f"\n{desc}\n")
    pipeline.visualize(results, save_name=args.output)


if __name__ == "__main__":
    test()