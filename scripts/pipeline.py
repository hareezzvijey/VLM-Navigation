from math import dist
from PIL import Image
from PIL import Image
from PIL import Image
from PIL import Image
from PIL import Image
from PIL import Image
from PIL import Image
from PIL import Image
from torchvision.ops import boxes
from PIL import Image
import cv2
import io
import sys
import torch
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os
import torchvision.transforms as T
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# [FIX-1]  DETECT CPU FALLBACK BEFORE GDINO IMPORT
# ─────────────────────────────────────────────────────────────────────────────
# GDINO prints the warning to stderr during its own module __init__.
# We capture it by wrapping stderr before the import happens.

_CPU_FALLBACK = False

class _StderrCapture(io.StringIO):
    """Tee: write to both the real stderr and our buffer."""
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

sys.stderr = _capture._real   # restore real stderr
_captured_text = _capture.getvalue()
if "Failed to load custom C++ ops" in _captured_text:
    _CPU_FALLBACK = True
    print("[Pipeline] ⚠  CPU fallback detected — thresholds will be auto-lowered")

try:
    from torchvision.ops import nms as hard_nms
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD PROFILES  (GPU vs CPU fallback)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY different thresholds?
#
# GDINO's vision backbone uses MultiScaleDeformableAttention (MSDA).
# On GPU, MSDA uses hand-written CUDA kernels → activations are well-scaled,
# sigmoid outputs for real objects cluster around 0.35–0.90.
#
# On CPU, MSDA falls back to a pure-Python slow path that skips the
# custom kernel's output normalisation step. The result: all sigmoid outputs
# compress into ~0.10–0.35 even for clearly visible objects.
#
# Threshold ladder: if the first pass returns 0 boxes, retry with lower values.

if _CPU_FALLBACK:
    BOX_THRESHOLD_DEFAULT  = 0.15   # GPU equivalent ≈ 0.30
    TEXT_THRESHOLD_DEFAULT = 0.12   # GPU equivalent ≈ 0.25
    THRESHOLD_LADDER       = [0.15, 0.10, 0.07]
else:
    BOX_THRESHOLD_DEFAULT  = 0.30
    TEXT_THRESHOLD_DEFAULT = 0.25
    THRESHOLD_LADDER       = [0.30, 0.25, 0.20]

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION PROMPT section
# ─────────────────────────────────────────────────────────────────────────────

MULTI_PROMPTS = [
    "person",
    "car",
    "bike",
    "bicycle",
    "road",
    "sidewalk",
    "tree"
]

def sanitise_prompt(prompt: str) -> str:
    prompt = prompt.lower().strip()
    prompt = " . ".join(p.strip() for p in prompt.split(".") if p.strip())
    if not prompt.endswith(" ."):
        prompt = prompt + " ."
    return prompt

# Masking Class
SAM_CLASSES = ["road", "sidewalk"]

# Per-class confidence thresholds (relative to GPU baseline)
PER_CLASS_THRESHOLDS = {
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
    """Scale a GPU threshold down proportionally for CPU fallback."""
    return round(threshold * 0.45, 3) if _CPU_FALLBACK else threshold

# ─────────────────────────────────────────────────────────────────────────────
# AREA FILTER (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────
MIN_ABSOLUTE_PIXELS = 900
MAX_RELATIVE_AREA   = 0.85
MAX_ASPECT_RATIO    = 12.0

def passes_area_filter(box, img_h, img_w):
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
    area = bw * bh; img_area = img_h * img_w
    if area < MIN_ABSOLUTE_PIXELS:
        return False, f"too small ({area}px)"
    if area / img_area > MAX_RELATIVE_AREA:
        return False, f"too large ({area/img_area:.0%})"
    if max(bw/bh, bh/bw) > MAX_ASPECT_RATIO:
        return False, f"sliver box"
    return True, ""

# ─────────────────────────────────────────────────────────────────────────────
# OCCLUSION (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Detection:
    box: list
    score: float
    label: str
    area: int = 0
    occluded: bool = False
    suppressed_by: Optional[int] = None

def compute_containment(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    return (ix2-ix1)*(iy2-iy1) / max((ax2-ax1)*(ay2-ay1), 1)

def run_occlusion_analysis(detections):
    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            c = compute_containment(detections[i].box, detections[j].box)
            if c >= 0.85:
                if detections[i].label == detections[j].label:
                    if detections[i].score < detections[j].score:
                        detections[i].suppressed_by = j
                else:
                    detections[i].occluded = True
    return detections

# ─────────────────────────────────────────────────────────────────────────────
# SOFT-NMS (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter=(ix2-ix1)*(iy2-iy1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(union,1)

def soft_nms(detections, sigma=0.5, score_gate=0.20):
    dets = [d for d in detections if d.suppressed_by is None]
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    for i in range(len(dets)):
        for j in range(i+1, len(dets)):
            overlap = _iou(dets[i].box, dets[j].box)
            if overlap > 0:
                dets[j].score *= np.exp(-(overlap**2) / sigma)
    return [d for d in dets if d.score >= score_gate]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class VLMPipeline:
    def __init__(self, soft_nms_sigma=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_nms_sigma = soft_nms_sigma
        self.cpu_fallback = _CPU_FALLBACK
        print(f"[Pipeline] Device        : {self.device}")
        print(f"[Pipeline] CPU fallback  : {self.cpu_fallback}")
        print(f"[Pipeline] Box threshold : {BOX_THRESHOLD_DEFAULT}")
        self._load_models()

    def _load_models(self):
        dino_config = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
        dino_ckpt   = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
        sam_ckpt    = os.path.join(WEIGHTS_DIR, "sam_vit_l_0b3195.pth")
        for p, n in [(dino_config,"DINO config"),(dino_ckpt,"DINO weights"),(sam_ckpt,"SAM weights")]:
            if not os.path.exists(p): raise FileNotFoundError(f"Missing {n}: {p}")
        print("[Pipeline] Loading Grounding DINO...")
        self.dino_model = load_model(dino_config, dino_ckpt, device=self.device)
        print("[Pipeline] Loading SAM...")
        sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        print("[Pipeline] Models loaded ✓")

    @staticmethod
    def _decode_boxes(raw_boxes, img_h, img_w):
        out = []
        for cx, cy, bw, bh in raw_boxes:
            x1 = max(0, int((cx - bw/2) * img_w))
            y1 = max(0, int((cy - bh/2) * img_h))
            x2 = min(img_w, int((cx + bw/2) * img_w))
            y2 = min(img_h, int((cy + bh/2) * img_h))
            if x2 > x1 and y2 > y1:
                out.append([x1, y1, x2, y2])
            else:
                out.append([x1, y1, max(x2, x1+1), max(y2, y1+1)])
        return out

    def _run_gdino(self, image_tensor, prompt, box_thr, text_thr):
        """Single GDINO inference call — returns (boxes_np, scores_np, labels)."""
        with torch.no_grad():
            raw_boxes, logits, phrases = predict(
                self.dino_model,
                image_tensor,
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

    def detect_and_segment(self, image_path, prompt=None, run_sam=True):
        # ── Load & validate image ────────────────────────────────────────────
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        MAX_SIZE = 800
        scale = MAX_SIZE/max(img_h, img_w)
        
        if scale < 1:
            image = cv2.resize(image, (int(img_w*scale), int(img_h*scale)))
        
        img_h, img_w = image.shape[:2]
        # ── [FIX-3] Force float32 tensor ────────────────────────────────────
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(Image.fromarray(image)).to(torch.float32).to(self.device)

        # ── [FIX-4] GDINO with threshold ladder ─────────────────────────────
        print("[Pipeline] Running Multi-Pass GDINO...")

        all_boxes = []
        all_scores = []
        all_labels = []

        for single_prompt in MULTI_PROMPTS:
            p = sanitise_prompt(single_prompt)

            for box_thr in THRESHOLD_LADDER:
                text_thr = box_thr * 0.8
                rb, sc, lb = self._run_gdino(image_tensor, p, box_thr, text_thr)

                if len(sc) > 0:
                    print(f"  [prompt='{single_prompt}'] → "
                        f"{len(sc)} detections (max={sc.max():.2f})")
                    all_boxes.extend(rb)
                    all_scores.extend(sc)
                    all_labels.extend(lb)
                    break
                else:
                    print(f"  [prompt='{single_prompt}'] → 0 detections at thr={box_thr:.2f}")

        if len(all_boxes) == 0:
            print("[Pipeline] ✗ No detections across all prompts.")
            return self._empty(image)

        raw_boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        labels = all_labels

        if raw_boxes is None or len(raw_boxes) == 0:
            print("[Pipeline] ✗ GDINO returned 0 boxes at all threshold levels.")
            print("            Check: (1) weights path correct?")
            print("                   (2) Image readable and non-empty?")
            print("                   (3) Prompt contains valid noun phrases?")
            return self._empty(image)

        # ── Decode cxcywh → xyxy ────────────────────────────────────────────
        boxes_xyxy = self._decode_boxes(raw_boxes, img_h, img_w)

        # ── Per-class confidence filter ──────────────────────────────────────
        step1 = []
        for box, score, label in zip(boxes_xyxy, scores, labels):
            thr = _cpu_adjust(PER_CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD))
            if score >= thr:
                area = (box[2]-box[0]) * (box[3]-box[1])
                step1.append(Detection(box=box, score=float(score), label=label, area=area))
            # else: silent drop (already visible in the probe log)
        print(f"  After confidence filter : {len(step1)}/{len(scores)} kept")

        # ── Area filter ──────────────────────────────────────────────────────
        step2 = []
        for det in step1:
            ok, reason = passes_area_filter(det.box, img_h, img_w)
            if ok:
                step2.append(det)
            else:
                print(f"  [area-drop] '{det.label}' {reason}")
        print(f"  After area filter       : {len(step2)}/{len(step1)} kept")

        if not step2:
            return self._empty(image)

        # ── Occlusion + Soft-NMS ─────────────────────────────────────────────
        step2 = run_occlusion_analysis(step2)
        step3 = soft_nms(step2, sigma=self.soft_nms_sigma)
        print(f"  After Soft-NMS          : {len(step3)}/{len(step2)} kept")
        print(f"  Final: {dict(Counter(d.label for d in step3))}")

        # ── SAM ──────────────────────────────────────────────────────────────
        # To mask everything
        # for box in boxes_xyxy:
        #     mask, _, _ = self.sam_predictor.predict(
        #         box=np.array(box),
        #         multimask_output=False
        #     )
        #     masks.append(mask[0])

        # ── SAM (FIXED) ──────────────────────────────────────────────
        masks = []

        if run_sam:
            self.sam_predictor.set_image(image)

        for det in step3:   # USE FINAL DETECTIONS ONLY
            box = det.box
            label = det.label

            if run_sam and label in SAM_CLASSES and det.score > 0.5:
                mask, _, _ = self.sam_predictor.predict(
                    box=np.array(box),
                    multimask_output=False
                )

                mask = mask[0]

                # Reject bad masks (prevents full blue screen)
                mask_area = mask.sum()
                total_area = mask.shape[0] * mask.shape[1]

                if mask_area / total_area > 0.6:
                    print(f"[SAM-drop] {label} mask too large")
                    masks.append(None)
                else:
                    masks.append(mask)

            else:
                masks.append(None)

        return {
            "image": image, "detections": step3, "masks": masks,
            "boxes": [d.box for d in step3],
            "scores": [d.score for d in step3],
            "phrases": [d.label for d in step3],
        }

    # Previous version
    # def build_scene_description(self, results):
    #     dets = results.get("detections", [])
    #     if not dets:
    #         return "Scene: No objects detected."
    #     img_h, img_w = results["image"].shape[:2]
    #     img_area = img_h * img_w
    #     lines = []
    #     for det in dets:
    #         x1, y1, x2, y2 = det.box
    #         cx = (x1+x2)/2
    #         h_pos = "left" if cx < img_w/3 else ("right" if cx > 2*img_w/3 else "center")
    #         rel = det.area / img_area
    #         dist = "near" if rel > 0.15 else ("far" if rel < 0.02 else "mid")
    #         occ  = " [occluded]" if det.occluded else ""
    #         lines.append(f"{det.label} at {h_pos} ({dist}){occ}")
    #     return "Scene: " + " | ".join(lines)

    def build_scene_description(self, results):
        dets = results.get("detections", [])
        if not dets:
            return "Scene: No objects detected."

        img_h, img_w = results["image"].shape[:2]
        img_area = img_h * img_w

        best = {}

        for det in dets:
            label = det.label

            # Keep highest score per label
            if label not in best or det.score > best[label].score:
                best[label] = det

        lines = []

        for det in best.values():
            x1, y1, x2, y2 = det.box

            cx = (x1 + x2) / 2
            h_pos = "left" if cx < img_w / 3 else (
                "right" if cx > 2 * img_w / 3 else "center"
            )

            rel = det.area / img_area
            dist = "near" if rel > 0.15 else (
                "far" if rel < 0.02 else "mid"
            )

            # ADD FILTER HERE
            if dist == "far" and det.label in ["car", "bike"]:
                continue

            occ = " [occluded]" if det.occluded else ""

            lines.append(f"{det.label} at {h_pos} ({dist}){occ}")

        return "Scene: " + " | ".join(lines)

    def visualize(self, results, save_name="output_7.png"):
        if not results or results.get("image") is None:
            return
        image = results["image"].copy()
        dets  = results.get("detections", [])
        masks = results.get("masks", [])

        palette = plt.cm.get_cmap("tab20", max(len({d.label for d in dets}), 1))
        labels_uniq = list({d.label for d in dets})
        colors = {lbl: tuple(int(c*255) for c in palette(i)[:3])
                  for i, lbl in enumerate(labels_uniq)}

        for mask, det in zip(masks, dets):

            if mask is None:
                continue

            if det.label not in ["road", "sidewalk"]:
                continue

            color = np.array(colors[det.label])

            mask_area = mask.sum()
            total_area = mask.shape[0] * mask.shape[1]

            if mask_area / total_area > 0.6:
                continue

            image[mask] = (image[mask] * 0.6 + color * 0.4).astype(np.uint8)

        for det in dets:
            x1,y1,x2,y2 = det.box
            color = colors[det.label]
            thick = 3 if not det.occluded else 1
            cv2.rectangle(image, (x1,y1),(x2,y2), color, thick)
            tag = f"{det.label}{'(occ)' if det.occluded else ''} {det.score:.2f}"
            (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
            cv2.putText(image, tag,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

        fig, axes = plt.subplots(1,2,figsize=(18,8))
        axes[0].imshow(results["image"]); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(image);            axes[1].set_title("Detections v3"); axes[1].axis("off")
        plt.tight_layout()
        path = os.path.join(OUTPUTS_DIR, save_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Visualize] Saved: {path}")

    @staticmethod
    def _empty(image):
        return {"image":image,"detections":[],"masks":[],"boxes":[],"scores":[],"phrases":[]}


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────
def test():
    pipeline = VLMPipeline()
    # url      = "http://images.cocodataset.org/val2017/000000312549.jpg"
    # img_path = os.path.join(BASE_DIR, "test.jpg")
    print(f"[Test] Downloading COCO sample...")
    # img_path = r"D:\Hareezzvijey\walkgpt_mini\examples\image.jpg"
    # img_path = r"C:\Users\Amma.DESKTOP-4K4SV7F\Downloads\sidewalk_img1.jpg"
    # img_path = r"D:\Hareezzvijey\walkgpt_mini\examples\sidewalk_images.jpg"
    # img_path = r"C:\Users\Amma.DESKTOP-4K4SV7F\Downloads\image3.jpg"
    # img_path = r"C:\Users\Amma.DESKTOP-4K4SV7F\Downloads\indian_road.jpg"
    # img_path = r"C:\Users\Amma.DESKTOP-4K4SV7F\Downloads\Sidewalk.jpg"
    img_path = r"C:\Users\Amma.DESKTOP-4K4SV7F\Downloads\image_5.jpg"
    # img_path = r"C:\Users\Amma.DESKTOP-4K4SV7F\Downloads\image_4.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    # urllib.request.urlretrieve(url, img_path)
    results  = pipeline.detect_and_segment(img_path)
    desc     = pipeline.build_scene_description(results)
    print(f"\n{desc}\n")
    pipeline.visualize(results)

if __name__ == "__main__":
    test()