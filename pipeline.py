# """
# Main VLM Pipeline — Detection + Segmentation + Depth + Free Space
# """
# from models import depth
# from navigation import free_space
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# import os
# import torchvision.transforms as T
# from collections import Counter
# from dataclasses import dataclass

# # Config imports
# from config import (
#     MULTI_PROMPTS,
#     PER_CLASS_THRESHOLDS,
#     DEFAULT_THRESHOLD,
#     OBJECT_ROLES,
#     SAM_CLASSES,
#     DINO_CONFIG,
#     DINO_CKPT,
#     SAM_CKPT,
#     OUTPUTS_DIR,
#     SOFT_NMS_SIGMA,
#     SOFT_NMS_SCORE_GATE,
#     PEDESTRIAN_SURFACES,
#     # ACCESSIBILITY_FEATURES,
#     # ACCESSIBILITY_HAZARDS,
# )
# from models.loader import load_models
# from utils.filters import (
#     Detection,
#     passes_area_filter,
#     deduplicate_cross_prompt,
#     run_occlusion_analysis,
#     soft_nms,
# )
# from utils.distance import (
#     estimate_distance,
#     estimate_distance_from_depth,
#     # estimate_metric_depth_from_map,
# )
# from utils.threshold_utils import cpu_adjust
# from navigation.free_space import analyse_free_space
# from navigation.navigation import build_navigation_description as build_nav_desc
# from navigation.navigation import build_walkgpt_description as build_walkgpt_desc
# from config.prompts import sanitise_prompt


# class VLMPipeline:
#     def __init__(
#         self,
#         soft_nms_sigma: float = SOFT_NMS_SIGMA,
#         max_image_size: int = 800,
#         enable_depth: bool = False,
#         depth_model_type: str = "MiDaS_small",
#     ):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.soft_nms_sigma = soft_nms_sigma
#         self.max_image_size = max_image_size
#         self.enable_depth = enable_depth
#         self.depth_model_type = depth_model_type
#         self.depth_estimator = None

#         print(f"[Pipeline] Device        : {self.device}")
#         print(f"[Pipeline] Max image size: {max_image_size}")
#         print(f"[Pipeline] Depth enabled : {enable_depth}")

#         self._load_models()

#     def _load_models(self):
#         """Load GroundingDINO, SAM, and optionally MiDaS depth models."""
#         for path, name in [(DINO_CONFIG, "DINO config"), (DINO_CKPT, "DINO weights"), (SAM_CKPT, "SAM weights")]:
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"Missing {name}: {path}")

#         self.dino_model, self.sam_predictor, self.cpu_fallback = load_models(
#             DINO_CONFIG, DINO_CKPT, SAM_CKPT, self.device
#         )

#         if self.enable_depth:
#             from models.depth import DepthEstimator
#             self.depth_estimator = DepthEstimator(
#                 model_type=self.depth_model_type,
#                 device=self.device,
#             )
#             if not self.depth_estimator.load():
#                 print("[Pipeline] Depth estimation disabled (model load failed)")
#                 self.depth_estimator = None
#                 self.enable_depth = False

#     @staticmethod
#     def _decode_boxes(raw_boxes: np.ndarray, img_h: int, img_w: int) -> list[list[int]]:
#         out = []
#         for cx, cy, bw, bh in raw_boxes:
#             x1 = max(0, int((cx - bw / 2) * img_w))
#             y1 = max(0, int((cy - bh / 2) * img_h))
#             x2 = min(img_w, int((cx + bw / 2) * img_w))
#             y2 = min(img_h, int((cy + bh / 2) * img_h))
#             out.append([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)])
#         return out

#     def _expand_box(self, box: list, img_h: int, img_w: int, scale: float = 1.15) -> list:
#         x1, y1, x2, y2 = box
#         cx = (x1 + x2) / 2
#         cy = (y1 + y2) / 2
#         w = (x2 - x1) * scale
#         h = (y2 - y1) * scale
#         return [
#             max(0, int(cx - w / 2)),
#             max(0, int(cy - h / 2)),
#             min(img_w, int(cx + w / 2)),
#             min(img_h, int(cy + h / 2))
#         ]

#     def _run_gdino(self, image_tensor, prompt, box_thr, text_thr):
#         from groundingdino.util.inference import predict
#         with torch.no_grad():
#             raw_boxes, logits, phrases = predict(
#                 self.dino_model, image_tensor, caption=prompt,
#                 box_threshold=box_thr, text_threshold=text_thr, device=self.device,
#             )
#         return (
#             raw_boxes.cpu().numpy(),
#             logits.cpu().numpy(),
#             [p.strip().replace(".", "").lower() for p in phrases]
#         )

#     def detect_and_segment(self, image_path: str, run_sam: bool = True) -> dict:
#         """Run full detection and segmentation pipeline."""
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Cannot load: {image_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img_h, img_w = image.shape[:2]

#         scale = self.max_image_size / max(img_h, img_w)
#         if scale < 1.0:
#             image = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))
#             img_h, img_w = image.shape[:2]

#         transform = T.Compose([
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])
#         image_tensor = transform(Image.fromarray(image)).to(torch.float32).to(self.device)

#         # ── Depth estimation ──────────────────────────────────────────────────────
#         depth_map = None
#         if self.enable_depth and self.depth_estimator is not None:
#             print("[Pipeline] Running MiDaS depth estimation...")
#             depth_map = self.depth_estimator.estimate_depth_map(image)
#             if depth_map is not None:
#                 print(f"  Depth map: {depth_map.shape}, range [{depth_map.min():.2f}, {depth_map.max():.2f}]")

#         print("[Pipeline] Running Multi-Pass GDINO...")
#         all_boxes_raw: list = []
#         all_scores: list = []
#         all_labels: list = []

#         # Dynamic thresholds for CPU fallback
#         if self.cpu_fallback:
#             threshold_ladder = [0.15, 0.10, 0.07]
#         else:
#             from config.thresholds import THRESHOLD_LADDER
#             threshold_ladder = THRESHOLD_LADDER

#         for single_prompt in MULTI_PROMPTS:
#             p = sanitise_prompt(single_prompt)
#             for box_thr in threshold_ladder:
#                 text_thr = box_thr * 1.2
#                 rb, sc, lb = self._run_gdino(image_tensor, p, box_thr, text_thr)
#                 if len(sc) > 0:
#                     print(f"  ['{single_prompt}'] {len(sc)} det(s) (max={sc.max():.2f}, thr={box_thr:.2f})")
#                     all_boxes_raw.extend(rb)
#                     all_scores.extend(sc.tolist())
#                     all_labels.extend(lb)
#                     break
#             else:
#                 print(f"  ['{single_prompt}'] 0 detections at all thresholds")

#         if not all_boxes_raw:
#             print("[Pipeline] ✗ No detections.")
#             return self._empty(image, depth_map)

#         boxes_xyxy = self._decode_boxes(np.array(all_boxes_raw), img_h, img_w)
#         boxes_xyxy, all_scores, all_labels = deduplicate_cross_prompt(boxes_xyxy, all_scores, all_labels)

#         # ── Confidence filtering ────────────────────────────────────────────
#         step1: list[Detection] = []
#         for box, score, label in zip(boxes_xyxy, all_scores, all_labels):
#             thr = cpu_adjust(PER_CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD))
#             if score >= thr:
#                 area = (box[2] - box[0]) * (box[3] - box[1])
#                 det = Detection(box=box, score=float(score), label=label, area=area)
#                 det.role = OBJECT_ROLES.get(label, "unknown")
#                 step1.append(det)
#         print(f"  After conf filter   : {len(step1)}/{len(all_scores)}")

#         # ── Area filter ─────────────────────────────────────────────────────
#         step2: list[Detection] = []
#         for det in step1:
#             ok, reason = passes_area_filter(det.box, img_h, img_w, det.label)
#             if ok:
#                 step2.append(det)
#             else:
#                 print(f"  [area-drop] '{det.label}' -- {reason}")
#         print(f"  After area filter   : {len(step2)}/{len(step1)}")

#         if not step2:
#             return self._empty(image, depth_map)

#         # ── Occlusion + NMS ─────────────────────────────────────────────────
#         step2 = run_occlusion_analysis(step2)
#         step3 = soft_nms(step2, sigma=self.soft_nms_sigma, score_gate=SOFT_NMS_SCORE_GATE)
#         print(f"  After Soft-NMS      : {len(step3)}/{len(step2)}")

#         # ── Distance estimation ─────────────────────────────────────────────
#         img_area = img_h * img_w
#         for det in step3:
#             if depth_map is not None:
#                 # Use depth-map based distance (WalkGPT-style)
#                 det.distance = estimate_distance_from_depth(det, depth_map)
#             else:
#                 # Fallback to heuristic
#                 det.distance = estimate_distance(det, img_h, img_w, img_area)

#         print(f"  Final: {dict(Counter(d.label for d in step3))}")

#         # ── SAM segmentation ────────────────────────────────────────────────
#         masks: list = []
#         mask_map: dict[int, np.ndarray] = {}

#         if run_sam:
#             self.sam_predictor.set_image(image)

#         for idx, det in enumerate(step3):
#             if not (run_sam and det.label in SAM_CLASSES):
#                 masks.append(None)
#                 continue
#             if det.score < 0.3:
#                 masks.append(None)
#                 continue

#             box_exp = self._expand_box(det.box, img_h, img_w, scale=1.15)
#             mask, _, _ = self.sam_predictor.predict(
#                 box=np.array(box_exp, dtype=np.float32), multimask_output=False,
#             )
#             mask = mask[0]
#             ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

#             # Allow large masks for surfaces
#             if det.label in PEDESTRIAN_SURFACES:
#                 if ratio < 0.01:
#                     print(f"[SAM-drop] '{det.label}' too small ratio={ratio:.2f}")
#                     masks.append(None)
#                     continue
#             else:
#                 if ratio > 0.95 or ratio < 0.02:
#                     print(f"[SAM-drop] '{det.label}' bad ratio={ratio:.2f}")
#                     masks.append(None)
#                     continue

#             masks.append(mask)
#             mask_map[idx] = mask

#         # ── Free-space analysis ────────────────────────────────────────────
#         free_space, col_bounds = analyse_free_space(image, step3, masks)
#         print(f"  Free-space: {free_space}")

#         return {
#             "image": image,
#             "detections": step3,
#             "masks": masks,
#             "mask_map": mask_map,
#             "col_bounds": col_bounds,
#             "free_space": free_space,
#             "depth_map": depth_map,
#             "boxes": [d.box for d in step3],
#             "scores": [d.score for d in step3],
#             "phrases": [d.label for d in step3],
#         }

#     def build_navigation_description(
#         self,
#         results: dict,
#         use_llm: bool = False,
#         llm_client = None,
#     ) -> dict:
#         """Build navigation description with optional LLM enhancement."""
#         return build_nav_desc(results, use_llm, llm_client)

#     def build_walkgpt_description(self, results: dict) -> dict:
#         """WalkGPT-style rich navigation output."""
#         return build_walkgpt_desc(results)

#     def visualize(self, results: dict, save_name: str = "output.png", show_depth: bool = False):
#         """Visualize detection + free-space + depth (improved debug view)."""

#         if not results or results.get("image") is None:
#             return

#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt

#         image = results["image"].copy()
#         dets = results.get("detections", [])
#         masks = results.get("masks", [])
#         free_space = results.get("free_space", {})
#         depth_map = results.get("depth_map", None)

#         # ─────────────────────────────
#         # COLOR SETUP
#         # ─────────────────────────────
#         unique_labels = list({d.label for d in dets})
#         cmap = matplotlib.colormaps.get_cmap("tab20")
#         colors = {
#             lbl: tuple(int(c * 255) for c in cmap(i / max(len(unique_labels), 1))[:3])
#             for i, lbl in enumerate(unique_labels)
#         }

#         from config import OBJECT_ROLES
#         surface_colors = {
#             "walkable": np.array([80, 200, 80]),
#             "semi_walkable": np.array([200, 160, 60]),
#             "non_walkable": np.array([200, 60, 60]),
#             "accessibility_hazard": np.array([255, 50, 50]),
#         }

#         # ─────────────────────────────
#         # SURFACE OVERLAY
#         # ─────────────────────────────
#         for mask, det in zip(masks, dets):
#             if mask is None:
#                 continue
#             role = OBJECT_ROLES.get(det.label, "unknown")
#             if role not in surface_colors:
#                 continue
#             if det.label != "sidewalk" and mask.sum() / mask.size > 0.85:
#                 continue

#             sc = surface_colors[role]
#             image[mask] = (image[mask] * 0.55 + sc * 0.45).astype(np.uint8)
#         object_colors = {
#             "person": (0, 0, 255),          # Red
#             "car": (0, 165, 255),           # Orange
#             "traffic cone": (0, 255, 255),  # Yellow
#             "bicycle": (255, 0, 0),         # Blue
#             "tree": (0, 200, 0),            # Green
#             "pothole": (255, 0, 255),       # Magenta
#             "barrier": (128, 0, 255),
#         }
#         default_color = (180, 180, 180)

#         color = object_colors.get(det.label, default_color)

#         # ─────────────────────────────
#         # DRAW DETECTIONS
#         # ─────────────────────────────
#         for det in dets:
#             x1, y1, x2, y2 = det.box

#             # distance intensity
#             if det.distance == "near":
#                 factor = 1.0
#             elif det.distance == "mid":
#                 factor = 0.7
#             else:
#                 factor = 0.5

#             base_color = object_colors.get(det.label, default_color)
#             color = tuple(int(c * factor) for c in base_color)

#             thickness = 3 if not det.occluded else 1

#             cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

#             tag = f"{det.label} {det.score:.2f} [{det.distance}]"
#             (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
#             cv2.putText(image, tag, (x1 + 2, y1 - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         # ─────────────────────────────
#         # FREE SPACE OVERLAY
#         # ─────────────────────────────
#         from navigation.free_space import _merge_surface_masks
#         from utils.geometry import extract_sidewalk_centerline, compute_corridor_bounds

#         img_h, img_w = image.shape[:2]
#         zone_top = int(img_h * 0.60)

#         sidewalk_mask = _merge_surface_masks(dets, masks, PEDESTRIAN_SURFACES)
#         centerline_fn, path_width_px, _ = extract_sidewalk_centerline(
#             sidewalk_mask, zone_top, img_h, img_w
#         )

#         col_bounds = compute_corridor_bounds(
#             sidewalk_mask, centerline_fn, path_width_px, img_h, img_w, zone_top
#         )

#         status_color = {
#             "walkable": (0, 100, 0),        # Dark Green
#             "crowded": (100, 100, 0),       # Dark Yellow
#             "blocked": (100, 0, 0),         # Dark Red
#             "uncertain": (80, 80, 120),     # Dark Purple
#             "unknown": (80, 80, 80),        # Dark Gray
#         }

#         for col, (cx1, cx2) in col_bounds.items():
#             status = free_space.get(col, "unknown")
#             color = status_color.get(status, (150, 150, 0))

#             overlay = image.copy()
#             cv2.rectangle(overlay, (cx1, zone_top), (cx2, img_h), color, -1)
#             image = cv2.addWeighted(overlay, 0.15, image, 0.85, 0)

#         # ─────────────────────────────
#         # DEPTH VISUALIZATION
#         # ─────────────────────────────
#         has_depth = show_depth and depth_map is not None
#         n_panels = 3 if has_depth else 2

#         fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 8))
#         axes = list(axes)

#         # Original
#         axes[0].imshow(results["image"])
#         axes[0].set_title("Original")
#         axes[0].axis("off")

#         # Detection + navigation
#         axes[1].imshow(image)
#         axes[1].set_title("Detection + Navigation")
#         axes[1].axis("off")

#         if has_depth:
#             from models.depth import DepthEstimator

#             # Heatmap
#             depth_heatmap = DepthEstimator.depth_to_colormap_static(
#                 depth_map,
#                 colormap=cv2.COLORMAP_TURBO
#             )

#             # Overlay with image (IMPORTANT)
#             overlay = cv2.addWeighted(results["image"], 0.5, depth_heatmap, 0.5, 0)

#             # Legend
#             h, w = overlay.shape[:2]
#             legend = np.linspace(1, 0, h).reshape(h, 1)
#             legend = np.repeat(legend, 30, axis=1)
#             legend_color = DepthEstimator.depth_to_colormap_static(legend)

#             overlay = np.hstack([overlay, legend_color])

#             cv2.putText(overlay, "FAR", (w + 5, 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#             cv2.putText(overlay, "NEAR", (w + 5, h - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#             axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#             axes[2].set_title("Depth Overlay (Blue=near → Red=far)")
#             axes[2].axis("off")

#         # ─────────────────────────────
#         # SAVE
#         # ─────────────────────────────
#         plt.tight_layout()
#         path = os.path.join(OUTPUTS_DIR, save_name)
#         plt.savefig(path, bbox_inches="tight", dpi=150)
#         plt.close()

#         print(f"[Visualize] Saved: {path}")

#     @staticmethod
#     def _empty(image: np.ndarray, depth_map=None) -> dict:
#         empty_fs = {"left": "uncertain", "center": "uncertain", "right": "uncertain"}
#         return {
#             "image": image,
#             "detections": [],
#             "masks": [],
#             "mask_map": {},
#             "col_bounds": None,
#             "free_space": empty_fs,
#             "depth_map": depth_map,
#             "boxes": [],
#             "scores": [],
#             "phrases": [],
#         }


"""
Main VLM Pipeline — FIXED VERSION

Key fixes:
1. Deduplicated detection logging (no repeated prompt prints)
2. Surface conflict resolution (sidewalk vs road)
3. Better SAM mask filtering (sidewalk masks no longer dropped)
4. Improved visualization (consistent colors, visible masks)
5. Combined depth+heuristic distance (safety-first)
6. Detection count preserved better through filtering pipeline
"""
import cv2
import torch
import numpy as np
from PIL import Image
import os
import torchvision.transforms as T
from collections import Counter
from dataclasses import dataclass

from config import (
    MULTI_PROMPTS,
    PER_CLASS_THRESHOLDS,
    DEFAULT_THRESHOLD,
    OBJECT_ROLES,
    SAM_CLASSES,
    DINO_CONFIG,
    DINO_CKPT,
    SAM_CKPT,
    OUTPUTS_DIR,
    SOFT_NMS_SIGMA,
    SOFT_NMS_SCORE_GATE,
    PEDESTRIAN_SURFACES,
)
from models.loader import load_models
from utils.filters import (
    Detection,
    passes_area_filter,
    deduplicate_cross_prompt,
    run_occlusion_analysis,
    soft_nms,
)
from utils.distance import estimate_distance_combined
from utils.threshold_utils import cpu_adjust
from navigation.free_space import analyse_free_space
from navigation.navigation import (
    build_navigation_description as build_nav_desc,
    build_walkgpt_description as build_walkgpt_desc,
)
from config.prompts import sanitise_prompt


class VLMPipeline:
    def __init__(
        self,
        soft_nms_sigma: float = SOFT_NMS_SIGMA,
        max_image_size: int = 800,
        enable_depth: bool = False,
        depth_model_type: str = "MiDaS_small",
    ):
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_nms_sigma  = soft_nms_sigma
        self.max_image_size  = max_image_size
        self.enable_depth    = enable_depth
        self.depth_model_type = depth_model_type
        self.depth_estimator = None

        print(f"[Pipeline] Device        : {self.device}")
        print(f"[Pipeline] Max image size: {max_image_size}")
        print(f"[Pipeline] Depth enabled : {enable_depth}")

        self._load_models()

    def _load_models(self):
        for path, name in [
            (DINO_CONFIG, "DINO config"),
            (DINO_CKPT,   "DINO weights"),
            (SAM_CKPT,    "SAM weights"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name}: {path}")

        self.dino_model, self.sam_predictor, self.cpu_fallback = load_models(
            DINO_CONFIG, DINO_CKPT, SAM_CKPT, self.device
        )

        if self.enable_depth:
            from models.depth import DepthEstimator
            self.depth_estimator = DepthEstimator(
                model_type=self.depth_model_type,
                device=self.device,
            )
            if not self.depth_estimator.load():
                print("[Pipeline] Depth estimation disabled (model load failed)")
                self.depth_estimator = None
                self.enable_depth    = False

    @staticmethod
    def _decode_boxes(raw_boxes: np.ndarray, img_h: int, img_w: int) -> list[list[int]]:
        out = []
        for cx, cy, bw, bh in raw_boxes:
            x1 = max(0, int((cx - bw / 2) * img_w))
            y1 = max(0, int((cy - bh / 2) * img_h))
            x2 = min(img_w, int((cx + bw / 2) * img_w))
            y2 = min(img_h, int((cy + bh / 2) * img_h))
            out.append([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)])
        return out

    def _expand_box(self, box: list, img_h: int, img_w: int, scale: float = 1.15) -> list:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = (x2 - x1) * scale
        h  = (y2 - y1) * scale
        return [
            max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
            min(img_w, int(cx + w / 2)), min(img_h, int(cy + h / 2)),
        ]

    def _run_gdino(self, image_tensor, prompt: str, box_thr: float, text_thr: float):
        from groundingdino.util.inference import predict
        with torch.no_grad():
            raw_boxes, logits, phrases = predict(
                self.dino_model, image_tensor, caption=prompt,
                box_threshold=box_thr, text_threshold=text_thr, device=self.device,
            )
        return (
            raw_boxes.cpu().numpy(),
            logits.cpu().numpy(),
            [p.strip().replace(".", "").lower() for p in phrases],
        )

    def detect_and_segment(self, image_path: str, run_sam: bool = True) -> dict:
        """Run full detection and segmentation pipeline."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        # ── Depth estimation ────────────────────────────────────────────────
        depth_map = None
        if self.enable_depth and self.depth_estimator is not None:
            print("[Pipeline] Running MiDaS depth estimation...")
            depth_map = self.depth_estimator.estimate_depth_map(image)
            if depth_map is not None:
                print(f"  Depth map: {depth_map.shape}, "
                      f"range [{depth_map.min():.2f}, {depth_map.max():.2f}]")

        # ── GDINO multi-pass ────────────────────────────────────────────────
        print("[Pipeline] Running Multi-Pass GDINO...")
        all_boxes_raw: list = []
        all_scores:    list = []
        all_labels:    list = []

        if self.cpu_fallback:
            threshold_ladder = [0.15, 0.10, 0.07]
        else:
            from config.thresholds import THRESHOLD_LADDER
            threshold_ladder = THRESHOLD_LADDER

        # FIXED: Track detections per prompt — deduplicated logging
        prompt_summary = {}

        for single_prompt in MULTI_PROMPTS:
            p = sanitise_prompt(single_prompt)
            detected_at_thr = None
            for box_thr in threshold_ladder:
                text_thr = box_thr * 1.2
                rb, sc, lb = self._run_gdino(image_tensor, p, box_thr, text_thr)
                if len(sc) > 0:
                    detected_at_thr = (box_thr, len(sc), float(sc.max()))
                    all_boxes_raw.extend(rb)
                    all_scores.extend(sc.tolist())
                    all_labels.extend(lb)
                    break

            # FIXED: Single log per prompt (not one per threshold attempt)
            if detected_at_thr:
                prompt_summary[single_prompt] = (
                    f"{detected_at_thr[1]} det(s) "
                    f"(max={detected_at_thr[2]:.2f}, thr={detected_at_thr[0]:.2f})"
                )
            else:
                prompt_summary[single_prompt] = "0 detections"

        # Print summary once
        for prompt_name, summary in prompt_summary.items():
            print(f"  ['{prompt_name}'] {summary}")

        if not all_boxes_raw:
            print("[Pipeline] ✗ No detections.")
            return self._empty(image, depth_map)

        boxes_xyxy = self._decode_boxes(np.array(all_boxes_raw), img_h, img_w)

        # FIXED: Surface conflict resolution happens inside deduplicate_cross_prompt
        boxes_xyxy, all_scores, all_labels = deduplicate_cross_prompt(
            boxes_xyxy, all_scores, all_labels
        )

        # ── Confidence filtering ────────────────────────────────────────────
        img_area = img_h * img_w
        step1: list[Detection] = []
        for box, score, label in zip(boxes_xyxy, all_scores, all_labels):
            thr = cpu_adjust(PER_CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD))
            if score >= thr:
                area = (box[2] - box[0]) * (box[3] - box[1])
                det  = Detection(
                    box=box, score=float(score), label=label,
                    area=area, area_ratio=area / max(img_area, 1),
                )
                det.role = OBJECT_ROLES.get(label, "unknown")
                step1.append(det)
        print(f"  After conf filter   : {len(step1)}/{len(all_scores)}")

        # ── Area filter ─────────────────────────────────────────────────────
        step2: list[Detection] = []
        for det in step1:
            ok, reason = passes_area_filter(det.box, img_h, img_w, det.label)
            if ok:
                step2.append(det)
            else:
                print(f"  [area-drop] '{det.label}' -- {reason}")
        print(f"  After area filter   : {len(step2)}/{len(step1)}")

        if not step2:
            return self._empty(image, depth_map)

        # ── Occlusion + NMS ─────────────────────────────────────────────────
        step2 = run_occlusion_analysis(step2)
        step3 = soft_nms(step2, sigma=self.soft_nms_sigma, score_gate=SOFT_NMS_SCORE_GATE)
        print(f"  After Soft-NMS      : {len(step3)}/{len(step2)}")

        # ── Distance estimation — FIXED: combined depth+heuristic ───────────
        for det in step3:
            det.distance = estimate_distance_combined(
                det, img_h, img_w, img_area, depth_map
            )

        label_counts = dict(Counter(d.label for d in step3))
        print(f"  Final detections    : {label_counts}")

        # ── SAM segmentation ────────────────────────────────────────────────
        masks:    list           = []
        mask_map: dict[int, np.ndarray] = {}

        if run_sam:
            self.sam_predictor.set_image(image)

        for idx, det in enumerate(step3):
            if not (run_sam and det.label in SAM_CLASSES):
                masks.append(None)
                continue
            if det.score < 0.25:   # FIXED: was 0.30, too strict for surfaces
                masks.append(None)
                continue

            box_exp = self._expand_box(det.box, img_h, img_w, scale=1.15)
            mask_pred, _, _ = self.sam_predictor.predict(
                box=np.array(box_exp, dtype=np.float32),
                multimask_output=False,
            )
            mask  = mask_pred[0]
            ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

            # FIXED: Separate thresholds for surfaces vs objects
            if det.label in PEDESTRIAN_SURFACES:
                # Sidewalk/crosswalk masks: allow very small to very large
                if ratio < 0.005:
                    print(f"[SAM-drop] '{det.label}' too small ratio={ratio:.3f}")
                    masks.append(None)
                    continue
                # No upper limit for pedestrian surfaces
            else:
                # Object masks: reject if covering whole image or tiny
                if ratio > 0.92 or ratio < 0.01:
                    print(f"[SAM-drop] '{det.label}' bad ratio={ratio:.3f}")
                    masks.append(None)
                    continue

            masks.append(mask)
            mask_map[idx] = mask

        # ── Free-space analysis ─────────────────────────────────────────────
        free_space_result, col_bounds = analyse_free_space(image, step3, masks)
        print(f"  Free-space          : {free_space_result}")

        return {
            "image":      image,
            "detections": step3,
            "masks":      masks,
            "mask_map":   mask_map,
            "col_bounds": col_bounds,
            "free_space": free_space_result,
            "depth_map":  depth_map,
            "boxes":      [d.box    for d in step3],
            "scores":     [d.score  for d in step3],
            "phrases":    [d.label  for d in step3],
        }

    def build_navigation_description(
        self,
        results: dict,
        use_llm: bool = False,
        llm_client=None,
    ) -> dict:
        return build_nav_desc(results, use_llm, llm_client)

    def build_walkgpt_description(self, results: dict) -> dict:
        return build_walkgpt_desc(results)

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATION — FIXED
    # ─────────────────────────────────────────────────────────────────────────
    def visualize(
        self,
        results: dict,
        save_name: str = "output.png",
        show_depth: bool = False,
    ):
        if not results or results.get("image") is None:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        image      = results["image"].copy()
        dets       = results.get("detections", [])
        masks      = results.get("masks", [])
        free_space = results.get("free_space", {})
        depth_map  = results.get("depth_map", None)
        img_h, img_w = image.shape[:2]

        # ── FIXED surface overlay colors ────────────────────────────────────
        surface_colors = {
            "walkable":             np.array([80, 200, 80]),     # green
            "semi_walkable":        np.array([180, 140, 40]),    # amber
            "non_walkable":         np.array([200, 60, 60]),     # red
            "accessibility_hazard": np.array([255, 50, 200]),    # magenta
        }

        # FIXED: Draw surface masks first, then objects on top
        for mask, det in zip(masks, dets):
            if mask is None:
                continue
            role = OBJECT_ROLES.get(det.label, "unknown")
            if role not in surface_colors:
                continue
            sc = surface_colors[role]
            # FIXED: stronger alpha for sidewalk (0.45 → 0.55) so it's visible
            alpha = 0.55 if det.label in PEDESTRIAN_SURFACES else 0.40
            image[mask] = (image[mask] * (1 - alpha) + sc * alpha).astype(np.uint8)

        # ── FIXED: Consistent object colors ─────────────────────────────────
        object_colors = {
            "person":       (0,   0,   255),   # Red   — highest danger
            "car":          (0,  165,  255),   # Orange
            "truck":        (0,  100,  200),   # Dark orange
            "bus":          (0,   80,  180),   # Dark orange
            "bicycle":      (255,  0,    0),   # Blue
            "motorcycle":   (200, 50,    0),   # Dark blue
            "traffic cone": (0,  255,  255),   # Yellow
            "barrier":      (128,  0,  255),   # Purple
            "tree":         (0,  160,    0),   # Dark green
            "pothole":      (255,  0,  255),   # Magenta
            "bollard":      (100, 100, 255),   # Light blue
            "bench":        (80,  80,  200),
        }
        default_obj_color = (180, 180, 180)

        for det in dets:
            x1, y1, x2, y2 = det.box

            # Distance-based intensity
            intensity = {"near": 1.0, "mid": 0.75, "far": 0.5}.get(det.distance, 0.6)
            base_color = object_colors.get(det.label, default_obj_color)
            color = tuple(int(c * intensity) for c in base_color)

            thickness = 3 if not det.occluded else 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # FIXED: Label background for readability
            tag = f"{det.label} {det.score:.2f} [{det.distance}]"
            if getattr(det, "on_path", False):
                tag = "⚠ " + tag
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(image, (x1, y1 - th - 5), (x1 + tw + 3, y1), color, -1)
            cv2.putText(image, tag, (x1 + 1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── Free-space corridor overlay ──────────────────────────────────────
        from navigation.free_space import _merge_surface_masks
        from utils.geometry import extract_sidewalk_centerline, compute_corridor_bounds

        zone_top     = int(img_h * 0.60)
        sidewalk_mask = _merge_surface_masks(dets, masks, PEDESTRIAN_SURFACES)
        centerline_fn, path_width_px, _ = extract_sidewalk_centerline(
            sidewalk_mask, zone_top, img_h, img_w
        )
        col_bounds = compute_corridor_bounds(
            sidewalk_mask, centerline_fn, path_width_px, img_h, img_w, zone_top
        )

        status_color = {
            "walkable":  (0,  120,    0),   # Dark green
            "crowded":   (120, 120,   0),   # Dark yellow
            "blocked":   (120,   0,   0),   # Dark red
            "uncertain": (80,   80, 120),   # Dark purple
            "unknown":   (80,   80,  80),   # Dark gray
        }

        for col, (cx1, cx2) in col_bounds.items():
            status = free_space.get(col, "unknown")
            color  = status_color.get(status, (120, 120, 0))
            overlay = image.copy()
            cv2.rectangle(overlay, (cx1, zone_top), (cx2, img_h), color, -1)
            image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)

            # FIXED: Add status label in corridor
            label_y = zone_top + 20
            cv2.putText(image, status.upper(), (cx1 + 3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ── Panels ──────────────────────────────────────────────────────────
        has_depth = show_depth and depth_map is not None
        n_panels  = 3 if has_depth else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 8))
        axes = list(axes)

        axes[0].imshow(results["image"])
        axes[0].set_title("Original", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(image)
        axes[1].set_title("Detection + Navigation", fontsize=13)
        axes[1].axis("off")

        if has_depth:
            from models.depth import DepthEstimator
            depth_heatmap = DepthEstimator.depth_to_colormap_static(
                depth_map, colormap=cv2.COLORMAP_TURBO
            )
            overlay = cv2.addWeighted(
                cv2.cvtColor(results["image"], cv2.COLOR_RGB2BGR), 0.45,
                depth_heatmap, 0.55, 0
            )
            # Legend
            h, w = overlay.shape[:2]
            legend = np.linspace(1, 0, h).reshape(h, 1)
            legend = np.repeat(legend, 35, axis=1)
            legend_color = DepthEstimator.depth_to_colormap_static(legend)
            overlay = np.hstack([overlay, legend_color])
            for text, y_pos in [("FAR", 20), ("NEAR", h - 10)]:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(overlay, (w + 1, y_pos - th - 2), (w + tw + 5, y_pos + 2), (0, 0, 0), -1)
                cv2.putText(overlay, text, (w + 3, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Depth Overlay  (blue=near → red=far)", fontsize=13)
            axes[2].axis("off")

        plt.tight_layout()
        path = os.path.join(OUTPUTS_DIR, save_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Visualize] Saved: {path}")

    @staticmethod
    def _empty(image: np.ndarray, depth_map=None) -> dict:
        empty_fs = {"left": "uncertain", "center": "uncertain", "right": "uncertain"}
        return {
            "image":      image,
            "detections": [],
            "masks":      [],
            "mask_map":   {},
            "col_bounds": None,
            "free_space": empty_fs,
            "depth_map":  depth_map,
            "boxes":      [],
            "scores":     [],
            "phrases":    [],
        }