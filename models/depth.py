# """
# Monocular Depth Estimation via MiDaS
# """
# import torch
# import numpy as np
# import cv2
# from typing import Optional


# class DepthEstimator:
#     """Monocular depth estimator backed by MiDaS."""

#     SUPPORTED_MODELS = {"MiDaS_small", "DPT_Hybrid", "DPT_Large"}

#     def __init__(
#         self,
#         model_type: str = "MiDaS_small",
#         device: Optional[torch.device] = None,
#     ):
#         if model_type not in self.SUPPORTED_MODELS:
#             raise ValueError(
#                 f"Unknown model_type '{model_type}'. "
#                 f"Choose from: {self.SUPPORTED_MODELS}"
#             )

#         self.model_type = model_type
#         self.device = device or torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )
#         self.model = None
#         self.transform = None
#         self._loaded = False

#     def load(self) -> bool:
#         """Load MiDaS model and transforms from torch.hub."""
#         if self._loaded:
#             return True

#         try:
#             print(f"[Depth] Loading MiDaS ({self.model_type})...")
#             self.model = torch.hub.load(
#                 "intel-isl/MiDaS", self.model_type, trust_repo=True
#             )
#             self.model.to(self.device)
#             self.model.eval()

#             midas_transforms = torch.hub.load(
#                 "intel-isl/MiDaS", "transforms", trust_repo=True
#             )

#             if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
#                 self.transform = midas_transforms.dpt_transform
#             else:
#                 self.transform = midas_transforms.small_transform

#             self._loaded = True
#             print(f"[Depth] MiDaS loaded on {self.device} ✓")
#             return True

#         except Exception as e:
#             print(f"[Depth] WARNING: Failed to load MiDaS: {e}")
#             return False

#     @property
#     def is_loaded(self) -> bool:
#         return self._loaded

#     def estimate_depth_map(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
#         """Produce a relative depth map from an RGB image."""
#         if not self._loaded:
#             return None

#         input_batch = self.transform(image_rgb).to(self.device)

#         with torch.no_grad():
#             prediction = self.model(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=image_rgb.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()

#         depth_map = prediction.cpu().numpy()

#         # MiDaS outputs inverse depth (close = high value).
#         # Invert so that high value = far, matching intuition for distance.
#         depth_map = depth_map.max() - depth_map
#         # Normalize to [0, 1]
#         d_min, d_max = depth_map.min(), depth_map.max()
#         if d_max - d_min > 1e-6:
#             depth_map = (depth_map - d_min) / (d_max - d_min)
#         else:
#             depth_map = np.zeros_like(depth_map)

#         return depth_map.astype(np.float32)

#     def get_object_depth(
#         self,
#         depth_map: np.ndarray,
#         box: list,
#         mask: Optional[np.ndarray] = None,
#     ) -> float:
#         """Return the median normalised depth for an object."""
#         if mask is not None and mask.any():
#             return float(np.median(depth_map[mask]))

#         x1, y1, x2, y2 = box
#         foot_top = int(y1 + (y2 - y1) * 0.6)
#         region = depth_map[foot_top:y2, x1:x2]
#         if region.size == 0:
#             region = depth_map[y1:y2, x1:x2]
#         return float(np.median(region)) if region.size > 0 else 0.5

#     def depth_to_bucket(self, depth_val: float) -> str:
#         """Map a normalised depth value to a distance bucket."""
#         if depth_val < 0.25:
#             return "near"
#         elif depth_val < 0.55:
#             return "mid"
#         else:
#             return "far"

#     # ── IMPROVED VISUALIZATION ──────────────────────────────────────────────

#     @staticmethod
#     def depth_to_colormap_static(
#         depth_map: np.ndarray,
#         colormap: int = cv2.COLORMAP_TURBO,  # better than JET
#     ) -> np.ndarray:
#         """
#         Convert depth map → colored heatmap (better perception)

#         0 (near) → blue
#         mid      → green/yellow
#         far      → red
#         """
#         depth_map = np.clip(depth_map, 0, 1)

#         depth_u8 = (depth_map * 255).astype(np.uint8)
#         heatmap = cv2.applyColorMap(depth_u8, colormap)

#         return heatmap


#     def overlay_depth_on_image(
#         self,
#         image_bgr: np.ndarray,
#         depth_map: np.ndarray,
#         alpha: float = 0.6,
#         colormap: int = cv2.COLORMAP_TURBO,
#     ) -> np.ndarray:
#         """
#         Overlay depth heatmap on original image

#         This is what you REALLY want for debugging navigation.
#         """
#         heatmap = self.depth_to_colormap_static(depth_map, colormap)

#         overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
#         return overlay


#     def draw_depth_legend(
#         self,
#         image: np.ndarray,
#         width: int = 30,
#     ) -> np.ndarray:
#         """
#         Add side legend showing near → far mapping
#         """
#         h, w = image.shape[:2]

#         legend = np.linspace(1, 0, h).reshape(h, 1)  # top = far, bottom = near
#         legend = np.repeat(legend, width, axis=1)

#         legend_color = self.depth_to_colormap_static(legend)

#         combined = np.hstack([image, legend_color])

#         # labels
#         cv2.putText(combined, "FAR", (w + 5, 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         cv2.putText(combined, "NEAR", (w + 5, h - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         return combined


#     def visualize_depth(
#         self,
#         image_bgr: np.ndarray,
#         depth_map: np.ndarray,
#         detections: Optional[list] = None,
#         masks: Optional[list] = None,
#     ) -> np.ndarray:
#         """
#         FINAL DEBUG VISUALIZATION

#         Combines:
#         - depth overlay
#         - optional object highlighting
#         - legend
#         """

#         # Step 1: overlay depth
#         vis = self.overlay_depth_on_image(image_bgr, depth_map, alpha=0.5)

#         # Step 2: draw detections (optional but VERY useful)
#         if detections is not None:
#             for det in detections:
#                 x1, y1, x2, y2 = map(int, det.box)

#                 color = (0, 255, 0)  # default

#                 # highlight based on distance
#                 if det.distance == "near":
#                     color = (0, 0, 255)      # RED = danger
#                 elif det.distance == "mid":
#                     color = (0, 255, 255)    # YELLOW
#                 else:
#                     color = (255, 255, 0)    # CYAN

#                 cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

#                 label = f"{det.label} ({det.distance})"
#                 cv2.putText(vis, label, (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#         # Step 3: add legend
#         vis = self.draw_depth_legend(vis)

#         return vis


"""
Monocular Depth Estimation via MiDaS — FIXED

Key fixes:
1. Depth inversion comment clarified (MiDaS outputs inverse depth)
2. Better percentile-based normalization (robust to outliers)
3. Improved overlay alpha for better visibility
4. Legend always visible
"""
import torch
import numpy as np
import cv2
from typing import Optional


class DepthEstimator:
    SUPPORTED_MODELS = {"MiDaS_small", "DPT_Hybrid", "DPT_Large"}

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: Optional[torch.device] = None,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model     = None
        self.transform = None
        self._loaded   = False

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            print(f"[Depth] Loading MiDaS ({self.model_type})...")
            self.model = torch.hub.load(
                "intel-isl/MiDaS", self.model_type, trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            if self.model_type in ("DPT_Large", "DPT_Hybrid"):
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self._loaded = True
            print(f"[Depth] MiDaS loaded on {self.device} ✓")
            return True
        except Exception as e:
            print(f"[Depth] WARNING: Failed to load MiDaS: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def estimate_depth_map(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Produce a relative depth map from an RGB image.

        Returns a float32 array in [0,1] where:
          0.0 = nearest object
          1.0 = farthest object
        """
        if not self._loaded:
            return None

        input_batch = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # MiDaS outputs INVERSE depth: high value = CLOSE, low value = FAR.
        # We invert so that 0.0 = near, 1.0 = far (matches human intuition).
        depth_map = depth_map.max() - depth_map

        # FIXED: Percentile-based normalization to ignore outlier pixels
        d_lo  = np.percentile(depth_map, 2)
        d_hi  = np.percentile(depth_map, 98)
        if d_hi - d_lo > 1e-6:
            depth_map = (depth_map - d_lo) / (d_hi - d_lo)
        else:
            depth_map = np.zeros_like(depth_map)

        depth_map = np.clip(depth_map, 0.0, 1.0)
        return depth_map.astype(np.float32)

    def get_object_depth(
        self,
        depth_map: np.ndarray,
        box: list,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        if mask is not None and mask.any():
            return float(np.median(depth_map[mask]))

        x1, y1, x2, y2 = box
        foot_top = int(y1 + (y2 - y1) * 0.6)
        region   = depth_map[foot_top:y2, x1:x2]
        if region.size == 0:
            region = depth_map[y1:y2, x1:x2]
        return float(np.percentile(region.flatten(), 30)) if region.size > 0 else 0.5

    def depth_to_bucket(self, depth_val: float) -> str:
        if depth_val < 0.30:
            return "near"
        elif depth_val < 0.60:
            return "mid"
        else:
            return "far"

    # ── VISUALIZATION ──────────────────────────────────────────────────────

    @staticmethod
    def depth_to_colormap_static(
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_TURBO,
    ) -> np.ndarray:
        """
        Convert [0,1] depth map to BGR heatmap.
        0 (near) → blue/purple  |  0.5 (mid) → green  |  1 (far) → red
        Uses TURBO colormap (perceptually superior to JET).
        """
        depth_map = np.clip(depth_map, 0, 1)
        depth_u8  = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, colormap)

    def overlay_depth_on_image(
        self,
        image_bgr: np.ndarray,
        depth_map: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_TURBO,
    ) -> np.ndarray:
        """
        FIXED: alpha=0.55 so depth is clearly visible while preserving image.
        """
        heatmap = self.depth_to_colormap_static(depth_map, colormap)
        return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)

    def draw_depth_legend(self, image: np.ndarray, width: int = 35) -> np.ndarray:
        """Add side legend: top=FAR (red), bottom=NEAR (blue)."""
        h, w   = image.shape[:2]
        legend = np.linspace(1, 0, h).reshape(h, 1)
        legend = np.repeat(legend, width, axis=1)
        legend_color = self.depth_to_colormap_static(legend)

        combined = np.hstack([image, legend_color])

        # FIXED: white background label boxes for legibility
        for text, y_pos in [("FAR", 20), ("NEAR", h - 10)]:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(combined, (w + 2, y_pos - th - 2), (w + tw + 6, y_pos + 2), (0, 0, 0), -1)
            cv2.putText(combined, text, (w + 4, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return combined

    def visualize_depth(
        self,
        image_bgr: np.ndarray,
        depth_map: np.ndarray,
        detections: Optional[list] = None,
        masks: Optional[list] = None,
    ) -> np.ndarray:
        """Full debug visualization: depth overlay + detections + legend."""

        vis = self.overlay_depth_on_image(image_bgr, depth_map, alpha=0.5)

        if detections is not None:
            for det in detections:
                x1, y1, x2, y2 = map(int, det.box)
                if det.distance == "near":
                    color = (0, 0, 255)      # Red = danger
                elif det.distance == "mid":
                    color = (0, 165, 255)    # Orange
                else:
                    color = (0, 255, 0)      # Green = safe

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{det.label} [{det.distance}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
                cv2.putText(vis, label, (x1 + 1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        vis = self.draw_depth_legend(vis)
        return vis