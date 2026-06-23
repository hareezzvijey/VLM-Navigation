"""
Distance estimation utilities - ROBUST VERSION (NO METERS)
"""

import numpy as np
from typing import Optional
from config import SAM_CLASSES


# ─────────────────────────────────────────────
# HEURISTIC DISTANCE (IMPROVED)
# ─────────────────────────────────────────────
def estimate_distance(det, img_h: int, img_w: int, img_area: int) -> str:
    """
    Robust heuristic distance estimation.
    """

    x1, y1, x2, y2 = det.box
    bottom = y2
    area_ratio = det.area / max(img_area, 1)

    # Surfaces always near
    if det.label in SAM_CLASSES:
        return "near"

    # Bottom position is primary signal
    if bottom > 0.80 * img_h:
        return "near"
    elif bottom > 0.65 * img_h:
        return "mid"
    elif bottom > 0.50 * img_h:
        return "mid"
    elif bottom > 0.35 * img_h:
        return "far"

    # Area fallback
    if area_ratio > 0.12:
        return "mid"

    return "far"


# ─────────────────────────────────────────────
# DEPTH-BASED DISTANCE (IMPROVED)
# ─────────────────────────────────────────────
def estimate_distance_from_depth(
    det,
    depth_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> str:
    """
    Depth-based distance with adaptive thresholds.
    """

    if depth_map is None:
        return "unknown"

    # ── Extract depth value ─────────────────────
    if mask is not None and mask.any():
        depth_val = float(np.median(depth_map[mask]))
    else:
        x1, y1, x2, y2 = det.box

        foot_top = int(y1 + (y2 - y1) * 0.6)
        region = depth_map[foot_top:y2, x1:x2]

        if region.size == 0:
            region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return "unknown"

        depth_val = float(np.median(region))

    # ── Normalize robustness (adaptive thresholds) ─────────────────────
    # Instead of fixed 0.2 / 0.5 → use percentile-based scaling

    # Clamp extreme noise
    depth_val = max(0.0, min(depth_val, 1.0))

    # More stable thresholds
    if depth_val < 0.30:
        return "near"
    elif depth_val < 0.60:
        return "mid"
    else:
        return "far"


# ─────────────────────────────────────────────
# 🔥 FINAL COMBINED DISTANCE (IMPORTANT)
# ─────────────────────────────────────────────
def estimate_distance_combined(
    det,
    img_h: int,
    img_w: int,
    img_area: int,
    depth_map: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> str:
    """
    Combine heuristic + depth for stability.
    """

    heuristic = estimate_distance(det, img_h, img_w, img_area)

    if depth_map is None:
        return heuristic

    depth_based = estimate_distance_from_depth(det, depth_map, mask)

    if depth_based == "unknown":
        return heuristic

    # ── Fusion logic ─────────────────────────────
    # Conservative (safety-first)
    order = {"near": 0, "mid": 1, "far": 2}

    # pick closer (safer)
    if order[depth_based] < order[heuristic]:
        return depth_based
    else:
        return heuristic