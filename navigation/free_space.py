"""
Free-space analysis: FINAL FIXED VERSION
"""

import numpy as np
from typing import Optional
from config import (
    PEDESTRIAN_SURFACES,
    # NON_WALKABLE_SURFACES,
    OBJECT_ROLES,
    HARD_BLOCK_ROLES,
    SOFT_BLOCK_ROLES,
    WALKABLE_COVERAGE,
    NON_WALKABLE_COVERAGE,
    # SW_BLOCK_COVERAGE,
    # SW_CROWD_COVERAGE,
    SW_FOOT_OVERLAP_RATIO,
)
from utils.geometry import extract_sidewalk_centerline, compute_corridor_bounds


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _merge_surface_masks(detections, masks, surface_set):
    combined = None
    for det, mask in zip(detections, masks):
        if mask is None or det.label not in surface_set:
            continue
        combined = mask.astype(bool) if combined is None else combined | mask.astype(bool)
    return combined


def _obstacle_on_sidewalk(det, sidewalk_mask):
    x1, y1, x2, y2 = det.box
    h, w = sidewalk_mask.shape

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w - 1, x2), min(h - 1, y2)

    foot_x = int((x1c + x2c) / 2)
    foot_y = y2c

    if sidewalk_mask[foot_y, foot_x]:
        return True

    region = sidewalk_mask[y1c:y2c + 1, x1c:x2c + 1]
    area = max((x2c - x1c + 1) * (y2c - y1c + 1), 1)

    return (region.sum() / area) >= SW_FOOT_OVERLAP_RATIO


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def analyse_free_space(image, detections, masks):

    img_h, img_w = image.shape[:2]
    zone_top = int(img_h * 0.60)

    sidewalk_mask = _merge_surface_masks(detections, masks, PEDESTRIAN_SURFACES)

    centerline_fn, path_width_px, _ = extract_sidewalk_centerline(
        sidewalk_mask, zone_top, img_h, img_w
    )

    col_bounds = compute_corridor_bounds(
        sidewalk_mask, centerline_fn, path_width_px, img_h, img_w, zone_top
    )

    # ─────────────────────────────
    # SURFACE COVERAGE
    # ─────────────────────────────
    walkable_pixels = {col: 0 for col in col_bounds}
    non_walkable_pixels = {col: 0 for col in col_bounds}
    semi_walkable_pixels = {col: 0 for col in col_bounds}

    for det, mask in zip(detections, masks):
        if mask is None:
            continue

        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in ("walkable", "semi_walkable", "non_walkable"):
            continue

        zone_mask = mask[zone_top:, :]

        for col, (cx1, cx2) in col_bounds.items():
            col_px = int(zone_mask[:, cx1:cx2].sum())

            if role == "walkable":
                walkable_pixels[col] += col_px
            elif role == "semi_walkable":
                semi_walkable_pixels[col] += col_px
            elif role == "non_walkable":
                non_walkable_pixels[col] += col_px

    # ─────────────────────────────
    # ON_PATH (with margin)
    # ─────────────────────────────
    cx1, cx2 = col_bounds["center"]
    margin = max(img_w * 0.08, 50)

    left_bound = max(0, cx1 - margin)
    right_bound = min(img_w, cx2 + margin)

    for det in detections:
        foot_x = (det.box[0] + det.box[2]) / 2
        det.on_path = (left_bound <= foot_x <= right_bound)

    # ─────────────────────────────
    # COLUMN CLASSIFICATION
    # ─────────────────────────────
    global_result = {}

    for col, (cx1, cx2) in col_bounds.items():

        col_area = max((img_h - zone_top) * (cx2 - cx1), 1)

        ped_cov = walkable_pixels[col] / col_area
        semi_cov = semi_walkable_pixels[col] / col_area
        nwk_cov = non_walkable_pixels[col] / col_area

        hard_count = 0
        soft_count = 0

        for det in detections:
            role = OBJECT_ROLES.get(det.label, "unknown")

            if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES:
                continue

            if det.distance not in ("near", "mid"):
                continue

            foot_x = (det.box[0] + det.box[2]) / 2
            if not (cx1 <= foot_x < cx2):
                continue

            if role in HARD_BLOCK_ROLES:
                # smarter cone handling
                if det.label == "traffic cone":
                    hard_count += 0.5   # weak blocker
                else:
                    hard_count += 1
            else:
                soft_count += 1

        # ─────────────────────────────
        # FINAL DECISION (FIXED LOGIC)
        # ─────────────────────────────

        if nwk_cov >= NON_WALKABLE_COVERAGE:
            state = "blocked"

        elif hard_count >= 2:
            state = "blocked"

        elif hard_count > 0 or soft_count >= 2:
            state = "crowded"

        elif ped_cov >= WALKABLE_COVERAGE:
            state = "walkable"

        elif semi_cov >= WALKABLE_COVERAGE:
            state = "walkable"

        else:
            state = "uncertain"

        global_result[col] = state

    return global_result, col_bounds