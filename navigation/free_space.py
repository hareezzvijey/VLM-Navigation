# """
# Free-space analysis: FINAL FIXED VERSION
# """

# import numpy as np
# from typing import Optional
# from config import (
#     PEDESTRIAN_SURFACES,
#     # NON_WALKABLE_SURFACES,
#     OBJECT_ROLES,
#     HARD_BLOCK_ROLES,
#     SOFT_BLOCK_ROLES,
#     WALKABLE_COVERAGE,
#     NON_WALKABLE_COVERAGE,
#     # SW_BLOCK_COVERAGE,
#     # SW_CROWD_COVERAGE,
#     SW_FOOT_OVERLAP_RATIO,
# )
# from utils.geometry import extract_sidewalk_centerline, compute_corridor_bounds


# # ─────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────

# def _merge_surface_masks(detections, masks, surface_set):
#     combined = None
#     for det, mask in zip(detections, masks):
#         if mask is None or det.label not in surface_set:
#             continue
#         combined = mask.astype(bool) if combined is None else combined | mask.astype(bool)
#     return combined


# def _obstacle_on_sidewalk(det, sidewalk_mask):
#     x1, y1, x2, y2 = det.box
#     h, w = sidewalk_mask.shape

#     x1c, y1c = max(0, x1), max(0, y1)
#     x2c, y2c = min(w - 1, x2), min(h - 1, y2)

#     foot_x = int((x1c + x2c) / 2)
#     foot_y = y2c

#     if sidewalk_mask[foot_y, foot_x]:
#         return True

#     region = sidewalk_mask[y1c:y2c + 1, x1c:x2c + 1]
#     area = max((x2c - x1c + 1) * (y2c - y1c + 1), 1)

#     return (region.sum() / area) >= SW_FOOT_OVERLAP_RATIO


# # ─────────────────────────────────────────────
# # MAIN FUNCTION
# # ─────────────────────────────────────────────

# def analyse_free_space(image, detections, masks):

#     img_h, img_w = image.shape[:2]
#     zone_top = int(img_h * 0.60)

#     sidewalk_mask = _merge_surface_masks(detections, masks, PEDESTRIAN_SURFACES)

#     centerline_fn, path_width_px, _ = extract_sidewalk_centerline(
#         sidewalk_mask, zone_top, img_h, img_w
#     )

#     col_bounds = compute_corridor_bounds(
#         sidewalk_mask, centerline_fn, path_width_px, img_h, img_w, zone_top
#     )

#     # ─────────────────────────────
#     # SURFACE COVERAGE
#     # ─────────────────────────────
#     walkable_pixels = {col: 0 for col in col_bounds}
#     non_walkable_pixels = {col: 0 for col in col_bounds}
#     semi_walkable_pixels = {col: 0 for col in col_bounds}

#     for det, mask in zip(detections, masks):
#         if mask is None:
#             continue

#         role = OBJECT_ROLES.get(det.label, "unknown")
#         if role not in ("walkable", "semi_walkable", "non_walkable"):
#             continue

#         zone_mask = mask[zone_top:, :]

#         for col, (cx1, cx2) in col_bounds.items():
#             col_px = int(zone_mask[:, cx1:cx2].sum())

#             if role == "walkable":
#                 walkable_pixels[col] += col_px
#             elif role == "semi_walkable":
#                 semi_walkable_pixels[col] += col_px
#             elif role == "non_walkable":
#                 non_walkable_pixels[col] += col_px

#     # ─────────────────────────────
#     # ON_PATH (with margin)
#     # ─────────────────────────────
#     cx1, cx2 = col_bounds["center"]
#     margin = max(img_w * 0.08, 50)

#     left_bound = max(0, cx1 - margin)
#     right_bound = min(img_w, cx2 + margin)

#     for det in detections:
#         foot_x = (det.box[0] + det.box[2]) / 2
#         det.on_path = (left_bound <= foot_x <= right_bound)

#     # ─────────────────────────────
#     # COLUMN CLASSIFICATION
#     # ─────────────────────────────
#     global_result = {}

#     for col, (cx1, cx2) in col_bounds.items():

#         col_area = max((img_h - zone_top) * (cx2 - cx1), 1)

#         ped_cov = walkable_pixels[col] / col_area
#         semi_cov = semi_walkable_pixels[col] / col_area
#         nwk_cov = non_walkable_pixels[col] / col_area

#         hard_count = 0
#         soft_count = 0

#         for det in detections:
#             role = OBJECT_ROLES.get(det.label, "unknown")

#             if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES:
#                 continue

#             if det.distance not in ("near", "mid"):
#                 continue

#             foot_x = (det.box[0] + det.box[2]) / 2
#             if not (cx1 <= foot_x < cx2):
#                 continue

#             if role in HARD_BLOCK_ROLES:
#                 # smarter cone handling
#                 if det.label == "traffic cone":
#                     hard_count += 0.5   # weak blocker
#                 else:
#                     hard_count += 1
#             else:
#                 soft_count += 1

#         # ─────────────────────────────
#         # FINAL DECISION (FIXED LOGIC)
#         # ─────────────────────────────

#         if nwk_cov >= NON_WALKABLE_COVERAGE:
#             state = "blocked"

#         elif hard_count >= 2:
#             state = "blocked"

#         elif hard_count > 0 or soft_count >= 2:
#             state = "crowded"

#         elif ped_cov >= WALKABLE_COVERAGE:
#             state = "walkable"

#         elif semi_cov >= WALKABLE_COVERAGE:
#             state = "walkable"

#         else:
#             state = "uncertain"

#         global_result[col] = state

#     return global_result, col_bounds

"""
Free-space analysis — FIXED VERSION

Key fixes:
1. Reduced "uncertain" states via surface-coverage fallback logic
2. Size-aware obstacle blocking (small cone ≠ full block)
3. Direction-specific blocking (obstacle on right doesn't block left)
4. Better sidewalk-to-road conflict handling in mask space
"""
import numpy as np
from typing import Optional
from config import (
    PEDESTRIAN_SURFACES,
    VEHICLE_SURFACES,
    OBJECT_ROLES,
    HARD_BLOCK_ROLES,
    SOFT_BLOCK_ROLES,
    WALKABLE_COVERAGE,
    NON_WALKABLE_COVERAGE,
    SW_FOOT_OVERLAP_RATIO,
    UNCERTAIN_MIN_SURFACE_SCORE,
    get_size_weight,
)
from utils.geometry import extract_sidewalk_centerline, compute_corridor_bounds


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _merge_surface_masks(detections, masks, surface_set) -> Optional[np.ndarray]:
    combined = None
    for det, mask in zip(detections, masks):
        if mask is None or det.label not in surface_set:
            continue
        combined = mask.astype(bool) if combined is None else combined | mask.astype(bool)
    return combined


def _obstacle_on_sidewalk(det, sidewalk_mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = det.box
    h, w = sidewalk_mask.shape

    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w - 1, x2); y2c = min(h - 1, y2)

    # Check foot point first
    foot_x = int((x1c + x2c) / 2)
    foot_y = y2c
    if 0 <= foot_y < h and 0 <= foot_x < w and sidewalk_mask[foot_y, foot_x]:
        return True

    # Check overlap ratio
    region = sidewalk_mask[y1c:y2c + 1, x1c:x2c + 1]
    area = max((x2c - x1c + 1) * (y2c - y1c + 1), 1)
    return (region.sum() / area) >= SW_FOOT_OVERLAP_RATIO


def _col_obstacle_weight(det, col_x1: int, col_x2: int, img_area: int) -> float:
    """
    FIXED: Size-aware obstacle weight per column.
    A tiny cone gets weight 0.1; a large vehicle gets weight 1.0.
    """
    foot_x = (det.box[0] + det.box[2]) / 2
    if not (col_x1 <= foot_x < col_x2):
        return 0.0

    area_ratio = det.area / max(img_area, 1)
    return get_size_weight(area_ratio)


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def analyse_free_space(image: np.ndarray, detections: list, masks: list):
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    zone_top = int(img_h * 0.60)

    # FIXED: prefer sidewalk masks; fall back to road only if no sidewalk found
    sidewalk_mask = _merge_surface_masks(detections, masks, PEDESTRIAN_SURFACES)
    road_mask     = _merge_surface_masks(detections, masks, VEHICLE_SURFACES)

    # If sidewalk exists in same region as road, subtract road from consideration
    # (pedestrian is on sidewalk, not road)
    if sidewalk_mask is not None and road_mask is not None:
        # Road areas that overlap with sidewalk are classified as sidewalk
        road_mask = road_mask & ~sidewalk_mask

    # Primary walkable surface for centerline extraction
    primary_surface_mask = sidewalk_mask if sidewalk_mask is not None else road_mask

    centerline_fn, path_width_px, _ = extract_sidewalk_centerline(
        primary_surface_mask, zone_top, img_h, img_w
    )

    col_bounds = compute_corridor_bounds(
        primary_surface_mask, centerline_fn, path_width_px, img_h, img_w, zone_top
    )

    # ─────────────────────────────
    # SURFACE COVERAGE PER COLUMN
    # ─────────────────────────────
    walkable_pixels    = {col: 0 for col in col_bounds}
    semi_walk_pixels   = {col: 0 for col in col_bounds}
    non_walkable_pixels = {col: 0 for col in col_bounds}

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
                semi_walk_pixels[col] += col_px
            elif role == "non_walkable":
                non_walkable_pixels[col] += col_px

    # ─────────────────────────────
    # ON_PATH ASSIGNMENT
    # ─────────────────────────────
    cx1, cx2 = col_bounds["center"]
    margin = max(img_w * 0.08, 50)
    left_bound  = max(0, cx1 - margin)
    right_bound = min(img_w, cx2 + margin)

    for det in detections:
        foot_x = (det.box[0] + det.box[2]) / 2
        det.on_path = (left_bound <= foot_x <= right_bound)

    # ─────────────────────────────
    # COLUMN CLASSIFICATION — FIXED
    # ─────────────────────────────
    global_result = {}

    for col, (cx1, cx2) in col_bounds.items():
        col_area = max((img_h - zone_top) * (cx2 - cx1), 1)

        ped_cov  = walkable_pixels[col]     / col_area
        semi_cov = semi_walk_pixels[col]    / col_area
        nwk_cov  = non_walkable_pixels[col] / col_area

        # ── Obstacle scoring (SIZE-AWARE) ──
        hard_weight = 0.0
        soft_weight = 0.0

        for det in detections:
            role = OBJECT_ROLES.get(det.label, "unknown")
            if role not in HARD_BLOCK_ROLES and role not in SOFT_BLOCK_ROLES:
                continue
            if det.distance not in ("near", "mid"):
                continue

            w = _col_obstacle_weight(det, cx1, cx2, img_area)
            if w == 0.0:
                continue

            if role in HARD_BLOCK_ROLES:
                # Traffic cones are soft-ish unless many
                if det.label == "traffic cone":
                    hard_weight += w * 0.5
                else:
                    hard_weight += w
            else:
                soft_weight += w

        # ── Decision logic ──
        # FIXED: "uncertain" only when truly ambiguous (no surface info at all)
        if nwk_cov >= NON_WALKABLE_COVERAGE:
            state = "blocked"

        elif hard_weight >= 1.5:
            state = "blocked"

        elif hard_weight >= 0.5 or soft_weight >= 1.5:
            state = "crowded"

        elif ped_cov >= WALKABLE_COVERAGE:
            state = "walkable"

        elif semi_cov >= WALKABLE_COVERAGE:
            # Road counts as walkable if no pedestrian surface at all
            state = "walkable" if sidewalk_mask is None else "crowded"

        elif (ped_cov + semi_cov) >= UNCERTAIN_MIN_SURFACE_SCORE:
            # FIXED: Small but non-zero surface coverage → walkable (not uncertain)
            state = "walkable"

        else:
            # Truly no surface info — use neighbor inference below
            state = "uncertain"

        global_result[col] = state

    # ─────────────────────────────
    # UNCERTAIN RESOLUTION — FIXED
    # Previously: uncertain → walkable (DANGEROUS)
    # Now: use neighbor states and a conservative default
    # ─────────────────────────────
    global_result = _resolve_uncertain(global_result)

    return global_result, col_bounds


def _resolve_uncertain(states: dict) -> dict:
    """
    FIXED: Replace blanket uncertain=walkable with context-aware inference.

    Rules:
    - If center is uncertain but neighbors are walkable → center = walkable
    - If center is uncertain but neighbors are blocked → center = crowded (safer)
    - If ALL uncertain → center = walkable, sides = uncertain (minimal assumption)
    - Never assume walkable when neighbors are blocked
    """
    result = dict(states)

    center = states.get("center", "uncertain")
    left   = states.get("left",   "uncertain")
    right  = states.get("right",  "uncertain")

    all_uncertain = all(v == "uncertain" for v in [center, left, right])

    if all_uncertain:
        # Minimal safe assumption: center might be walkable, sides unknown
        result["center"] = "walkable"
        # left/right stay uncertain
        return result

    # Resolve individual uncertain columns
    for col in ["left", "center", "right"]:
        if result[col] != "uncertain":
            continue

        neighbors = [v for c, v in states.items() if c != col and v != "uncertain"]
        if not neighbors:
            result[col] = "uncertain"
            continue

        walkable_count  = neighbors.count("walkable")
        blocked_count   = neighbors.count("blocked")
        crowded_count   = neighbors.count("crowded")

        if blocked_count > 0:
            result[col] = "crowded"   # conservative
        elif walkable_count >= len(neighbors):
            result[col] = "walkable"  # all neighbors walkable
        elif crowded_count > 0:
            result[col] = "crowded"
        else:
            result[col] = "walkable"

    return result