# """
# Navigation description builder - FINAL STABLE VERSION (UPDATED)
# """
# from collections import defaultdict
# from typing import Dict
# from config import (
#     OBJECT_ROLES,
#     OBSTACLE_ROLES_SET,
#     SEVERITY,
#     NAV_CONDITIONAL,
#     ACCESSIBILITY_FEATURES,
#     ACCESSIBILITY_HAZARDS,
#     SURFACE_QUALITY,
# )
# from utils.geometry import get_hpos_x, classify_hpos
# from utils.distance import estimate_distance
# from navigation.decision import decide_action, get_direction_confidence
# from config.labels import get_display_label


# # ─────────────────────────────────────────────
# # GROUP DETECTIONS
# # ─────────────────────────────────────────────
# def group_detections(detections: list) -> dict[str, list]:
#     groups = defaultdict(list)
#     for det in detections:
#         groups[det.label].append(det)
#     return dict(groups)


# # ─────────────────────────────────────────────
# # MAIN FUNCTION
# # ─────────────────────────────────────────────
# def build_navigation_description(
#     results: dict,
#     use_llm: bool = False,
#     llm_client=None,
# ) -> dict:

#     dets = results.get("detections", [])
#     free_space = results.get("free_space", {"left": "unknown", "center": "unknown", "right": "unknown"})
#     mask_map = results.get("mask_map", {})
#     img_h, img_w = results["image"].shape[:2]
#     img_area = img_h * img_w

#     # ─────────────────────────────
#     # ACTION
#     # ─────────────────────────────
#     action = decide_action(free_space, dets, img_w)

#     base_score_map = {
#         "walkable": 3.0,
#         "crowded": 1.5,
#         "uncertain": 1.0,
#         "blocked": -5.0,
#         "unknown": 0.0,
#     }

#     scores = {
#         "left": base_score_map.get(free_space.get("left", "unknown"), 0),
#         "center": base_score_map.get(free_space.get("center", "unknown"), 0),
#         "right": base_score_map.get(free_space.get("right", "unknown"), 0),
#     }

#     # FIX 1: use ON_PATH in scoring
#     for det in dets:
#         role = OBJECT_ROLES.get(det.label, "unknown")
#         if role not in OBSTACLE_ROLES_SET:
#             continue

#         foot_x = (det.box[0] + det.box[2]) / 2
#         pos = classify_hpos(foot_x, img_w)

#         sev = SEVERITY.get(det.label, 1.0)
#         dist_w = {"near": 1.0, "mid": 0.6, "far": 0.2}.get(det.distance, 0.5)

#         path_boost = 1.5 if getattr(det, "on_path", False) else 1.0

#         scores[pos] -= sev * dist_w * path_boost

#     confidence = get_direction_confidence(scores)

#     # ─────────────────────────────
#     # GROUP DETECTIONS
#     # ─────────────────────────────
#     groups = group_detections(dets)

#     obstacle_descriptions = []
#     surface_descriptions = []

#     dist_rank = {"near": 0, "mid": 1, "far": 2, "unknown": 3}

#     # ─────────────────────────────
#     # BUILD DESCRIPTIONS
#     # ─────────────────────────────
#     for label, group in groups.items():

#         role = OBJECT_ROLES.get(label, "unknown")

#         rep = min(group, key=lambda d: (dist_rank.get(d.distance, 3), -d.score))
#         rep_idx = dets.index(rep) if rep in dets else -1
#         mask = mask_map.get(rep_idx)

#         hpos_x = get_hpos_x(rep, mask, img_w)
#         h_pos = classify_hpos(hpos_x, img_w)

#         dist = rep.distance if rep.distance != "unknown" \
#             else estimate_distance(rep, img_h, img_w, img_area)

#         # FIX 2: STRONG FILTER
#         if dist == "far" and not rep.on_path:
#             continue

#         # Skip weak trees
#         if label == "tree" and (dist != "near" or not rep.on_path):
#             continue

#         count = len(group)

#         # FIX 3: CLEAN FORMAT (NO plural inside)
#         if count > 1:
#             compact = f"{label}({dist}, {h_pos})"
#         else:
#             compact = f"{label}({dist}, {h_pos})"

#         # Add count outside (only for display, not inside structure)
#         if count > 1:
#             compact = f"{count}x {compact}"

#         on_path_any = any(d.on_path for d in group)

#         is_relevant = (
#             dist in ("near", "mid") or
#             on_path_any
#         )

#         if role in OBSTACLE_ROLES_SET and is_relevant:
#             obstacle_descriptions.append(compact)

#         elif role in ("walkable", "semi_walkable"):
#             if h_pos == "center":
#                 surface_descriptions.append(compact)

#     # ─────────────────────────────
#     # RISK (IMPROVED)
#     # ─────────────────────────────
#     caution_score = 0.0
#     obstacle_counts = {}

#     for det in dets:
#         if det.distance == "far":
#             continue
#         obstacle_counts[det.label] = obstacle_counts.get(det.label, 0) + 1

#     for label, count in obstacle_counts.items():

#         role = OBJECT_ROLES.get(label, "unknown")
#         if role not in OBSTACLE_ROLES_SET:
#             continue

#         group = [d for d in dets if d.label == label and d.distance != "far"]

#         # FIX 4: ON_PATH PRIORITY
#         on_path = any(getattr(d, "on_path", False) for d in group)

#         distances = [d.distance for d in group]

#         if "near" in distances:
#             dist_w = 1.0
#         elif "mid" in distances:
#             dist_w = 0.6
#         else:
#             dist_w = 0.0

#         if on_path:
#             dist_w *= 1.3

#         sev = SEVERITY.get(label, 1.0)

#         crowd_boost = 1.0
#         if label == "person":
#             if count >= 5:
#                 crowd_boost = 2.0
#             elif count >= 3:
#                 crowd_boost = 1.5

#         if label == "traffic cone" and count >= 3:
#             crowd_boost = 1.8

#         caution_score += sev * dist_w * crowd_boost

#     if caution_score >= 6.0:
#         risk = "urgent"
#     elif caution_score >= 4.0:
#         risk = "high"
#     elif caution_score >= 2.0:
#         risk = "medium"
#     elif caution_score > 0.5:
#         risk = "low"
#     else:
#         risk = "none"

#     # Safe override
#     if risk == "urgent":
#         action = "stop"
#     elif risk == "high" and action == "move_forward":
#         action = "move_forward_cautious"

#     # ─────────────────────────────
#     # FINAL TEXT
#     # ─────────────────────────────
#     scene_text = "\n".join([
#         f"ACTION: {action}",
#         f"RISK: {risk}",
#         f"PATH: center {free_space.get('center')}, left {free_space.get('left')}, right {free_space.get('right')}",
#         "OBSTACLES: " + ("; ".join(obstacle_descriptions) if obstacle_descriptions else "none"),
#         "ENV: " + ("; ".join(surface_descriptions) if surface_descriptions else "none"),
#     ])

#     result = {
#         "free_space": free_space,
#         "action": action,
#         "risk": risk,
#         "confidence": confidence,
#         "obstacles": obstacle_descriptions,
#         "surfaces": surface_descriptions,
#         "scene_text": scene_text,
#         "guidance_source": "rule_based",
#     }

#     return result


# # ─────────────────────────────────────────────
# # WALKGPT-STYLE OUTPUT
# # ─────────────────────────────────────────────
# def build_walkgpt_description(results: dict) -> dict:
#     """WalkGPT-style rich navigation output."""
    
#     # Get base navigation description
#     base = build_navigation_description(results)

#     dets = results.get("detections", [])
#     masks = results.get("masks", [])
#     mask_map = results.get("mask_map", {})
#     free_space = results.get("free_space", {"left": "unknown", "center": "unknown", "right": "unknown"})
#     col_bounds = results.get("col_bounds", None)
#     depth_map = results.get("depth_map", None)
#     img_h, img_w = results["image"].shape[:2]

#     # ── Accessibility analysis ──────────────────────────────────────────────
#     detected_features = []
#     detected_hazards = []
#     surface_types = set()

#     for det in dets:
#         if det.label in ACCESSIBILITY_FEATURES:
#             detected_features.append(get_display_label(det.label))
#         if det.label in ACCESSIBILITY_HAZARDS:
#             detected_hazards.append(
#                 f"{get_display_label(det.label)} ({det.distance})"
#             )
#         sq = SURFACE_QUALITY.get(det.label)
#         if sq:
#             surface_types.add(sq)

#     surface_priority = ["smooth", "textured", "rough", "wet", "damaged", "uneven", "soft", "stepped"]
#     primary_surface = "unknown"
#     for sp in surface_priority:
#         if sp in surface_types:
#             primary_surface = sp
#             break

#     width_assessment = "unknown"
#     if col_bounds:
#         center_bounds = col_bounds.get("center", (0, img_w))
#         center_width_ratio = (center_bounds[1] - center_bounds[0]) / max(img_w, 1)
#         if center_width_ratio >= 0.08:
#             width_assessment = "adequate (>1.2m)"
#         elif center_width_ratio >= 0.04:
#             width_assessment = "narrow (<1.2m)"
#         else:
#             width_assessment = "very narrow — single file"

#     accessibility = {
#         "surface": primary_surface,
#         "hazards": detected_hazards,
#         "features": detected_features,
#         "width_assessment": width_assessment,
#     }

#     # ── Spatial map ─────────────────────────────────────────────────────────
#     spatial_map = {}
#     for col_name in ["left", "center", "right"]:
#         col_objects = []
#         if col_bounds:
#             cx1, cx2 = col_bounds.get(col_name, (0, img_w))
#             for det in dets:
#                 foot_x = (det.box[0] + det.box[2]) / 2
#                 if cx1 <= foot_x < cx2:
#                     role = OBJECT_ROLES.get(det.label, "unknown")
#                     if role in OBSTACLE_ROLES_SET:
#                         col_objects.append(
#                             f"{get_display_label(det.label)}({det.distance})"
#                         )

#         spatial_map[col_name] = {
#             "status": free_space.get(col_name, "unknown"),
#             "objects": col_objects[:5],
#         }

#     return {
#         **base,
#         "accessibility": accessibility,
#         "spatial_map": spatial_map,
#     }


"""
Navigation description builder — FIXED VERSION

Key fixes:
1. Direction-specific risk scoring (obstacle right ≠ risk left)
2. Size-aware obstacle scoring
3. Dynamic cluster boost from risk_config
4. Cleaner obstacle description format
5. No hallucinated guidance
"""
from collections import defaultdict
from typing import Dict
from config import (
    OBJECT_ROLES,
    OBSTACLE_ROLES_SET,
    SEVERITY,
    NAV_CONDITIONAL,
    ACCESSIBILITY_FEATURES,
    ACCESSIBILITY_HAZARDS,
    SURFACE_QUALITY,
    get_cluster_boost,
    get_size_weight,
)
from utils.geometry import get_hpos_x, classify_hpos
from utils.distance import estimate_distance
from navigation.decision import decide_action, get_direction_confidence
from config.labels import get_display_label


def group_detections(detections: list) -> dict[str, list]:
    groups = defaultdict(list)
    for det in detections:
        groups[det.label].append(det)
    return dict(groups)


def build_navigation_description(
    results: dict,
    use_llm: bool = False,
    llm_client=None,
) -> dict:

    dets = results.get("detections", [])
    free_space = results.get("free_space", {"left": "unknown", "center": "unknown", "right": "unknown"})
    mask_map   = results.get("mask_map", {})
    img_h, img_w = results["image"].shape[:2]
    img_area = img_h * img_w

    # ─────────────────────────────
    # ACTION
    # ─────────────────────────────
    action = decide_action(free_space, dets, img_w)

    base_score_map = {
        "walkable":  3.0,
        "crowded":   1.5,
        "uncertain": 0.5,
        "blocked":  -5.0,
        "unknown":   0.0,
    }
    scores = {
        k: base_score_map.get(free_space.get(k, "unknown"), 0)
        for k in ["left", "center", "right"]
    }

    for det in dets:
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in OBSTACLE_ROLES_SET:
            continue
        foot_x = (det.box[0] + det.box[2]) / 2
        pos    = classify_hpos(foot_x, img_w)
        sev    = SEVERITY.get(det.label, 1.0)
        dist_w = {"near": 1.0, "mid": 0.6, "far": 0.2}.get(det.distance, 0.5)
        area_ratio = det.area / max(img_area, 1)
        size_w = get_size_weight(area_ratio)
        path_boost = 1.5 if getattr(det, "on_path", False) else 1.0
        scores[pos] -= sev * dist_w * size_w * path_boost

    confidence = get_direction_confidence(scores)

    # ─────────────────────────────
    # GROUP DETECTIONS
    # ─────────────────────────────
    groups = group_detections(dets)

    obstacle_descriptions = []
    surface_descriptions  = []
    dist_rank = {"near": 0, "mid": 1, "far": 2, "unknown": 3}

    for label, group in groups.items():
        role = OBJECT_ROLES.get(label, "unknown")

        rep = min(group, key=lambda d: (dist_rank.get(d.distance, 3), -d.score))
        rep_idx = dets.index(rep) if rep in dets else -1
        mask = mask_map.get(rep_idx)

        hpos_x = get_hpos_x(rep, mask, img_w)
        h_pos  = classify_hpos(hpos_x, img_w)

        dist = rep.distance if rep.distance != "unknown" \
            else estimate_distance(rep, img_h, img_w, img_area)

        # Skip far objects not on path
        if dist == "far" and not rep.on_path:
            continue

        # Skip weak trees unless near and on path
        if label == "tree" and (dist != "near" or not rep.on_path):
            continue

        count  = len(group)
        compact = f"{get_display_label(label)}({dist}, {h_pos})"
        if count > 1:
            compact = f"{count}x {compact}"

        on_path_any = any(d.on_path for d in group)
        is_relevant = dist in ("near", "mid") or on_path_any

        if role in OBSTACLE_ROLES_SET and is_relevant:
            obstacle_descriptions.append(compact)
        elif role in ("walkable", "semi_walkable"):
            if h_pos == "center":
                surface_descriptions.append(compact)

    # ─────────────────────────────
    # RISK — DIRECTION-SPECIFIC + DYNAMIC CLUSTER BOOST
    # ─────────────────────────────
    caution_score    = 0.0
    obstacle_counts  = {}

    for det in dets:
        if det.distance == "far":
            continue
        obstacle_counts[det.label] = obstacle_counts.get(det.label, 0) + 1

    for label, count in obstacle_counts.items():
        role = OBJECT_ROLES.get(label, "unknown")
        if role not in OBSTACLE_ROLES_SET:
            continue

        group = [d for d in dets if d.label == label and d.distance != "far"]
        on_path = any(getattr(d, "on_path", False) for d in group)
        distances = [d.distance for d in group]

        if "near" in distances:
            dist_w = 1.0
        elif "mid" in distances:
            dist_w = 0.6
        else:
            dist_w = 0.0

        if on_path:
            dist_w *= 1.3

        # Average size weight across group
        avg_area_ratio = sum(
            d.area / max(img_area, 1) for d in group
        ) / max(len(group), 1)
        size_w = get_size_weight(avg_area_ratio)

        sev    = SEVERITY.get(label, 1.0)
        cluster_boost = get_cluster_boost(label, count)

        caution_score += sev * dist_w * size_w * cluster_boost

    if caution_score >= 6.5:
        risk = "urgent"
    elif caution_score >= 4.0:
        risk = "high"
    elif caution_score >= 2.0:
        risk = "medium"
    elif caution_score > 0.7:
        risk = "low"
    else:
        risk = "none"

    # Safe overrides
    if risk == "urgent":
        action = "stop"
    elif risk == "high" and action == "move_forward":
        action = "move_forward_cautious"

    # ─────────────────────────────
    # FINAL TEXT
    # ─────────────────────────────
    scene_text = "\n".join([
        f"ACTION: {action}",
        f"RISK: {risk}",
        f"PATH: center {free_space.get('center')}, "
        f"left {free_space.get('left')}, right {free_space.get('right')}",
        "OBSTACLES: " + ("; ".join(obstacle_descriptions) if obstacle_descriptions else "none"),
        "ENV: " + ("; ".join(surface_descriptions) if surface_descriptions else "none"),
    ])

    result = {
        "free_space": free_space,
        "action":     action,
        "risk":       risk,
        "confidence": confidence,
        "obstacles":  obstacle_descriptions,
        "surfaces":   surface_descriptions,
        "scene_text": scene_text,
        "guidance_source": "rule_based",
    }

    # ─────────────────────────────
    # OPTIONAL LLM ENHANCEMENT
    # ─────────────────────────────
    if use_llm and llm_client is not None and llm_client.is_loaded():
        from llm.prompt import build_navigation_prompt, parse_navigation_response, SYSTEM_PROMPT
        from llm.fallback import should_use_llm

        if should_use_llm(result):
            prompt   = build_navigation_prompt(result)
            response = llm_client.generate(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=60,
                temperature=0.0,
            )
            if response:
                parsed = parse_navigation_response(response, action, risk)
                result["guidance"]        = parsed["guidance"]
                result["guidance_source"] = parsed["source"]
        else:
            from llm.fallback import generate_fallback_guidance
            result["guidance"]        = generate_fallback_guidance(result)
            result["guidance_source"] = "rule_based"

    return result


def build_walkgpt_description(results: dict) -> dict:
    """WalkGPT-style rich navigation output."""
    base = build_navigation_description(results)

    dets      = results.get("detections", [])
    free_space = results.get("free_space", {"left": "unknown", "center": "unknown", "right": "unknown"})
    col_bounds = results.get("col_bounds", None)
    img_h, img_w = results["image"].shape[:2]

    # ── Accessibility analysis ────────────────────────────────────────────
    detected_features = []
    detected_hazards  = []
    surface_types     = set()

    for det in dets:
        if det.label in ACCESSIBILITY_FEATURES:
            detected_features.append(get_display_label(det.label))
        if det.label in ACCESSIBILITY_HAZARDS:
            detected_hazards.append(f"{get_display_label(det.label)} ({det.distance})")
        sq = SURFACE_QUALITY.get(det.label)
        if sq:
            surface_types.add(sq)

    surface_priority = ["smooth", "textured", "rough", "wet", "damaged", "uneven", "soft"]
    primary_surface  = "unknown"
    for sp in surface_priority:
        if sp in surface_types:
            primary_surface = sp
            break

    width_assessment = "unknown"
    if col_bounds:
        center_bounds     = col_bounds.get("center", (0, img_w))
        center_width_ratio = (center_bounds[1] - center_bounds[0]) / max(img_w, 1)
        if center_width_ratio >= 0.08:
            width_assessment = "adequate (>1.2m)"
        elif center_width_ratio >= 0.04:
            width_assessment = "narrow (<1.2m)"
        else:
            width_assessment = "very narrow — single file"

    accessibility = {
        "surface":          primary_surface,
        "hazards":          detected_hazards,
        "features":         detected_features,
        "width_assessment": width_assessment,
    }

    # ── Spatial map ──────────────────────────────────────────────────────
    spatial_map = {}
    for col_name in ["left", "center", "right"]:
        col_objects = []
        if col_bounds:
            cx1, cx2 = col_bounds.get(col_name, (0, img_w))
            for det in dets:
                foot_x = (det.box[0] + det.box[2]) / 2
                if cx1 <= foot_x < cx2:
                    role = OBJECT_ROLES.get(det.label, "unknown")
                    if role in OBSTACLE_ROLES_SET:
                        col_objects.append(
                            f"{get_display_label(det.label)}({det.distance})"
                        )
        spatial_map[col_name] = {
            "status":  free_space.get(col_name, "unknown"),
            "objects": col_objects[:5],
        }

    return {
        **base,
        "accessibility": accessibility,
        "spatial_map":   spatial_map,
    }
