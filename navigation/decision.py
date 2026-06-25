# """
# Decision Engine - Scoring-based action selection (FINAL FIXED)
# """
# from config import OBJECT_ROLES, OBSTACLE_ROLES_SET, SEVERITY
# from utils.geometry import classify_hpos


# def decide_action(free_space: dict, detections: list, img_w: int) -> str:
#     """
#     Optimal path selection using scoring (improved stability + accuracy)
#     """

#     # ─────────────────────────────
#     # 0. SMART UNCERTAINTY HANDLING
#     # ─────────────────────────────
#     if all(v == "uncertain" for v in free_space.values()):
#         free_space["center"] = "walkable"

#     elif (
#         free_space.get("center") == "uncertain"
#         and free_space.get("left") == "uncertain"
#     ):
#         free_space["center"] = "walkable"

#     # ─────────────────────────────
#     # 1. BASE SCORES
#     # ─────────────────────────────
#     base_score_map = {
#         "walkable": 3.0,
#         "crowded": 1.2,
#         "uncertain": 0.8,
#         "blocked": -6.0,
#         "unknown": 0.0,
#     }

#     scores = {
#         k: base_score_map.get(free_space.get(k, "unknown"), 0.0)
#         for k in ["left", "center", "right"]
#     }

#     # ─────────────────────────────
#     # 2. OBSTACLE PENALTY (IMPROVED)
#     # ─────────────────────────────
#     for det in detections:
#         role = OBJECT_ROLES.get(det.label, "unknown")
#         if role not in OBSTACLE_ROLES_SET:
#             continue

#         foot_x = (det.box[0] + det.box[2]) / 2
#         pos = classify_hpos(foot_x, img_w)

#         sev = SEVERITY.get(det.label, 1.0)

#         dist_w = {
#             "near": 1.0,
#             "mid": 0.7,
#             "far": 0.2
#         }.get(det.distance, 0.5)

#         # CRITICAL: ON-PATH BOOST
#         path_boost = 1.5 if getattr(det, "on_path", False) else 1.0

#         penalty = sev * dist_w * path_boost

#         if pos in scores:
#             scores[pos] -= penalty

#     # ─────────────────────────────
#     # 3. CROWD PENALTY (LOCALIZED)
#     # ─────────────────────────────
#     crowd_by_dir = {"left": 0, "center": 0, "right": 0}

#     for det in detections:
#         if det.label != "person":
#             continue

#         foot_x = (det.box[0] + det.box[2]) / 2
#         pos = classify_hpos(foot_x, img_w)

#         if pos in crowd_by_dir:
#             crowd_by_dir[pos] += 1

#     for direction, count in crowd_by_dir.items():
#         if count >= 3:
#             scores[direction] -= 0.4 * count

#     # ─────────────────────────────
#     # 4. CENTER PRIORITY (SAFE WALKING)
#     # ─────────────────────────────
#     if free_space.get("center") in ("walkable", "crowded"):
#         scores["center"] += 0.6

#     # ─────────────────────────────
#     # 5. HARD BLOCK OVERRIDE
#     # ─────────────────────────────
#     if free_space.get("center") == "blocked":
#         scores["center"] -= 3.0

#     # ─────────────────────────────
#     # 6. FINAL DECISION
#     # ─────────────────────────────
#     best_dir = max(scores, key=scores.get)
#     best_score = scores[best_dir]

#     # Stop if everything is bad
#     if best_score < -2.5:
#         return "stop"

#     # ─────────────────────────────
#     # 7. ACTION MAPPING
#     # ─────────────────────────────
#     if best_dir == "center":
#         if free_space.get("center") != "blocked":
#             return "move_forward"

#         # fallback to next best
#         sorted_dirs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#         for d, s in sorted_dirs:
#             if d != "center" and s > -2:
#                 return f"move_{d}"

#         return "stop"

#     return f"move_{best_dir}"


# # ─────────────────────────────
# # CONFIDENCE (UNCHANGED BUT SAFER)
# # ─────────────────────────────
# def get_direction_confidence(scores: dict) -> dict:
#     """
#     Normalize scores into probabilities.
#     """
#     positive_scores = {k: max(0, v) for k, v in scores.items()}
#     total = sum(positive_scores.values()) + 1e-6

#     return {
#         k: round(v / total, 2) for k, v in positive_scores.items()
#     }


"""
Decision Engine — FIXED: Direction-specific risk, no dangerous assumptions

Key fixes:
1. Removed blanket "uncertain → walkable" (dangerous)
2. Direction-specific obstacle penalty (cone right ≠ affect left)
3. Size-aware penalty
4. Center bias only when center is actually safe
5. Better stop threshold
"""
from config import (
    OBJECT_ROLES, OBSTACLE_ROLES_SET, SEVERITY,
    get_size_weight, DIRECTION_SPILLOVER,
)
from utils.geometry import classify_hpos


def decide_action(free_space: dict, detections: list, img_w: int) -> str:
    """
    Scoring-based action selection with direction-specific penalties.
    """

    # ─────────────────────────────
    # BASE SCORES
    # ─────────────────────────────
    base_score_map = {
        "walkable":  3.0,
        "crowded":   1.2,
        "uncertain": 0.5,   # FIXED: was 0.8; uncertain is NOT walkable
        "blocked":  -6.0,
        "unknown":   0.0,
    }

    scores = {
        k: base_score_map.get(free_space.get(k, "unknown"), 0.0)
        for k in ["left", "center", "right"]
    }

    # ─────────────────────────────
    # OBSTACLE PENALTY (DIRECTION-SPECIFIC + SIZE-AWARE)
    # FIXED: obstacle in "right" only minimally affects "left"
    # ─────────────────────────────
    img_area_estimate = img_w * img_w  # rough; actual img_h not available here

    for det in detections:
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in OBSTACLE_ROLES_SET:
            continue

        foot_x = (det.box[0] + det.box[2]) / 2
        obs_dir = classify_hpos(foot_x, img_w)

        sev = SEVERITY.get(det.label, 1.0)
        dist_w = {"near": 1.0, "mid": 0.7, "far": 0.15}.get(det.distance, 0.5)

        # Size weight: large vehicle = 1.0, tiny cone = 0.1–0.3
        det_area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
        area_ratio = det_area / max(img_area_estimate, 1)
        size_w = get_size_weight(area_ratio)

        # on_path boost (conservative)
        path_boost = 1.4 if getattr(det, "on_path", False) else 1.0

        base_penalty = sev * dist_w * size_w * path_boost

        # FIXED: Apply penalty with direction spillover
        spillover = DIRECTION_SPILLOVER.get(obs_dir, {obs_dir: 1.0})
        for target_dir, spill_w in spillover.items():
            scores[target_dir] -= base_penalty * spill_w

    # ─────────────────────────────
    # CROWD PENALTY (DIRECTIONAL)
    # ─────────────────────────────
    crowd_by_dir = {"left": 0, "center": 0, "right": 0}
    for det in detections:
        if det.label != "person":
            continue
        foot_x = (det.box[0] + det.box[2]) / 2
        pos = classify_hpos(foot_x, img_w)
        if pos in crowd_by_dir:
            crowd_by_dir[pos] += 1

    for direction, count in crowd_by_dir.items():
        if count >= 3:
            scores[direction] -= 0.4 * count
        elif count >= 2:
            scores[direction] -= 0.3 * count

    # ─────────────────────────────
    # CENTER PRIORITY — FIXED
    # Only boost if center is actually safe
    # ─────────────────────────────
    if free_space.get("center") in ("walkable",):
        scores["center"] += 0.5
    elif free_space.get("center") == "crowded":
        scores["center"] += 0.1   # slight bias still
    # Do NOT boost if blocked or uncertain

    # ─────────────────────────────
    # HARD BLOCK OVERRIDE
    # ─────────────────────────────
    if free_space.get("center") == "blocked":
        scores["center"] -= 4.0

    # ─────────────────────────────
    # FINAL DECISION
    # ─────────────────────────────
    best_dir = max(scores, key=scores.get)
    best_score = scores[best_dir]

    # FIXED: More conservative stop threshold
    if best_score < -3.0:
        return "stop"

    # All blocked
    if all(free_space.get(d, "unknown") == "blocked" for d in ["left", "center", "right"]):
        return "stop"

    # ─────────────────────────────
    # ACTION MAPPING
    # ─────────────────────────────
    if best_dir == "center":
        if free_space.get("center") not in ("blocked",):
            return "move_forward"
        # Fallback to next best
        sorted_dirs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for d, s in sorted_dirs:
            if d != "center" and s > -2.5:
                return f"move_{d}"
        return "stop"

    return f"move_{best_dir}"


def get_direction_confidence(scores: dict) -> dict:
    """Normalize scores into probabilities."""
    positive_scores = {k: max(0.0, v) for k, v in scores.items()}
    total = sum(positive_scores.values()) + 1e-6
    return {k: round(v / total, 2) for k, v in positive_scores.items()}