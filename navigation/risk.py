# """
# Risk calculation and spatial relations
# """
# from config import (
#     OBJECT_ROLES,
#     OBSTACLE_ROLES_SET,
#     SEVERITY,
#     DIST_WEIGHT,
#     DEFAULT_SEVERITY,
#     CAUTION_URGENT,
#     CAUTION_HIGH,
#     CAUTION_MEDIUM,
#     CAUTION_LOW,
# )


# def compute_inter_object_relations(
#     detections: list,
#     top_n: int = 3,
# ) -> list[str]:
#     """
#     [SB1] Return list of spatial relation strings for top-N on-path obstacles.
#     """
#     on_path_obs = [
#         d for d in detections
#         if d.on_path
#         and OBJECT_ROLES.get(d.label, "unknown") in OBSTACLE_ROLES_SET
#         and d.distance in ("near", "mid")  # Include mid
#     ]

#     on_path_obs.sort(
#         key=lambda d: SEVERITY.get(d.label, 1) * DIST_WEIGHT.get(d.distance, 0),
#         reverse=True,
#     )
#     on_path_obs = on_path_obs[:top_n]

#     relations = []
#     for i in range(len(on_path_obs)):
#         for j in range(i + 1, len(on_path_obs)):
#             a = on_path_obs[i]
#             b = on_path_obs[j]

#             if a.label == b.label:
#                 continue

#             ax = (a.box[0] + a.box[2]) / 2
#             bx = (b.box[0] + b.box[2]) / 2

#             if ax < bx - 20:
#                 h_rel = f"{a.label} is left of {b.label}"
#             elif ax > bx + 20:
#                 h_rel = f"{a.label} is right of {b.label}"
#             else:
#                 h_rel = f"{a.label} and {b.label} are side by side"

#             dist_order = {"near": 0, "mid": 1, "far": 2}
#             da = dist_order.get(a.distance, 1)
#             db = dist_order.get(b.distance, 1)
#             if da < db:
#                 d_rel = f"{a.label} is closer"
#             elif da > db:
#                 d_rel = f"{b.label} is closer"
#             else:
#                 d_rel = None

#             relations.append(h_rel)
#             if d_rel:
#                 relations.append(d_rel)

#     # Remove duplicates
#     return list(set(relations))


# def compute_caution(detections: list, img_h: int, img_w: int, img_area: int) -> tuple[float, str]:
#     """
#     [DI2] Compute scene-level caution score and urgency label.
#     """
#     from utils.distance import estimate_distance

#     max_caution = 0.0
#     for det in detections:
#         role = OBJECT_ROLES.get(det.label, "unknown")
#         if role not in {"obstacle", "hazard", "dynamic_hazard"}:
#             continue
#         sev = SEVERITY.get(det.label, DEFAULT_SEVERITY)
#         dist = det.distance if det.distance != "unknown" else estimate_distance(det, img_h, img_w, img_area)
#         dw = DIST_WEIGHT.get(dist, 0.0)
#         max_caution = max(max_caution, sev * dw)

#     if max_caution >= CAUTION_URGENT:
#         urgency = "URGENT"
#     elif max_caution >= CAUTION_HIGH:
#         urgency = "HIGH"
#     elif max_caution >= CAUTION_MEDIUM:
#         urgency = "MEDIUM"
#     elif max_caution > 0:
#         urgency = "LOW"
#     else:
#         urgency = "NONE"

#     return max_caution, urgency


"""
Risk calculation and spatial relations — FIXED

Key fix: Uses direction-specific risk and size-aware scoring.
"""
from config import (
    OBJECT_ROLES,
    OBSTACLE_ROLES_SET,
    SEVERITY,
    DIST_WEIGHT,
    DEFAULT_SEVERITY,
    CAUTION_URGENT,
    CAUTION_HIGH,
    CAUTION_MEDIUM,
    CAUTION_LOW,
    get_size_weight,
)


def compute_inter_object_relations(
    detections: list,
    top_n: int = 3,
) -> list[str]:
    """Return spatial relation strings for top-N on-path obstacles."""
    on_path_obs = [
        d for d in detections
        if d.on_path
        and OBJECT_ROLES.get(d.label, "unknown") in OBSTACLE_ROLES_SET
        and d.distance in ("near", "mid")
    ]

    on_path_obs.sort(
        key=lambda d: SEVERITY.get(d.label, 1) * DIST_WEIGHT.get(d.distance, 0),
        reverse=True,
    )
    on_path_obs = on_path_obs[:top_n]

    relations = []
    for i in range(len(on_path_obs)):
        for j in range(i + 1, len(on_path_obs)):
            a = on_path_obs[i]
            b = on_path_obs[j]
            if a.label == b.label:
                continue
            ax = (a.box[0] + a.box[2]) / 2
            bx = (b.box[0] + b.box[2]) / 2
            if ax < bx - 20:
                h_rel = f"{a.label} is left of {b.label}"
            elif ax > bx + 20:
                h_rel = f"{a.label} is right of {b.label}"
            else:
                h_rel = f"{a.label} and {b.label} are side by side"
            dist_order = {"near": 0, "mid": 1, "far": 2}
            da = dist_order.get(a.distance, 1)
            db = dist_order.get(b.distance, 1)
            if da < db:
                d_rel = f"{a.label} is closer"
            elif da > db:
                d_rel = f"{b.label} is closer"
            else:
                d_rel = None
            relations.append(h_rel)
            if d_rel:
                relations.append(d_rel)

    return list(set(relations))


def compute_caution(
    detections: list, img_h: int, img_w: int, img_area: int
) -> tuple[float, str]:
    """Compute scene-level caution score with size awareness."""
    from utils.distance import estimate_distance

    max_caution = 0.0
    for det in detections:
        role = OBJECT_ROLES.get(det.label, "unknown")
        if role not in {"obstacle", "hazard", "dynamic_hazard"}:
            continue
        sev  = SEVERITY.get(det.label, DEFAULT_SEVERITY)
        dist = det.distance if det.distance != "unknown" \
            else estimate_distance(det, img_h, img_w, img_area)
        dw   = DIST_WEIGHT.get(dist, 0.0)
        size_w = get_size_weight(det.area / max(img_area, 1))
        max_caution = max(max_caution, sev * dw * size_w)

    if max_caution >= CAUTION_URGENT:
        urgency = "URGENT"
    elif max_caution >= CAUTION_HIGH:
        urgency = "HIGH"
    elif max_caution >= CAUTION_MEDIUM:
        urgency = "MEDIUM"
    elif max_caution > 0:
        urgency = "LOW"
    else:
        urgency = "NONE"

    return max_caution, urgency
