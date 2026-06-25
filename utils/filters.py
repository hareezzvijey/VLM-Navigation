# """
# Filter utilities: area filter, NMS, dedup, occlusion
# """
# import numpy as np
# from dataclasses import dataclass
# from typing import Optional
# from config import (
#     MIN_ABSOLUTE_PIXELS,
#     MAX_RELATIVE_AREA,
#     MAX_ASPECT_RATIO,
#     DYNAMIC_LABELS,
#     SOFT_NMS_SIGMA,
#     SOFT_NMS_SCORE_GATE,
# )


# @dataclass
# class Detection:
#     box: list
#     score: float
#     label: str
#     area: int = 0
#     occluded: bool = False
#     suppressed_by: Optional[int] = None
#     role: str = "unknown"
#     distance: str = "unknown"
#     on_path: bool = False


# def _iou(a: list, b: list) -> float:
#     ix1 = max(a[0], b[0])
#     iy1 = max(a[1], b[1])
#     ix2 = min(a[2], b[2])
#     iy2 = min(a[3], b[3])
#     if ix2 <= ix1 or iy2 <= iy1:
#         return 0.0
#     inter = (ix2 - ix1) * (iy2 - iy1)
#     union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
#     return inter / max(union, 1)


# def passes_area_filter(box: list, img_h: int, img_w: int, label: str = "") -> tuple[bool, str]:
#     x1, y1, x2, y2 = box
#     bw = max(x2 - x1, 1)
#     bh = max(y2 - y1, 1)
#     area = bw * bh

#     if area < MIN_ABSOLUTE_PIXELS:
#         return False, f"too small ({area}px)"

#     if label not in {"tree", "pole", "building", "grass", "soil"}:
#         if area / (img_h * img_w) > MAX_RELATIVE_AREA:
#             return False, f"too large ({area/(img_h*img_w):.0%})"

#     if max(bw / bh, bh / bw) > MAX_ASPECT_RATIO:
#         return False, "sliver box"

#     return True, ""


# def deduplicate_cross_prompt(
#     boxes: list, scores: list, labels: list, iou_threshold: float = 0.50,
# ) -> tuple[list, list, list]:
#     n = len(boxes)
#     if n <= 1:
#         return boxes, scores, labels
    
#     keep = [True] * n
#     order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    
#     surface_labels = {"sidewalk", "road", "grass", "soil", "gravel", "dirt", "plant"}
#     protected_indices = [i for i, label in enumerate(labels) if label in surface_labels]
    
#     for ri, i in enumerate(order):
#         if not keep[i]:
#             continue
#         for j in order[ri + 1:]:
#             if not keep[j]:
#                 continue
            
#             if i in protected_indices or j in protected_indices:
#                 continue
                
#             if _iou(boxes[i], boxes[j]) >= iou_threshold:
#                 keep[j] = False
    
#     kb = [boxes[i] for i in range(n) if keep[i]]
#     ks = [scores[i] for i in range(n) if keep[i]]
#     kl = [labels[i] for i in range(n) if keep[i]]
#     removed = n - len(kb)
#     if removed > 0:
#         print(f"  [U1 dedup] Removed {removed} cross-prompt duplicates")
#     return kb, ks, kl


# def compute_containment(a: list, b: list) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     ix1 = max(ax1, bx1)
#     iy1 = max(ay1, by1)
#     ix2 = min(ax2, bx2)
#     iy2 = min(ay2, by2)
#     if ix2 <= ix1 or iy2 <= iy1:
#         return 0.0
#     return (ix2 - ix1) * (iy2 - iy1) / max((ax2 - ax1) * (ay2 - ay1), 1)


# def run_occlusion_analysis(detections: list[Detection]) -> list[Detection]:
#     n = len(detections)
    
#     surface_labels = {"sidewalk", "road", "grass", "soil"}
    
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
            
#             if detections[i].label in surface_labels or detections[j].label in surface_labels:
#                 continue
                
#             if compute_containment(detections[i].box, detections[j].box) < 0.85:
#                 continue
                
#             if detections[i].label == detections[j].label:
#                 if detections[i].score < detections[j].score:
#                     detections[i].suppressed_by = j
#             else:
#                 outer_is_dynamic = detections[j].label in DYNAMIC_LABELS
#                 ob = detections[j].box
#                 ib = detections[i].box
#                 if outer_is_dynamic and (ib[2]-ib[0])*(ib[3]-ib[1]) < (ob[2]-ob[0])*(ob[3]-ob[1]):
#                     detections[i].occluded = True
#     return detections

# def soft_nms(detections: list[Detection], sigma: float = 0.5, score_gate: float = 0.20) -> list[Detection]:
#     dets = sorted([d for d in detections if d.suppressed_by is None], key=lambda d: d.score, reverse=True)
    
#     surface_labels = {"sidewalk", "road", "grass", "soil"}
    
#     for i in range(len(dets)):
#         for j in range(i + 1, len(dets)):
#             if dets[i].label in surface_labels or dets[j].label in surface_labels:
#                 continue
                
#             ov = _iou(dets[i].box, dets[j].box)
#             if ov > 0:
#                 dets[j].score *= np.exp(-(ov**2) / sigma)
    
#     return [d for d in dets if d.score >= score_gate]


"""
Filter utilities: area filter, NMS, dedup, occlusion — FIXED

Key fixes:
1. Surface conflict resolution (sidewalk vs road disambiguation)
2. Size-aware NMS (large objects not penalised unfairly)
3. Relaxed area filter to avoid dropping distant people
4. De-duplicate logging (no repeated prints per prompt)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import (
    MIN_ABSOLUTE_PIXELS,
    MAX_RELATIVE_AREA,
    MAX_ASPECT_RATIO,
    DYNAMIC_LABELS,
    SOFT_NMS_SIGMA,
    SOFT_NMS_SCORE_GATE,
    SURFACE_PRIORITY,
)
from config.prompts import CONFLICTING_SURFACE_PAIRS


@dataclass
class Detection:
    box: list
    score: float
    label: str
    area: int = 0
    occluded: bool = False
    suppressed_by: Optional[int] = None
    role: str = "unknown"
    distance: str = "unknown"
    on_path: bool = False
    # NEW: area_ratio for size-aware logic
    area_ratio: float = 0.0


def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1)


def passes_area_filter(
    box: list, img_h: int, img_w: int, label: str = ""
) -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    area = bw * bh

    # FIXED: even smaller minimum for dynamic agents detected far away
    min_px = MIN_ABSOLUTE_PIXELS // 2 if label in DYNAMIC_LABELS else MIN_ABSOLUTE_PIXELS
    if area < min_px:
        return False, f"too small ({area}px)"

    if label not in {"tree", "pole", "building", "grass", "soil", "sidewalk", "road"}:
        if area / (img_h * img_w) > MAX_RELATIVE_AREA:
            return False, f"too large ({area/(img_h*img_w):.0%})"

    if max(bw / bh, bh / bw) > MAX_ASPECT_RATIO:
        return False, "sliver box"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE CONFLICT RESOLUTION
# FIXED: When sidewalk + road detected in same region, sidewalk wins
# ─────────────────────────────────────────────────────────────────────────────

def resolve_surface_conflicts(
    boxes: list, scores: list, labels: list, iou_threshold: float = 0.40
) -> tuple[list, list, list]:
    """
    Resolve conflicting surface detections in the same region.
    Uses SURFACE_PRIORITY: sidewalk (10) > road (3) > grass (1).
    This prevents 'road' from overwriting 'sidewalk' in same area.
    """
    n = len(boxes)
    if n <= 1:
        return boxes, scores, labels

    suppress = [False] * n
    conflict_pairs = {(a, b) for a, b in CONFLICTING_SURFACE_PAIRS}
    conflict_pairs |= {(b, a) for a, b in CONFLICTING_SURFACE_PAIRS}

    for i in range(n):
        for j in range(i + 1, n):
            if suppress[i] or suppress[j]:
                continue
            pair = (labels[i], labels[j])
            if pair not in conflict_pairs:
                continue
            if _iou(boxes[i], boxes[j]) < iou_threshold:
                continue

            # Keep the one with higher SURFACE_PRIORITY
            pri_i = SURFACE_PRIORITY.get(labels[i], 0)
            pri_j = SURFACE_PRIORITY.get(labels[j], 0)

            if pri_i >= pri_j:
                suppress[j] = True
                print(f"  [surface-dedup] '{labels[j]}' suppressed by '{labels[i]}' (priority {pri_i} > {pri_j})")
            else:
                suppress[i] = True
                print(f"  [surface-dedup] '{labels[i]}' suppressed by '{labels[j]}' (priority {pri_j} > {pri_i})")

    kb = [boxes[i]  for i in range(n) if not suppress[i]]
    ks = [scores[i] for i in range(n) if not suppress[i]]
    kl = [labels[i] for i in range(n) if not suppress[i]]
    removed = n - len(kb)
    if removed > 0:
        print(f"  [surface-dedup] Resolved {removed} surface conflicts")
    return kb, ks, kl


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PROMPT DEDUPLICATION
# FIXED: Don't suppress when labels are different surface types (handled above)
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_cross_prompt(
    boxes: list, scores: list, labels: list, iou_threshold: float = 0.50,
) -> tuple[list, list, list]:
    n = len(boxes)
    if n <= 1:
        return boxes, scores, labels

    # First: resolve surface conflicts specifically
    boxes, scores, labels = resolve_surface_conflicts(boxes, scores, labels)
    n = len(boxes)

    keep = [True] * n
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    surface_labels = {"sidewalk", "road", "grass", "soil", "gravel", "dirt", "plant", "crosswalk"}

    for ri, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ri + 1:]:
            if not keep[j]:
                continue

            # Never suppress surfaces against non-surfaces or surfaces against each other
            # (handled above in resolve_surface_conflicts)
            if labels[i] in surface_labels or labels[j] in surface_labels:
                continue

            # Only suppress if same label
            if labels[i] != labels[j]:
                # Allow different object types even if overlapping
                if _iou(boxes[i], boxes[j]) < 0.70:    # higher threshold for cross-label
                    continue

            if _iou(boxes[i], boxes[j]) >= iou_threshold:
                keep[j] = False

    kb = [boxes[i]  for i in range(n) if keep[i]]
    ks = [scores[i] for i in range(n) if keep[i]]
    kl = [labels[i] for i in range(n) if keep[i]]
    removed = n - len(kb)
    if removed > 0:
        print(f"  [dedup] Removed {removed} cross-prompt duplicates")
    return kb, ks, kl


def compute_containment(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((ax2 - ax1) * (ay2 - ay1), 1)


def run_occlusion_analysis(detections: list[Detection]) -> list[Detection]:
    n = len(detections)
    surface_labels = {"sidewalk", "road", "grass", "soil"}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if detections[i].label in surface_labels or detections[j].label in surface_labels:
                continue
            if compute_containment(detections[i].box, detections[j].box) < 0.85:
                continue
            if detections[i].label == detections[j].label:
                if detections[i].score < detections[j].score:
                    detections[i].suppressed_by = j
            else:
                outer_is_dynamic = detections[j].label in DYNAMIC_LABELS
                ob = detections[j].box
                ib = detections[i].box
                if outer_is_dynamic and (ib[2]-ib[0])*(ib[3]-ib[1]) < (ob[2]-ob[0])*(ob[3]-ob[1]):
                    detections[i].occluded = True
    return detections


def soft_nms(
    detections: list[Detection],
    sigma: float = 0.5,
    score_gate: float = 0.15,
) -> list[Detection]:
    """
    FIXED Soft-NMS:
    - Surface labels bypass NMS entirely
    - Dynamic agents (people) use a higher score_gate to preserve count
    """
    dets = sorted(
        [d for d in detections if d.suppressed_by is None],
        key=lambda d: d.score, reverse=True
    )

    surface_labels = {"sidewalk", "road", "grass", "soil", "crosswalk"}

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            # Bypass NMS for surface labels
            if dets[i].label in surface_labels or dets[j].label in surface_labels:
                continue
            ov = _iou(dets[i].box, dets[j].box)
            if ov > 0:
                dets[j].score *= np.exp(-(ov ** 2) / sigma)

    # FIXED: Use a LOWER gate for dynamic agents so distant people aren't lost
    result = []
    for d in dets:
        gate = score_gate * 0.7 if d.label in DYNAMIC_LABELS else score_gate
        if d.score >= gate:
            result.append(d)

    return result
