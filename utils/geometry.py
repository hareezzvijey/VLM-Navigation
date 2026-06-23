"""
Geometry utilities: centerline, corridor, position classification
"""
import numpy as np
from typing import Optional, Callable, Tuple
from config import COL_LEFT_END, COL_RIGHT_START


def classify_hpos(x: float, img_w: int) -> str:
    """[P1] Narrow center zone: 38–62% instead of 33–67%."""
    ratio = x / img_w
    if ratio < COL_LEFT_END:
        return "left"
    elif ratio > COL_RIGHT_START:
        return "right"
    else:
        return "center"


def get_hpos_x(det, mask: Optional[np.ndarray], img_w: int) -> float:
    """
    Return the x-coordinate to use for horizontal position classification.
    Priority:
      1. SAM mask centroid (if mask available) — [P4]
      2. Box foot-point centre (x1+x2)/2      — existing behaviour
    """
    if mask is not None:
        cols = np.where(mask.any(axis=0))[0]
        if len(cols) > 0:
            return float(cols.mean())
    return float((det.box[0] + det.box[2]) / 2)


def extract_sidewalk_centerline(
    sidewalk_mask: Optional[np.ndarray],
    zone_top: int,
    img_h: int,
    img_w: int,
) -> Tuple[Optional[Callable], int, Optional[Tuple]]:
    """
    [DI1] Extract sidewalk centerline via per-row midpoint regression.
    """
    if sidewalk_mask is None:
        return None, 0, None

    zone_mask = sidewalk_mask[zone_top:, :]
    rows_y: list[int] = []
    mid_x: list[float] = []
    widths: list[int] = []

    for r in range(zone_mask.shape[0]):
        cols = np.where(zone_mask[r])[0]
        if len(cols) >= 5:
            rows_y.append(r + zone_top)
            mid_x.append((int(cols[0]) + int(cols[-1])) / 2.0)
            widths.append(int(cols[-1]) - int(cols[0]))

    if len(rows_y) < 10:
        return None, 0, None

    poly_deg = 2 if len(rows_y) >= 20 else 1
    coeffs = np.polyfit(rows_y, mid_x, poly_deg)
    slope, intercept = (coeffs[-2], coeffs[-1])
    path_width_px = int(np.median(widths))

    def centerline_fn(y: float) -> float:
        return float(slope * y + intercept)

    return centerline_fn, path_width_px, (slope, intercept)


def compute_corridor_bounds(
    sidewalk_mask: Optional[np.ndarray],
    centerline_fn: Optional[Callable],
    path_width_px: int,
    img_h: int,
    img_w: int,
    zone_top: int,
) -> dict[str, tuple[int, int]]:

    if centerline_fn is not None and path_width_px >= 30:
        y_eval = int(img_h * 0.85)
        cl_x = centerline_fn(y_eval)

        # 🔥 FIX: INVALID centerline → fallback
        if not (0 <= cl_x <= img_w):
            print(f"  [DI1] Invalid centerline (x={cl_x:.0f}) — fallback used")
            return {
                "left": (0, int(img_w * COL_LEFT_END)),
                "center": (int(img_w * COL_LEFT_END), int(img_w * COL_RIGHT_START)),
                "right": (int(img_w * COL_RIGHT_START), img_w),
            }

        half_pw = path_width_px / 2.0
        left_edge = max(0, int(cl_x - half_pw))
        right_edge = min(img_w, int(cl_x + half_pw))

        # 🔥 FIX: TOO NARROW → fallback
        if (right_edge - left_edge) < img_w * 0.25:
            print("  [DI1] Corridor too narrow — fallback used")
            return {
                "left": (0, int(img_w * COL_LEFT_END)),
                "center": (int(img_w * COL_LEFT_END), int(img_w * COL_RIGHT_START)),
                "right": (int(img_w * COL_RIGHT_START), img_w),
            }

        third = max((right_edge - left_edge) // 3, 1)

        bounds = {
            "left": (left_edge, left_edge + third),
            "center": (left_edge + third, left_edge + 2 * third),
            "right": (left_edge + 2 * third, right_edge),
        }

        print(f"  [DI1] Centerline at y={y_eval}: x={cl_x:.0f}, "
              f"width={path_width_px}px -> corridor {left_edge}-{right_edge}")

        return bounds

    # DEFAULT fallback
    return {
        "left": (0, int(img_w * COL_LEFT_END)),
        "center": (int(img_w * COL_LEFT_END), int(img_w * COL_RIGHT_START)),
        "right": (int(img_w * COL_RIGHT_START), img_w),
    }