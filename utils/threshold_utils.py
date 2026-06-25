# """Threshold utility functions for CPU/GPU adaptive thresholds."""
# from models.loader import get_cpu_fallback


# def cpu_adjust(threshold: float) -> float:
#     """Scale GPU threshold down proportionally for CPU fallback mode.

#     When running on CPU (GroundingDINO C++ ops unavailable), the model
#     produces lower-confidence predictions, so thresholds are scaled to ~45%
#     of their GPU values to avoid dropping all detections.
#     """
#     return round(threshold * 0.45, 3) if get_cpu_fallback() else threshold


# def get_box_thresholds() -> tuple[float, list[float]]:
#     """Return (default_box_threshold, threshold_ladder) adjusted for device.

#     Returns:
#         A tuple of (BOX_THRESHOLD_DEFAULT, THRESHOLD_LADDER) that is
#         pre-adjusted for CPU vs GPU runtime.
#     """
#     if get_cpu_fallback():
#         return 0.15, [0.15, 0.10, 0.07]
#     else:
#         from config.thresholds import BOX_THRESHOLD_DEFAULT, THRESHOLD_LADDER
#         return BOX_THRESHOLD_DEFAULT, THRESHOLD_LADDER


"""Threshold utility functions for CPU/GPU adaptive thresholds."""
from models.loader import get_cpu_fallback


def cpu_adjust(threshold: float) -> float:
    """Scale GPU threshold down for CPU fallback mode."""
    return round(threshold * 0.45, 3) if get_cpu_fallback() else threshold


def get_box_thresholds() -> tuple[float, list[float]]:
    """Return (default_box_threshold, threshold_ladder) adjusted for device."""
    if get_cpu_fallback():
        return 0.15, [0.15, 0.10, 0.07]
    else:
        from config.thresholds import BOX_THRESHOLD_DEFAULT, THRESHOLD_LADDER
        return BOX_THRESHOLD_DEFAULT, THRESHOLD_LADDER
