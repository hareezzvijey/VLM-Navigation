"""Display labels for LLM output"""
from typing import Dict

LABEL_DISPLAY_NAMES: Dict[str, str] = {
    "bicycle": "bike",
    "motorcycle": "motorbike",
    "traffic cone": "cone",
    "traffic light": "traffic light",
    "traffic sign": "sign",
    "stop sign": "stop sign",
    "fire hydrant": "hydrant",
    # Accessibility
    "construction zone": "construction",
    "tactile paving": "tactile strip",
    "curb cut": "curb cut",
    "uneven surface": "uneven ground",
}


def get_display_label(label: str) -> str:
    """Get display-friendly label name for LLM output."""
    return LABEL_DISPLAY_NAMES.get(label, label)