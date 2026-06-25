# """
# Guidance text generator — WalkGPT-style natural language navigation
# ====================================================================
# Template-based natural language generation for pedestrian navigation
# guidance. Produces conversational, accessibility-aware descriptions
# that replicate WalkGPT's output style.

# No external LLM required — deterministic, zero-latency templates.
# """
# from typing import Optional
# from config import (
#     OBJECT_ROLES,
#     ACCESSIBILITY_FEATURES,
#     ACCESSIBILITY_HAZARDS,
#     SURFACE_QUALITY,
# )
# from config.labels import get_display_label


# # ── Action → Guidance Templates ─────────────────────────────────────────────

# _ACTION_TEMPLATES = {
#     "move_forward": "The path ahead is clear. Continue walking forward.",
#     "move_forward_crowded": (
#         "The path ahead is passable but crowded. "
#         "Proceed forward with caution, maintaining awareness of others."
#     ),
#     "move_forward_cautious": (
#         "The path ahead is partially obstructed. "
#         "Move forward carefully and watch your step."
#     ),
#     "move_left": "Shift to the left side of the path — it's clearer there.",
#     "move_left_(crowded)": (
#         "Move to the left. It's busy but passable."
#     ),
#     "move_left_(cautious)": (
#         "Try moving left cautiously — the main path is blocked."
#     ),
#     "move_right": "Shift to the right side of the path — it's clearer there.",
#     "move_right_(crowded)": (
#         "Move to the right. It's busy but passable."
#     ),
#     "move_right_(cautious)": (
#         "Try moving right cautiously — the main path is blocked."
#     ),
#     "stop": (
#         "Stop immediately. There is a significant obstruction ahead. "
#         "Reassess your surroundings before proceeding."
#     ),
#     "stop_path_unclear": (
#         "Stop. The path ahead is unclear or fully blocked. "
#         "Look for alternative routes or wait for the path to clear."
#     ),
# }

# _RISK_CONTEXT = {
#     "none": "",
#     "low": "Minimal risk detected. ",
#     "medium": "Moderate risk ahead — stay alert. ",
#     "high": "High risk detected — proceed with extreme caution. ",
#     "urgent": "URGENT: Immediate hazard detected. Stop and reassess. ",
# }


# # ── Main Guidance Generator ─────────────────────────────────────────────────

# def generate_guidance_text(nav_data: dict) -> str:
#     """Generate WalkGPT-style conversational navigation guidance.

#     Args:
#         nav_data: Output dict from build_walkgpt_description(), containing
#                   'action', 'risk', 'obstacles', 'surfaces',
#                   'accessibility', 'depth_info', 'spatial_map'.

#     Returns:
#         Multi-sentence natural language guidance string.
#     """
#     parts: list[str] = []

#     # 1. Risk context (if non-trivial)
#     risk = nav_data.get("risk", "none")
#     risk_text = _RISK_CONTEXT.get(risk, "")
#     if risk_text:
#         parts.append(risk_text)

#     # 2. Primary action guidance
#     action = nav_data.get("action", "stop_path_unclear")
#     action_text = _ACTION_TEMPLATES.get(action, _ACTION_TEMPLATES["stop_path_unclear"])
#     parts.append(action_text)

#     # 3. Obstacle descriptions
#     obstacles = nav_data.get("obstacles", [])
#     if obstacles:
#         obstacle_text = _format_obstacle_sentences(obstacles)
#         if obstacle_text:
#             parts.append(obstacle_text)

#     # 4. Surface / accessibility info
#     accessibility = nav_data.get("accessibility", {})
#     acc_text = format_accessibility_note(
#         accessibility.get("features", []),
#         accessibility.get("hazards", []),
#         accessibility.get("surface", ""),
#         accessibility.get("width_assessment", ""),
#     )
#     if acc_text:
#         parts.append(acc_text)

#     # 5. Depth-based clearance
#     depth_info = nav_data.get("depth_info", {})
#     clearance_m = depth_info.get("path_clearance_m")
#     if clearance_m is not None and clearance_m < 5.0:
#         parts.append(
#             f"Path clearance is approximately {clearance_m:.0f} metres ahead."
#         )

#     return " ".join(parts)


# # ── Obstacle Formatting ─────────────────────────────────────────────────────

# def _format_obstacle_sentences(obstacles: list[str]) -> str:
#     """Convert compact obstacle strings like 'person(3m, center)' into
#     natural language sentences."""
#     if not obstacles:
#         return ""

#     sentences = []
#     for obs in obstacles[:4]:  # Limit to top-4 most relevant
#         # Parse compact format: "label(dist, pos)" or "2x label(dist, pos)"
#         sentences.append(f"Detected: {obs}.")

#     if len(obstacles) > 4:
#         sentences.append(f"Plus {len(obstacles) - 4} other objects nearby.")

#     return " ".join(sentences)


# # ── Accessibility Note ───────────────────────────────────────────────────────

# def format_accessibility_note(
#     features: list[str],
#     hazards: list[str],
#     surface: str = "",
#     width: str = "",
# ) -> str:
#     """Format accessibility information into a readable note.

#     Args:
#         features: Positive accessibility features detected (e.g. curb cuts).
#         hazards: Accessibility hazards detected (e.g. potholes).
#         surface: Surface quality description.
#         width: Path width assessment.

#     Returns:
#         A sentence or empty string.
#     """
#     parts = []

#     if surface:
#         parts.append(f"Surface: {surface}.")

#     if features:
#         feat_str = ", ".join(features[:3])
#         parts.append(f"Accessibility features: {feat_str}.")

#     if hazards:
#         haz_str = ", ".join(hazards[:3])
#         parts.append(f"Accessibility concern: {haz_str}.")

#     if width:
#         parts.append(f"Path width: {width}.")

#     return " ".join(parts)


# # ── Spatial Summary ──────────────────────────────────────────────────────────

# def format_spatial_summary(spatial_map: dict) -> str:
#     """Format spatial_map into a one-line directional summary.

#     Args:
#         spatial_map: {"left": {"status": "...", "objects": [...]}, ...}
#     """
#     parts = []
#     for direction in ["left", "center", "right"]:
#         info = spatial_map.get(direction, {})
#         status = info.get("status", "unknown")
#         objects = info.get("objects", [])

#         if objects:
#             obj_str = ", ".join(objects[:2])
#             parts.append(f"{direction}: {status} ({obj_str})")
#         else:
#             parts.append(f"{direction}: {status}")

#     return " | ".join(parts)


"""
Guidance text generator — WalkGPT-style natural language navigation
"""
from typing import Optional
from config import (
    OBJECT_ROLES,
    ACCESSIBILITY_FEATURES,
    ACCESSIBILITY_HAZARDS,
    SURFACE_QUALITY,
)
from config.labels import get_display_label


_ACTION_TEMPLATES = {
    "move_forward":          "The path ahead is clear. Continue walking forward.",
    "move_forward_crowded":  "The path ahead is passable but crowded. Proceed forward with caution.",
    "move_forward_cautious": "The path ahead is partially obstructed. Move forward carefully.",
    "move_left":             "Shift to the left side of the path — it is clearer there.",
    "move_left_crowded":     "Move to the left. It is busy but passable.",
    "move_left_cautious":    "Try moving left cautiously — the main path is blocked.",
    "move_right":            "Shift to the right side of the path — it is clearer there.",
    "move_right_crowded":    "Move to the right. It is busy but passable.",
    "move_right_cautious":   "Try moving right cautiously — the main path is blocked.",
    "stop":                  "Stop immediately. There is a significant obstruction ahead. Reassess your surroundings.",
    "stop_path_unclear":     "Stop. The path ahead is unclear or fully blocked. Look for an alternative route.",
}

_RISK_CONTEXT = {
    "none":   "",
    "low":    "Minimal risk detected. ",
    "medium": "Moderate risk ahead — stay alert. ",
    "high":   "High risk detected — proceed with extreme caution. ",
    "urgent": "URGENT: Immediate hazard detected. Stop and reassess. ",
}


def generate_guidance_text(nav_data: dict) -> str:
    """Generate WalkGPT-style conversational navigation guidance."""
    parts: list[str] = []

    risk = nav_data.get("risk", "none")
    risk_text = _RISK_CONTEXT.get(risk, "")
    if risk_text:
        parts.append(risk_text)

    action = nav_data.get("action", "stop_path_unclear")
    parts.append(_ACTION_TEMPLATES.get(action, _ACTION_TEMPLATES["stop_path_unclear"]))

    obstacles = nav_data.get("obstacles", [])
    if obstacles:
        obstacle_text = _format_obstacle_sentences(obstacles)
        if obstacle_text:
            parts.append(obstacle_text)

    accessibility = nav_data.get("accessibility", {})
    acc_text = format_accessibility_note(
        accessibility.get("features", []),
        accessibility.get("hazards", []),
        accessibility.get("surface", ""),
        accessibility.get("width_assessment", ""),
    )
    if acc_text:
        parts.append(acc_text)

    return " ".join(parts)


def _format_obstacle_sentences(obstacles: list[str]) -> str:
    if not obstacles:
        return ""
    sentences = [f"Detected: {obs}." for obs in obstacles[:4]]
    if len(obstacles) > 4:
        sentences.append(f"Plus {len(obstacles) - 4} other objects nearby.")
    return " ".join(sentences)


def format_accessibility_note(
    features: list[str],
    hazards: list[str],
    surface: str = "",
    width: str = "",
) -> str:
    parts = []
    if surface:
        parts.append(f"Surface: {surface}.")
    if features:
        parts.append(f"Accessibility features: {', '.join(features[:3])}.")
    if hazards:
        parts.append(f"Accessibility concern: {', '.join(hazards[:3])}.")
    if width:
        parts.append(f"Path width: {width}.")
    return " ".join(parts)


def format_spatial_summary(spatial_map: dict) -> str:
    parts = []
    for direction in ["left", "center", "right"]:
        info    = spatial_map.get(direction, {})
        status  = info.get("status", "unknown")
        objects = info.get("objects", [])
        if objects:
            parts.append(f"{direction}: {status} ({', '.join(objects[:2])})")
        else:
            parts.append(f"{direction}: {status}")
    return " | ".join(parts)