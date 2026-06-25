# """
# Rule-based fallback guidance generator - ACTION ALIGNED
# """
# from typing import Dict, Any


# def generate_fallback_guidance(nav_data: Dict[str, Any]) -> str:
#     """Generate rule-based guidance when LLM is unavailable."""
#     action = nav_data.get("action", "move_forward")
#     risk = nav_data.get("risk", "none")
#     free_space = nav_data.get("free_space", {})
#     obstacles = nav_data.get("obstacles", [])
    
#     # ── Always start with action ──────────────────────────────────────────
#     action_map = {
#         "move_forward": "Move forward.",
#         "move_forward_crowded": "Move forward.",
#         "move_forward_cautious": "Move forward cautiously.",
#         "move_left": "Move left.",
#         "move_left_crowded": "Move left.",
#         "move_left_cautious": "Move left cautiously.",
#         "move_right": "Move right.",
#         "move_right_crowded": "Move right.",
#         "move_right_cautious": "Move right cautiously.",
#         "stop": "Stop.",
#         "stop_path_unclear": "Stop.",
#     }
    
#     action_text = action_map.get(action, "Move forward.")
    
#     # ── Add obstacle context ──────────────────────────────────────────────
#     obstacle_text = ""
#     if obstacles:
#         main_obs = obstacles[0]
#         obstacle_text = f" {main_obs}."
    
#     # ── Add risk ──────────────────────────────────────────────────────────
#     risk_text = f" Risk: {risk}."
    
#     return f"{action_text}{obstacle_text}{risk_text}"


# def should_use_llm(nav_data: Dict[str, Any]) -> bool:
#     """
#     Determine if LLM should be used based on data complexity.
#     """
#     obstacles = nav_data.get("obstacles", [])
#     risk = nav_data.get("risk", "none")
#     free_space = nav_data.get("free_space", {})
    
#     # Use LLM for complex scenes
#     return (
#         len(obstacles) >= 3 or                # Multiple obstacles
#         risk in ("urgent", "high") or         # High risk
#         len([s for s in free_space.values() if s in ("blocked", "crowded")]) >= 2  # Multiple issues
#     )


"""
Rule-based fallback guidance — FIXED: direction-aware, more informative
"""
from typing import Dict, Any


def generate_fallback_guidance(nav_data: Dict[str, Any]) -> str:
    """Generate rule-based guidance when LLM is unavailable."""
    action     = nav_data.get("action", "move_forward")
    risk       = nav_data.get("risk", "none")
    free_space = nav_data.get("free_space", {})
    obstacles  = nav_data.get("obstacles", [])

    action_map = {
        "move_forward":          "Move forward.",
        "move_forward_crowded":  "Move forward.",
        "move_forward_cautious": "Move forward cautiously.",
        "move_left":             "Move left.",
        "move_left_crowded":     "Move left.",
        "move_left_cautious":    "Move left cautiously.",
        "move_right":            "Move right.",
        "move_right_crowded":    "Move right.",
        "move_right_cautious":   "Move right cautiously.",
        "stop":                  "Stop.",
        "stop_path_unclear":     "Stop.",
    }

    action_text = action_map.get(action, "Move forward.")

    # Build obstacle context (up to 2 obstacles)
    obstacle_text = ""
    if obstacles:
        obs_parts = [obs for obs in obstacles[:2]]
        obstacle_text = " " + "; ".join(obs_parts) + "."

    # FIXED: Include direction info in fallback
    center_state = free_space.get("center", "unknown")
    direction_note = ""
    if action in ("move_left", "move_left_crowded", "move_left_cautious"):
        direction_note = " Left path is clearer."
    elif action in ("move_right", "move_right_crowded", "move_right_cautious"):
        direction_note = " Right path is clearer."

    risk_text = f" Risk: {risk}."

    return f"{action_text}{obstacle_text}{direction_note}{risk_text}"


def should_use_llm(nav_data: Dict[str, Any]) -> bool:
    """
    Determine if LLM should be used based on scene complexity.
    FIXED: use LLM for more cases to improve guidance quality.
    """
    obstacles  = nav_data.get("obstacles", [])
    risk       = nav_data.get("risk", "none")
    free_space = nav_data.get("free_space", {})

    blocked_count = sum(1 for s in free_space.values() if s == "blocked")
    crowded_count = sum(1 for s in free_space.values() if s == "crowded")

    return (
        len(obstacles) >= 2 or                        # Multiple obstacles
        risk in ("urgent", "high", "medium") or       # Any notable risk
        blocked_count >= 1 or                         # Any blocked path
        crowded_count >= 2                             # Multiple crowded
    )
