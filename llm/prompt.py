# """
# Prompt Templates for Phi-2 - STRICT MODE (FINAL STABLE)
# """
# from typing import Dict, Any


# # ─────────────────────────────────────────────
# # SYSTEM PROMPT
# # ─────────────────────────────────────────────
# SYSTEM_PROMPT = """STRICT NAVIGATION ASSISTANT

# RULES:
# 1. Output EXACTLY 1 sentence
# 2. MUST start with the given ACTION
# 3. MUST include the given RISK
# 4. MUST mention the main obstacle
# 5. DO NOT change the action

# FORMAT:
# ACTION. Obstacle info. Risk: <risk>

# EXAMPLES:
# Move left. Traffic cone at mid right. Risk: low.
# Stop. Person at near center. Risk: urgent.
# Move forward. Path clear. Risk: none.

# NO explanations. NO extra sentences.
# """


# # ─────────────────────────────────────────────
# # NAVIGATION PROMPT
# # ─────────────────────────────────────────────
# def build_navigation_prompt(nav_data: Dict[str, Any]) -> str:
#     """Strict prompt enforcing action + risk + obstacle."""

#     action = nav_data.get("action", "move_forward")
#     risk = nav_data.get("risk", "none")
#     obstacles = nav_data.get("obstacles", [])
#     free_space = nav_data.get("free_space", {})

#     action_text = action.replace("_", " ")

#     # pick only ONE obstacle (main)
#     main_obstacle = obstacles[0] if obstacles else "path clear"

#     prompt = f"""STRICT NAVIGATION TASK

# FOLLOW RULES EXACTLY:

# - Start with: "{action_text}"
# - Include risk: "{risk}"
# - Mention obstacle: "{main_obstacle}"
# - Output ONLY 1 sentence

# DATA:
# Action = {action_text}
# Risk = {risk}
# Obstacle = {main_obstacle}
# Path = center:{free_space.get('center','?')} left:{free_space.get('left','?')} right:{free_space.get('right','?')}

# OUTPUT:
# """

#     return prompt


# # ─────────────────────────────────────────────
# # WALKGPT PROMPT
# # ─────────────────────────────────────────────
# def build_walkgpt_prompt(nav_data: Dict[str, Any]) -> str:
#     """Strict WalkGPT-style prompt (controlled)."""

#     action = nav_data.get("action", "move_forward")
#     risk = nav_data.get("risk", "none")
#     obstacles = nav_data.get("obstacles", [])
#     spatial_map = nav_data.get("spatial_map", {})

#     action_text = action.replace("_", " ")
#     main_obstacle = obstacles[0] if obstacles else "path clear"

#     # compact spatial summary (minimal tokens)
#     spatial_summary = []
#     for direction in ["left", "center", "right"]:
#         d = spatial_map.get(direction, {})
#         status = d.get("status", "?")
#         spatial_summary.append(f"{direction}:{status}")

#     prompt = f"""STRICT NAVIGATION TASK

# Start with: "{action_text}"
# Include risk: "{risk}"
# Mention obstacle: "{main_obstacle}"

# Map: {' '.join(spatial_summary)}

# OUTPUT:
# """

#     return prompt


# # ─────────────────────────────────────────────
# # RESPONSE PARSER (CRITICAL)
# # ─────────────────────────────────────────────
# def parse_navigation_response(
#     response: str,
#     expected_action: str = "move_forward",
#     expected_risk: str = "none"
# ) -> Dict[str, str]:
#     """
#     Enforce correctness AFTER LLM generation.
#     This prevents hallucination + action mismatch.
#     """

#     if not response:
#         return {
#             "guidance": f"{expected_action.replace('_',' ')}. Risk: {expected_risk}.",
#             "source": "fallback"
#         }

#     response = response.strip()
#     action_text = expected_action.replace("_", " ")

#     # ─────────────────────────────
#     # 1. FORCE ACTION AT START
#     # ─────────────────────────────
#     if not response.lower().startswith(action_text.lower()):
#         # remove existing first sentence if conflicting
#         parts = response.split(".")
#         if len(parts) > 1:
#             response = parts[1].strip()
#         response = f"{action_text}. {response}"

#     # ─────────────────────────────
#     # 2. REMOVE MULTI-SENTENCE OUTPUT
#     # ─────────────────────────────
#     sentences = [s.strip() for s in response.split(".") if s.strip()]
#     if len(sentences) > 1:
#         # keep only first meaningful sentence
#         response = sentences[0] + "."
#     else:
#         response = sentences[0] + "."

#     # ─────────────────────────────
#     # 3. FORCE RISK PRESENCE
#     # ─────────────────────────────
#     if "risk:" not in response.lower():
#         response = response.rstrip(".") + f". Risk: {expected_risk}."

#     # ─────────────────────────────
#     # 4. REMOVE BAD PHRASES
#     # ─────────────────────────────
#     forbidden_phrases = [
#         "i suggest",
#         "you should",
#         "be careful",
#         "watch out",
#         "stay alert",
#         "keep an eye",
#         "it seems",
#         "the person is",
#     ]

#     lower_resp = response.lower()
#     for phrase in forbidden_phrases:
#         if phrase in lower_resp:
#             return {
#                 "guidance": f"{action_text}. Risk: {expected_risk}.",
#                 "source": "fallback_clean"
#             }

#     # ─────────────────────────────
#     # 5. FINAL CLEAN
#     # ─────────────────────────────
#     response = response.replace("  ", " ").strip()

#     return {
#         "guidance": response,
#         "source": "llm"
#     }


"""
Prompt Templates — FIXED: richer context, stricter parsing, no hallucinations

Key fixes:
1. Full spatial map included in prompt (not just main obstacle)
2. Multiple obstacle context passed
3. Response parser catches more edge cases
4. Forbidden phrase list expanded
"""
from typing import Dict, Any


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a pedestrian navigation assistant for visually impaired users.

STRICT OUTPUT RULES:
1. Output EXACTLY 1 sentence (ending with a period).
2. MUST start with the exact ACTION word(s) given.
3. MUST include "Risk: <level>" at the end.
4. MUST mention the main obstacle if one exists.
5. DO NOT change or rephrase the action.
6. DO NOT add extra sentences or explanations.
7. DO NOT say "I suggest", "you should", "be careful", "watch out".

FORMAT:
<action>. <obstacle info>. Risk: <risk>.

EXAMPLES:
Move forward. Path is clear. Risk: none.
Move left. Traffic cone ahead at center. Risk: low.
Stop. Person blocking path at near center. Risk: urgent.
Move forward cautiously. Bicycle at mid right. Risk: medium.
"""


# ─────────────────────────────────────────────
# NAVIGATION PROMPT — FIXED: includes full spatial context
# ─────────────────────────────────────────────
def build_navigation_prompt(nav_data: Dict[str, Any]) -> str:
    action     = nav_data.get("action", "move_forward")
    risk       = nav_data.get("risk", "none")
    obstacles  = nav_data.get("obstacles", [])
    free_space = nav_data.get("free_space", {})

    action_text    = action.replace("_", " ")
    main_obstacle  = obstacles[0] if obstacles else "path clear"

    # FIXED: include all obstacles (up to 3) for richer context
    other_obstacles = ""
    if len(obstacles) > 1:
        other_obstacles = f"\nOther obstacles: {', '.join(obstacles[1:3])}"

    prompt = f"""NAVIGATION TASK — Follow rules exactly.

ACTION = "{action_text}"
RISK   = "{risk}"
MAIN OBSTACLE = "{main_obstacle}"{other_obstacles}
PATH STATUS = center:{free_space.get('center','?')} | left:{free_space.get('left','?')} | right:{free_space.get('right','?')}

Start your response with "{action_text}" and end with "Risk: {risk}."
Output ONE sentence only.

OUTPUT:
"""
    return prompt


# ─────────────────────────────────────────────
# WALKGPT PROMPT — FIXED: richer spatial map
# ─────────────────────────────────────────────
def build_walkgpt_prompt(nav_data: Dict[str, Any]) -> str:
    action     = nav_data.get("action", "move_forward")
    risk       = nav_data.get("risk", "none")
    obstacles  = nav_data.get("obstacles", [])
    spatial_map = nav_data.get("spatial_map", {})
    accessibility = nav_data.get("accessibility", {})

    action_text   = action.replace("_", " ")
    main_obstacle = obstacles[0] if obstacles else "path clear"

    # Build compact spatial summary
    spatial_lines = []
    for direction in ["left", "center", "right"]:
        d       = spatial_map.get(direction, {})
        status  = d.get("status", "?")
        objects = d.get("objects", [])
        obj_str = ", ".join(objects[:2]) if objects else "clear"
        spatial_lines.append(f"  {direction}: {status} ({obj_str})")

    spatial_summary = "\n".join(spatial_lines)
    surface = accessibility.get("surface", "unknown")
    width   = accessibility.get("width_assessment", "unknown")

    prompt = f"""WALKGPT NAVIGATION TASK

ACTION = "{action_text}"
RISK   = "{risk}"
MAIN OBSTACLE = "{main_obstacle}"

SPATIAL MAP:
{spatial_summary}

SURFACE: {surface} | WIDTH: {width}

Start with "{action_text}". Mention obstacle. End with "Risk: {risk}."
ONE sentence only.

OUTPUT:
"""
    return prompt


# ─────────────────────────────────────────────
# RESPONSE PARSER — FIXED: catches more edge cases
# ─────────────────────────────────────────────
def parse_navigation_response(
    response: str,
    expected_action: str = "move_forward",
    expected_risk: str = "none",
) -> Dict[str, str]:
    """
    Enforce correctness AFTER LLM generation.
    Prevents hallucination and action mismatch.
    """
    if not response or len(response.strip()) < 5:
        return {
            "guidance": f"{expected_action.replace('_', ' ')}. Risk: {expected_risk}.",
            "source": "fallback"
        }

    response = response.strip()
    action_text = expected_action.replace("_", " ")

    # ── 1. Remove prompt leakage ──────────────────────────────
    # Sometimes LLM repeats the prompt
    for marker in ["OUTPUT:", "OUTPUT :", "Response:", "Answer:"]:
        if marker in response:
            response = response.split(marker)[-1].strip()

    # ── 2. Force action at start ──────────────────────────────
    if not response.lower().startswith(action_text.lower()):
        # Try to find the action somewhere in the first sentence
        parts = response.split(".")
        found = False
        for idx, part in enumerate(parts):
            if action_text.lower() in part.lower():
                response = part.strip() + "." + ".".join(parts[idx+1:])
                found = True
                break
        if not found:
            response = f"{action_text}. {response}"

    # ── 3. Keep only first sentence ───────────────────────────
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if sentences:
        response = sentences[0] + "."
    else:
        response = f"{action_text}. Risk: {expected_risk}."

    # ── 4. Force risk at end ──────────────────────────────────
    if "risk:" not in response.lower():
        response = response.rstrip(".") + f". Risk: {expected_risk}."

    # ── 5. Remove bad phrases ─────────────────────────────────
    forbidden_phrases = [
        "i suggest", "you should", "be careful", "watch out",
        "stay alert", "keep an eye", "it seems", "the person is",
        "please", "i recommend", "make sure", "i would", "remember",
        "always", "never", "try to",
    ]
    lower_resp = response.lower()
    for phrase in forbidden_phrases:
        if phrase in lower_resp:
            return {
                "guidance": f"{action_text}. Risk: {expected_risk}.",
                "source": "fallback_clean"
            }

    # ── 6. Length guard (max ~150 chars for one sentence) ────
    if len(response) > 160:
        # Truncate at last period before 150
        truncated = response[:150]
        last_dot = truncated.rfind(".")
        if last_dot > 20:
            response = truncated[:last_dot + 1]
        else:
            response = f"{action_text}. Risk: {expected_risk}."

    # ── 7. Final cleanup ──────────────────────────────────────
    response = response.replace("  ", " ").strip()
    if not response.endswith("."):
        response += "."

    return {
        "guidance": response,
        "source":   "llm"
    }
