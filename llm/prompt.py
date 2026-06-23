"""
Prompt Templates for Phi-2 - STRICT MODE (FINAL STABLE)
"""
from typing import Dict, Any


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """STRICT NAVIGATION ASSISTANT

RULES:
1. Output EXACTLY 1 sentence
2. MUST start with the given ACTION
3. MUST include the given RISK
4. MUST mention the main obstacle
5. DO NOT change the action

FORMAT:
ACTION. Obstacle info. Risk: <risk>

EXAMPLES:
Move left. Traffic cone at mid right. Risk: low.
Stop. Person at near center. Risk: urgent.
Move forward. Path clear. Risk: none.

NO explanations. NO extra sentences.
"""


# ─────────────────────────────────────────────
# NAVIGATION PROMPT
# ─────────────────────────────────────────────
def build_navigation_prompt(nav_data: Dict[str, Any]) -> str:
    """Strict prompt enforcing action + risk + obstacle."""

    action = nav_data.get("action", "move_forward")
    risk = nav_data.get("risk", "none")
    obstacles = nav_data.get("obstacles", [])
    free_space = nav_data.get("free_space", {})

    action_text = action.replace("_", " ")

    # pick only ONE obstacle (main)
    main_obstacle = obstacles[0] if obstacles else "path clear"

    prompt = f"""STRICT NAVIGATION TASK

FOLLOW RULES EXACTLY:

- Start with: "{action_text}"
- Include risk: "{risk}"
- Mention obstacle: "{main_obstacle}"
- Output ONLY 1 sentence

DATA:
Action = {action_text}
Risk = {risk}
Obstacle = {main_obstacle}
Path = center:{free_space.get('center','?')} left:{free_space.get('left','?')} right:{free_space.get('right','?')}

OUTPUT:
"""

    return prompt


# ─────────────────────────────────────────────
# WALKGPT PROMPT
# ─────────────────────────────────────────────
def build_walkgpt_prompt(nav_data: Dict[str, Any]) -> str:
    """Strict WalkGPT-style prompt (controlled)."""

    action = nav_data.get("action", "move_forward")
    risk = nav_data.get("risk", "none")
    obstacles = nav_data.get("obstacles", [])
    spatial_map = nav_data.get("spatial_map", {})

    action_text = action.replace("_", " ")
    main_obstacle = obstacles[0] if obstacles else "path clear"

    # compact spatial summary (minimal tokens)
    spatial_summary = []
    for direction in ["left", "center", "right"]:
        d = spatial_map.get(direction, {})
        status = d.get("status", "?")
        spatial_summary.append(f"{direction}:{status}")

    prompt = f"""STRICT NAVIGATION TASK

Start with: "{action_text}"
Include risk: "{risk}"
Mention obstacle: "{main_obstacle}"

Map: {' '.join(spatial_summary)}

OUTPUT:
"""

    return prompt


# ─────────────────────────────────────────────
# RESPONSE PARSER (CRITICAL)
# ─────────────────────────────────────────────
def parse_navigation_response(
    response: str,
    expected_action: str = "move_forward",
    expected_risk: str = "none"
) -> Dict[str, str]:
    """
    Enforce correctness AFTER LLM generation.
    This prevents hallucination + action mismatch.
    """

    if not response:
        return {
            "guidance": f"{expected_action.replace('_',' ')}. Risk: {expected_risk}.",
            "source": "fallback"
        }

    response = response.strip()
    action_text = expected_action.replace("_", " ")

    # ─────────────────────────────
    # 1. FORCE ACTION AT START
    # ─────────────────────────────
    if not response.lower().startswith(action_text.lower()):
        # remove existing first sentence if conflicting
        parts = response.split(".")
        if len(parts) > 1:
            response = parts[1].strip()
        response = f"{action_text}. {response}"

    # ─────────────────────────────
    # 2. REMOVE MULTI-SENTENCE OUTPUT
    # ─────────────────────────────
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if len(sentences) > 1:
        # keep only first meaningful sentence
        response = sentences[0] + "."
    else:
        response = sentences[0] + "."

    # ─────────────────────────────
    # 3. FORCE RISK PRESENCE
    # ─────────────────────────────
    if "risk:" not in response.lower():
        response = response.rstrip(".") + f". Risk: {expected_risk}."

    # ─────────────────────────────
    # 4. REMOVE BAD PHRASES
    # ─────────────────────────────
    forbidden_phrases = [
        "i suggest",
        "you should",
        "be careful",
        "watch out",
        "stay alert",
        "keep an eye",
        "it seems",
        "the person is",
    ]

    lower_resp = response.lower()
    for phrase in forbidden_phrases:
        if phrase in lower_resp:
            return {
                "guidance": f"{action_text}. Risk: {expected_risk}.",
                "source": "fallback_clean"
            }

    # ─────────────────────────────
    # 5. FINAL CLEAN
    # ─────────────────────────────
    response = response.replace("  ", " ").strip()

    return {
        "guidance": response,
        "source": "llm"
    }