"""Detection prompts and sanitization

WalkGPT-style multi-prompt list covering:
  - Pedestrians & vehicles (dynamic hazards)
  - Fixed obstacles
  - Surface types (walkable / semi / non-walkable)
  - Accessibility features & hazards
"""

MULTI_PROMPTS = [
    # Dynamic agents
    "person", "car", "bicycle",
    # Fixed obstacles
    "traffic cone", "barrier",
    # Surfaces
    "sidewalk", "road",
    "grass",
    # Natural obstacles
    "tree",
    # Accessibility features
    "pothole",
    "puddle",
    "construction zone",
]


def sanitise_prompt(prompt: str) -> str:
    """Lowercase, normalise spaces around dots, guarantee trailing ' .'"""
    prompt = prompt.lower().strip()
    prompt = " . ".join(p.strip() for p in prompt.split(".") if p.strip())
    if not prompt.endswith(" ."):
        prompt += " ."
    return prompt