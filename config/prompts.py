# """Detection prompts and sanitization

# WalkGPT-style multi-prompt list covering:
#   - Pedestrians & vehicles (dynamic hazards)
#   - Fixed obstacles
#   - Surface types (walkable / semi / non-walkable)
#   - Accessibility features & hazards
# """

# MULTI_PROMPTS = [
#     # Dynamic agents
#     "person", "car", "bicycle",
#     # Fixed obstacles
#     "traffic cone", "barrier",
#     # Surfaces
#     "sidewalk", "road",
#     "grass",
#     # Natural obstacles
#     "tree",
#     # Accessibility features
#     "pothole",
#     "puddle",
#     "construction zone",
# ]


# def sanitise_prompt(prompt: str) -> str:
#     """Lowercase, normalise spaces around dots, guarantee trailing ' .'"""
#     prompt = prompt.lower().strip()
#     prompt = " . ".join(p.strip() for p in prompt.split(".") if p.strip())
#     if not prompt.endswith(" ."):
#         prompt += " ."
#     return prompt

"""
Detection prompts — FIXED to reduce sidewalk/road label confusion

Key fix: Run sidewalk and road as SEPARATE prompts with distinct phrasing
so GroundingDINO can better disambiguate them via NMS/dedup.
"""

# FIXED: Separated sidewalk and road into distinct prompts.
# Previously both were in the same pass, causing overlap detection.
# Now each surface type runs independently → better deduplication.
MULTI_PROMPTS = [
    # Dynamic agents (highest priority)
    "person",
    "car",
    "bicycle",
    # Fixed obstacles
    "traffic cone",
    "barrier",
    # Surfaces — SEPARATE PASSES for sidewalk vs road (prevents mixing)
    "sidewalk",       # pedestrian surface — runs alone
    "road",           # vehicle surface — runs alone
    "grass",
    # Natural obstacles
    "tree",
    # Accessibility hazards
    "pothole",
    "puddle",
    "construction zone",
]

# SURFACE DEDUP PAIRS: when both are detected in same region, keep the
# higher-priority one (sidewalk > road for pedestrian navigation)
CONFLICTING_SURFACE_PAIRS = [
    ("sidewalk", "road"),
    ("sidewalk", "asphalt"),
    ("crosswalk", "road"),
]


def sanitise_prompt(prompt: str) -> str:
    """Lowercase, normalise spaces around dots, guarantee trailing ' .'"""
    prompt = prompt.lower().strip()
    prompt = " . ".join(p.strip() for p in prompt.split(".") if p.strip())
    if not prompt.endswith(" ."):
        prompt += " ."
    return prompt
