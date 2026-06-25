# """Navigation: corridor split, free-space thresholds, path width"""
# from typing import Dict

# # ── Corridor Split ─────────────────────────────────────────────────
# # [P1] Narrow center zone: L=0–38%, C=38–62%, R=62–100%
# COL_LEFT_END = 0.38
# COL_RIGHT_START = 0.62

# # ── Free-space thresholds ──────────────────────────────────────────
# WALKABLE_COVERAGE = 0.25          # Ped surface must cover ≥25%
# NON_WALKABLE_COVERAGE = 0.30      # Non-walkable ≥30% → blocked
# SW_BLOCK_COVERAGE = 0.50          # Obstacle covers ≥50% of column
# SW_CROWD_COVERAGE = 0.20          # Obstacle covers ≥20% → crowded
# SW_FOOT_OVERLAP_RATIO = 0.15      # Overlap with sidewalk foot

# # ── Path Width ─────────────────────────────────────────────────────
# # [DI5] ADA 2010 §403.5: min 1.2m / 48in clear path
# NARROW_RATIO = 0.08               # < 8% of image → constrained
# TIGHT_RATIO = 0.04                # < 4% of image → single-file


"""Navigation: corridor split, free-space thresholds, path width — FIXED"""
from typing import Dict

# ── Corridor Split ─────────────────────────────────────────────────────────
# Narrow center zone: L=0–38%, C=38–62%, R=62–100%
COL_LEFT_END    = 0.38
COL_RIGHT_START = 0.62

# ── Free-space thresholds ──────────────────────────────────────────────────
# FIXED: Relaxed WALKABLE_COVERAGE so "uncertain" appears less often
WALKABLE_COVERAGE     = 0.15   # was 0.25 — too strict, caused many "uncertain"
NON_WALKABLE_COVERAGE = 0.35   # was 0.30 — slightly more permissive
SW_BLOCK_COVERAGE     = 0.45   # obstacle covers ≥45% → blocked
SW_CROWD_COVERAGE     = 0.18   # obstacle covers ≥18% → crowded
SW_FOOT_OVERLAP_RATIO = 0.12   # overlap with sidewalk foot

# ── Path Width ─────────────────────────────────────────────────────────────
# ADA 2010 §403.5: min 1.2m / 48in clear path
NARROW_RATIO = 0.08    # < 8% of image → constrained
TIGHT_RATIO  = 0.04    # < 4% of image → single-file

# ── Uncertain resolution ───────────────────────────────────────────────────
# FIXED: Instead of assuming uncertain=walkable (dangerous),
# use a confidence-based approach
UNCERTAIN_MIN_SURFACE_SCORE = 0.08   # if surface coverage ≥ this, treat as walkable