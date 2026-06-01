"""
src/data/constants.py
---------------------
Pure, immutable constants used across PathPilot AI.

No external dependencies — safe to import anywhere.
"""

# ---------------------------------------------------------------------------
# Project-idea scoring
# ---------------------------------------------------------------------------

#: Weights for the 7 scoring dimensions.  Must sum to 1.0.
#: Adjust here to rebalance priorities without touching business logic.
SCORING_WEIGHTS: dict[str, float] = {
    "resume_value":            0.20,
    "agentic_fit":             0.15,
    "buildability_4_6_weeks":  0.15,
    "personal_relevance":      0.10,
    "technical_depth":         0.15,
    "differentiation":         0.10,
    "recruiter_explainability": 0.15,
}

assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 1e-9, (
    "SCORING_WEIGHTS must sum to exactly 1.0"
)

# ---------------------------------------------------------------------------
# JD seniority inference
# ---------------------------------------------------------------------------

#: Keywords that suggest a senior-level role when found in a JD.
SENIORITY_SIGNALS: dict[str, list[str]] = {
    "junior": ["entry level", "0-2 years", "junior", "associate", "new grad"],
    "mid":    ["2-5 years", "mid-level", "mid level", "3+ years"],
    "senior": ["5+ years", "senior", "lead", "principal", "staff", "architect"],
}

# ---------------------------------------------------------------------------
# Roadmap defaults
# ---------------------------------------------------------------------------

#: Hard cap on the number of roadmap weeks the validator will accept.
MAX_ROADMAP_WEEKS: int = 12

#: Minimum number of pasted JDs before gap analysis runs.
MIN_JD_COUNT: int = 1

#: Recommended minimum for best gap coverage.
RECOMMENDED_JD_COUNT: int = 3

# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------

#: Fuzzy-match threshold (0–100) used by rapidfuzz when comparing skills.
#: Scores above this are treated as a match.
SKILL_MATCH_THRESHOLD: int = 80
