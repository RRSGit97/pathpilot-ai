"""
tests/test_skill_categorizer.py
--------------------------------
Tests for skill categorisation via the real module path.

NOTE: The old stub at this path imported from a non-existent
`src.logic.skill_categorizer` module. This file now tests the
actual `map_skill_gaps` function from `src.tools.skill_gap_mapper`.

For complete coverage of the skill gap mapper, see:
  tests/test_skill_gap_mapper.py

This file retains the original filename for compatibility but
re-exports a small set of focused categorisation sanity checks.
"""

from __future__ import annotations

import pytest

from src.data.enums import SkillLevel
from src.data.schemas import AggregatedRoleAnalysis, UserProfile
from src.tools.skill_gap_mapper import map_skill_gaps


def _role(required: list[str]) -> AggregatedRoleAnalysis:
    return AggregatedRoleAnalysis(
        source_jd_count=1,
        consensus_title="Test Role",
        required_skills=required,
        skill_jd_counts={s: 1 for s in required},
    )


def _profile(skills: list[str]) -> UserProfile:
    return UserProfile(
        name="Test",
        current_skills=skills,
        target_roles=["Test Role"],
        weekly_hours_available=5,
    )


def test_empty_inputs_return_zero_relevance():
    """Consistent with old test spirit: empty everything → perfect or no-op score."""
    role = _role([])
    profile = _profile([])
    report = map_skill_gaps(role, profile)
    # No required skills → relevance defaults to 1.0 (perfect)
    assert report.relevance_score == 1.0


def test_all_present_returns_high_relevance():
    role = _role(["python", "fastapi"])
    profile = _profile(["Python", "FastAPI"])
    report = map_skill_gaps(role, profile)
    assert report.relevance_score == 1.0


def test_nothing_present_returns_zero_relevance():
    role = _role(["langgraph", "qdrant"])
    profile = _profile([])
    report = map_skill_gaps(role, profile)
    assert report.relevance_score == 0.0
