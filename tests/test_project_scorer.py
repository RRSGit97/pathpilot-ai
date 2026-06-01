"""
tests/test_project_scorer.py
-----------------------------
Tests for project scoring — uses real module paths.

NOTE: The old stub at this path imported from non-existent
`src.logic.project_scorer` and `src.schemas.project` paths.
This file now uses the correct:
  - `src.tools.scoring.score_project`
  - `src.data.schemas.ProjectIdea`

For comprehensive scoring tests, see: tests/test_scoring.py.
This file retains the filename for compatibility but provides
correct, working tests.
"""

from __future__ import annotations

import pytest

from src.data.enums import ProjectDifficulty, SkillLevel
from src.data.schemas import ProjectIdea, ProjectScore, SkillGapItem, SkillGapReport, UserProfile
from src.tools.scoring import score_project


def _empty_gap() -> SkillGapReport:
    return SkillGapReport(target_role="AI Engineer", items=[], relevance_score=0.0)


def _simple_profile() -> UserProfile:
    return UserProfile(
        name="Test",
        current_skills=["Python"],
        target_roles=["AI Engineer"],
        weekly_hours_available=5,
    )


def test_score_project_returns_project_score():
    """Replaces the broken placeholder — uses correct 3-arg signature."""
    idea = ProjectIdea(
        title="Test Project",
        description="A simple project to test the scoring pipeline end-to-end.",
        technologies=["Python"],
        architecture_overview="Single service with REST API.",
        difficulty=ProjectDifficulty.INTERMEDIATE,
    )
    gap = _empty_gap()
    profile = _simple_profile()
    scored = score_project(idea, gap, profile)
    assert isinstance(scored, ProjectScore)


def test_composite_score_set_after_scoring():
    """composite_score must be computed, not left at default 0."""
    idea = ProjectIdea(
        title="Agent Project",
        description="A LangGraph multi-agent project with FastAPI and Qdrant database backend.",
        technologies=["LangGraph", "FastAPI", "Qdrant"],
        architecture_overview="Multi-agent orchestrator with API gateway and vector storage.",
        difficulty=ProjectDifficulty.INTERMEDIATE,
    )
    gap = SkillGapReport(
        target_role="AI Engineer",
        items=[SkillGapItem(skill="langgraph", level=SkillLevel.MISSING_REQUIRED)],
        relevance_score=0.5,
    )
    profile = _simple_profile()
    scored = score_project(idea, gap, profile)
    # composite score should be > 0 for a well-described agentic project
    assert scored.composite_score > 0.0


def test_cliche_project_low_differentiation():
    """Todo + chatbot should trigger the cliché penalty."""
    idea = ProjectIdea(
        title="Chatbot Todo App",
        description="A chatbot that lets you manage your todo list via conversation.",
        technologies=["Python"],
        architecture_overview="Simple wrapper",
        difficulty=ProjectDifficulty.BEGINNER,
    )
    scored = score_project(idea, _empty_gap(), _simple_profile())
    assert scored.differentiation < 5.0
