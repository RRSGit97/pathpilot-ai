"""
tests/test_scoring.py
---------------------
Tests for src/tools/scoring.py.

All scoring functions are deterministic; no LLM calls.

Covers:
- score_project: each dimension individually where testable
- score_project: composite score computation
- Boundary: cliché project scores low on differentiation
- Boundary: beginner project scores high on buildability
- Boundary: agentic keywords elevate agentic_fit
- rank_projects: ordering guarantee (highest composite first)
- explain_top_projects: output shape
"""

from __future__ import annotations

import pytest

from src.data.enums import ProjectDifficulty
from src.data.schemas import (
    ProjectIdea,
    ProjectScore,
    SkillGapItem,
    SkillGapReport,
    UserProfile,
)
from src.data.enums import SkillLevel
from src.tools.scoring import explain_top_projects, rank_projects, score_project


# ---------------------------------------------------------------------------
# Shared fixtures (inline — small and self-explanatory)
# ---------------------------------------------------------------------------

def _make_gap(
    missing_req: list[str] | None = None,
    missing_opt: list[str] | None = None,
    partial: list[str] | None = None,
    strong: list[str] | None = None,
) -> SkillGapReport:
    items = []
    for s in (missing_req or []):
        items.append(SkillGapItem(skill=s, level=SkillLevel.MISSING_REQUIRED))
    for s in (missing_opt or []):
        items.append(SkillGapItem(skill=s, level=SkillLevel.MISSING_OPTIONAL))
    for s in (partial or []):
        items.append(SkillGapItem(skill=s, level=SkillLevel.PARTIAL))
    for s in (strong or []):
        items.append(SkillGapItem(skill=s, level=SkillLevel.STRONG))
    return SkillGapReport(target_role="AI Engineer", items=items)


def _make_profile(
    skills: list[str] | None = None,
    pain_points: str = "",
    target_roles: list[str] | None = None,
) -> UserProfile:
    return UserProfile(
        name="Test",
        current_skills=skills or [],
        target_roles=target_roles or ["AI Engineer"],
        weekly_hours_available=10,
        pain_points=pain_points,
    )


def _make_project(
    title: str = "My Project",
    description: str = "A well described project with enough detail to evaluate.",
    technologies: list[str] | None = None,
    architecture: str = "Clean three-tier architecture with API, logic, and database layers.",
    difficulty: ProjectDifficulty = ProjectDifficulty.INTERMEDIATE,
) -> ProjectIdea:
    return ProjectIdea(
        title=title,
        description=description,
        technologies=technologies or [],
        architecture_overview=architecture,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# score_project: resume_value dimension
# ---------------------------------------------------------------------------

class TestResumeValueDimension:
    def test_missing_required_tech_gives_high_resume_value(self):
        gap = _make_gap(missing_req=["langgraph"])
        profile = _make_profile()
        project = _make_project(technologies=["LangGraph"])  # uses missing required tech
        score = score_project(project, gap, profile)
        assert score.resume_value > 0.0

    def test_no_missing_skills_gives_zero_resume_value(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(technologies=["Python"])
        # Python is not in gap → resume_value stays 0
        score = score_project(project, gap, profile)
        assert score.resume_value == 0.0

    def test_resume_value_capped_at_10(self):
        # Flood with many missing required skills
        missing = [f"skill_{i}" for i in range(5)]
        gap = _make_gap(missing_req=missing)
        profile = _make_profile()
        project = _make_project(technologies=[f"skill_{i}" for i in range(5)])
        score = score_project(project, gap, profile)
        assert score.resume_value <= 10.0

    def test_partial_skill_gives_medium_reward(self):
        gap = _make_gap(partial=["pydantic"])
        profile = _make_profile()
        project = _make_project(technologies=["pydantic"])
        score = score_project(project, gap, profile)
        assert score.resume_value > 0.0


# ---------------------------------------------------------------------------
# score_project: agentic_fit dimension
# ---------------------------------------------------------------------------

class TestAgenticFitDimension:
    def test_langgraph_in_description_boosts_agentic_fit(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(
            description="Build an autonomous LangGraph multi-agent system for workflow automation.",
            technologies=["LangGraph"],
        )
        score = score_project(project, gap, profile)
        assert score.agentic_fit > 5.0

    def test_no_agentic_keywords_gives_low_score(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(
            description="A simple data dashboard showing sales metrics.",
            technologies=["Pandas", "Matplotlib"],
        )
        score = score_project(project, gap, profile)
        assert score.agentic_fit < 5.0

    def test_agentic_fit_capped_at_10(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(
            description="Agent autonomous multi-agent LangGraph agentic autonomous tool-calling agentic agent.",
            technologies=["LangGraph", "OpenAI"],
        )
        score = score_project(project, gap, profile)
        assert score.agentic_fit <= 10.0


# ---------------------------------------------------------------------------
# score_project: buildability dimension
# ---------------------------------------------------------------------------

class TestBuildabilityDimension:
    def test_beginner_project_scores_high_buildability(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(difficulty=ProjectDifficulty.BEGINNER, technologies=["Python"])
        score = score_project(project, gap, profile)
        assert score.buildability_4_6_weeks >= 9.0

    def test_advanced_project_scores_lower_buildability(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(difficulty=ProjectDifficulty.ADVANCED)
        score = score_project(project, gap, profile)
        assert score.buildability_4_6_weeks < 7.0

    def test_many_technologies_lowers_buildability(self):
        gap = _make_gap()
        profile = _make_profile()
        many_techs = [f"tech{i}" for i in range(10)]  # 10 technologies
        project_lean = _make_project(technologies=["Python", "FastAPI"])
        project_heavy = _make_project(technologies=many_techs)
        score_lean = score_project(project_lean, gap, profile)
        score_heavy = score_project(project_heavy, gap, profile)
        assert score_heavy.buildability_4_6_weeks < score_lean.buildability_4_6_weeks

    def test_buildability_never_negative(self):
        gap = _make_gap()
        profile = _make_profile()
        huge_tech_list = [f"t{i}" for i in range(30)]
        project = _make_project(difficulty=ProjectDifficulty.ADVANCED, technologies=huge_tech_list)
        score = score_project(project, gap, profile)
        assert score.buildability_4_6_weeks >= 0.0


# ---------------------------------------------------------------------------
# score_project: differentiation dimension
# ---------------------------------------------------------------------------

class TestDifferentiationDimension:
    def test_cliche_project_scores_low_differentiation(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(
            title="Todo List Chatbot",
            description="A simple chatbot wrapper for managing a todo CRUD list.",
        )
        score = score_project(project, gap, profile)
        assert score.differentiation < 5.0

    def test_original_project_scores_10_differentiation(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(
            title="Multi-Agent Pipeline Orchestrator",
            description="Novel system that orchestrates multi-agent pipelines for autonomous decisions.",
        )
        score = score_project(project, gap, profile)
        assert score.differentiation == 10.0

    def test_differentiation_never_negative(self):
        gap = _make_gap()
        profile = _make_profile()
        project = _make_project(
            title="Chatbot Todo Calculator Weather Wrapper CRUD",
            description="Chatbot todo calculator weather wrapper tic-tac-toe crud app.",
        )
        score = score_project(project, gap, profile)
        assert score.differentiation >= 0.0


# ---------------------------------------------------------------------------
# score_project: personal_relevance dimension
# ---------------------------------------------------------------------------

class TestPersonalRelevanceDimension:
    def test_target_role_in_text_boosts_relevance(self):
        gap = _make_gap()
        profile = _make_profile(target_roles=["AI Engineer"])
        project = _make_project(
            description="This project builds an AI Engineer portfolio demonstration system.",
        )
        score = score_project(project, gap, profile)
        assert score.personal_relevance > 3.0  # baseline is 3.0

    def test_pain_point_words_boost_relevance(self):
        gap = _make_gap()
        profile = _make_profile(pain_points="struggle with multi-agent state management")
        project = _make_project(
            description="Manage state for multi-agent workflows with clear routing.",
        )
        score_with_pain = score_project(project, gap, profile)
        profile_no_pain = _make_profile(pain_points="")
        score_no_pain = score_project(project, gap, profile_no_pain)
        assert score_with_pain.personal_relevance >= score_no_pain.personal_relevance


# ---------------------------------------------------------------------------
# score_project: composite score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_composite_score_within_valid_range(self):
        gap = _make_gap(missing_req=["langgraph"])
        profile = _make_profile()
        project = _make_project(technologies=["LangGraph"])
        score = score_project(project, gap, profile)
        assert 0.0 <= score.composite_score <= 10.0

    def test_composite_is_set_by_compute_composite(self):
        score = ProjectScore(project_id="test")
        assert score.composite_score == 0.0  # default before compute
        score.compute_composite()
        assert isinstance(score.composite_score, float)

    def test_high_scoring_project_has_higher_composite(self, agentic_project, sample_gap_report, sample_profile):
        agentic_score = score_project(agentic_project, sample_gap_report, sample_profile)
        assert agentic_score.composite_score > 0.0

    def test_cliche_project_ranks_lower_than_agentic(
        self, agentic_project, cliche_project, sample_gap_report, sample_profile
    ):
        agentic_score = score_project(agentic_project, sample_gap_report, sample_profile)
        cliche_score = score_project(cliche_project, sample_gap_report, sample_profile)
        assert agentic_score.composite_score >= cliche_score.composite_score


# ---------------------------------------------------------------------------
# rank_projects
# ---------------------------------------------------------------------------

class TestRankProjects:
    def test_returns_sorted_descending(self, agentic_project, cliche_project, sample_gap_report, sample_profile):
        pairs = rank_projects([cliche_project, agentic_project], sample_gap_report, sample_profile)
        scores = [score.composite_score for _, score in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_returns_tuples_of_project_and_score(self, agentic_project, sample_gap_report, sample_profile):
        pairs = rank_projects([agentic_project], sample_gap_report, sample_profile)
        assert len(pairs) == 1
        proj, score = pairs[0]
        assert isinstance(proj, ProjectIdea)
        assert isinstance(score, ProjectScore)

    def test_empty_list_returns_empty(self, sample_gap_report, sample_profile):
        assert rank_projects([], sample_gap_report, sample_profile) == []


# ---------------------------------------------------------------------------
# explain_top_projects
# ---------------------------------------------------------------------------

class TestExplainTopProjects:
    def test_returns_string(self, agentic_project, sample_gap_report, sample_profile):
        pairs = rank_projects([agentic_project], sample_gap_report, sample_profile)
        explanation = explain_top_projects(pairs)
        assert isinstance(explanation, str)

    def test_contains_project_title(self, agentic_project, sample_gap_report, sample_profile):
        pairs = rank_projects([agentic_project], sample_gap_report, sample_profile)
        explanation = explain_top_projects(pairs)
        assert agentic_project.title in explanation

    def test_empty_list_returns_no_projects_message(self):
        explanation = explain_top_projects([])
        assert "No projects" in explanation

    def test_top_n_limits_output(self, agentic_project, cliche_project, sample_gap_report, sample_profile):
        pairs = rank_projects([agentic_project, cliche_project], sample_gap_report, sample_profile)
        explanation = explain_top_projects(pairs, top_n=1)
        # Only 1 project explained — check second one's title is absent
        assert explanation.count("#1:") == 1
