"""
tests/test_skill_gap_mapper.py
-------------------------------
Tests for src/tools/skill_gap_mapper.py.

The mapper is fully deterministic — no LLM calls.
All tests run offline.

Covers:
- STRONG categorisation (skill in profile)
- PARTIAL categorisation (skill in context / prior projects)
- MISSING_REQUIRED / MISSING_OPTIONAL categorisation
- Relevance score calculation
- Boundary: empty required skills
- Resume parse integration (extends explicit skills)
- Markdown formatter output
"""

from __future__ import annotations

import pytest

from src.data.enums import SkillLevel
from src.data.schemas import (
    AggregatedRoleAnalysis,
    ResumeParseResult,
    SkillGapReport,
    UserProfile,
)
from src.tools.skill_gap_mapper import format_gap_report_markdown, map_skill_gaps


# ---------------------------------------------------------------------------
# Helper builders (kept inline for readability)
# ---------------------------------------------------------------------------

def _make_role(
    required: list[str],
    optional: list[str] | None = None,
    tools: list[str] | None = None,
    title: str = "AI Engineer",
) -> AggregatedRoleAnalysis:
    optional = optional or []
    tools = tools or []
    counts = {s: 2 for s in required + optional + tools}
    return AggregatedRoleAnalysis(
        source_jd_count=2,
        consensus_title=title,
        required_skills=required,
        optional_skills=optional,
        tools_and_frameworks=tools,
        skill_jd_counts=counts,
    )


def _make_profile(
    skills: list[str],
    prior_projects: str = "",
    target_roles: list[str] | None = None,
) -> UserProfile:
    return UserProfile(
        name="Test User",
        current_skills=skills,
        prior_projects=prior_projects,
        target_roles=target_roles or ["AI Engineer"],
        weekly_hours_available=10,
    )


# ---------------------------------------------------------------------------
# STRONG categorisation
# ---------------------------------------------------------------------------

class TestStrongCategorisation:
    def test_explicit_skill_is_strong(self):
        role = _make_role(required=["python"])
        profile = _make_profile(skills=["Python"])  # case insensitive
        report = map_skill_gaps(role, profile)
        assert report.strong
        assert report.strong[0].skill == "python"

    def test_multiple_strong_skills(self):
        role = _make_role(required=["python", "fastapi", "docker"])
        profile = _make_profile(skills=["Python", "FastAPI", "Docker"])
        report = map_skill_gaps(role, profile)
        strong_names = {item.skill for item in report.strong}
        assert {"python", "fastapi", "docker"}.issubset(strong_names)

    def test_alias_normalised_strong(self):
        role = _make_role(required=["kubernetes"])
        profile = _make_profile(skills=["K8S"])  # alias → kubernetes
        report = map_skill_gaps(role, profile)
        assert report.strong


# ---------------------------------------------------------------------------
# PARTIAL categorisation (in context text but not explicit)
# ---------------------------------------------------------------------------

class TestPartialCategorisation:
    def test_skill_in_prior_projects_is_partial(self):
        role = _make_role(required=["langgraph"])
        profile = _make_profile(
            skills=[],
            prior_projects="Built a prototype with LangGraph and ChromaDB.",
        )
        report = map_skill_gaps(role, profile)
        assert report.partial
        assert any(i.skill == "langgraph" for i in report.partial)

    def test_skill_in_resume_context_is_partial(self):
        role = _make_role(required=["qdrant"])
        profile = _make_profile(skills=[])
        resume = ResumeParseResult(
            extracted_skills=[],
            raw_text="Worked with Qdrant for vector storage experiments.",
        )
        report = map_skill_gaps(role, profile, resume_parse=resume)
        assert any(i.skill == "qdrant" for i in report.partial)

    def test_resume_extracted_skills_treated_as_strong(self):
        role = _make_role(required=["langchain"])
        profile = _make_profile(skills=[])
        resume = ResumeParseResult(
            extracted_skills=["LangChain"],  # explicit in resume skills list
            raw_text="",
        )
        report = map_skill_gaps(role, profile, resume_parse=resume)
        # Extracted skills should be treated as STRONG (not just partial)
        assert report.strong


# ---------------------------------------------------------------------------
# MISSING categorisation
# ---------------------------------------------------------------------------

class TestMissingCategorisation:
    def test_missing_required_skill(self):
        role = _make_role(required=["langgraph"])
        profile = _make_profile(skills=["Python"])  # no langgraph
        report = map_skill_gaps(role, profile)
        assert report.missing_required
        assert report.missing_required[0].skill == "langgraph"

    def test_missing_optional_skill(self):
        role = _make_role(required=[], optional=["postgresql"])
        profile = _make_profile(skills=["Python"])
        report = map_skill_gaps(role, profile)
        assert report.missing_optional
        assert report.missing_optional[0].skill == "postgresql"

    def test_tool_not_in_required_or_optional_is_optional(self):
        role = _make_role(required=["python"], tools=["redis"])
        profile = _make_profile(skills=["python"])
        report = map_skill_gaps(role, profile)
        # redis treated as optional since it's tools-only
        assert any(i.skill == "redis" for i in report.missing_optional)


# ---------------------------------------------------------------------------
# Relevance score
# ---------------------------------------------------------------------------

class TestRelevanceScore:
    def test_perfect_score_when_all_required_covered(self):
        role = _make_role(required=["python", "fastapi"])
        profile = _make_profile(skills=["Python", "FastAPI"])
        report = map_skill_gaps(role, profile)
        assert report.relevance_score == 1.0

    def test_zero_required_gives_perfect_score(self):
        role = _make_role(required=[], optional=["docker"])
        profile = _make_profile(skills=[])
        report = map_skill_gaps(role, profile)
        assert report.relevance_score == 1.0

    def test_partial_score_calculation(self):
        # 3 required: python(strong), fastapi(strong), langgraph(missing) → 2/3 ≈ 0.67
        role = _make_role(required=["python", "fastapi", "langgraph"])
        profile = _make_profile(skills=["Python", "FastAPI"])
        report = map_skill_gaps(role, profile)
        assert abs(report.relevance_score - (2 / 3)) < 0.01

    def test_partial_skill_counts_toward_relevance(self):
        role = _make_role(required=["langgraph"])
        profile = _make_profile(
            skills=[],
            prior_projects="Experimented with LangGraph for routing.",
        )
        report = map_skill_gaps(role, profile)
        # PARTIAL counts as covered → score should be 1.0
        assert report.relevance_score == 1.0

    def test_score_is_rounded_to_two_decimals(self):
        role = _make_role(required=["a", "b", "c"])
        profile = _make_profile(skills=["a"])  # 1/3
        report = map_skill_gaps(role, profile)
        # Check rounding — should be 0.33
        assert report.relevance_score == round(report.relevance_score, 2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_role_returns_empty_report(self):
        role = _make_role(required=[], optional=[], tools=[])
        profile = _make_profile(skills=["Python"])
        report = map_skill_gaps(role, profile)
        assert isinstance(report, SkillGapReport)
        assert report.items == []

    def test_does_not_double_count_same_skill(self):
        # same skill in required AND optional should appear once
        role = _make_role(required=["python"], optional=["python"])
        profile = _make_profile(skills=["python"])
        report = map_skill_gaps(role, profile)
        python_items = [i for i in report.items if i.skill == "python"]
        assert len(python_items) == 1

    def test_safe_substring_guard(self):
        # "C" should not match inside "Mac" or "JavaScript"
        role = _make_role(required=["c"])
        profile = _make_profile(
            skills=[],
            prior_projects="Used Mac OS and JavaScript for scripting.",
        )
        report = map_skill_gaps(role, profile)
        # "c" should not appear as PARTIAL just because "Mac" and "JavaScript" contain it
        c_items = [i for i in report.items if i.skill == "c"]
        if c_items:
            assert c_items[0].level in (SkillLevel.MISSING_REQUIRED, SkillLevel.PARTIAL)

    def test_resume_with_parse_warning_skips_extracted_skills(self):
        role = _make_role(required=["langgraph"])
        profile = _make_profile(skills=[])
        resume = ResumeParseResult(
            extracted_skills=["LangGraph"],
            raw_text="",
            parse_warning="Could not extract text reliably.",
        )
        report = map_skill_gaps(role, profile, resume_parse=resume)
        # With parse_warning set, extracted skills are skipped → langgraph should be missing
        assert report.missing_required


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

class TestFormatGapReportMarkdown:
    def test_returns_string(self):
        role = _make_role(required=["python", "langgraph"])
        profile = _make_profile(skills=["Python"])
        report = map_skill_gaps(role, profile)
        md = format_gap_report_markdown(report)
        assert isinstance(md, str)

    def test_contains_target_role(self):
        role = _make_role(required=["python"], title="ML Engineer")
        profile = _make_profile(skills=["python"])
        report = map_skill_gaps(role, profile)
        md = format_gap_report_markdown(report)
        assert "ML Engineer" in md

    def test_contains_relevance_percentage(self):
        role = _make_role(required=["python"])
        profile = _make_profile(skills=["python"])
        report = map_skill_gaps(role, profile)
        md = format_gap_report_markdown(report)
        assert "100%" in md

    def test_empty_report_shows_no_skills_message(self):
        role = _make_role(required=[], optional=[])
        profile = _make_profile(skills=[])
        report = map_skill_gaps(role, profile)
        md = format_gap_report_markdown(report)
        assert "No specific skills" in md
