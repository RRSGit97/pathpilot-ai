"""
tests/test_schemas.py
----------------------
Tests for Pydantic schema validation in src/data/schemas.py.

No LLM or external services needed.

Covers:
- LearningRoadmap: total_weeks must match number of weeks
- SixWeekPlan: must have exactly 6 weeks
- ProjectScore: compute_composite uses SCORING_WEIGHTS
- ApprovalState: all_approved() logic
- PortfolioOutputs: mark_approved / is_approved
- UserProfile: strips blank skills from lists
- SkillGapReport: convenience properties (strong, partial, etc.)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data.enums import ApprovalStatus, SkillLevel
from src.data.schemas import (
    ApprovalState,
    LearningRoadmap,
    LearningRoadmapWeek,
    PortfolioOutputs,
    ProjectScore,
    SkillGapItem,
    SkillGapReport,
    SixWeekPlan,
    SixWeekPlanWeek,
    UserProfile,
)
from src.data.enums import ArtifactType


# ---------------------------------------------------------------------------
# LearningRoadmap
# ---------------------------------------------------------------------------

class TestLearningRoadmap:
    def _week(self, n: int) -> LearningRoadmapWeek:
        return LearningRoadmapWeek(week_number=n, focus_topic=f"Week {n}", estimated_hours=8)

    def test_valid_roadmap_passes_validation(self):
        roadmap = LearningRoadmap(
            title="Test",
            target_role="AI Engineer",
            total_weeks=2,
            weeks=[self._week(1), self._week(2)],
        )
        assert roadmap.total_weeks == 2

    def test_weeks_mismatch_raises_validation_error(self):
        with pytest.raises(ValidationError, match="total_weeks"):
            LearningRoadmap(
                title="Test",
                target_role="AI Engineer",
                total_weeks=3,  # says 3 but provides 2
                weeks=[self._week(1), self._week(2)],
            )

    def test_empty_weeks_list_skips_validator(self):
        # If weeks is empty, the mismatch validator is skipped
        roadmap = LearningRoadmap(
            title="No weeks yet",
            target_role="AI Engineer",
            total_weeks=4,
        )
        assert roadmap.total_weeks == 4

    def test_total_weeks_capped_at_12(self):
        with pytest.raises(ValidationError):
            LearningRoadmap(
                title="Too long",
                target_role="AI Engineer",
                total_weeks=13,  # exceeds max
            )


# ---------------------------------------------------------------------------
# SixWeekPlan
# ---------------------------------------------------------------------------

class TestSixWeekPlan:
    def _plan_week(self, n: int) -> SixWeekPlanWeek:
        return SixWeekPlanWeek(week_number=n, goals=[f"Goal {n}"], tasks=[f"Task {n}"], deliverable=f"Del {n}")

    def test_valid_six_week_plan(self):
        plan = SixWeekPlan(
            project_id="proj-001",
            project_title="My Project",
            weeks=[self._plan_week(i) for i in range(1, 7)],
        )
        assert len(plan.weeks) == 6

    def test_wrong_number_of_weeks_raises(self):
        with pytest.raises(ValidationError, match="6 weeks"):
            SixWeekPlan(
                project_id="proj-001",
                project_title="My Project",
                weeks=[self._plan_week(i) for i in range(1, 5)],  # 4 weeks
            )

    def test_empty_weeks_skips_validator(self):
        plan = SixWeekPlan(project_id="proj-001", project_title="Draft")
        assert plan.weeks == []


# ---------------------------------------------------------------------------
# ProjectScore.compute_composite
# ---------------------------------------------------------------------------

class TestProjectScore:
    def test_composite_zero_when_all_dimensions_zero(self):
        score = ProjectScore(project_id="p")
        score.compute_composite()
        assert score.composite_score == 0.0

    def test_composite_positive_when_dimensions_set(self):
        score = ProjectScore(
            project_id="p",
            resume_value=8.0,
            agentic_fit=9.0,
            buildability_4_6_weeks=7.0,
            personal_relevance=6.0,
            technical_depth=7.0,
            differentiation=8.0,
            recruiter_explainability=8.0,
        )
        score.compute_composite()
        assert score.composite_score > 0.0

    def test_composite_within_range(self):
        score = ProjectScore(
            project_id="p",
            resume_value=10.0,
            agentic_fit=10.0,
            buildability_4_6_weeks=10.0,
            personal_relevance=10.0,
            technical_depth=10.0,
            differentiation=10.0,
            recruiter_explainability=10.0,
        )
        score.compute_composite()
        assert 0.0 <= score.composite_score <= 10.0

    def test_as_dict_returns_seven_dimensions(self):
        score = ProjectScore(project_id="p", resume_value=5.0)
        d = score.as_dict()
        assert len(d) == 7
        assert all(isinstance(v, float) for v in d.values())


# ---------------------------------------------------------------------------
# ApprovalState
# ---------------------------------------------------------------------------

class TestApprovalState:
    def test_all_pending_by_default(self):
        state = ApprovalState()
        assert not state.all_approved()

    def test_all_approved_returns_true_when_all_set(self):
        state = ApprovalState(
            target_role=ApprovalStatus.APPROVED,
            skill_gap_report=ApprovalStatus.APPROVED,
            learning_roadmap=ApprovalStatus.APPROVED,
            chosen_project=ApprovalStatus.APPROVED,
            build_plan=ApprovalStatus.APPROVED,
            portfolio_outputs=ApprovalStatus.APPROVED,
        )
        assert state.all_approved()

    def test_all_approved_false_if_any_pending(self):
        state = ApprovalState(
            target_role=ApprovalStatus.APPROVED,
            skill_gap_report=ApprovalStatus.APPROVED,
            learning_roadmap=ApprovalStatus.APPROVED,
            chosen_project=ApprovalStatus.PENDING,  # still pending
            build_plan=ApprovalStatus.APPROVED,
            portfolio_outputs=ApprovalStatus.APPROVED,
        )
        assert not state.all_approved()


# ---------------------------------------------------------------------------
# PortfolioOutputs
# ---------------------------------------------------------------------------

class TestPortfolioOutputs:
    def test_is_approved_false_by_default(self):
        portfolio = PortfolioOutputs(project_id="p")
        assert not portfolio.is_approved(ArtifactType.RESUME_BULLETS)

    def test_mark_approved_sets_artifact(self):
        portfolio = PortfolioOutputs(project_id="p")
        portfolio.mark_approved(ArtifactType.RESUME_BULLETS)
        assert portfolio.is_approved(ArtifactType.RESUME_BULLETS)

    def test_mark_approved_is_idempotent(self):
        portfolio = PortfolioOutputs(project_id="p")
        portfolio.mark_approved(ArtifactType.README_OUTLINE)
        portfolio.mark_approved(ArtifactType.README_OUTLINE)
        count = portfolio.approved_artifacts.count(ArtifactType.README_OUTLINE)
        assert count == 1


# ---------------------------------------------------------------------------
# SkillGapReport convenience properties
# ---------------------------------------------------------------------------

class TestSkillGapReportProperties:
    def _report(self) -> SkillGapReport:
        items = [
            SkillGapItem(skill="python",     level=SkillLevel.STRONG),
            SkillGapItem(skill="langgraph",  level=SkillLevel.PARTIAL),
            SkillGapItem(skill="qdrant",     level=SkillLevel.MISSING_REQUIRED),
            SkillGapItem(skill="postgresql", level=SkillLevel.MISSING_OPTIONAL),
        ]
        return SkillGapReport(target_role="AI Engineer", items=items)

    def test_strong_property(self):
        report = self._report()
        assert len(report.strong) == 1
        assert report.strong[0].skill == "python"

    def test_partial_property(self):
        report = self._report()
        assert len(report.partial) == 1
        assert report.partial[0].skill == "langgraph"

    def test_missing_required_property(self):
        report = self._report()
        assert len(report.missing_required) == 1
        assert report.missing_required[0].skill == "qdrant"

    def test_missing_optional_property(self):
        report = self._report()
        assert len(report.missing_optional) == 1
        assert report.missing_optional[0].skill == "postgresql"


# ---------------------------------------------------------------------------
# UserProfile: skills list cleaning
# ---------------------------------------------------------------------------

class TestUserProfile:
    def test_blank_skills_stripped_from_list(self):
        profile = UserProfile(
            name="Test",
            current_skills=["Python", "", "  ", "FastAPI"],
            target_roles=["AI Engineer"],
            weekly_hours_available=10,
        )
        # Blank entries should be removed
        assert "" not in profile.current_skills
        assert "Python" in profile.current_skills

    def test_weekly_hours_must_be_positive(self):
        with pytest.raises(ValidationError):
            UserProfile(
                name="Test",
                current_skills=["Python"],
                target_roles=["AI Engineer"],
                weekly_hours_available=0,  # must be >= 1
            )
