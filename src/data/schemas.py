"""
src/data/schemas.py
-------------------
All Pydantic v2 models for PathPilot AI.

Design rules:
- Every model uses ``model_config = ConfigDict(frozen=False, extra="forbid")``
  so unknown fields surface as errors early.
- IDs are UUIDs stored as strings; callers generate them with ``str(uuid4())``.
- Optional fields have defaults of ``None`` or sensible empty values so
  in-progress state objects can be built incrementally.
- No LLM calls, DB access, or side effects live here — pure data only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.enums import (
    ArtifactType,
    ApprovalStatus,
    LearningStyle,
    ProjectDifficulty,
    SkillLevel,
)
from src.data.constants import SCORING_WEIGHTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a fresh UUID string."""
    return str(uuid4())


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.utcnow().isoformat()


_BASE_CONFIG = ConfigDict(frozen=False, extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# User & Resume
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """Everything PathPilot needs to know about the user before analysis begins."""

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id, description="Unique profile ID.")
    name: str = Field(description="User's display name.")
    current_skills: list[str] = Field(
        default_factory=list,
        description="Raw skill tags the user self-reports (e.g. 'Python', 'SQL').",
    )
    prior_projects: str = Field(
        default="",
        description="Free-text description of past projects and work experience.",
    )
    target_roles: list[str] = Field(
        default_factory=list,
        description="Job titles the user is actively targeting (e.g. 'AI Engineer').",
    )
    weekly_hours_available: Annotated[int, Field(ge=1, le=80)] = Field(
        default=10,
        description="Hours per week the user can dedicate to learning and building.",
    )
    learning_style: LearningStyle = Field(
        default=LearningStyle.MIXED,
        description="Preferred mode of absorbing new material.",
    )
    pain_points: str = Field(
        default="",
        description="Self-described blockers or frustrations the user wants to overcome.",
    )
    resume_text: str | None = Field(
        default=None,
        description="Plain-text content extracted from an uploaded PDF resume.",
    )
    created_at: str = Field(default_factory=_now_iso)

    @field_validator("current_skills", "target_roles", mode="before")
    @classmethod
    def _strip_blanks(cls, v: list[str]) -> list[str]:
        """Remove empty strings that can sneak in from form inputs."""
        return [s.strip() for s in v if s and s.strip()]


class ResumeParseResult(BaseModel):
    """Structured output produced by the resume parser from raw PDF/DOCX/text input."""

    model_config = _BASE_CONFIG

    extracted_skills: list[str] = Field(
        default_factory=list,
        description="Skills identified in the resume text (heuristic pre-pass only).",
    )
    work_experience_summary: str = Field(
        default="",
        description="Short paragraph summarising roles and responsibilities.",
    )
    education_summary: str = Field(
        default="",
        description="Highest degree and relevant certifications.",
    )
    raw_text: str = Field(
        default="",
        description="Original text passed to the parser (stored for traceability).",
    )
    parse_warning: str | None = Field(
        default=None,
        description=(
            "Set when extraction produced suspiciously little text (e.g. scanned PDF). "
            "When non-None, extracted_skills is intentionally empty — do not proceed "
            "as if parsing succeeded."
        ),
    )


# ---------------------------------------------------------------------------
# Job Descriptions
# ---------------------------------------------------------------------------

class JobDescriptionInput(BaseModel):
    """A single pasted job description before LLM extraction."""

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    title: str = Field(default="", description="Job title, if known or typed by user.")
    raw_text: str = Field(description="The full copy-pasted JD text.")
    source_url: str | None = Field(
        default=None,
        description="Optional URL the user pasted from (not fetched automatically).",
    )

    @field_validator("raw_text")
    @classmethod
    def _min_length(cls, v: str) -> str:
        if len(v.strip()) < 50:
            raise ValueError("Job description text is too short (< 50 characters).")
        return v.strip()


class RoleRequirementAnalysis(BaseModel):
    """LLM-extracted structure from one or more job descriptions."""

    model_config = _BASE_CONFIG

    jd_id: str = Field(description="ID of the source JobDescriptionInput.")
    inferred_title: str = Field(default="", description="Role title inferred by the LLM.")
    seniority: str = Field(
        default="",
        description="Inferred seniority level (junior / mid / senior / staff).",
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills explicitly listed as required or must-have.",
    )
    optional_skills: list[str] = Field(
        default_factory=list,
        description="Skills listed as nice-to-have, preferred, or bonus.",
    )
    tools_and_frameworks: list[str] = Field(
        default_factory=list,
        description="Specific technologies, libraries, or platforms mentioned.",
    )
    project_expectations: list[str] = Field(
        default_factory=list,
        description="Bullet points describing the work or deliverables expected.",
    )


class AggregatedRoleAnalysis(BaseModel):
    """
    Merged role requirements extracted from 1–5 job descriptions.

    This is the primary output of the skill-extraction pipeline.  All
    skill lists are deduplicated and sorted by frequency (most-mentioned
    JDs first).  Evidence snippets give the UI something to show the user
    when they ask "why was this skill flagged?"
    """

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    source_jd_count: int = Field(
        description="Number of source JDs that were analyzed.",
    )
    consensus_title: str = Field(
        default="",
        description="Most representative job title across all JDs.",
    )
    all_titles: list[str] = Field(
        default_factory=list,
        description="Every distinct job title seen (deduplicated).",
    )
    seniority: str = Field(
        default="unknown",
        description="Consensus seniority level: junior / mid / senior / staff / unknown.",
    )
    seniority_signals: list[str] = Field(
        default_factory=list,
        description="Verbatim phrases from JDs that indicate experience level.",
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills required by at least one JD, sorted by how many JDs mention them. "
            "A skill is required if it appears in ANY JD's required section."
        ),
    )
    optional_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills mentioned as optional / nice-to-have in JDs, "
            "not present in required_skills. Sorted by mention frequency."
        ),
    )
    tools_and_frameworks: list[str] = Field(
        default_factory=list,
        description="Specific technologies mentioned across JDs, sorted by frequency.",
    )
    project_expectations: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of work expectations extracted from JDs.",
    )
    # ---- Evidence / frequency metadata ----
    skill_jd_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Maps each normalised skill name to the number of source JDs "
            "that mentioned it.  Useful for ranking and UI badges."
        ),
    )
    evidence_snippets: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Maps each normalised skill name to up to 3 context snippets "
            "pulled directly from the source JD text. "
            "Shows the user *why* a skill was extracted."
        ),
    )
    created_at: str = Field(default_factory=_now_iso)

    def skills_by_frequency(self) -> list[tuple[str, int]]:
        """
        Return all required + optional skills as (skill, count) pairs,
        sorted by JD mention count descending.

        Useful for ranking skill gaps by market demand.
        """
        all_skills = list(dict.fromkeys(self.required_skills + self.optional_skills))
        return sorted(
            [(s, self.skill_jd_counts.get(s, 1)) for s in all_skills],
            key=lambda x: x[1],
            reverse=True,
        )

    def top_required_skills(self, n: int = 10) -> list[str]:
        """Return the top-n required skills by mention frequency."""
        ranked = [s for s, _ in self.skills_by_frequency() if s in self.required_skills]
        return ranked[:n]




# ---------------------------------------------------------------------------
# Skill Gap Analysis
# ---------------------------------------------------------------------------

class SkillGapItem(BaseModel):
    """One skill mapped to the user's proficiency level relative to a target role."""

    model_config = _BASE_CONFIG

    skill: str = Field(description="Normalised skill name.")
    level: SkillLevel = Field(description="Categorised proficiency level.")
    source_jds: list[str] = Field(
        default_factory=list,
        description="JD IDs where this skill appeared.",
    )
    notes: str = Field(
        default="",
        description="Optional context (e.g. 'Used in personal project only').",
    )


class SkillGapReport(BaseModel):
    """Aggregated gap analysis across all submitted job descriptions."""

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    target_role: str = Field(description="Agreed target role for this analysis.")
    items: list[SkillGapItem] = Field(default_factory=list)
    relevance_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description=(
            "Fraction of required skills the user already has at STRONG or PARTIAL level. "
            "Computed deterministically by the skill categoriser."
        ),
    )
    created_at: str = Field(default_factory=_now_iso)

    # Convenience properties
    @property
    def strong(self) -> list[SkillGapItem]:
        return [i for i in self.items if i.level == SkillLevel.STRONG]

    @property
    def partial(self) -> list[SkillGapItem]:
        return [i for i in self.items if i.level == SkillLevel.PARTIAL]

    @property
    def missing_required(self) -> list[SkillGapItem]:
        return [i for i in self.items if i.level == SkillLevel.MISSING_REQUIRED]

    @property
    def missing_optional(self) -> list[SkillGapItem]:
        return [i for i in self.items if i.level == SkillLevel.MISSING_OPTIONAL]


# ---------------------------------------------------------------------------
# Learning Roadmap
# ---------------------------------------------------------------------------

class LearningRoadmapWeek(BaseModel):
    """One week's slice of the personalised learning plan."""

    model_config = _BASE_CONFIG

    week_number: Annotated[int, Field(ge=1)] = Field(description="1-indexed week number.")
    focus_topic: str = Field(description="Primary subject or theme for this week.")
    skills_covered: list[str] = Field(
        default_factory=list,
        description="Skills the learner should make progress on this week.",
    )
    tasks: list[str] = Field(
        default_factory=list,
        description="Concrete actions: read X, watch Y, build Z.",
    )
    estimated_hours: Annotated[int, Field(ge=1)] = Field(
        description="Total hours budgeted for this week's work.",
    )
    resources: list[str] = Field(
        default_factory=list,
        description="Links or titles of recommended resources.",
    )


class LearningRoadmap(BaseModel):
    """Complete week-by-week learning plan generated by PathPilot."""

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    title: str = Field(description="Human-readable roadmap title.")
    target_role: str = Field(description="Role this roadmap is designed for.")
    total_weeks: Annotated[int, Field(ge=1, le=12)] = Field(
        description="Number of weeks (capped at 12 by the validator).",
    )
    weeks: list[LearningRoadmapWeek] = Field(default_factory=list)
    version: int = Field(default=1, description="Incremented on each regeneration.")
    created_at: str = Field(default_factory=_now_iso)

    @model_validator(mode="after")
    def _weeks_match_total(self) -> "LearningRoadmap":
        if self.weeks and len(self.weeks) != self.total_weeks:
            raise ValueError(
                f"total_weeks={self.total_weeks} but {len(self.weeks)} week entries found."
            )
        return self


# ---------------------------------------------------------------------------
# Project Ideas & Scoring
# ---------------------------------------------------------------------------

class ProjectIdea(BaseModel):
    """A raw project concept before scoring."""

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    title: str = Field(description="Short project name.")
    description: str = Field(description="2–4 sentence overview of what it does.")
    technologies: list[str] = Field(
        default_factory=list,
        description="Primary languages, frameworks, and tools involved.",
    )
    architecture_overview: str = Field(
        default="",
        description="Brief description of how components fit together.",
    )
    difficulty: ProjectDifficulty = Field(
        default=ProjectDifficulty.INTERMEDIATE,
        description="Self-assessed or LLM-assessed complexity.",
    )


class ProjectScore(BaseModel):
    """Seven-dimension score for a single project idea.

    All dimensions are 0–10.  The composite score is computed from
    ``SCORING_WEIGHTS`` in ``src/data/constants.py`` — never from the LLM.
    """

    model_config = _BASE_CONFIG

    project_id: str = Field(description="ID of the scored ProjectIdea.")

    # Scored dimensions (0-10 each)
    resume_value: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0
    agentic_fit: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0
    buildability_4_6_weeks: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0
    personal_relevance: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0
    technical_depth: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0
    differentiation: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0
    recruiter_explainability: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0

    # Computed by the deterministic scorer, not the LLM
    composite_score: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0

    def compute_composite(self) -> float:
        """Recalculate and update composite_score from the canonical weights."""
        w = SCORING_WEIGHTS
        self.composite_score = round(
            self.resume_value            * w["resume_value"]
            + self.agentic_fit           * w["agentic_fit"]
            + self.buildability_4_6_weeks * w["buildability_4_6_weeks"]
            + self.personal_relevance    * w["personal_relevance"]
            + self.technical_depth       * w["technical_depth"]
            + self.differentiation       * w["differentiation"]
            + self.recruiter_explainability * w["recruiter_explainability"],
            2,
        )
        return self.composite_score

    def as_dict(self) -> dict[str, float]:
        """Dimension scores as a plain dict (useful for charting)."""
        return {
            "Resume Value":              self.resume_value,
            "Agentic Fit":               self.agentic_fit,
            "Buildability (4-6 wks)":    self.buildability_4_6_weeks,
            "Personal Relevance":        self.personal_relevance,
            "Technical Depth":           self.technical_depth,
            "Differentiation":           self.differentiation,
            "Recruiter Explainability":  self.recruiter_explainability,
        }


# ---------------------------------------------------------------------------
# Six-Week Build Plan
# ---------------------------------------------------------------------------

class SixWeekPlanWeek(BaseModel):
    """One week of the 6-week project build plan."""

    model_config = _BASE_CONFIG

    week_number: Annotated[int, Field(ge=1, le=6)] = Field(description="1-indexed week.")
    goals: list[str] = Field(
        default_factory=list,
        description="High-level outcomes to achieve by end of the week.",
    )
    tasks: list[str] = Field(
        default_factory=list,
        description="Concrete day-level tasks.",
    )
    deliverable: str = Field(
        default="",
        description="A tangible artifact or milestone that marks the week done.",
    )


class SixWeekPlan(BaseModel):
    """Complete 6-week build plan for the chosen project."""

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    project_id: str = Field(description="ID of the chosen ProjectIdea.")
    project_title: str = Field(description="Human-readable title (denormalised for display).")
    weeks: list[SixWeekPlanWeek] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)

    @model_validator(mode="after")
    def _must_have_six_weeks(self) -> "SixWeekPlan":
        if self.weeks and len(self.weeks) != 6:
            raise ValueError(
                f"SixWeekPlan must have exactly 6 weeks; got {len(self.weeks)}."
            )
        return self


# ---------------------------------------------------------------------------
# Portfolio Artifacts
# ---------------------------------------------------------------------------

class PortfolioOutputs(BaseModel):
    """All human-approvable portfolio documents for a chosen project."""

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    project_id: str = Field(description="ID of the source ProjectIdea.")

    readme_outline: str = Field(
        default="",
        description="Markdown outline for the project README.",
    )
    resume_bullets: list[str] = Field(
        default_factory=list,
        description="2–4 resume bullet points; must be approved before use.",
    )
    architecture_summary: str = Field(
        default="",
        description="1–2 paragraph plain-English description of the system.",
    )
    demo_script: str = Field(
        default="",
        description="Step-by-step walkthrough for a live or recorded demo.",
    )
    interview_explanation_30s: str = Field(
        default="",
        description="Elevator-pitch version of what the project is and why it matters.",
    )
    interview_explanation_2m: str = Field(
        default="",
        description="Detailed technical narrative for a panel or design interview.",
    )

    # Track which artifacts have been human-approved
    approved_artifacts: list[ArtifactType] = Field(
        default_factory=list,
        description="Artifact types the user has explicitly approved.",
    )
    created_at: str = Field(default_factory=_now_iso)

    def is_approved(self, artifact: ArtifactType) -> bool:
        """Return True only if the artifact has been explicitly approved."""
        return artifact in self.approved_artifacts

    def mark_approved(self, artifact: ArtifactType) -> None:
        """Add an artifact to the approved set (idempotent)."""
        if artifact not in self.approved_artifacts:
            self.approved_artifacts.append(artifact)


# ---------------------------------------------------------------------------
# Workflow / Session State
# ---------------------------------------------------------------------------

class ApprovalState(BaseModel):
    """Tracks human-in-the-loop approval status for each workflow gate."""

    model_config = _BASE_CONFIG

    target_role: ApprovalStatus = ApprovalStatus.PENDING
    skill_gap_report: ApprovalStatus = ApprovalStatus.PENDING
    learning_roadmap: ApprovalStatus = ApprovalStatus.PENDING
    chosen_project: ApprovalStatus = ApprovalStatus.PENDING
    build_plan: ApprovalStatus = ApprovalStatus.PENDING
    portfolio_outputs: ApprovalStatus = ApprovalStatus.PENDING

    human_feedback: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Gate name → free-text feedback from the user. "
            "Populated when status is REVISION_REQUESTED."
        ),
    )

    def all_approved(self) -> bool:
        """Return True only when every gate has been approved."""
        statuses = [
            self.target_role,
            self.skill_gap_report,
            self.learning_roadmap,
            self.chosen_project,
            self.build_plan,
            self.portfolio_outputs,
        ]
        return all(s == ApprovalStatus.APPROVED for s in statuses)


class ProgressLogEntry(BaseModel):
    """
    A weekly progress log entry for the 6-week build phase.

    Also supports simple event logging (set ``event`` only) for
    lightweight activity tracking.
    """

    model_config = _BASE_CONFIG

    id: str = Field(default_factory=_new_id)
    project_id: str = Field(default="", description="ID of the project being built.")
    week_number: int = Field(default=0, ge=0, le=12, description="Build week (0 = pre-build).")
    date: str = Field(default_factory=_now_iso, description="ISO date of this log entry.")
    tasks_completed: list[str] = Field(default_factory=list, description="Tasks finished this week.")
    blockers: list[str] = Field(default_factory=list, description="Blockers encountered.")
    next_steps: list[str] = Field(default_factory=list, description="Planned next actions.")
    event: str = Field(default="", description="Short event summary (for simple logging).")
    detail: str = Field(default="", description="Optional longer note.")
    is_error: bool = Field(default=False)
    created_at: str = Field(default_factory=_now_iso)
