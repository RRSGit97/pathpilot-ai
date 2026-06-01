"""
src/graph/state.py
------------------
Typed state object that flows through every LangGraph node.

Design rules
------------
- Every field is Optional or has a default so the graph can be
  initialised with minimal input (just raw_profile_text + raw_jds).
- Fields use our Pydantic models from ``src.data.schemas``.
- ``status`` and ``error`` provide visible progress tracking.
- ``approvals`` uses the ``ApprovalState`` Pydantic model to enforce
  type-safe approval tracking at each pipeline milestone.
- No LLM calls or side effects — pure data container.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict

from src.data.schemas import (
    AggregatedRoleAnalysis,
    ApprovalState,
    LearningRoadmap,
    PortfolioOutputs,
    ProjectIdea,
    ProjectScore,
    ResumeParseResult,
    SixWeekPlan,
    SkillGapReport,
    UserProfile,
)


class PipelineState(TypedDict, total=False):
    """
    Full state object threaded through the PathPilot LangGraph workflow.

    LangGraph merges the dict returned by each node into this state,
    so nodes only need to return the keys they change.

    ── Inputs (set by the UI before the graph starts) ──────────────
    raw_profile_text   : Freeform self-description from the intake form.
    resume_bytes       : Raw resume file bytes (PDF/DOCX), or None.
    resume_filename    : Original filename for type detection.
    raw_jd_texts       : 3–5 pasted job description strings.

    ── Computed by nodes (built up incrementally) ─────────────────
    user_profile       : Structured profile (from Profile Agent).
    resume_parse       : Parsed resume data (from resume_parser).
    role_analysis      : Aggregated role requirements (from skill_extractor).
    gap_report         : Skill gap analysis (from skill_gap_mapper).
    gap_narrative      : LLM-written narrative of the gap report.
    learning_roadmap   : Week-by-week learning plan.
    project_ideas      : Generated project ideas (3–5).
    project_scores     : Scores for each idea.
    ranked_projects    : (ProjectIdea, ProjectScore) pairs sorted desc.
    chosen_project_idx : Index into ranked_projects the user picked.
    project_critique   : LLM critique of the chosen project.
    build_plan         : 6-week build plan for the chosen project.
    portfolio_outputs  : README, resume bullets, demo script, etc.

    ── Approval & control ─────────────────────────────────────────
    approvals          : Per-milestone approval flags.
    human_feedback     : Latest feedback string from the user.
    status             : Current pipeline status message (for UI).
    error              : Error message if a node fails, else None.
    current_node       : Name of the node currently executing.
    """

    # ── Inputs ────────────────────────────────────────────────────
    raw_profile_text: str
    resume_bytes: Optional[bytes]
    resume_filename: Optional[str]
    raw_jd_texts: list[str]

    # ── Computed ──────────────────────────────────────────────────
    user_profile: Optional[UserProfile]
    resume_parse: Optional[ResumeParseResult]
    role_analysis: Optional[AggregatedRoleAnalysis]
    gap_report: Optional[SkillGapReport]
    gap_narrative: Optional[str]
    learning_roadmap: Optional[LearningRoadmap]
    project_ideas: Optional[list[ProjectIdea]]
    project_scores: Optional[list[ProjectScore]]
    ranked_projects: Optional[list[dict]]  # serialised (idea, score) pairs
    chosen_project_idx: Optional[int]
    project_critique: Optional[dict]
    build_plan: Optional[SixWeekPlan]
    portfolio_outputs: Optional[PortfolioOutputs]

    # ── Approval & control ────────────────────────────────────────
    approvals: Optional[dict]  # serialised ApprovalState
    human_feedback: Optional[str]
    status: str
    error: Optional[str]
    current_node: str
