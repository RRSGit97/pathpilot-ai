"""
src/graph/edges.py
------------------
Edge routing functions for the PathPilot LangGraph workflow.

In the MVP the graph is purely linear — edges connect nodes in sequence
without conditional branching.  This file exists as the home for future
conditional edges (e.g. retry on revision, skip optional steps).

The routing functions below are ready to be wired into
``graph.add_conditional_edges()`` when needed.
"""

from __future__ import annotations

from src.data.enums import ApprovalStatus
from src.graph.state import PipelineState


def route_after_role_review(state: PipelineState) -> str:
    """
    After review_role: proceed to roadmap or loop back for revision.

    Currently unused in the linear MVP graph, but ready for when we
    add conditional edges.
    """
    approvals = state.get("approvals") or {}
    if approvals.get("target_role") == ApprovalStatus.APPROVED.value:
        return "generate_roadmap"
    return "extract_jds"  # re-run JD extraction with feedback


def route_after_roadmap_review(state: PipelineState) -> str:
    """After review_roadmap: proceed to project ideation or regenerate."""
    approvals = state.get("approvals") or {}
    if approvals.get("learning_roadmap") == ApprovalStatus.APPROVED.value:
        return "generate_projects"
    return "generate_roadmap"


def route_after_project_review(state: PipelineState) -> str:
    """After review_project: critique chosen project or regenerate ideas."""
    approvals = state.get("approvals") or {}
    if approvals.get("chosen_project") == ApprovalStatus.APPROVED.value:
        return "critique_project"
    return "generate_projects"


def route_after_portfolio_review(state: PipelineState) -> str:
    """After review_portfolio: finish or regenerate artifacts."""
    approvals = state.get("approvals") or {}
    if approvals.get("portfolio_outputs") == ApprovalStatus.APPROVED.value:
        return "__end__"
    return "generate_portfolio"
