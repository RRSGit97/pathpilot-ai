"""
src/graph/nodes/reviews.py
--------------------------
Human-in-the-loop review nodes.

Each review node uses ``langgraph.types.interrupt()`` to pause the graph
and present data to the user for approval.  When the user resumes the
graph (via ``graph.invoke(Command(resume=...))``), the interrupt returns
the user's feedback.

Pattern
-------
1. Node prepares a summary dict for the UI to display.
2. Calls ``interrupt(summary)`` which pauses execution.
3. When resumed, ``interrupt()`` returns the user's response.
4. Node updates approval flags and returns.

The Streamlit UI will:
- Detect that the graph is paused (via checkpoint metadata)
- Display the summary to the user
- Collect approval / feedback
- Resume the graph with ``Command(resume={"approved": True, ...})``
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from src.data.enums import ApprovalStatus
from src.data.schemas import ProjectIdea, ProjectScore
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def _get_approvals(state: PipelineState) -> dict:
    """Get current approvals dict, initialising if needed."""
    return state.get("approvals") or {
        "target_role": ApprovalStatus.PENDING.value,
        "skill_gap_report": ApprovalStatus.PENDING.value,
        "learning_roadmap": ApprovalStatus.PENDING.value,
        "chosen_project": ApprovalStatus.PENDING.value,
        "build_plan": ApprovalStatus.PENDING.value,
        "portfolio_outputs": ApprovalStatus.PENDING.value,
    }


# ──────────────────────────────────────────────────────────────────
# Review 1: Target Role + Extracted Requirements
# ──────────────────────────────────────────────────────────────────

def review_role_node(state: PipelineState) -> dict:
    """
    Present role analysis to the user for approval.
    """
    logger.info("Node: review_role (waiting for human)")

    role_analysis = state.get("role_analysis")
    gap_report = state.get("gap_report")

    summary = {
        "review_type": "role_and_gaps",
        "title": role_analysis.consensus_title if role_analysis else "Unknown",
        "seniority": role_analysis.seniority if role_analysis else "unknown",
        "required_skills": role_analysis.required_skills[:10] if role_analysis else [],
        "optional_skills": role_analysis.optional_skills[:10] if role_analysis else [],
        "gap_narrative": state.get("gap_narrative", ""),
        "relevance_score": gap_report.relevance_score if gap_report else 0,
        "prompt": (
            "Please review the target role, extracted requirements, and skill gap "
            "analysis above. Do they look accurate?\n\n"
            "Reply with: {\"approved\": true} or {\"approved\": false, \"feedback\": \"...\"}"
        ),
    }

    # ── INTERRUPT: waits for user to approve ──
    user_response: dict = interrupt(summary)

    approved = user_response.get("approved", False)
    feedback = user_response.get("feedback", "")

    approvals = _get_approvals(state)
    approvals["target_role"] = ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REVISION_REQUESTED.value
    approvals["skill_gap_report"] = approvals["target_role"]

    logger.info("Role review: approved=%s, feedback=%r", approved, feedback[:80] if feedback else "")

    return {
        "approvals": approvals,
        "human_feedback": feedback,
        "status": "✅ Role & gaps approved." if approved else "🔄 Revision requested for role analysis.",
        "current_node": "review_role",
    }


# ──────────────────────────────────────────────────────────────────
# Review 2: Learning Roadmap
# ──────────────────────────────────────────────────────────────────

def review_roadmap_node(state: PipelineState) -> dict:
    """
    Present the learning roadmap for optional review.
    """
    logger.info("Node: review_roadmap (waiting for human)")

    roadmap = state.get("learning_roadmap")

    weeks_summary = []
    if roadmap:
        for w in roadmap.weeks:
            weeks_summary.append({
                "week": w.week_number,
                "focus": w.focus_topic,
                "hours": w.estimated_hours,
                "skills": w.skills_covered,
            })

    summary = {
        "review_type": "learning_roadmap",
        "title": roadmap.title if roadmap else "No roadmap",
        "total_weeks": roadmap.total_weeks if roadmap else 0,
        "weeks": weeks_summary,
        "prompt": (
            "Review the learning roadmap. You can approve it or request changes.\n\n"
            "Reply with: {\"approved\": true} or {\"approved\": false, \"feedback\": \"...\"}"
        ),
    }

    user_response: dict = interrupt(summary)

    approved = user_response.get("approved", True)  # Optional: default approve
    feedback = user_response.get("feedback", "")

    approvals = _get_approvals(state)
    approvals["learning_roadmap"] = ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REVISION_REQUESTED.value

    return {
        "approvals": approvals,
        "human_feedback": feedback,
        "status": "✅ Roadmap approved." if approved else "🔄 Roadmap revision requested.",
        "current_node": "review_roadmap",
    }


# ──────────────────────────────────────────────────────────────────
# Review 3: Project Selection
# ──────────────────────────────────────────────────────────────────

def review_project_node(state: PipelineState) -> dict:
    """
    Present ranked projects and let the user choose one.
    """
    logger.info("Node: review_project (waiting for human)")

    ranked = state.get("ranked_projects", [])

    projects_summary = []
    for i, pair in enumerate(ranked):
        idea = pair["idea"]
        score = pair["score"]
        projects_summary.append({
            "index": i,
            "title": idea["title"],
            "description": idea["description"],
            "composite_score": score["composite_score"],
            "technologies": idea.get("technologies", []),
        })

    summary = {
        "review_type": "project_selection",
        "projects": projects_summary,
        "prompt": (
            "Review the scored project ideas above. Choose one to build.\n\n"
            "Reply with: {\"approved\": true, \"chosen_index\": 0}"
        ),
    }

    user_response: dict = interrupt(summary)

    approved = user_response.get("approved", False)
    chosen_idx = user_response.get("chosen_index", 0)
    feedback = user_response.get("feedback", "")

    approvals = _get_approvals(state)
    approvals["chosen_project"] = ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REVISION_REQUESTED.value

    return {
        "approvals": approvals,
        "chosen_project_idx": chosen_idx,
        "human_feedback": feedback,
        "status": f"✅ Project #{chosen_idx + 1} selected." if approved else "🔄 Project selection pending.",
        "current_node": "review_project",
    }


# ──────────────────────────────────────────────────────────────────
# Review 4: Portfolio Outputs (REQUIRED — most sensitive)
# ──────────────────────────────────────────────────────────────────

def review_portfolio_node(state: PipelineState) -> dict:
    """
    Present portfolio artifacts for final approval before public use.
    """
    logger.info("Node: review_portfolio (waiting for human)")

    portfolio = state.get("portfolio_outputs")

    summary = {
        "review_type": "portfolio_outputs",
        "resume_bullets": portfolio.resume_bullets if portfolio else [],
        "readme_preview": (portfolio.readme_outline[:500] + "...") if portfolio and portfolio.readme_outline else "",
        "interview_30s": portfolio.interview_explanation_30s if portfolio else "",
        "prompt": (
            "⚠️ IMPORTANT: Review all portfolio artifacts carefully before using "
            "them publicly. Ensure no claims are exaggerated beyond what you built.\n\n"
            "Reply with: {\"approved\": true} or {\"approved\": false, \"feedback\": \"...\"}"
        ),
    }

    user_response: dict = interrupt(summary)

    approved = user_response.get("approved", False)
    feedback = user_response.get("feedback", "")

    approvals = _get_approvals(state)
    approvals["portfolio_outputs"] = ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REVISION_REQUESTED.value

    return {
        "approvals": approvals,
        "human_feedback": feedback,
        "status": "✅ Portfolio approved — ready to use!" if approved else "🔄 Portfolio revision requested.",
        "current_node": "review_portfolio",
    }
