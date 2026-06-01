"""
src/graph/nodes/scoring.py
--------------------------
Node: Score and rank project ideas deterministically.

Purely deterministic — no LLM calls.  Uses the scoring module.
"""

from __future__ import annotations

import logging

from src.graph.state import PipelineState
from src.tools.scoring import rank_projects, explain_top_projects

logger = logging.getLogger(__name__)


def score_projects_node(state: PipelineState) -> dict:
    """
    Score each project idea and rank them by composite score.
    """
    logger.info("Node: score_projects")

    project_ideas = state.get("project_ideas")
    gap_report = state.get("gap_report")
    user_profile = state.get("user_profile")

    if not project_ideas or not gap_report or not user_profile:
        return {
            "error": "Missing project ideas, gap report, or profile for scoring.",
            "status": "❌ Project scoring failed — missing inputs.",
            "current_node": "score_projects",
        }

    try:
        ranked = rank_projects(project_ideas, gap_report, user_profile)

        # Serialise for state storage (TypedDict can't hold tuples)
        ranked_dicts = []
        scores = []
        for idea, score in ranked:
            ranked_dicts.append({
                "idea": idea.model_dump(),
                "score": score.model_dump(),
            })
            scores.append(score)

        best_title = ranked[0][0].title if ranked else "Unknown"
        best_score = ranked[0][1].composite_score if ranked else 0

        logger.info(
            "Ranked %d projects. Top: '%s' (%.1f/10).",
            len(ranked), best_title, best_score,
        )

        return {
            "project_scores": scores,
            "ranked_projects": ranked_dicts,
            "status": f"✅ Scored {len(ranked)} projects. Top: {best_title} ({best_score:.1f}/10).",
            "error": None,
            "current_node": "score_projects",
        }

    except Exception as exc:
        logger.exception("Project scoring failed")
        return {
            "error": f"Project scoring error: {exc}",
            "status": "❌ Project scoring failed.",
            "current_node": "score_projects",
        }
