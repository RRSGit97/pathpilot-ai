"""
src/graph/nodes/critique.py
----------------------------
Node: LLM-based critique of the user's chosen project idea.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.data.prompts import PROJECT_CRITIQUE_SYSTEM_PROMPT, PROJECT_CRITIQUE_USER_TEMPLATE
from src.data.schemas import ProjectIdea, ProjectScore
from src.graph.state import PipelineState
from src.tools.skill_extractor import build_llm

logger = logging.getLogger(__name__)


def critique_project_node(state: PipelineState) -> dict:
    """
    Critique the user's chosen project for feasibility and risks.
    """
    logger.info("Node: critique_project")

    ranked = state.get("ranked_projects")
    chosen_idx = state.get("chosen_project_idx", 0)
    gap_report = state.get("gap_report")

    if not ranked:
        return {
            "error": "No ranked projects available for critique.",
            "status": "❌ Critique failed — no projects ranked.",
            "current_node": "critique_project",
        }

    try:
        pair = ranked[chosen_idx]
        idea = ProjectIdea(**pair["idea"])
        score = ProjectScore(**pair["score"])

        missing_str = ", ".join(i.skill for i in gap_report.missing_required) if gap_report else "None"
        partial_str = ", ".join(i.skill for i in gap_report.partial) if gap_report else "None"

        llm = build_llm(temperature=0.3)

        user_msg = PROJECT_CRITIQUE_USER_TEMPLATE.format(
            project_title=idea.title,
            project_description=idea.description,
            technologies=", ".join(idea.technologies),
            architecture_overview=idea.architecture_overview,
            difficulty=idea.difficulty.value,
            resume_value=score.resume_value,
            agentic_fit=score.agentic_fit,
            buildability=score.buildability_4_6_weeks,
            personal_relevance=score.personal_relevance,
            technical_depth=score.technical_depth,
            differentiation=score.differentiation,
            recruiter_explainability=score.recruiter_explainability,
            composite_score=score.composite_score,
            missing_skills=missing_str,
            partial_skills=partial_str,
        )

        response = llm.invoke([
            SystemMessage(content=PROJECT_CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        content = response.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        critique = json.loads(content)

        logger.info("Critique verdict: %s", critique.get("overall_verdict", "unknown"))

        return {
            "project_critique": critique,
            "status": f"✅ Project critiqued: {critique.get('overall_verdict', 'reviewed')}.",
            "error": None,
            "current_node": "critique_project",
        }

    except Exception as exc:
        logger.exception("Project critique failed")
        return {
            "project_critique": {"overall_verdict": "error", "risks": [str(exc)]},
            "error": f"Critique error: {exc}",
            "status": "⚠️ Project critique failed — proceeding with caution.",
            "current_node": "critique_project",
        }
