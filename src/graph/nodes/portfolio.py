"""
src/graph/nodes/portfolio.py
----------------------------
Node: Generate portfolio artifacts via the Portfolio Agent LLM.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.data.prompts import PORTFOLIO_SYSTEM_PROMPT, PORTFOLIO_USER_TEMPLATE
from src.data.schemas import PortfolioOutputs, ProjectIdea, SixWeekPlan
from src.graph.state import PipelineState
from src.tools.skill_extractor import build_llm

logger = logging.getLogger(__name__)


def _summarize_build_plan(plan: SixWeekPlan) -> str:
    """Create a text summary of the build plan for the portfolio prompt."""
    lines = []
    for week in plan.weeks:
        goals = "; ".join(week.goals) if week.goals else "No goals"
        lines.append(f"Week {week.week_number}: {goals} → Deliverable: {week.deliverable}")
    return "\n".join(lines)


def generate_portfolio_node(state: PipelineState) -> dict:
    """
    Generate portfolio outputs (README, resume bullets, demo script, etc.).
    """
    logger.info("Node: generate_portfolio")

    ranked = state.get("ranked_projects")
    chosen_idx = state.get("chosen_project_idx", 0)
    build_plan = state.get("build_plan")
    role_analysis = state.get("role_analysis")

    if not ranked or not build_plan:
        return {
            "error": "Missing project or build plan for portfolio generation.",
            "status": "❌ Portfolio generation failed — missing inputs.",
            "current_node": "generate_portfolio",
        }

    try:
        pair = ranked[chosen_idx]
        idea = ProjectIdea(**pair["idea"])

        plan_summary = _summarize_build_plan(build_plan)
        target_role = role_analysis.consensus_title if role_analysis else "Target Role"

        # Confirmed features = build plan deliverables (conservative)
        confirmed = [w.deliverable for w in build_plan.weeks if w.deliverable]
        confirmed_str = "\n".join(f"- {f}" for f in confirmed) if confirmed else "No features confirmed yet."

        llm = build_llm(temperature=0.3)

        user_msg = PORTFOLIO_USER_TEMPLATE.format(
            project_title=idea.title,
            project_description=idea.description,
            technologies=", ".join(idea.technologies),
            architecture_overview=idea.architecture_overview,
            build_plan_summary=plan_summary,
            confirmed_features=confirmed_str,
            target_role=target_role,
        )

        response = llm.invoke([
            SystemMessage(content=PORTFOLIO_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        content = response.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        data = json.loads(content)

        portfolio = PortfolioOutputs(
            project_id=idea.id,
            readme_outline=data.get("readme_outline", ""),
            resume_bullets=data.get("resume_bullets", []),
            architecture_summary=data.get("architecture_summary", ""),
            demo_script=data.get("demo_script", ""),
            interview_explanation_30s=data.get("interview_explanation_30s", ""),
            interview_explanation_2m=data.get("interview_explanation_2m", ""),
        )

        logger.info(
            "Portfolio generated: %d resume bullets, %d chars README.",
            len(portfolio.resume_bullets), len(portfolio.readme_outline),
        )

        return {
            "portfolio_outputs": portfolio,
            "status": f"✅ Portfolio artifacts generated for '{idea.title}'.",
            "error": None,
            "current_node": "generate_portfolio",
        }

    except Exception as exc:
        logger.exception("Portfolio generation failed")
        return {
            "error": f"Portfolio generation error: {exc}",
            "status": "❌ Portfolio generation failed.",
            "current_node": "generate_portfolio",
        }
