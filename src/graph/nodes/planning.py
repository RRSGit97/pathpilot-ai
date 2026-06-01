"""
src/graph/nodes/planning.py
----------------------------
Node: Generate a 6-week build plan via the Planning Agent LLM.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.data.prompts import PLANNING_SYSTEM_PROMPT, PLANNING_USER_TEMPLATE
from src.data.schemas import ProjectIdea, SixWeekPlan, SixWeekPlanWeek
from src.graph.state import PipelineState
from src.tools.skill_extractor import build_llm

logger = logging.getLogger(__name__)


def generate_build_plan_node(state: PipelineState) -> dict:
    """
    Generate a 6-week build plan for the chosen project.
    """
    logger.info("Node: generate_build_plan")

    ranked = state.get("ranked_projects")
    chosen_idx = state.get("chosen_project_idx", 0)
    critique = state.get("project_critique", {})
    user_profile = state.get("user_profile")
    gap_report = state.get("gap_report")

    if not ranked or not user_profile:
        return {
            "error": "Missing ranked projects or profile for planning.",
            "status": "❌ Build plan generation failed — missing inputs.",
            "current_node": "generate_build_plan",
        }

    try:
        pair = ranked[chosen_idx]
        idea = ProjectIdea(**pair["idea"])

        skills_to_learn = ", ".join(i.skill for i in gap_report.missing_required) if gap_report else "None"
        current_skills = ", ".join(user_profile.current_skills) or "None"

        llm = build_llm(temperature=0.3)

        user_msg = PLANNING_USER_TEMPLATE.format(
            project_title=idea.title,
            project_description=idea.description,
            technologies=", ".join(idea.technologies),
            architecture_overview=idea.architecture_overview,
            difficulty=idea.difficulty.value,
            risks=", ".join(critique.get("risks", ["None identified"])),
            suggestions=", ".join(critique.get("suggestions", ["None"])),
            scope_warning=critique.get("scope_warning", "None"),
            weekly_hours=user_profile.weekly_hours_available,
            current_skills=current_skills,
            skills_to_learn=skills_to_learn,
        )

        response = llm.invoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        content = response.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        data = json.loads(content)

        weeks = [SixWeekPlanWeek(**w) for w in data.get("weeks", [])]
        plan = SixWeekPlan(
            project_id=idea.id,
            project_title=data.get("project_title", idea.title),
            weeks=weeks,
        )

        logger.info("Build plan generated: %d weeks.", len(plan.weeks))

        return {
            "build_plan": plan,
            "status": f"✅ 6-week build plan generated for '{idea.title}'.",
            "error": None,
            "current_node": "generate_build_plan",
        }

    except Exception as exc:
        logger.exception("Build plan generation failed")
        return {
            "error": f"Build plan error: {exc}",
            "status": "❌ Build plan generation failed.",
            "current_node": "generate_build_plan",
        }
