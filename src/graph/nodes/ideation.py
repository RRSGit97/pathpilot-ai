"""
src/graph/nodes/ideation.py
---------------------------
Node: Generate 3–5 project ideas via the Project Ideation Agent LLM.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.data.prompts import PROJECT_IDEATION_SYSTEM_PROMPT, PROJECT_IDEATION_USER_TEMPLATE
from src.data.schemas import ProjectIdea
from src.graph.state import PipelineState
from src.tools.skill_extractor import build_llm

logger = logging.getLogger(__name__)


def generate_projects_node(state: PipelineState) -> dict:
    """
    Generate 3–5 project ideas tailored to the user's profile and gaps.
    """
    logger.info("Node: generate_projects")

    user_profile = state.get("user_profile")
    gap_report = state.get("gap_report")
    role_analysis = state.get("role_analysis")

    if not user_profile or not gap_report or not role_analysis:
        return {
            "error": "Missing profile, gap report, or role analysis for ideation.",
            "status": "❌ Project ideation failed — missing inputs.",
            "current_node": "generate_projects",
        }

    try:
        llm = build_llm(temperature=0.7)  # higher temp for creativity

        target_role = role_analysis.consensus_title or "Target Role"
        seniority = role_analysis.seniority or "unknown"
        current = ", ".join(user_profile.current_skills) or "None"
        missing_req = ", ".join(i.skill for i in gap_report.missing_required) or "None"
        partial = ", ".join(i.skill for i in gap_report.partial) or "None"
        missing_opt = ", ".join(i.skill for i in gap_report.missing_optional) or "None"
        tools = ", ".join(role_analysis.tools_and_frameworks[:10]) or "None"

        user_msg = PROJECT_IDEATION_USER_TEMPLATE.format(
            count=4,
            target_role=target_role,
            seniority=seniority,
            current_skills=current,
            prior_projects=user_profile.prior_projects or "None",
            pain_points=user_profile.pain_points or "None",
            weekly_hours=user_profile.weekly_hours_available,
            missing_required=missing_req,
            partial_skills=partial,
            missing_optional=missing_opt,
            tools_and_frameworks=tools,
        )

        response = llm.invoke([
            SystemMessage(content=PROJECT_IDEATION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        content = response.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        raw_ideas = json.loads(content)
        if isinstance(raw_ideas, dict) and "projects" in raw_ideas:
            raw_ideas = raw_ideas["projects"]

        ideas = []
        for raw in raw_ideas:
            try:
                idea = ProjectIdea(**raw)
                ideas.append(idea)
            except Exception as parse_err:
                logger.warning("Skipping malformed idea: %s", parse_err)

        if not ideas:
            return {
                "error": "LLM returned no valid project ideas.",
                "status": "❌ Project ideation returned no valid ideas.",
                "current_node": "generate_projects",
            }

        logger.info("Generated %d project ideas.", len(ideas))

        return {
            "project_ideas": ideas,
            "status": f"✅ Generated {len(ideas)} project ideas.",
            "error": None,
            "current_node": "generate_projects",
        }

    except Exception as exc:
        logger.exception("Project ideation failed")
        return {
            "error": f"Project ideation error: {exc}",
            "status": "❌ Project ideation failed.",
            "current_node": "generate_projects",
        }
