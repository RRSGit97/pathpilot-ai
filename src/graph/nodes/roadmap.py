"""
src/graph/nodes/roadmap.py
--------------------------
Node: Generate a personalised learning roadmap via the Resource Agent LLM.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.data.prompts import ROADMAP_SYSTEM_PROMPT, ROADMAP_USER_TEMPLATE
from src.data.schemas import LearningRoadmap, LearningRoadmapWeek
from src.graph.state import PipelineState
from src.tools.skill_extractor import build_llm

logger = logging.getLogger(__name__)


def generate_roadmap_node(state: PipelineState) -> dict:
    """
    Generate a week-by-week learning roadmap using the Resource Agent LLM.
    """
    logger.info("Node: generate_roadmap")

    user_profile = state.get("user_profile")
    gap_report = state.get("gap_report")
    role_analysis = state.get("role_analysis")

    if not user_profile or not gap_report:
        return {
            "error": "Missing profile or gap report for roadmap generation.",
            "status": "❌ Roadmap generation failed — missing inputs.",
            "current_node": "generate_roadmap",
        }

    try:
        llm = build_llm(temperature=0.4)

        target_role = (role_analysis.consensus_title if role_analysis else "Target Role")
        missing_req = ", ".join(i.skill for i in gap_report.missing_required) or "None"
        partial = ", ".join(i.skill for i in gap_report.partial) or "None"
        current = ", ".join(user_profile.current_skills) or "None"

        # Scale weeks: more gaps → more weeks (4–8)
        total_gaps = len(gap_report.missing_required) + len(gap_report.partial)
        total_weeks = min(8, max(4, total_gaps))

        user_msg = ROADMAP_USER_TEMPLATE.format(
            total_weeks=total_weeks,
            target_role=target_role,
            current_skills=current,
            weekly_hours=user_profile.weekly_hours_available,
            learning_style=user_profile.learning_style.value,
            missing_required=missing_req,
            partial_skills=partial,
        )

        response = llm.invoke([
            SystemMessage(content=ROADMAP_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        # Parse response into our schema
        content = response.content
        # Strip markdown fences if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        data = json.loads(content)

        weeks = [LearningRoadmapWeek(**w) for w in data.get("weeks", [])]
        roadmap = LearningRoadmap(
            title=data.get("title", f"Learning Roadmap for {target_role}"),
            target_role=data.get("target_role", target_role),
            total_weeks=total_weeks,
            weeks=weeks,
        )

        logger.info("Roadmap generated: %d weeks.", len(roadmap.weeks))

        return {
            "learning_roadmap": roadmap,
            "status": f"✅ {total_weeks}-week learning roadmap generated.",
            "error": None,
            "current_node": "generate_roadmap",
        }

    except Exception as exc:
        logger.exception("Roadmap generation failed")
        return {
            "error": f"Roadmap generation error: {exc}",
            "status": "❌ Roadmap generation failed.",
            "current_node": "generate_roadmap",
        }
