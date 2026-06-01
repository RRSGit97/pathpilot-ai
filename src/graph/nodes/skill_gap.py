"""
src/graph/nodes/skill_gap.py
----------------------------
Node: Run deterministic skill gap mapping + LLM narrative generation.

1. Deterministic: skill_gap_mapper.map_skill_gaps → SkillGapReport
2. LLM: Skill Gap Agent generates a human-readable narrative.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.data.prompts import SKILL_GAP_SYSTEM_PROMPT, SKILL_GAP_USER_TEMPLATE
from src.data.schemas import SkillGapReport
from src.graph.state import PipelineState
from src.tools.skill_extractor import build_llm
from src.tools.skill_gap_mapper import format_gap_report_markdown, map_skill_gaps

logger = logging.getLogger(__name__)


def map_skill_gaps_node(state: PipelineState) -> dict:
    """
    Compute skill gaps deterministically, then generate a narrative via LLM.
    """
    logger.info("Node: map_skill_gaps")

    user_profile = state.get("user_profile")
    role_analysis = state.get("role_analysis")

    if not user_profile or not role_analysis:
        return {
            "error": "Missing user profile or role analysis for gap mapping.",
            "status": "❌ Skill gap mapping failed — missing inputs.",
            "current_node": "map_skill_gaps",
        }

    try:
        # Step 1: Deterministic gap computation
        resume_parse = state.get("resume_parse")
        gap_report = map_skill_gaps(role_analysis, user_profile, resume_parse)

        logger.info(
            "Gap report: strong=%d, partial=%d, missing_req=%d, missing_opt=%d, score=%.0f%%",
            len(gap_report.strong), len(gap_report.partial),
            len(gap_report.missing_required), len(gap_report.missing_optional),
            gap_report.relevance_score * 100,
        )

        # Step 2: LLM narrative (best-effort — pipeline works without it)
        narrative = format_gap_report_markdown(gap_report)  # fallback
        try:
            llm = build_llm(temperature=0.3)

            strong_str = ", ".join(i.skill for i in gap_report.strong) or "None"
            partial_str = ", ".join(i.skill for i in gap_report.partial) or "None"
            missing_req_str = ", ".join(i.skill for i in gap_report.missing_required) or "None"
            missing_opt_str = ", ".join(i.skill for i in gap_report.missing_optional) or "None"

            user_msg = SKILL_GAP_USER_TEMPLATE.format(
                target_role=role_analysis.consensus_title or "Target Role",
                relevance_score=int(gap_report.relevance_score * 100),
                strong_skills=strong_str,
                partial_skills=partial_str,
                missing_required=missing_req_str,
                missing_optional=missing_opt_str,
            )

            response = llm.invoke([
                SystemMessage(content=SKILL_GAP_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            narrative = response.content
            logger.info("Gap narrative generated via LLM.")

        except Exception as llm_exc:
            logger.warning("LLM narrative failed, using markdown fallback: %s", llm_exc)

        return {
            "gap_report": gap_report,
            "gap_narrative": narrative,
            "status": f"✅ Skill gap mapped — relevance score: {gap_report.relevance_score * 100:.0f}%.",
            "error": None,
            "current_node": "map_skill_gaps",
        }

    except Exception as exc:
        logger.exception("Skill gap mapping failed")
        return {
            "error": f"Skill gap error: {exc}",
            "status": "❌ Skill gap mapping failed.",
            "current_node": "map_skill_gaps",
        }
