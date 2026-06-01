"""
src/graph/nodes/jd_extraction.py
--------------------------------
Node: Parse raw JD texts + extract aggregated role requirements.

Combines two steps:
1. Deterministic JD cleaning (job_parser.parse_multiple_jds)
2. LLM extraction + deterministic aggregation (skill_extractor.extract_role_requirements)
"""

from __future__ import annotations

import logging

from src.graph.state import PipelineState
from src.tools.job_parser import parse_multiple_jds
from src.tools.skill_extractor import extract_role_requirements

logger = logging.getLogger(__name__)


def extract_jds_node(state: PipelineState) -> dict:
    """
    raw_jd_texts → parse → extract → AggregatedRoleAnalysis.
    """
    logger.info("Node: extract_jds")

    raw_jds = state.get("raw_jd_texts", [])
    if not raw_jds or not any(jd.strip() for jd in raw_jds):
        return {
            "error": "No job descriptions provided.",
            "status": "❌ JD extraction failed — no text provided.",
            "current_node": "extract_jds",
        }

    try:
        # Step 1: Deterministic cleaning + metadata inference
        parsed_jds = parse_multiple_jds(raw_jds)
        logger.info("Cleaned %d JDs from %d raw inputs.", len(parsed_jds), len(raw_jds))

        if not parsed_jds:
            return {
                "error": "All JDs were too short or invalid after cleaning.",
                "status": "❌ No valid JDs after cleaning.",
                "current_node": "extract_jds",
            }

        # Step 2: LLM extraction per JD + deterministic aggregation
        role_analysis = extract_role_requirements(parsed_jds)

        logger.info(
            "Role analysis: title=%r, required=%d, optional=%d, seniority=%s",
            role_analysis.consensus_title,
            len(role_analysis.required_skills),
            len(role_analysis.optional_skills),
            role_analysis.seniority,
        )

        return {
            "role_analysis": role_analysis,
            "status": (
                f"✅ Extracted requirements from {role_analysis.source_jd_count} JDs. "
                f"Target: {role_analysis.consensus_title or 'Unknown Role'}."
            ),
            "error": None,
            "current_node": "extract_jds",
        }

    except Exception as exc:
        logger.exception("JD extraction failed")
        return {
            "error": f"JD extraction error: {exc}",
            "status": "❌ JD extraction failed.",
            "current_node": "extract_jds",
        }
