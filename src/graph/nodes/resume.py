"""
src/graph/nodes/resume.py
-------------------------
Node: Parse a resume file (PDF/DOCX/text) into ResumeParseResult.

Purely deterministic — no LLM calls.  Uses the resume_parser module.
"""

from __future__ import annotations

import logging

from src.data.schemas import ResumeParseResult
from src.graph.state import PipelineState
from src.tools.resume_parser import ResumeSourceType, parse_resume

logger = logging.getLogger(__name__)


def _detect_source_type(filename: str | None) -> ResumeSourceType:
    """Infer ResumeSourceType from the file extension."""
    if not filename:
        return ResumeSourceType.PLAIN_TEXT
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return ResumeSourceType.PDF
    if lower.endswith(".docx"):
        return ResumeSourceType.DOCX
    return ResumeSourceType.PLAIN_TEXT


def parse_resume_node(state: PipelineState) -> dict:
    """
    Parse resume_bytes → ResumeParseResult.

    Skips cleanly if no resume was uploaded (resume is optional in the MVP).
    Sets ``resume_parse.parse_warning`` if extraction was thin (e.g. scanned PDF).
    """
    logger.info("Node: parse_resume")

    resume_bytes = state.get("resume_bytes")
    if not resume_bytes:
        logger.info("No resume uploaded — skipping.")
        return {
            "resume_parse": None,
            "status": "⏭️ No resume uploaded — skipping resume parsing.",
            "current_node": "parse_resume",
        }

    filename = state.get("resume_filename")
    source_type = _detect_source_type(filename)

    try:
        result: ResumeParseResult = parse_resume(
            source=resume_bytes,
            source_type=source_type,
        )

        if result.parse_warning:
            logger.warning("Resume parse warning: %s", result.parse_warning)
            return {
                "resume_parse": result,
                "status": f"⚠️ Resume parsed with warning: {result.parse_warning}",
                "error": None,
                "current_node": "parse_resume",
            }

        logger.info(
            "Resume parsed: %d skills, %d chars raw text.",
            len(result.extracted_skills), len(result.raw_text),
        )
        return {
            "resume_parse": result,
            "status": f"✅ Resume parsed — {len(result.extracted_skills)} skills extracted.",
            "error": None,
            "current_node": "parse_resume",
        }

    except Exception as exc:
        logger.exception("Resume parsing failed")
        return {
            "resume_parse": None,
            "error": f"Resume parsing error: {exc}",
            "status": "⚠️ Resume parsing failed — continuing without resume data.",
            "current_node": "parse_resume",
        }
