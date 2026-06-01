"""
src/graph/nodes/profile.py
--------------------------
Node: Ingest the user's freeform self-description into a UserProfile.

Uses the Profile Agent (LLM) to parse unstructured text into the
UserProfile Pydantic model.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.data.prompts import PROFILE_SYSTEM_PROMPT, PROFILE_USER_TEMPLATE
from src.data.schemas import UserProfile
from src.graph.state import PipelineState
from src.tools.skill_extractor import build_llm

logger = logging.getLogger(__name__)


def ingest_profile_node(state: PipelineState) -> dict:
    """
    Parse raw_profile_text → UserProfile via LLM structured output.

    Falls back to a minimal profile if the LLM call fails, so the
    pipeline can still proceed with whatever the user typed.
    """
    logger.info("Node: ingest_profile")

    raw_text = state.get("raw_profile_text", "")
    if not raw_text.strip():
        return {
            "error": "No profile text provided.",
            "status": "❌ Profile intake failed — no text provided.",
            "current_node": "ingest_profile",
        }

    try:
        llm = build_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(UserProfile)

        user_msg = PROFILE_USER_TEMPLATE.format(raw_input=raw_text)
        profile: UserProfile = structured_llm.invoke([
            SystemMessage(content=PROFILE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        logger.info(
            "Profile parsed: name=%s, skills=%d, roles=%d",
            profile.name, len(profile.current_skills), len(profile.target_roles),
        )

        return {
            "user_profile": profile,
            "status": f"✅ Profile created for {profile.name}.",
            "error": None,
            "current_node": "ingest_profile",
        }

    except Exception as exc:
        logger.exception("Profile ingestion failed")
        # Graceful fallback: create a minimal profile from the raw text
        fallback = UserProfile(
            name="User",
            current_skills=[],
            target_roles=[],
        )
        return {
            "user_profile": fallback,
            "error": f"Profile parsing error (using fallback): {exc}",
            "status": "⚠️ Profile parsed with fallback — please review.",
            "current_node": "ingest_profile",
        }
