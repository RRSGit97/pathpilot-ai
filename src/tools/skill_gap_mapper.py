"""
src/tools/skill_gap_mapper.py
-----------------------------
Deterministic mapper that compares a user's skills against aggregated role requirements.

This module uses pure Python logic to categorize skills into Strong, Partial, or Missing.
No LLMs are used here to ensure the gap analysis remains grounded in factual evidence.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.data.enums import SkillLevel
from src.data.schemas import (
    AggregatedRoleAnalysis,
    ResumeParseResult,
    SkillGapItem,
    SkillGapReport,
    UserProfile,
)
from src.tools.text_utils import normalize_skill_token

logger = logging.getLogger(__name__)


def _is_safe_substring(skill: str, text: str) -> bool:
    """
    Check if a skill appears in the text as a discrete token.
    Avoids matching "C" inside "Mac" or "Java" inside "JavaScript".
    """
    if not skill or not text:
        return False
    # Use negative lookbehind/lookahead for alphanumeric chars
    # This allows matching "C++" correctly (since + isn't alphanumeric)
    pattern = rf"(?<![a-zA-Z0-9]){re.escape(skill)}(?![a-zA-Z0-9])"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def map_skill_gaps(
    role_analysis: AggregatedRoleAnalysis,
    user_profile: UserProfile,
    resume_parse: Optional[ResumeParseResult] = None,
) -> SkillGapReport:
    """
    Compare user skills to aggregated role requirements deterministically.

    Categorization Rules:
    - STRONG: Skill is explicitly listed in user profile or extracted from resume.
    - PARTIAL: Skill isn't explicit, but appears in prior projects or resume text.
    - MISSING_REQUIRED: Skill is required by JDs but user lacks it.
    - MISSING_OPTIONAL: Skill is optional/tool-only and user lacks it.
    """
    # 1. Pool explicit skills (Strong candidates)
    explicit_raw = list(user_profile.current_skills)
    
    if resume_parse and not resume_parse.parse_warning:
        explicit_raw.extend(resume_parse.extracted_skills)
        
    explicit_normalized = {normalize_skill_token(s) for s in explicit_raw if s}

    # 2. Pool context text (Partial candidates)
    context_parts = [user_profile.prior_projects]
    if resume_parse:
        context_parts.append(resume_parse.work_experience_summary)
        context_parts.append(resume_parse.education_summary)
        context_parts.append(resume_parse.raw_text)
        
    full_context_text = " ".join(filter(None, context_parts)).lower()

    # 3. Gather all target skills from role analysis
    # Order matters: check required first, then optional, then tools.
    required_set = set(role_analysis.required_skills)
    optional_set = set(role_analysis.optional_skills)
    tools_set = set(role_analysis.tools_and_frameworks)
    
    # Tools that are neither required nor optional are treated as optional.
    for tool in tools_set:
        if tool not in required_set and tool not in optional_set:
            optional_set.add(tool)

    # Dedup target skills maintaining order of importance
    all_target_skills = []
    for s in role_analysis.required_skills:
        if s not in all_target_skills:
            all_target_skills.append(s)
    for s in role_analysis.optional_skills:
        if s not in all_target_skills:
            all_target_skills.append(s)
    for s in tools_set:
        if s not in all_target_skills:
            all_target_skills.append(s)

    items: list[SkillGapItem] = []
    
    for skill in all_target_skills:
        norm_skill = normalize_skill_token(skill)
        if not norm_skill:
            continue
            
        is_required = norm_skill in required_set
        
        # Determine level
        if norm_skill in explicit_normalized:
            level = SkillLevel.STRONG
            notes = "Explicitly listed in profile or resume."
        elif _is_safe_substring(norm_skill, full_context_text) or _is_safe_substring(skill, full_context_text):
            level = SkillLevel.PARTIAL
            notes = "Mentioned in prior projects or experience."
        else:
            level = SkillLevel.MISSING_REQUIRED if is_required else SkillLevel.MISSING_OPTIONAL
            notes = "Required by JDs." if is_required else "Optional/Preferred in JDs."
            
        jd_count = role_analysis.skill_jd_counts.get(norm_skill, 1)
        notes += f" Mentioned in {jd_count} job description(s)."

        items.append(
            SkillGapItem(
                skill=norm_skill,
                level=level,
                notes=notes,
            )
        )

    # 4. Calculate Relevance Score (based on required skills only)
    required_items = [item for item in items if item.skill in required_set]
    covered_required = [item for item in required_items if item.level in (SkillLevel.STRONG, SkillLevel.PARTIAL)]
    
    relevance_score = 1.0
    if required_items:
        relevance_score = len(covered_required) / len(required_items)

    return SkillGapReport(
        target_role=role_analysis.consensus_title,
        items=items,
        relevance_score=round(relevance_score, 2),
    )


def format_gap_report_markdown(report: SkillGapReport) -> str:
    """
    Format a SkillGapReport into a clean Markdown summary for the UI.
    """
    lines = [
        f"### Gap Analysis for: {report.target_role}",
        f"**Overall Relevance Score:** {report.relevance_score * 100:.0f}%\n",
    ]
    
    # Helper to format a section
    def _add_section(title: str, items: list[SkillGapItem]):
        if not items:
            return
        lines.append(f"#### {title}")
        for item in items:
            lines.append(f"- **{item.skill.title()}**: {item.notes}")
        lines.append("")

    _add_section("🌟 Strong Matches", report.strong)
    _add_section("⚠️ Partial Matches", report.partial)
    _add_section("❌ Missing Required Skills", report.missing_required)
    _add_section("💡 Missing Optional Skills", report.missing_optional)
    
    if not report.items:
        lines.append("*No specific skills were extracted from the job descriptions.*")

    return "\n".join(lines)
