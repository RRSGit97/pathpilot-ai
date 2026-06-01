"""
src/tools/scoring.py
--------------------
Deterministic project scoring module.

Scores `ProjectIdea` objects across seven dimensions (0–10 scale) without LLM calls.
Uses clear, understandable heuristics based on keyword matching, gap reports, and 
project difficulty. This ensures scoring is fast, repeatable, and easily explainable.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.data.enums import ProjectDifficulty
from src.data.schemas import ProjectIdea, ProjectScore, SkillGapReport, UserProfile
from src.tools.text_utils import normalize_skill_token


def _count_matches(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear as whole words in the text."""
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(kw.lower())}(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            count += 1
    return count


def score_project(
    project: ProjectIdea,
    gap_report: SkillGapReport,
    user_profile: UserProfile,
) -> ProjectScore:
    """
    Score a single project deterministically across 7 dimensions (0.0 to 10.0).
    """
    
    # Pre-process text fields for easier matching
    full_text = f"{project.title} {project.description} {project.architecture_overview}".lower()
    tech_normalized = {normalize_skill_token(t) for t in project.technologies if normalize_skill_token(t)}
    
    # ---------------------------------------------------------
    # 1. Resume Value (0-10)
    # ---------------------------------------------------------
    # Rewards projects that use skills the user is missing or only partially knows.
    resume_val = 0.0
    missing_req = {normalize_skill_token(item.skill) for item in gap_report.missing_required}
    missing_opt = {normalize_skill_token(item.skill) for item in gap_report.missing_optional}
    partial_skills = {normalize_skill_token(item.skill) for item in gap_report.partial}
    
    for tech in tech_normalized:
        if tech in missing_req:
            resume_val += 4.0  # High reward for missing required skills
        elif tech in partial_skills:
            resume_val += 2.0  # Medium reward for strengthening partial skills
        elif tech in missing_opt:
            resume_val += 1.0  # Low reward for optional skills

    resume_val = min(10.0, resume_val)

    # ---------------------------------------------------------
    # 2. Agentic AI Fit (0-10)
    # ---------------------------------------------------------
    # Rewards terms indicating multi-agent or autonomous workflows.
    high_agentic = ["agent", "multi-agent", "langgraph", "autonomous", "tool-calling", "agentic"]
    medium_agentic = ["llm", "rag", "langchain", "prompt", "openai", "gemini", "claude"]
    
    agentic_score = 0.0
    agentic_score += _count_matches(full_text, high_agentic) * 3.5
    agentic_score += _count_matches(full_text, medium_agentic) * 1.5
    for tech in tech_normalized:
        if tech in high_agentic:
            agentic_score += 3.5
        elif tech in medium_agentic:
            agentic_score += 1.5
            
    agentic_score = min(10.0, agentic_score)

    # ---------------------------------------------------------
    # 3. Buildability in 4-6 Weeks (0-10)
    # ---------------------------------------------------------
    # Advanced projects are harder to finish in 6 weeks. 
    # Too many technologies also lower buildability.
    if project.difficulty == ProjectDifficulty.BEGINNER:
        build_score = 9.0
    elif project.difficulty == ProjectDifficulty.INTERMEDIATE:
        build_score = 7.0
    else:
        build_score = 4.0
        
    tech_count = len(project.technologies)
    if tech_count > 5:
        build_score -= (tech_count - 5) * 1.0  # -1 for each tech over 5
        
    build_score = max(0.0, min(10.0, build_score))

    # ---------------------------------------------------------
    # 4. Personal Relevance (0-10)
    # ---------------------------------------------------------
    # Match project text against user pain points and target roles.
    rel_score = 3.0  # Baseline
    if user_profile.pain_points:
        # Split pain points into words > 3 chars
        pain_words = [w for w in re.split(r'\W+', user_profile.pain_points.lower()) if len(w) > 3]
        rel_score += _count_matches(full_text, pain_words) * 2.0
        
    for role in user_profile.target_roles:
        if role.lower() in full_text:
            rel_score += 3.0

    rel_score = min(10.0, rel_score)

    # ---------------------------------------------------------
    # 5. Technical Depth (0-10)
    # ---------------------------------------------------------
    if project.difficulty == ProjectDifficulty.ADVANCED:
        depth_score = 8.0
    elif project.difficulty == ProjectDifficulty.INTERMEDIATE:
        depth_score = 5.0
    else:
        depth_score = 2.0
        
    depth_keywords = ["database", "api", "auth", "cloud", "docker", "deploy", "pipeline", "async", "cache", "aws", "gcp"]
    depth_score += _count_matches(full_text, depth_keywords) * 1.0
    depth_score = min(10.0, depth_score)

    # ---------------------------------------------------------
    # 6. Differentiation (0-10)
    # ---------------------------------------------------------
    # Penalize highly cliché beginner projects.
    diff_score = 10.0
    cliche_words = ["chatbot", "todo", "calculator", "weather", "wrapper", "tic-tac-toe", "crud"]
    diff_score -= _count_matches(project.title.lower() + " " + project.description.lower(), cliche_words) * 3.0
    diff_score = max(0.0, diff_score)

    # ---------------------------------------------------------
    # 7. Recruiter Explainability (0-10)
    # ---------------------------------------------------------
    # Good explainability comes from clear, appropriately sized descriptions and action verbs.
    exp_score = 7.0
    desc_len = len(project.description)
    if desc_len < 40 or desc_len > 300:
        exp_score -= 2.0
    else:
        exp_score += 1.0
        
    if project.architecture_overview and len(project.architecture_overview) > 40:
        exp_score += 1.0
        
    action_verbs = ["build", "create", "automate", "predict", "design", "deploy", "integrate", "orchestrate"]
    exp_score += _count_matches(full_text, action_verbs) * 0.5
    exp_score = min(10.0, exp_score)

    # ---------------------------------------------------------
    # Compile Score
    # ---------------------------------------------------------
    score = ProjectScore(
        project_id=project.id,
        resume_value=round(resume_val, 1),
        agentic_fit=round(agentic_score, 1),
        buildability_4_6_weeks=round(build_score, 1),
        personal_relevance=round(rel_score, 1),
        technical_depth=round(depth_score, 1),
        differentiation=round(diff_score, 1),
        recruiter_explainability=round(exp_score, 1),
    )
    score.compute_composite()
    return score


def rank_projects(
    projects: list[ProjectIdea],
    gap_report: SkillGapReport,
    user_profile: UserProfile,
) -> list[tuple[ProjectIdea, ProjectScore]]:
    """
    Score a list of projects and return them sorted by composite score (highest first).
    """
    scored = []
    for p in projects:
        score = score_project(p, gap_report, user_profile)
        scored.append((p, score))
        
    # Sort descending by composite score, tiebreaker on agentic_fit
    scored.sort(key=lambda x: (x[1].composite_score, x[1].agentic_fit), reverse=True)
    return scored


def explain_top_projects(ranked_pairs: list[tuple[ProjectIdea, ProjectScore]], top_n: int = 3) -> str:
    """
    Generate a markdown explanation of why the top projects scored well.
    Highlights the standout dimensions (>= 8.0) for each top project.
    """
    if not ranked_pairs:
        return "No projects to explain."
        
    limit = min(len(ranked_pairs), top_n)
    top_projects = ranked_pairs[:limit]
    
    lines = [f"### Top {limit} Project Recommendations"]
    
    for rank, (proj, score) in enumerate(top_projects, start=1):
        lines.append(f"\n#### #{rank}: {proj.title} (Score: {score.composite_score:.1f}/10)")
        lines.append(f"*{proj.description}*")
        
        # Find standout dimensions
        standouts = []
        for dim_name, dim_val in score.as_dict().items():
            if dim_val >= 8.0:
                standouts.append(f"**{dim_name} ({dim_val:.1f})**")
                
        if standouts:
            lines.append("\n**Why it ranks highly:**")
            lines.append(f"- Outstanding in: {', '.join(standouts)}")
            
        # Add a tailored reason based on the highest specific sub-scores
        reasons = []
        if score.resume_value >= 8.0:
            reasons.append("Fills critical skill gaps directly from your target roles.")
        if score.agentic_fit >= 8.0:
            reasons.append("Perfectly demonstrates modern Agentic AI workflows.")
        if score.differentiation >= 8.0:
            reasons.append("Avoids common clichés, making your portfolio stand out.")
        if score.buildability_4_6_weeks >= 8.0:
            reasons.append("Highly achievable within a 4-6 week sprint.")
            
        for reason in reasons[:2]: # Show max 2 reasons to keep it brief
            lines.append(f"- {reason}")
            
    return "\n".join(lines)
