from typing import List
from src.schemas.skill_gap import SkillGapReport

def categorize_skills(profile_skills: List[str], required_skills: List[str], optional_skills: List[str]) -> SkillGapReport:
    # Placeholder categorizer logic
    return SkillGapReport(
        strong_skills=[],
        partial_skills=[],
        missing_required=[],
        missing_optional=[],
        relevance_score=0.0
    )
