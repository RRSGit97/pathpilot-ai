from pydantic import BaseModel
from typing import List, Dict

class SkillGapReport(BaseModel):
    strong_skills: List[str]
    partial_skills: List[str]
    missing_required: List[str]
    missing_optional: List[str]
    relevance_score: float
