from pydantic import BaseModel
from typing import List

class RoadmapWeek(BaseModel):
    week_number: int
    focus_topic: str
    skills_covered: List[str]
    tasks: List[str]
    estimated_hours: int
    resources: List[str]

class LearningRoadmap(BaseModel):
    title: str
    target_role: str
    total_weeks: int
    weeks: List[RoadmapWeek]
