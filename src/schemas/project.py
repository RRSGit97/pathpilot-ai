from pydantic import BaseModel
from typing import List

class ProjectScores(BaseModel):
    resume_value: float
    agentic_fit: float
    buildability_4_6_weeks: float
    personal_relevance: float
    technical_depth: float
    differentiation: float
    recruiter_explainability: float

class ProjectIdea(BaseModel):
    title: str
    description: str
    technologies: List[str]
    architecture_overview: str
    difficulty: str

class ScoredProject(BaseModel):
    idea: ProjectIdea
    scores: ProjectScores
    composite_score: float

class BuildPlanWeek(BaseModel):
    week_number: int
    goals: List[str]
    tasks: List[str]
    deliverable: str

class BuildPlan(BaseModel):
    project_title: str
    weeks: List[BuildPlanWeek]
