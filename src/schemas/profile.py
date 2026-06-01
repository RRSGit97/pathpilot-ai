from pydantic import BaseModel
from typing import List, Optional

class UserProfile(BaseModel):
    name: str
    current_skills: List[str]
    prior_projects: str
    target_roles: List[str]
    weekly_hours: int
    learning_style: str
    pain_points: str
    resume_text: Optional[str] = None
