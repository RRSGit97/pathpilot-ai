from pydantic import BaseModel
from typing import List

class PortfolioArtifacts(BaseModel):
    readme_outline: str
    resume_bullets: List[str]
    architecture_summary: str
    demo_script: str
    interview_explanation_30s: str
    interview_explanation_2m: str
