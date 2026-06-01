from pydantic import BaseModel
from typing import List

class ExtractedJD(BaseModel):
    title: str
    seniority: str
    required_skills: List[str]
    optional_skills: List[str]
    tools_frameworks: List[str]
    project_expectations: List[str]
