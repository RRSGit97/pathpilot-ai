"""
src/data
--------
Shared data layer: enums, constants, and all Pydantic schemas.

Import from here for convenience:
    from src.data import SkillLevel, UserProfile, AppSettings
"""
from src.data.enums import (
    SkillLevel,
    LearningStyle,
    ProjectDifficulty,
    ApprovalStatus,
    LLMProvider,
    ArtifactType,
)
from src.data.constants import SCORING_WEIGHTS, SENIORITY_SIGNALS

__all__ = [
    # Enums
    "SkillLevel",
    "LearningStyle",
    "ProjectDifficulty",
    "ApprovalStatus",
    "LLMProvider",
    "ArtifactType",
    # Constants
    "SCORING_WEIGHTS",
    "SENIORITY_SIGNALS",
]
