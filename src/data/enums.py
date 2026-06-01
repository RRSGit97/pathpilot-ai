"""
src/data/enums.py
-----------------
All enumerations used across PathPilot AI.

Keeping enums separate from schemas prevents circular imports and makes
it easy to import a single lightweight module in logic layers that
should not depend on Pydantic.
"""
from enum import Enum


class SkillLevel(str, Enum):
    """Categorises a user's proficiency relative to a job requirement."""

    STRONG = "strong"
    """User has demonstrated or strong self-assessed proficiency."""

    PARTIAL = "partial"
    """User has exposure but not mastery (e.g., used once, academic only)."""

    MISSING_REQUIRED = "missing_required"
    """Skill appears in the required section of JDs but user has none."""

    MISSING_OPTIONAL = "missing_optional"
    """Skill appears only in optional/preferred JD sections; user lacks it."""


class LearningStyle(str, Enum):
    """User's preferred mode of absorbing new material."""

    PROJECT_BASED = "project_based"
    THEORETICAL = "theoretical"
    VIDEO_TUTORIAL = "video_tutorial"
    MIXED = "mixed"


class ProjectDifficulty(str, Enum):
    """Self-assessed complexity of a project idea."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ApprovalStatus(str, Enum):
    """Tracks whether a human has reviewed and accepted a workflow stage."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class LLMProvider(str, Enum):
    """Supported LLM backends.  Add new values here to enable new providers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class ArtifactType(str, Enum):
    """The distinct portfolio documents PathPilot can produce."""

    README_OUTLINE = "readme_outline"
    RESUME_BULLETS = "resume_bullets"
    ARCHITECTURE_SUMMARY = "architecture_summary"
    DEMO_SCRIPT = "demo_script"
    INTERVIEW_SHORT = "interview_short"   # 30-second explanation
    INTERVIEW_LONG = "interview_long"     # 2-minute explanation
