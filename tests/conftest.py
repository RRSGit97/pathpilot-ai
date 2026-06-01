"""
tests/conftest.py
-----------------
Shared pytest fixtures for PathPilot AI tests.

Design goals
------------
- Zero live external services: no OpenAI/Gemini calls, no Qdrant, no network.
- Use an in-memory SQLite database so tests are isolated and fast.
- Provide ready-made Pydantic model instances so individual test files
  don't repeat boilerplate construction.
- Patch settings.database_path to a temp file so storage tests never
  touch the real application DB.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.data.enums import LearningStyle, ProjectDifficulty, SkillLevel
from src.data.schemas import (
    AggregatedRoleAnalysis,
    LearningRoadmap,
    LearningRoadmapWeek,
    PortfolioOutputs,
    ProgressLogEntry,
    ProjectIdea,
    ProjectScore,
    ResumeParseResult,
    SkillGapItem,
    SkillGapReport,
    SixWeekPlan,
    SixWeekPlanWeek,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Environment guard: ensure vector store is disabled for all tests
# ---------------------------------------------------------------------------

os.environ.setdefault("VECTOR_STORE_ENABLED", "false")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")


# ---------------------------------------------------------------------------
# Sample profile fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile() -> UserProfile:
    """A realistic transitioning AI engineer profile (matches sample_data/)."""
    return UserProfile(
        name="Alex Mercer",
        current_skills=["Python", "SQL", "Pandas", "NumPy", "Scikit-Learn", "Git", "Docker", "FastAPI"],
        prior_projects=(
            "Built a basic Retrieval-Augmented Generation (RAG) prototype using "
            "LangChain, ChromaDB, and OpenAI API. Also built a Streamlit dashboard "
            "for sales trend visualisation."
        ),
        target_roles=["AI Engineer", "Applied AI Engineer", "Machine Learning Engineer"],
        weekly_hours_available=15,
        learning_style=LearningStyle.PROJECT_BASED,
        pain_points=(
            "I struggle with multi-agent state management, LangGraph routing, "
            "and optimising vector databases like Qdrant."
        ),
    )


# ---------------------------------------------------------------------------
# Sample role analysis fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_role_analysis() -> AggregatedRoleAnalysis:
    """Aggregated role requirements used by gap-mapping and scoring tests."""
    return AggregatedRoleAnalysis(
        source_jd_count=3,
        consensus_title="AI Engineer",
        all_titles=["AI Engineer", "Applied AI Engineer"],
        seniority="mid",
        required_skills=["python", "langgraph", "fastapi", "qdrant", "pydantic"],
        optional_skills=["docker", "postgresql"],
        tools_and_frameworks=["langchain", "openai"],
        project_expectations=["Build multi-agent routing engines"],
        skill_jd_counts={
            "python": 3,
            "langgraph": 3,
            "fastapi": 2,
            "qdrant": 2,
            "pydantic": 2,
            "docker": 1,
            "postgresql": 1,
            "langchain": 1,
            "openai": 1,
        },
        evidence_snippets={},
    )


# ---------------------------------------------------------------------------
# Sample skill gap report fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gap_report() -> SkillGapReport:
    """Pre-built gap report for Alex Mercer vs the AI Engineer role."""
    items = [
        SkillGapItem(skill="python",    level=SkillLevel.STRONG,           notes="Explicitly listed."),
        SkillGapItem(skill="langgraph", level=SkillLevel.MISSING_REQUIRED, notes="Required by JDs."),
        SkillGapItem(skill="fastapi",   level=SkillLevel.STRONG,           notes="Explicitly listed."),
        SkillGapItem(skill="qdrant",    level=SkillLevel.MISSING_REQUIRED, notes="Required by JDs."),
        SkillGapItem(skill="pydantic",  level=SkillLevel.PARTIAL,          notes="In prior projects."),
        SkillGapItem(skill="docker",    level=SkillLevel.STRONG,           notes="Explicitly listed."),
        SkillGapItem(skill="postgresql",level=SkillLevel.MISSING_OPTIONAL, notes="Optional."),
        SkillGapItem(skill="langchain", level=SkillLevel.PARTIAL,          notes="In prior projects."),
        SkillGapItem(skill="openai",    level=SkillLevel.PARTIAL,          notes="In prior projects."),
    ]
    # relevance = covered required (python, fastapi, pydantic) / total required (5) = 3/5 = 0.6
    return SkillGapReport(
        target_role="AI Engineer",
        items=items,
        relevance_score=0.6,
    )


# ---------------------------------------------------------------------------
# Sample project idea fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agentic_project() -> ProjectIdea:
    """A well-crafted, agentic project idea that should score highly."""
    return ProjectIdea(
        title="Multi-Agent Job Application Assistant",
        description=(
            "Build a LangGraph-powered multi-agent system that automatically searches "
            "job boards, scores roles against a user profile, and drafts tailored "
            "cover letters using the OpenAI API. Deploy as a FastAPI microservice."
        ),
        technologies=["LangGraph", "OpenAI", "FastAPI", "Qdrant", "Pydantic"],
        architecture_overview=(
            "Supervisor agent orchestrates three sub-agents: a search agent, "
            "a scoring agent, and a drafting agent. State is persisted via SQLite "
            "checkpoint and exposed through a REST API."
        ),
        difficulty=ProjectDifficulty.INTERMEDIATE,
    )


@pytest.fixture
def cliche_project() -> ProjectIdea:
    """A boring todo-list chatbot that should score low on differentiation."""
    return ProjectIdea(
        title="Todo List Chatbot",
        description="A simple chatbot that wraps the OpenAI API to manage a todo list via natural language.",
        technologies=["Python", "OpenAI"],
        architecture_overview="Chatbot wrapper",
        difficulty=ProjectDifficulty.BEGINNER,
    )


# ---------------------------------------------------------------------------
# Temp SQLite DB fixture (isolated per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Yield a path to a fresh, temporary SQLite file.

    Tests that import `sqlite_store` or `checkpoint_store` should use this
    fixture and patch `settings.database_path` to point here so they never
    touch the real application database.
    """
    db_file = tmp_path / "test_pathpilot.db"
    yield db_file


@pytest.fixture
def initialized_db(temp_db_path: Path, monkeypatch):
    """
    Patch the settings database path and run init_db() to create tables.
    Returns the temp Path so tests can open a raw sqlite3 connection if needed.
    """
    # Patch settings BEFORE importing sqlite_store functions
    monkeypatch.setattr("src.config.settings.database_path", temp_db_path)
    monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", temp_db_path)

    from src.storage.sqlite_store import init_db
    init_db()
    return temp_db_path


# ---------------------------------------------------------------------------
# Minimal resume parse result
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_resume_parse() -> ResumeParseResult:
    return ResumeParseResult(
        extracted_skills=["Python", "LangChain", "ChromaDB"],
        work_experience_summary="2 years of data analysis and junior SWE roles.",
        education_summary="BSc Computer Science.",
        raw_text="Worked with Python, LangChain, ChromaDB. Built a RAG pipeline.",
    )
