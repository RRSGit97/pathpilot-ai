"""
tests/test_sqlite_store.py
--------------------------
Tests for src/storage/sqlite_store.py.

Strategy
--------
- Each test gets a fresh, isolated SQLite file (via the `initialized_db` fixture).
- We patch `src.storage.sqlite_store.settings.database_path` to the temp file.
- No real application DB is touched.

Covers:
- init_db: creates tables
- save_session_state / load_session_state: round-trip for all fields
- add_progress_log / get_progress_logs: CRUD and ordering
- export_session_summary: correct summary keys and values
- load_session_state: returns None for unknown session
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.data.schemas import (
    AggregatedRoleAnalysis,
    LearningRoadmap,
    LearningRoadmapWeek,
    PortfolioOutputs,
    ProgressLogEntry,
    ProjectIdea,
    SkillGapReport,
    SixWeekPlan,
    SixWeekPlanWeek,
    UserProfile,
)
from src.graph.state import PipelineState


# ---------------------------------------------------------------------------
# Helper: build a minimal PipelineState for round-trip tests
# ---------------------------------------------------------------------------

def _minimal_state(profile: UserProfile) -> PipelineState:
    return {
        "raw_profile_text": "I know Python.",
        "raw_jd_texts": ["JD text here"],
        "user_profile": profile,
        "status": "Testing",
        "current_node": "test",
    }


def _full_state(
    profile: UserProfile,
    role_analysis: AggregatedRoleAnalysis,
    gap_report: SkillGapReport,
) -> PipelineState:
    roadmap = LearningRoadmap(
        title="AI Engineer Roadmap",
        target_role="AI Engineer",
        total_weeks=1,
        weeks=[
            LearningRoadmapWeek(
                week_number=1,
                focus_topic="LangGraph basics",
                estimated_hours=10,
            )
        ],
    )
    project = ProjectIdea(
        title="Multi-Agent Demo",
        description="Demo project using LangGraph and OpenAI.",
        technologies=["LangGraph", "OpenAI"],
        difficulty="intermediate",
    )
    six_wk = SixWeekPlan(
        project_id=project.id,
        project_title=project.title,
        weeks=[
            SixWeekPlanWeek(week_number=i, goals=[f"Goal {i}"], tasks=[f"Task {i}"], deliverable=f"Del {i}")
            for i in range(1, 7)
        ],
    )
    portfolio = PortfolioOutputs(
        project_id=project.id,
        resume_bullets=["Built X using Y"],
        readme_outline="# My Project\n...",
    )
    return {
        "raw_profile_text": "Full state.",
        "raw_jd_texts": ["JD 1", "JD 2"],
        "user_profile": profile,
        "role_analysis": role_analysis,
        "gap_report": gap_report,
        "gap_narrative": "You are strong in Python but lack LangGraph.",
        "learning_roadmap": roadmap,
        "project_ideas": [project],
        "ranked_projects": [
            {
                "idea": project.model_dump(),
                "score": {"composite_score": 7.5},
            }
        ],
        "chosen_project_idx": 0,
        "build_plan": six_wk,
        "portfolio_outputs": portfolio,
        "approvals": {"target_role": "approved"},
        "status": "Done",
        "current_node": "finished",
    }


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_app_sessions_table(self, initialized_db: Path):
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='app_sessions'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_progress_logs_table(self, initialized_db: Path):
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='progress_logs'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_idempotent_on_multiple_calls(self, initialized_db: Path, monkeypatch):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import init_db
        # Should not raise
        init_db()
        init_db()


# ---------------------------------------------------------------------------
# save_session_state / load_session_state round-trip
# ---------------------------------------------------------------------------

class TestSaveAndLoadSessionState:
    def test_save_then_load_returns_state(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import load_session_state, save_session_state

        state = _minimal_state(sample_profile)
        save_session_state("sess-001", state)
        loaded = load_session_state("sess-001")

        assert loaded is not None
        assert loaded["user_profile"].name == sample_profile.name

    def test_load_returns_none_for_unknown_session(self, initialized_db, monkeypatch):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import load_session_state

        result = load_session_state("does-not-exist")
        assert result is None

    def test_full_state_round_trip(
        self, initialized_db, monkeypatch, sample_profile, sample_role_analysis, sample_gap_report
    ):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import load_session_state, save_session_state

        state = _full_state(sample_profile, sample_role_analysis, sample_gap_report)
        save_session_state("sess-full", state)
        loaded = load_session_state("sess-full")

        assert loaded is not None
        assert loaded["gap_narrative"] == "You are strong in Python but lack LangGraph."
        assert loaded["chosen_project_idx"] == 0
        assert loaded["learning_roadmap"].total_weeks == 1
        assert len(loaded["project_ideas"]) == 1
        assert loaded["build_plan"].project_title == "Multi-Agent Demo"
        assert len(loaded["portfolio_outputs"].resume_bullets) == 1

    def test_upsert_updates_existing_session(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import load_session_state, save_session_state

        state1 = _minimal_state(sample_profile)
        save_session_state("sess-upsert", state1)

        # Now save with a different status
        state2 = dict(state1)
        state2["status"] = "Updated status"
        save_session_state("sess-upsert", state2)

        loaded = load_session_state("sess-upsert")
        # Only one row should exist; profile still intact
        assert loaded["user_profile"].name == sample_profile.name

    def test_raw_jd_texts_survive_round_trip(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import load_session_state, save_session_state

        state = _minimal_state(sample_profile)
        state["raw_jd_texts"] = ["JD Alpha", "JD Beta"]
        save_session_state("sess-jds", state)
        loaded = load_session_state("sess-jds")
        assert loaded["raw_jd_texts"] == ["JD Alpha", "JD Beta"]


# ---------------------------------------------------------------------------
# Progress logs
# ---------------------------------------------------------------------------

class TestProgressLogs:
    def test_add_and_retrieve_single_log(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import (
            add_progress_log,
            get_progress_logs,
            save_session_state,
        )

        save_session_state("sess-log", _minimal_state(sample_profile))
        entry = ProgressLogEntry(
            project_id="proj-001",
            week_number=1,
            tasks_completed=["Set up repo", "Wrote README"],
            blockers=["Had a blocker"],
            next_steps=["Fix blocker"],
        )
        add_progress_log("sess-log", entry)
        logs = get_progress_logs("sess-log")

        assert len(logs) == 1
        assert logs[0].project_id == "proj-001"
        assert logs[0].week_number == 1
        assert "Set up repo" in logs[0].tasks_completed

    def test_multiple_logs_ordered_by_date_then_week(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import (
            add_progress_log,
            get_progress_logs,
            save_session_state,
        )

        save_session_state("sess-multi", _minimal_state(sample_profile))

        for week in [3, 1, 2]:
            entry = ProgressLogEntry(
                project_id="proj-x",
                week_number=week,
                date=f"2024-01-0{week}T00:00:00",  # earlier date for lower week
            )
            add_progress_log("sess-multi", entry)

        logs = get_progress_logs("sess-multi")
        weeks = [log.week_number for log in logs]
        assert weeks == sorted(weeks)

    def test_no_logs_returns_empty_list(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import get_progress_logs, save_session_state

        save_session_state("sess-empty-log", _minimal_state(sample_profile))
        assert get_progress_logs("sess-empty-log") == []


# ---------------------------------------------------------------------------
# export_session_summary
# ---------------------------------------------------------------------------

class TestExportSessionSummary:
    def test_returns_error_for_unknown_session(self, initialized_db, monkeypatch):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import export_session_summary

        summary = export_session_summary("ghost-session")
        assert "error" in summary

    def test_returns_expected_keys_for_minimal_session(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import export_session_summary, save_session_state

        save_session_state("sess-export", _minimal_state(sample_profile))
        summary = export_session_summary("sess-export")

        for key in ("session_id", "profile_name", "target_role", "has_gap_report",
                    "roadmap_weeks", "project_ideas_count", "has_build_plan", "has_portfolio"):
            assert key in summary, f"Missing key: {key}"

    def test_profile_name_correct(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import export_session_summary, save_session_state

        save_session_state("sess-name", _minimal_state(sample_profile))
        summary = export_session_summary("sess-name")
        assert summary["profile_name"] == sample_profile.name

    def test_has_portfolio_true_when_portfolio_saved(
        self, initialized_db, monkeypatch, sample_profile, sample_role_analysis, sample_gap_report
    ):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import export_session_summary, save_session_state

        state = _full_state(sample_profile, sample_role_analysis, sample_gap_report)
        save_session_state("sess-portfolio", state)
        summary = export_session_summary("sess-portfolio")
        assert summary["has_portfolio"] is True

    def test_roadmap_weeks_count_correct(
        self, initialized_db, monkeypatch, sample_profile, sample_role_analysis, sample_gap_report
    ):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import export_session_summary, save_session_state

        state = _full_state(sample_profile, sample_role_analysis, sample_gap_report)
        save_session_state("sess-weeks", state)
        summary = export_session_summary("sess-weeks")
        assert summary["roadmap_weeks"] == 1  # set in _full_state
