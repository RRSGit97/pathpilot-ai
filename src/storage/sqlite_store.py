"""
src/storage/sqlite_store.py
---------------------------
SQLite-backed persistence for structured application memory.

Stores business objects (Pydantic models) mapped to a session ID.
This allows users to resume their career discovery sessions across app restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from src.config import settings
from src.data.schemas import (
    AggregatedRoleAnalysis,
    ApprovalState,
    LearningRoadmap,
    PortfolioOutputs,
    ProgressLogEntry,
    ProjectIdea,
    ProjectScore,
    SixWeekPlan,
    SkillGapReport,
    UserProfile,
)
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)

# Ensure the database directory exists
settings.database_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection():
    """Yield a configured SQLite connection and close it afterwards."""
    conn = sqlite3.connect(str(settings.database_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_sessions (
                session_id TEXT PRIMARY KEY,
                profile_json TEXT,
                raw_jds_json TEXT,
                role_analysis_json TEXT,
                skill_gap_report_json TEXT,
                gap_narrative TEXT,
                roadmap_json TEXT,
                project_ideas_json TEXT,
                ranked_projects_json TEXT,
                chosen_project_idx INTEGER,
                project_critique_json TEXT,
                build_plan_json TEXT,
                portfolio_outputs_json TEXT,
                approvals_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS progress_logs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                week_number INTEGER,
                date TEXT,
                tasks_completed_json TEXT,
                blockers_json TEXT,
                next_steps_json TEXT,
                FOREIGN KEY(session_id) REFERENCES app_sessions(session_id)
            )
            """
        )
        conn.commit()
    logger.info("SQLite application memory initialized.")


# ── CRUD Helpers ─────────────────────────────────────────────────────────────

def _dump_json(obj: Any) -> Optional[str]:
    """Helper to dump Pydantic models or standard Python types to JSON."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump_json"):
        # Single Pydantic model
        return obj.model_dump_json()
    if isinstance(obj, list) and obj and hasattr(obj[0], "model_dump"):
        # list[BaseModel] — serialize each element to a plain dict first
        return json.dumps([item.model_dump() for item in obj])
    return json.dumps(obj)


def save_session_state(session_id: str, state: PipelineState) -> None:
    """Upsert the current LangGraph PipelineState into SQLite application memory."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO app_sessions (
                session_id, profile_json, raw_jds_json, role_analysis_json,
                skill_gap_report_json, gap_narrative, roadmap_json, project_ideas_json,
                ranked_projects_json, chosen_project_idx, project_critique_json,
                build_plan_json, portfolio_outputs_json, approvals_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                raw_jds_json = excluded.raw_jds_json,
                role_analysis_json = excluded.role_analysis_json,
                skill_gap_report_json = excluded.skill_gap_report_json,
                gap_narrative = excluded.gap_narrative,
                roadmap_json = excluded.roadmap_json,
                project_ideas_json = excluded.project_ideas_json,
                ranked_projects_json = excluded.ranked_projects_json,
                chosen_project_idx = excluded.chosen_project_idx,
                project_critique_json = excluded.project_critique_json,
                build_plan_json = excluded.build_plan_json,
                portfolio_outputs_json = excluded.portfolio_outputs_json,
                approvals_json = excluded.approvals_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                session_id,
                _dump_json(state.get("user_profile")),
                _dump_json(state.get("raw_jd_texts")),
                _dump_json(state.get("role_analysis")),
                _dump_json(state.get("gap_report")),
                state.get("gap_narrative"),
                _dump_json(state.get("learning_roadmap")),
                _dump_json(state.get("project_ideas")),
                _dump_json(state.get("ranked_projects")),
                state.get("chosen_project_idx"),
                _dump_json(state.get("project_critique")),
                _dump_json(state.get("build_plan")),
                _dump_json(state.get("portfolio_outputs")),
                _dump_json(state.get("approvals")),
            ),
        )
        conn.commit()
    logger.info("Saved application state for session '%s'", session_id)


def load_session_state(session_id: str) -> Optional[PipelineState]:
    """Load the PipelineState from SQLite application memory for a given session."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_sessions WHERE session_id = ?", (session_id,)).fetchone()
        
    if not row:
        return None

    def _load_json(val: Optional[str]) -> Any:
        return json.loads(val) if val else None

    # Reconstruct the typed state
    state: PipelineState = {
        "raw_jd_texts": _load_json(row["raw_jds_json"]) or [],
        "gap_narrative": row["gap_narrative"],
        "chosen_project_idx": row["chosen_project_idx"],
        "project_critique": _load_json(row["project_critique_json"]),
        "approvals": _load_json(row["approvals_json"]),
    }

    # Pydantic reconstruction
    if row["profile_json"]:
        state["user_profile"] = UserProfile.model_validate_json(row["profile_json"])
    if row["role_analysis_json"]:
        state["role_analysis"] = AggregatedRoleAnalysis.model_validate_json(row["role_analysis_json"])
    if row["skill_gap_report_json"]:
        state["gap_report"] = SkillGapReport.model_validate_json(row["skill_gap_report_json"])
    if row["roadmap_json"]:
        state["learning_roadmap"] = LearningRoadmap.model_validate_json(row["roadmap_json"])
    if row["project_ideas_json"]:
        # project_ideas is a list of ProjectIdea
        raw_list = json.loads(row["project_ideas_json"])
        state["project_ideas"] = [ProjectIdea(**i) for i in raw_list]
    if row["ranked_projects_json"]:
        state["ranked_projects"] = json.loads(row["ranked_projects_json"])
    if row["build_plan_json"]:
        state["build_plan"] = SixWeekPlan.model_validate_json(row["build_plan_json"])
    if row["portfolio_outputs_json"]:
        state["portfolio_outputs"] = PortfolioOutputs.model_validate_json(row["portfolio_outputs_json"])

    return state


def add_progress_log(session_id: str, log_entry: ProgressLogEntry) -> None:
    """Save a single progress log entry."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO progress_logs (
                id, session_id, project_id, week_number, date,
                tasks_completed_json, blockers_json, next_steps_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_entry.id,
                session_id,
                log_entry.project_id,
                log_entry.week_number,
                log_entry.date,
                json.dumps(log_entry.tasks_completed),
                json.dumps(log_entry.blockers),
                json.dumps(log_entry.next_steps),
            ),
        )
        conn.commit()


def get_progress_logs(session_id: str) -> List[ProgressLogEntry]:
    """Retrieve all progress logs for a session, ordered by date."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM progress_logs WHERE session_id = ? ORDER BY date ASC, week_number ASC", 
            (session_id,)
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            ProgressLogEntry(
                id=row["id"],
                project_id=row["project_id"],
                week_number=row["week_number"],
                date=row["date"],
                tasks_completed=json.loads(row["tasks_completed_json"]),
                blockers=json.loads(row["blockers_json"]),
                next_steps=json.loads(row["next_steps_json"]),
            )
        )
    return results


def export_session_summary(session_id: str) -> Dict[str, Any]:
    """
    Export the latest saved session summary for display in the UI.
    This provides a clean dictionary summary without requiring the caller
    to re-instantiate or know about the Pydantic models.
    """
    state = load_session_state(session_id)
    if not state:
        return {"error": "Session not found."}

    summary = {
        "session_id": session_id,
        "profile_name": state.get("user_profile").name if state.get("user_profile") else "Unknown",
        "target_role": state.get("role_analysis").consensus_title if state.get("role_analysis") else "Unknown",
        "has_gap_report": bool(state.get("gap_report")),
        "roadmap_weeks": state.get("learning_roadmap").total_weeks if state.get("learning_roadmap") else 0,
        "project_ideas_count": len(state.get("project_ideas", [])),
    }

    # If they have a chosen project, summarize it
    if state.get("chosen_project_idx") is not None and state.get("ranked_projects"):
        try:
            chosen = state["ranked_projects"][state["chosen_project_idx"]]["idea"]
            summary["chosen_project_title"] = chosen.get("title", "Unknown")
        except (IndexError, KeyError):
            pass

    summary["has_build_plan"] = bool(state.get("build_plan"))
    summary["has_portfolio"] = bool(state.get("portfolio_outputs"))
    
    return summary
