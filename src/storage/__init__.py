"""
src/storage/__init__.py
-----------------------
Storage module for PathPilot AI.

Provides SQLite-based persistence for structured application memory
and LangGraph checkpointing.
"""

from src.storage.sqlite_store import (
    init_db,
    save_session_state,
    load_session_state,
    add_progress_log,
    get_progress_logs,
    export_session_summary,
)
from src.storage.checkpoint_store import (
    get_checkpointer,
    setup_checkpoint_tables,
)

def initialize_all_storage() -> None:
    """Run all database initialization logic (create tables)."""
    init_db()
    setup_checkpoint_tables()

__all__ = [
    "initialize_all_storage",
    "init_db",
    "save_session_state",
    "load_session_state",
    "add_progress_log",
    "get_progress_logs",
    "export_session_summary",
    "get_checkpointer",
    "setup_checkpoint_tables",
]
