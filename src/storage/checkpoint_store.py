"""
src/storage/checkpoint_store.py
-------------------------------
SQLite-backed checkpointer for the LangGraph workflow.

Uses langgraph-checkpoint-sqlite to persist graph state at each step.
This is separate from the application memory (sqlite_store.py).
While sqlite_store.py stores the structured business objects,
the checkpointer stores the raw graph traversal state, interrupt checkpoints,
and node history to enable resuming and time-travel.

We store the checkpoints in the same SQLite file for simplicity, 
using a separate set of tables managed by SqliteSaver.
"""

from __future__ import annotations

import sqlite3
import logging
from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings

logger = logging.getLogger(__name__)

# Ensure the database directory exists
settings.database_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_checkpointer():
    """
    Context manager that yields a configured SqliteSaver.
    
    Usage:
        with get_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            graph.invoke(..., config=...)
    """
    # Connect using sqlite3 standard library
    # SqliteSaver accepts a sqlite3 connection directly
    conn = sqlite3.connect(str(settings.database_path), check_same_thread=False)
    
    try:
        # SqliteSaver will automatically create its tables (checkpoints, writes, etc.)
        # if they don't exist by default.
        saver = SqliteSaver(conn)
        yield saver
    finally:
        conn.close()


def setup_checkpoint_tables() -> None:
    """
    Explicitly run the LangGraph setup to create checkpoint tables.
    This can be called during initial application bootstrap.
    """
    with sqlite3.connect(str(settings.database_path)) as conn:
        SqliteSaver(conn).setup()
    logger.info("LangGraph SQLite checkpoint tables initialized.")
