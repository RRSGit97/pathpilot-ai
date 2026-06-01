"""
tests/test_db_repositories.py
------------------------------
Tests for SQLite storage repository layer (src/storage/sqlite_store.py).

NOTE: The old stub at this path imported from non-existent
`src.db.repositories.profile_repo.ProfileRepository`. That class
does not exist — the real storage module is `src.storage.sqlite_store`.

This file provides focused CRUD smoke tests that complement the
comprehensive round-trip tests in tests/test_sqlite_store.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.schemas import UserProfile


def _minimal_state(profile: UserProfile) -> dict:
    return {
        "raw_profile_text": "Test profile text.",
        "raw_jd_texts": [],
        "user_profile": profile,
        "status": "test",
        "current_node": "test",
    }


class TestBasicStorageOperations:
    """Thin smoke tests that cover the main CRUD path."""

    def test_save_and_load_profile(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import load_session_state, save_session_state

        state = _minimal_state(sample_profile)
        save_session_state("repo-test-001", state)
        loaded = load_session_state("repo-test-001")

        assert loaded is not None
        assert loaded["user_profile"].name == sample_profile.name

    def test_load_nonexistent_session_returns_none(self, initialized_db, monkeypatch):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import load_session_state

        result = load_session_state("nonexistent-session-xyz")
        assert result is None

    def test_export_summary_for_missing_session_returns_error(self, initialized_db, monkeypatch):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import export_session_summary

        summary = export_session_summary("ghost")
        assert "error" in summary

    def test_export_summary_contains_profile_name(self, initialized_db, monkeypatch, sample_profile):
        monkeypatch.setattr("src.storage.sqlite_store.settings.database_path", initialized_db)
        from src.storage.sqlite_store import export_session_summary, save_session_state

        save_session_state("repo-summary", _minimal_state(sample_profile))
        summary = export_session_summary("repo-summary")
        assert summary["profile_name"] == sample_profile.name
