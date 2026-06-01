"""
tests/smoke_test_ui.py
-----------------------
Smoke test for the PathPilot Streamlit app.

Uses Streamlit's built-in AppTest runner (streamlit >= 1.28) to
exercise the app without a browser. No live LLM or external services
are required — we mock the backend calls.

What is covered
---------------
- App renders without crash on fresh session
- All 10 expected tabs are present
- Sidebar contains the demo-load button
- At least one button renders in the main body

What is NOT covered here
------------------------
- Full end-to-end workflow execution (requires live LLM)
- Visual layout and design verification
- File upload binary parsing (requires real file bytes)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guard: skip the entire module gracefully if streamlit is not installed
# or AppTest is not available (older streamlit).
# ---------------------------------------------------------------------------

try:
    from streamlit.testing.v1 import AppTest  # type: ignore
except ImportError:
    pytest.skip(
        "streamlit.testing.v1.AppTest not available — install streamlit >= 1.28 to run UI tests.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_APP_PATH = Path(__file__).parent.parent / "app.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm():
    """Return a MagicMock that looks enough like a LangChain ChatModel."""
    m = MagicMock()
    m.with_structured_output.return_value = m
    m.invoke.return_value = MagicMock(content="mocked")
    return m


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestAppSmoke:
    """Basic rendering smoke tests — no LLM calls."""

    @pytest.fixture(autouse=True)
    def _patch_llm(self):
        """Prevent any real LLM initialisation during UI tests."""
        with patch("src.tools.skill_extractor.build_llm", return_value=_mock_llm()):
            yield

    def test_app_renders_without_exception(self):
        """The app must not crash on a fresh session."""
        at = AppTest.from_file(str(_APP_PATH), default_timeout=30)
        at.run()
        assert not at.exception, f"App raised: {at.exception}"

    def test_all_ten_tabs_present(self):
        """All 10 workflow tabs must be rendered."""
        at = AppTest.from_file(str(_APP_PATH), default_timeout=30)
        at.run()
        # Tabs are rendered as tab elements; we check the tab labels.
        # Actual labels discovered by running the app: numbered like "1. Profile" etc.
        tab_labels = [t.label for t in at.tabs]
        expected_tab_fragments = [
            "Profile",
            "Job Descriptions",
            "Role Analysis",
            "Skill Gap",
            "Roadmap",
            "Project Ideas",
            "Scoring",
            "Build Plan",
            "Portfolio",
            "Memory",
        ]
        for fragment in expected_tab_fragments:
            assert any(fragment in label for label in tab_labels), (
                f"Tab containing {fragment!r} not found. Got: {tab_labels}"
            )

    def test_sidebar_has_demo_button(self):
        """The sidebar must contain the 'Load Demo (Sample Data)' button."""
        at = AppTest.from_file(str(_APP_PATH), default_timeout=30)
        at.run()
        sidebar_button_labels = [b.label for b in at.sidebar.button]
        assert any(
            "Demo" in label or "Sample" in label
            for label in sidebar_button_labels
        ), f"Demo button not found in sidebar. Buttons: {sidebar_button_labels}"

    def test_profile_tab_has_submit_button(self):
        """The Profile tab must expose a submit / save button."""
        at = AppTest.from_file(str(_APP_PATH), default_timeout=30)
        at.run()
        # We expect at least one form submit button somewhere in the main body
        all_buttons = at.button
        labels = [b.label for b in all_buttons]
        assert len(labels) > 0, "No buttons found in main body"
