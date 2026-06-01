"""
app.py
------
Main Streamlit entrypoint for PathPilot AI.

Architecture
============
1. ``setup_db()`` runs once (cached) to create SQLite tables.
2. ``init_session_state()`` sets up session-level variables.
3. ``_sync_api_key()`` bridges ``st.session_state`` → ``settings`` so
   the runtime key override is re-applied on every Streamlit rerun.
4. If no valid API key exists, ``render_api_key_setup()`` shows a
   password input screen **instead of** the main tabs.  The key is
   stored *only* in ``st.session_state`` (never on disk).
5. A ``get_checkpointer()`` context wraps each Streamlit run so
   the LangGraph ``SqliteSaver`` has a live connection.
6. ``get_pipeline_state()`` reads the accumulated state from the
   LangGraph checkpointer (authoritative) with SQLite fallback.
7. ``get_pending_interrupt()`` detects if the graph is paused.
8. Every tab receives ``(state, interrupt_node, graph, config)``
   so data flow is explicit and traceable in the IDE.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

from src.config import settings
from src.graph.workflow import build_graph
from src.storage import export_session_summary, get_checkpointer, initialize_all_storage
from src.ui.components import render_status_bar, render_step_progress
from src.ui.pages import (
    render_build_plan_tab,
    render_jds_tab,
    render_portfolio_tab,
    render_profile_tab,
    render_progress_tab,
    render_project_ideas_tab,
    render_roadmap_tab,
    render_role_analysis_tab,
    render_scoring_tab,
    render_skill_gap_tab,
)
from src.ui.state_helpers import (
    get_pending_interrupt,
    get_pipeline_state,
    init_session_state,
    load_sample_data,
)

# ── Bootstrap ─────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@st.cache_resource
def setup_db():
    """Run once per process to create / migrate SQLite tables."""
    initialize_all_storage()
    logger.info("Database initialized.")


# ── API Key Management ────────────────────────────────────────────

def _sync_api_key() -> None:
    """
    Synchronize the API key from ``st.session_state`` into the
    ``settings`` singleton on every Streamlit rerun.

    Streamlit reruns the full script on every interaction, but
    ``settings`` is a module-level singleton that persists across
    reruns.  This function ensures the runtime key override is
    always current.

    SECURITY: The key lives only in ``st.session_state`` (in-memory).
    It is NEVER written to .env, SQLite, logs, or any persistent file.
    """
    session_key = st.session_state.get("openai_api_key")
    if session_key:
        settings.set_runtime_api_key(session_key)


def _mask_key(key: str) -> str:
    """Return a safely masked version of the key for display."""
    if len(key) <= 8:
        return "sk-****"
    return f"{key[:7]}...{key[-4:]}"


def render_api_key_setup() -> None:
    """
    Render the API key setup screen.

    Shown instead of the main app when no valid API key is available.
    Accepts the key via a password input, validates the format, and
    stores it in session state only.
    """
    st.title("🧭 PathPilot AI")
    st.caption("Your Agentic Career Discovery & Project-Planning Assistant")

    st.divider()

    st.header("🔑 OpenAI API Key Setup")
    st.markdown(
        "PathPilot AI uses OpenAI's models to analyze your profile, extract "
        "skills from job descriptions, and generate personalized career plans.  \n\n"
        "Enter your API key below to get started. Your key is stored **only in "
        "this browser session** — it is never saved to disk, logged, or shared."
    )

    # ── Key input form ────────────────────────────────────────────
    with st.form("api_key_form"):
        key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            help=(
                "Get your key from https://platform.openai.com/api-keys.  "
                "It will start with 'sk-' or 'sk-proj-'."
            ),
        )
        submitted = st.form_submit_button(
            "🔐 Save API Key for This Session",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        key = key_input.strip()

        # Validation
        if not key:
            st.error("Please enter your API key.")
            return

        if not key.startswith("sk-"):
            st.error(
                "That doesn't look like a valid OpenAI API key.  "
                "It should start with `sk-` or `sk-proj-`."
            )
            return

        # ── Valid key — save to session state only ────────────────
        # SECURITY: This key is stored ONLY in st.session_state
        # (Streamlit's in-memory session store).  It is NOT written
        # to .env, SQLite, logs, README, sample files, or any file.
        st.session_state["openai_api_key"] = key
        settings.set_runtime_api_key(key)
        st.success("✅ API key saved for this session!")
        st.rerun()

    # ── Alternative: .env hint ────────────────────────────────────
    with st.expander("💡 Alternative: Use a `.env` file"):
        st.markdown(
            "If you prefer, create a `.env` file in the project root:\n\n"
            "```env\n"
            "LLM_PROVIDER=openai\n"
            "LLM_MODEL=gpt-4o-mini\n"
            "OPENAI_API_KEY=sk-proj-your-key-here\n"
            "```\n\n"
            "Then restart the Streamlit server.  The app will detect the "
            "key automatically."
        )


# ── Main ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PathPilot AI",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    setup_db()
    init_session_state()

    # Re-apply the session-stored API key on every rerun
    _sync_api_key()

    # ── API Key Gate ──────────────────────────────────────────────
    # If no valid key is available (neither from session nor .env),
    # show the setup screen and stop.  Main tabs are NOT rendered.
    if not settings.has_valid_api_key():
        render_api_key_setup()
        return

    # ── Graph + checkpointer ──────────────────────────────────────
    # get_checkpointer() yields a SqliteSaver with a live connection.
    # We compile the graph inside this context so every invoke/read
    # can reach the checkpoint tables.
    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": st.session_state.session_id}}

        # Read current state & interrupt status
        state = get_pipeline_state(graph, config)
        interrupt_node = get_pending_interrupt(graph, config)

        # ── Header ────────────────────────────────────────────────
        st.title("🧭 PathPilot AI")
        st.caption("Your Agentic Career Discovery & Project-Planning Assistant")

        # Status bar (shows last success / error from pipeline)
        render_status_bar()

        # ── Tabs ──────────────────────────────────────────────────
        # All 10 tabs receive the same (state, interrupt_node, graph, config)
        # so every render function can trigger pipeline steps or read state
        # without hidden coupling.
        tabs = st.tabs([
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
        ])

        with tabs[0]:
            render_profile_tab(state, interrupt_node, graph, config)
        with tabs[1]:
            render_jds_tab(state, interrupt_node, graph, config)
        with tabs[2]:
            render_role_analysis_tab(state, interrupt_node, graph, config)
        with tabs[3]:
            render_skill_gap_tab(state, interrupt_node, graph, config)
        with tabs[4]:
            render_roadmap_tab(state, interrupt_node, graph, config)
        with tabs[5]:
            render_project_ideas_tab(state, interrupt_node, graph, config)
        with tabs[6]:
            render_scoring_tab(state, interrupt_node, graph, config)
        with tabs[7]:
            render_build_plan_tab(state, interrupt_node, graph, config)
        with tabs[8]:
            render_portfolio_tab(state, interrupt_node, graph, config)
        with tabs[9]:
            render_progress_tab(state, interrupt_node, graph, config)

        # ── Sidebar ───────────────────────────────────────────────
        with st.sidebar:
            st.header("🧭 PathPilot AI")

            # API key status indicator
            st.subheader("🔑 API Key")
            session_key = st.session_state.get("openai_api_key")
            if session_key:
                st.success(f"Key loaded: `{_mask_key(session_key)}`")
            elif settings.active_api_key():
                st.info("Using key from `.env` file.")
            if st.button("🔓 Clear API Key", use_container_width=True):
                st.session_state.pop("openai_api_key", None)
                settings.clear_runtime_api_key()
                st.rerun()

            st.divider()

            # Pipeline status
            st.subheader("Pipeline Status")
            if st.session_state.pipeline_running:
                st.warning("⏳ Pipeline is running…")
            elif interrupt_node:
                st.info(f"⏸️ Waiting for approval at: **{interrupt_node}**")
            elif state.get("approvals", {}).get("portfolio_outputs") == "approved":
                st.success("🎉 Pipeline complete!")
            else:
                st.caption("Pipeline idle — ready for input.")

            st.divider()

            # Step progress checklist
            st.subheader("Progress")
            render_step_progress()

            st.divider()

            # Session summary from SQLite
            st.subheader("Session Info")
            summary = export_session_summary(st.session_state.session_id)
            if "error" not in summary:
                st.write(f"**Profile:** {summary.get('profile_name', '—')}")
                st.write(f"**Target Role:** {summary.get('target_role', '—')}")
                st.write(f"**Ideas:** {summary.get('project_ideas_count', 0)}")
                if summary.get("chosen_project_title"):
                    st.write(f"**Chosen:** {summary['chosen_project_title']}")
                st.write(f"**Build Plan:** {'✅' if summary.get('has_build_plan') else '—'}")
                st.write(f"**Portfolio:** {'✅' if summary.get('has_portfolio') else '—'}")

            st.divider()

            # Load sample data button
            if st.button("💾 Load Demo (Sample Data)", use_container_width=True):
                load_sample_data()
                st.success("Sample data loaded! Go to Job Descriptions (Tab 2) and click Run Full Analysis.")
                st.rerun()

            # Reset button
            if st.button("🔄 Start New Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main()
