"""
src/ui/state_helpers.py
-----------------------
Helper functions for managing Streamlit session state and bridging
the LangGraph runtime with the SQLite application memory.

Key responsibilities:
1. Initialize session state variables on first load.
2. Read the current pipeline state from the LangGraph checkpointer.
3. Persist pipeline state snapshots to SQLite after each major step.
4. Detect pending LangGraph interrupt checkpoints.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

import streamlit as st

from src.storage.sqlite_store import load_session_state, save_session_state

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize necessary session state variables on first Streamlit load."""

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info("New session: %s", st.session_state.session_id)

    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False

    # ── User input storage ────────────────────────────────────────
    if "profile_input" not in st.session_state:
        st.session_state.profile_input = ""
    if "jd_inputs" not in st.session_state:
        st.session_state.jd_inputs = ""
    if "resume_bytes" not in st.session_state:
        st.session_state.resume_bytes = None
    if "resume_filename" not in st.session_state:
        st.session_state.resume_filename = None

    # ── Pipeline phase tracking ───────────────────────────────────
    # Tracks which phases have been executed so the UI can show correct states.
    if "completed_phases" not in st.session_state:
        st.session_state.completed_phases = set()

    # ── Last error ────────────────────────────────────────────────
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "last_status" not in st.session_state:
        st.session_state.last_status = None


def get_pipeline_state(graph, config) -> Dict[str, Any]:
    """
    Read the current pipeline state from the LangGraph checkpointer.

    This is the *authoritative* source of truth for data produced by nodes.
    The LangGraph checkpointer stores the full accumulated state dict after
    each node execution, which we read directly.

    Falls back to the SQLite application store if no graph state exists yet
    (e.g. the app was restarted and checkpointer was MemorySaver).
    """
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return dict(snapshot.values)
    except Exception as exc:
        logger.debug("Could not read graph state: %s", exc)

    # Fallback to persisted SQLite application memory
    db_state = load_session_state(st.session_state.session_id)
    return db_state or {}


def persist_state_snapshot(state: Dict[str, Any]) -> None:
    """
    Save the current pipeline state to SQLite application memory.

    This is called after each significant graph invocation so that:
    - The sidebar session summary stays up-to-date.
    - State survives app restarts (even if using MemorySaver).
    - The UI can display data while the graph is paused at an interrupt.
    """
    try:
        save_session_state(st.session_state.session_id, state)
    except Exception as exc:
        logger.error("Failed to persist state snapshot: %s", exc)


def get_pending_interrupt(graph, config) -> Optional[str]:
    """
    Check if the LangGraph workflow is currently paused at an interrupt.

    Returns the name of the node the graph is waiting *before*, or None.
    """
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.next:
            return snapshot.next[0]
    except Exception as exc:
        logger.debug("Could not check interrupt state: %s", exc)
    return None


def run_pipeline_phase(
    graph,
    config: dict,
    input_data: Dict[str, Any] | None = None,
    *,
    resume_data: Dict[str, Any] | None = None,
    phase_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Execute a pipeline phase (either start or resume) with full error handling,
    status tracking, and automatic state persistence.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph workflow.
    config : dict
        LangGraph config with thread_id.
    input_data : dict, optional
        Initial state dict for starting the pipeline.
    resume_data : dict, optional
        Resume payload for continuing from an interrupt.
    phase_name : str
        Human-readable name for logging / status messages.

    Returns
    -------
    dict
        The full pipeline state after execution.
    """
    from langgraph.types import Command

    st.session_state.pipeline_running = True
    st.session_state.last_error = None
    st.session_state.last_status = None

    try:
        if resume_data is not None:
            logger.info("Resuming pipeline at phase '%s'", phase_name)
            result = graph.invoke(Command(resume=resume_data), config=config)
        elif input_data is not None:
            logger.info("Starting pipeline phase '%s'", phase_name)
            result = graph.invoke(input_data, config=config)
        else:
            raise ValueError("Must provide either input_data or resume_data")

        # Read the full accumulated state after execution
        state = get_pipeline_state(graph, config)

        # Persist to SQLite
        persist_state_snapshot(state)

        # Track completion
        st.session_state.completed_phases.add(phase_name)
        st.session_state.last_status = state.get("status", f"✅ {phase_name} complete.")
        st.session_state.last_error = state.get("error")

        logger.info(
            "Phase '%s' complete. Status: %s",
            phase_name,
            st.session_state.last_status,
        )
        return state

    except Exception as exc:
        logger.exception("Pipeline phase '%s' failed", phase_name)
        st.session_state.last_error = str(exc)
        st.session_state.last_status = f"❌ {phase_name} failed: {exc}"
        return get_pipeline_state(graph, config)

    finally:
        st.session_state.pipeline_running = False


def load_sample_data() -> None:
    """
    Load sample data files into session state and optionally Qdrant.
    Resets the current session to ensure a clean sandbox environment.
    """
    import json
    import uuid
    from pathlib import Path

    # 1. Reset current session state to clear any stale checkpointer memory
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Re-initialize clean session variables
    init_session_state()

    sample_dir = Path(__file__).parent.parent.parent / "sample_data"
    profile_path = sample_dir / "sample_profile.json"
    jds_path = sample_dir / "sample_job_descriptions.json"
    resources_path = sample_dir / "sample_resources.json"

    # 2. Load Profile
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
                st.session_state.profile_input = profile_data.get("raw_profile_text", "")
                st.session_state.completed_phases.add("profile_submitted")
                logger.info("Sample profile loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load sample profile: %s", exc)
            st.session_state.last_error = f"Failed to load sample profile: {exc}"
    else:
        logger.warning("Sample profile file not found at: %s", profile_path)

    # 3. Load JDs
    if jds_path.exists():
        try:
            with open(jds_path, "r", encoding="utf-8") as f:
                jds_data = json.load(f)
                jd_texts = []
                for jd in jds_data:
                    if isinstance(jd, dict):
                        text = jd.get("description") or jd.get("raw_text") or ""
                        if not text and "title" in jd:
                            text = f"Title: {jd['title']}\nCompany: {jd.get('company', '')}\nDescription: {jd.get('description', '')}"
                        jd_texts.append(text)
                    elif isinstance(jd, str):
                        jd_texts.append(jd)
                st.session_state.jd_inputs = "\n\n---\n\n".join(jd_texts)
                st.session_state.completed_phases.add("jds_submitted")
                logger.info("Sample job descriptions loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load sample JDs: %s", exc)
            st.session_state.last_error = f"Failed to load sample JDs: {exc}"
    else:
        logger.warning("Sample job descriptions file not found at: %s", jds_path)

    # 4. Optional Vector Store Ingestion
    if resources_path.exists():
        try:
            from src.tools.vector_store import vector_store
            if vector_store.enabled:
                with open(resources_path, "r", encoding="utf-8") as f:
                    resources_data = json.load(f)

                texts = []
                metadatas = []
                for idx, res in enumerate(resources_data):
                    text_repr = (
                        f"Title: {res.get('title', '')}\n"
                        f"URL: {res.get('url', '')}\n"
                        f"Type: {res.get('type', '')}\n"
                        f"Skill Covered: {res.get('skill_covered', '')}\n"
                        f"Description: {res.get('description', '')}"
                    )
                    texts.append(text_repr)
                    metadatas.append({
                        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"res-sample-{idx}")),
                        "title": res.get("title", ""),
                        "url": res.get("url", ""),
                        "type": res.get("type", ""),
                        "skill_covered": res.get("skill_covered", ""),
                    })

                # Ingest to Qdrant collection
                vector_store.init_collections()
                vector_store.upsert(
                    collection_name="resource_catalog_chunks",
                    texts=texts,
                    metadatas=metadatas
                )
                logger.info("Ingested %d sample resources into vector store.", len(texts))
        except Exception as exc:
            logger.warning("Optional vector store ingestion skipped for sample resources: %s", exc)
    else:
        logger.warning("Sample resources file not found at: %s", resources_path)

