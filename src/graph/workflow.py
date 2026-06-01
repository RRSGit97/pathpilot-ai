"""
src/graph/workflow.py
---------------------
Compiles the PathPilot AI LangGraph workflow.

Graph topology
==============
::

    ┌──────────────┐
    │ ingest_profile│
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ parse_resume  │  (skips if no resume)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ extract_jds   │  (deterministic clean + LLM extraction)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ map_skill_gaps│  (deterministic mapping + LLM narrative)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ review_role   │  ◄── INTERRUPT: user approves role + gaps
    └──────┬───────┘
           ▼
    ┌────────────────┐
    │ generate_roadmap│
    └──────┬─────────┘
           ▼
    ┌────────────────┐
    │ review_roadmap  │  ◄── INTERRUPT: user reviews roadmap (optional)
    └──────┬─────────┘
           ▼
    ┌──────────────────┐
    │ generate_projects │
    └──────┬───────────┘
           ▼
    ┌──────────────────┐
    │ score_projects    │  (deterministic scoring)
    └──────┬───────────┘
           ▼
    ┌──────────────────┐
    │ review_project    │  ◄── INTERRUPT: user picks a project
    └──────┬───────────┘
           ▼
    ┌──────────────────┐
    │ critique_project  │
    └──────┬───────────┘
           ▼
    ┌──────────────────────┐
    │ generate_build_plan   │
    └──────┬───────────────┘
           ▼
    ┌──────────────────────┐
    │ generate_portfolio    │
    └──────┬───────────────┘
           ▼
    ┌──────────────────────┐
    │ review_portfolio      │  ◄── INTERRUPT: user approves artifacts
    └──────┬───────────────┘
           ▼
         [END]

Human-in-the-loop
==================
The graph uses ``interrupt_before`` on the four review nodes.
When a review node is reached, the graph pauses and saves its
checkpoint.  The Streamlit UI detects the pause, shows the
relevant data, and resumes with ``Command(resume={...})``.

Usage
=====
::

    from src.graph.workflow import build_graph

    graph = build_graph()

    # Start the pipeline
    initial_state = {
        "raw_profile_text": "I know Python, want to be an AI engineer...",
        "raw_jd_texts": [jd1, jd2, jd3],
        "status": "Starting...",
    }
    config = {"configurable": {"thread_id": "user-session-1"}}

    # First run — will pause at review_role
    result = graph.invoke(initial_state, config=config)

    # Resume after user approves role
    from langgraph.types import Command
    result = graph.invoke(
        Command(resume={"approved": True}),
        config=config,
    )
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    critique_project_node,
    extract_jds_node,
    generate_build_plan_node,
    generate_portfolio_node,
    generate_projects_node,
    generate_roadmap_node,
    ingest_profile_node,
    map_skill_gaps_node,
    parse_resume_node,
    review_portfolio_node,
    review_project_node,
    review_roadmap_node,
    review_role_node,
    score_projects_node,
)
from src.graph.state import PipelineState

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None):
    """
    Build and compile the PathPilot AI career discovery graph.

    Parameters
    ----------
    checkpointer : optional
        A LangGraph checkpointer for state persistence.
        Defaults to ``MemorySaver()`` (in-memory, lost on restart).
        For production, pass a SQLite or Redis checkpointer.

    Returns
    -------
    CompiledStateGraph
        Ready to call ``.invoke()`` or ``.stream()``.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("ingest_profile", ingest_profile_node)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("extract_jds", extract_jds_node)
    graph.add_node("map_skill_gaps", map_skill_gaps_node)
    graph.add_node("review_role", review_role_node)
    graph.add_node("generate_roadmap", generate_roadmap_node)
    graph.add_node("review_roadmap", review_roadmap_node)
    graph.add_node("generate_projects", generate_projects_node)
    graph.add_node("score_projects", score_projects_node)
    graph.add_node("review_project", review_project_node)
    graph.add_node("critique_project", critique_project_node)
    graph.add_node("generate_build_plan", generate_build_plan_node)
    graph.add_node("generate_portfolio", generate_portfolio_node)
    graph.add_node("review_portfolio", review_portfolio_node)

    # ── Linear edges (the main happy path) ────────────────────────
    graph.set_entry_point("ingest_profile")

    graph.add_edge("ingest_profile", "parse_resume")
    graph.add_edge("parse_resume", "extract_jds")
    graph.add_edge("extract_jds", "map_skill_gaps")
    graph.add_edge("map_skill_gaps", "review_role")
    graph.add_edge("review_role", "generate_roadmap")
    graph.add_edge("generate_roadmap", "review_roadmap")
    graph.add_edge("review_roadmap", "generate_projects")
    graph.add_edge("generate_projects", "score_projects")
    graph.add_edge("score_projects", "review_project")
    graph.add_edge("review_project", "critique_project")
    graph.add_edge("critique_project", "generate_build_plan")
    graph.add_edge("generate_build_plan", "generate_portfolio")
    graph.add_edge("generate_portfolio", "review_portfolio")
    graph.add_edge("review_portfolio", END)

    # ── Compile with interrupt points ─────────────────────────────
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "review_role",
            "review_roadmap",
            "review_project",
            "review_portfolio",
        ],
    )

    logger.info("PathPilot graph compiled: 14 nodes, 4 interrupt points.")
    return compiled
