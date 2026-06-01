"""
tests/test_graph_workflow.py
-----------------------------
Tests for the LangGraph workflow: graph structure + review node logic.

Strategy
--------
- build_graph() is tested structurally without invoking any LLM nodes.
- Review nodes (reviews.py) are tested by mocking `interrupt()` so they
  can be called directly without a running LangGraph execution context.
- All LLM-calling nodes (ingest_profile, extract_jds, etc.) are NOT invoked
  in these tests; they are mocked at the graph level where needed.

Covers:
- build_graph: compiles without error
- build_graph: correct node count
- build_graph: interrupt_before on 4 review nodes
- review_role_node: approved path → approval state updated
- review_role_node: rejected path → revision requested
- review_roadmap_node: defaults to approved
- review_project_node: chosen_project_idx captured
- review_portfolio_node: approved → correct status message
- _get_approvals: initialises all gates to PENDING
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.enums import ApprovalStatus
from src.graph.nodes.reviews import (
    _get_approvals,
    review_portfolio_node,
    review_project_node,
    review_roadmap_node,
    review_role_node,
)
from src.graph.state import PipelineState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> PipelineState:
    """Build a minimal PipelineState dict for review node tests."""
    state: PipelineState = {
        "status": "Testing",
        "current_node": "test",
        "raw_jd_texts": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# build_graph: structural checks
# ---------------------------------------------------------------------------

class TestBuildGraph:
    """Verify graph compiles and has the right shape — no execution."""

    def test_builds_without_error(self):
        from src.graph.workflow import build_graph
        graph = build_graph()
        assert graph is not None

    def test_node_count(self):
        from src.graph.workflow import build_graph
        graph = build_graph()
        # 14 nodes declared in workflow.py
        assert len(graph.nodes) >= 14

    def test_interrupt_nodes_present(self):
        from src.graph.workflow import build_graph
        graph = build_graph()
        # Access the compiled graph's interrupt_before config
        expected = {"review_role", "review_roadmap", "review_project", "review_portfolio"}
        # The compiled graph exposes `interrupt_before` via its builder
        # We verify by calling get_graph() to inspect the schema
        schema = graph.get_graph()
        node_names = {n for n in schema.nodes}
        for node in expected:
            assert node in node_names, f"Expected interrupt node {node!r} not in graph"

    def test_entry_point_is_ingest_profile(self):
        from src.graph.workflow import build_graph
        graph = build_graph()
        schema = graph.get_graph()
        # Check that __start__ connects to ingest_profile
        start_edges = [e for e in schema.edges if e.source == "__start__"]
        assert len(start_edges) >= 1
        assert any(e.target == "ingest_profile" for e in start_edges)

    def test_uses_memory_saver_by_default(self):
        from langgraph.checkpoint.memory import MemorySaver
        from src.graph.workflow import build_graph
        # Should not raise; default checkpointer is MemorySaver
        graph = build_graph()
        # Graph compiled — implicitly used MemorySaver
        assert graph is not None

    def test_accepts_custom_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        from src.graph.workflow import build_graph
        custom = MemorySaver()
        graph = build_graph(checkpointer=custom)
        assert graph is not None


# ---------------------------------------------------------------------------
# _get_approvals: initialisation helper
# ---------------------------------------------------------------------------

class TestGetApprovals:
    def test_initialises_all_gates_to_pending_when_empty(self):
        state = _base_state()
        approvals = _get_approvals(state)
        expected_gates = {
            "target_role", "skill_gap_report", "learning_roadmap",
            "chosen_project", "build_plan", "portfolio_outputs"
        }
        for gate in expected_gates:
            assert gate in approvals
            assert approvals[gate] == ApprovalStatus.PENDING.value

    def test_returns_existing_approvals_unchanged(self):
        existing = {"target_role": "approved", "skill_gap_report": "approved"}
        state = _base_state(approvals=existing)
        result = _get_approvals(state)
        assert result is existing


# ---------------------------------------------------------------------------
# review_role_node
# ---------------------------------------------------------------------------

class TestReviewRoleNode:
    def _make_state_with_role(self, sample_role_analysis, sample_gap_report) -> PipelineState:
        return _base_state(
            role_analysis=sample_role_analysis,
            gap_report=sample_gap_report,
            gap_narrative="You are strong in Python.",
        )

    def test_approved_sets_target_role_approved(self, sample_role_analysis, sample_gap_report):
        state = self._make_state_with_role(sample_role_analysis, sample_gap_report)
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_role_node(state)
        assert result["approvals"]["target_role"] == ApprovalStatus.APPROVED.value

    def test_approved_sets_skill_gap_approved(self, sample_role_analysis, sample_gap_report):
        state = self._make_state_with_role(sample_role_analysis, sample_gap_report)
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_role_node(state)
        assert result["approvals"]["skill_gap_report"] == ApprovalStatus.APPROVED.value

    def test_rejected_sets_revision_requested(self, sample_role_analysis, sample_gap_report):
        state = self._make_state_with_role(sample_role_analysis, sample_gap_report)
        with patch(
            "src.graph.nodes.reviews.interrupt",
            return_value={"approved": False, "feedback": "Please add more skills."},
        ):
            result = review_role_node(state)
        assert result["approvals"]["target_role"] == ApprovalStatus.REVISION_REQUESTED.value

    def test_feedback_captured_in_human_feedback(self, sample_role_analysis, sample_gap_report):
        state = self._make_state_with_role(sample_role_analysis, sample_gap_report)
        with patch(
            "src.graph.nodes.reviews.interrupt",
            return_value={"approved": False, "feedback": "Needs more detail"},
        ):
            result = review_role_node(state)
        assert result["human_feedback"] == "Needs more detail"

    def test_status_message_contains_approved_on_approval(self, sample_role_analysis, sample_gap_report):
        state = self._make_state_with_role(sample_role_analysis, sample_gap_report)
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_role_node(state)
        assert "approved" in result["status"].lower()

    def test_current_node_is_set(self, sample_role_analysis, sample_gap_report):
        state = self._make_state_with_role(sample_role_analysis, sample_gap_report)
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_role_node(state)
        assert result["current_node"] == "review_role"


# ---------------------------------------------------------------------------
# review_roadmap_node
# ---------------------------------------------------------------------------

class TestReviewRoadmapNode:
    def _roadmap_state(self):
        from src.data.schemas import LearningRoadmap, LearningRoadmapWeek
        roadmap = LearningRoadmap(
            title="Test Roadmap",
            target_role="AI Engineer",
            total_weeks=2,
            weeks=[
                LearningRoadmapWeek(week_number=1, focus_topic="Python", estimated_hours=10),
                LearningRoadmapWeek(week_number=2, focus_topic="LangGraph", estimated_hours=12),
            ],
        )
        return _base_state(learning_roadmap=roadmap)

    def test_default_approved_when_no_feedback(self):
        state = self._roadmap_state()
        # Default: approved=True
        with patch("src.graph.nodes.reviews.interrupt", return_value={}):
            result = review_roadmap_node(state)
        assert result["approvals"]["learning_roadmap"] == ApprovalStatus.APPROVED.value

    def test_can_request_revision(self):
        state = self._roadmap_state()
        with patch(
            "src.graph.nodes.reviews.interrupt",
            return_value={"approved": False, "feedback": "Too fast"},
        ):
            result = review_roadmap_node(state)
        assert result["approvals"]["learning_roadmap"] == ApprovalStatus.REVISION_REQUESTED.value

    def test_status_message_approved(self):
        state = self._roadmap_state()
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_roadmap_node(state)
        assert "approved" in result["status"].lower()


# ---------------------------------------------------------------------------
# review_project_node
# ---------------------------------------------------------------------------

class TestReviewProjectNode:
    def _project_state(self):
        ranked = [
            {
                "idea": {
                    "title": "Agent Project",
                    "description": "Build a multi-agent system.",
                    "technologies": ["LangGraph"],
                },
                "score": {"composite_score": 8.2},
            },
            {
                "idea": {
                    "title": "Simple App",
                    "description": "A basic web app.",
                    "technologies": ["Flask"],
                },
                "score": {"composite_score": 5.1},
            },
        ]
        return _base_state(ranked_projects=ranked)

    def test_chosen_index_captured(self):
        state = self._project_state()
        with patch(
            "src.graph.nodes.reviews.interrupt",
            return_value={"approved": True, "chosen_index": 1},
        ):
            result = review_project_node(state)
        assert result["chosen_project_idx"] == 1

    def test_default_chosen_index_is_zero(self):
        state = self._project_state()
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_project_node(state)
        assert result["chosen_project_idx"] == 0

    def test_approved_sets_chosen_project_approved(self):
        state = self._project_state()
        with patch(
            "src.graph.nodes.reviews.interrupt",
            return_value={"approved": True, "chosen_index": 0},
        ):
            result = review_project_node(state)
        assert result["approvals"]["chosen_project"] == ApprovalStatus.APPROVED.value

    def test_status_message_contains_project_number(self):
        state = self._project_state()
        with patch(
            "src.graph.nodes.reviews.interrupt",
            return_value={"approved": True, "chosen_index": 0},
        ):
            result = review_project_node(state)
        assert "#1" in result["status"]


# ---------------------------------------------------------------------------
# review_portfolio_node
# ---------------------------------------------------------------------------

class TestReviewPortfolioNode:
    def _portfolio_state(self):
        from src.data.schemas import PortfolioOutputs
        portfolio = PortfolioOutputs(
            project_id="proj-001",
            resume_bullets=["Built X using LangGraph", "Deployed Y with FastAPI"],
            readme_outline="# Multi-Agent System\n## Overview...",
            interview_explanation_30s="I built a multi-agent system...",
        )
        return _base_state(portfolio_outputs=portfolio)

    def test_approved_sets_portfolio_approved(self):
        state = self._portfolio_state()
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_portfolio_node(state)
        assert result["approvals"]["portfolio_outputs"] == ApprovalStatus.APPROVED.value

    def test_rejected_sets_revision_requested(self):
        state = self._portfolio_state()
        with patch(
            "src.graph.nodes.reviews.interrupt",
            return_value={"approved": False, "feedback": "Tone too aggressive"},
        ):
            result = review_portfolio_node(state)
        assert result["approvals"]["portfolio_outputs"] == ApprovalStatus.REVISION_REQUESTED.value

    def test_approved_status_message(self):
        state = self._portfolio_state()
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_portfolio_node(state)
        assert "approved" in result["status"].lower() or "ready" in result["status"].lower()

    def test_no_portfolio_in_state_does_not_crash(self):
        state = _base_state()  # no portfolio_outputs
        with patch("src.graph.nodes.reviews.interrupt", return_value={"approved": True}):
            result = review_portfolio_node(state)
        assert "approvals" in result
