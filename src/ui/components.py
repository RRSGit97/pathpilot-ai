"""
src/ui/components.py
--------------------
Reusable Streamlit UI components for PathPilot AI.

- render_empty_state: Friendly placeholder for tabs with no data yet.
- render_status_bar: Persistent status + error display bar.
- render_approval_form: Human-in-the-loop approval/rejection form.
- download_button: One-click markdown file export.
- render_step_progress: Visual pipeline progress indicator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


# ── Pipeline step definitions (order matters) ─────────────────────
PIPELINE_STEPS = [
    ("profile_submitted", "Profile"),
    ("jds_submitted", "Job Descriptions"),
    ("analysis_complete", "Analysis"),
    ("role_approved", "Role Approved"),
    ("roadmap_approved", "Roadmap Approved"),
    ("project_selected", "Project Selected"),
    ("plan_generated", "Build Plan"),
    ("portfolio_generated", "Portfolio"),
    ("portfolio_approved", "Portfolio Approved"),
]


def render_empty_state(message: str, next_action: str = "") -> None:
    """Render a friendly empty state when a tab has no data yet."""
    st.info(f"⏳ {message}")
    if next_action:
        st.caption(f"**Next step:** {next_action}")


def render_status_bar() -> None:
    """Show the most recent status message and any errors from session state."""
    status = st.session_state.get("last_status")
    error = st.session_state.get("last_error")

    # Only show errors that are associated with a pipeline run that the user triggered.
    # Mid-pipeline node warnings (e.g. "missing profile") are already surfaced inline
    # in each tab — showing them in the global banner too causes persistent confusion.
    _pipeline_ran = "analysis_complete" in st.session_state.get("completed_phases", set())

    if error and _pipeline_ran:
        st.error(f"⚠️ {error}")
    elif status and _pipeline_ran:
        # Success status — only show after a pipeline run
        if status.startswith("✅"):
            st.success(status)
        elif status.startswith("❌") or status.startswith("⚠️"):
            st.warning(status)


def render_step_progress() -> None:
    """Render a visual progress indicator in the sidebar."""
    completed = st.session_state.get("completed_phases", set())

    for key, label in PIPELINE_STEPS:
        if key in completed:
            st.markdown(f"✅ {label}")
        else:
            st.markdown(f"⬜ {label}")


def render_approval_form(
    review_type: str,
    prompt_text: str,
    on_submit_callback,
    key_prefix: str,
    extra_content_callback=None,
) -> None:
    """
    Render a human-in-the-loop approval form for a LangGraph interrupt.

    Args:
        review_type: Identifier for what is being reviewed.
        prompt_text: The instruction text for the user.
        on_submit_callback: Called with {"approved": bool, "feedback": str, ...}.
        key_prefix: Unique string for widget keys (avoids Streamlit key collisions).
        extra_content_callback: Optional — renders custom form fields and returns extra data dict.
    """
    st.warning("⚠️ **Human Approval Required**")
    st.markdown(prompt_text)

    with st.form(key=f"form_{key_prefix}_{review_type}"):
        # Optionally allow caller to inject extra fields (e.g. project selection radio)
        extra_data = {}
        if extra_content_callback:
            extra_data = extra_content_callback() or {}

        feedback = st.text_area(
            "Feedback or Requested Changes (Optional)",
            key=f"feedback_{key_prefix}_{review_type}",
            help="If you request changes, describe what should be different.",
        )

        col1, col2 = st.columns(2)
        with col1:
            approve = st.form_submit_button("✅ Approve", use_container_width=True)
        with col2:
            reject = st.form_submit_button("🔄 Request Changes", use_container_width=True)

        if approve:
            on_submit_callback({"approved": True, "feedback": feedback, **extra_data})
        elif reject:
            on_submit_callback({"approved": False, "feedback": feedback, **extra_data})


def download_button(
    label: str, data: str, filename: str, mime: str = "text/markdown"
) -> None:
    """Standard Streamlit download button wrapper."""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        use_container_width=True,
    )
