"""
src/ui/pages.py
---------------
Render functions for each tab in the PathPilot AI Streamlit UI.

Each function receives:
- state: the current PipelineState dict (read from LangGraph checkpointer / SQLite)
- interrupt_node: name of the node the graph is paused before, or None
- graph: the compiled LangGraph CompiledStateGraph
- config: the LangGraph config dict with thread_id

Data flow:
    Tab button click  →  run_pipeline_phase()  →  graph.invoke()  →  nodes execute
                                                →  persist_state_snapshot()
                                                →  st.rerun()  →  UI re-reads state
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.config import settings
from src.ui.components import download_button, render_approval_form, render_empty_state
from src.ui.state_helpers import run_pipeline_phase

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 1: PROFILE INTAKE                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_profile_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("👤 Your Profile")
    st.markdown("Tell PathPilot about your background, skills, and goals.")

    # ── Show parsed profile if it exists ──────────────────────────
    profile = state.get("user_profile")
    if profile:
        # Detect LLM fallback: name is still "User" and skills are empty
        llm_fallback = (profile.name == "User" and not profile.current_skills and not profile.target_roles)
        if llm_fallback:
            st.warning(
                "⚠️ **Profile saved but LLM parsing did not run yet.**  "
                "This usually means the API key wasn't loaded yet. "
                "Re-submit your profile below after confirming your `.env` key is set and the server has restarted."
            )
        else:
            st.success(f"Profile loaded for **{profile.name}**.")
        with st.expander("📋 View Parsed Profile", expanded=llm_fallback):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {profile.name}")
                st.write(f"**Weekly Hours:** {profile.weekly_hours_available}")
                st.write(f"**Target Roles:** {', '.join(profile.target_roles) if profile.target_roles else 'Not specified'}")
            with col2:
                st.write(f"**Current Skills:** {', '.join(profile.current_skills[:15]) if profile.current_skills else 'None listed'}")
                if profile.prior_projects:
                    st.write(f"**Prior Projects:** {len(profile.prior_projects)}")
    elif "profile_submitted" in st.session_state.get("completed_phases", set()):
        st.info("✅ Profile saved. Head to **Tab 2 — Job Descriptions** to continue.")

    # ── Resume upload ─────────────────────────────────────────────
    if settings.resume_upload_enabled:
        uploaded_file = st.file_uploader(
            "Upload your Resume (Optional)",
            type=["pdf", "txt", "docx"],
            help="Parsed automatically to extract skills and experience.",
        )
        if uploaded_file is not None:
            st.session_state.resume_bytes = uploaded_file.getvalue()
            st.session_state.resume_filename = uploaded_file.name
            st.caption(f"📄 File ready: {uploaded_file.name} ({len(uploaded_file.getvalue()):,} bytes)")

    # ── Profile text form ─────────────────────────────────────────
    with st.form("profile_intake_form"):
        profile_text = st.text_area(
            "Describe your current skills, background, and career goals",
            value=st.session_state.profile_input,
            height=200,
            placeholder=(
                "Example: I'm a Python developer with 2 years of experience in "
                "data analysis. I know pandas, SQL, and basic ML. I want to transition "
                "into an AI/ML Engineer role. I've built a sentiment analysis project "
                "and a recommendation engine prototype..."
            ),
        )

        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        if submitted:
            if not profile_text.strip() and not st.session_state.resume_bytes:
                st.error("Please provide a profile description or upload a resume.")
                return

            st.session_state.profile_input = profile_text
            st.session_state.completed_phases.add("profile_submitted")
            st.success("✅ Profile saved! Now go to **Tab 2 — Job Descriptions** to continue.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 2: JOB DESCRIPTIONS                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_jds_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("📄 Target Roles / Job Descriptions")
    st.markdown("Paste **3–5 job descriptions** separated by `---` (three dashes on a new line).")
    st.info(
        "💡 **These should be job postings from companies you want to work at** — "
        "copy from LinkedIn, Glassdoor, or company career pages. "
        "Do **not** paste your own resume here (that goes in Tab 1)."
    )

    # ── JD input area ─────────────────────────────────────────────
    with st.form("jd_intake_form"):
        jd_text = st.text_area(
            "Job Descriptions",
            value=st.session_state.jd_inputs,
            height=350,
            placeholder=(
                "Paste job description 1 here...\n\n---\n\n"
                "Paste job description 2 here...\n\n---\n\n"
                "Paste job description 3 here..."
            ),
        )
        submitted = st.form_submit_button("💾 Save Job Descriptions", use_container_width=True)
        if submitted:
            st.session_state.jd_inputs = jd_text
            jds = [jd.strip() for jd in jd_text.split("---") if jd.strip()]
            if len(jds) < 1:
                st.error("Please paste at least one job description.")
                return
            st.session_state.completed_phases.add("jds_submitted")
            st.success(f"✅ Saved **{len(jds)}** job description(s). Ready to run analysis below.")

    # ── Run Full Analysis Button ──────────────────────────────────
    st.divider()
    profile_ready = "profile_submitted" in st.session_state.get("completed_phases", set())
    jds_ready = "jds_submitted" in st.session_state.get("completed_phases", set())

    if not profile_ready:
        st.info("Complete **Tab 1 (Profile)** first before running analysis.")
    elif not jds_ready:
        st.info("Save your job descriptions above, then click **Run Full Analysis**.")
    else:
        jds = [jd.strip() for jd in st.session_state.jd_inputs.split("---") if jd.strip()]
        st.info(f"Ready: **{len(jds)}** JDs + profile loaded. This will run through role extraction, skill gap mapping, and pause for your approval.")

        if st.button("🚀 Run Full Analysis", use_container_width=True, type="primary"):
            with st.spinner("Running profile parsing → resume parsing → JD extraction → skill gap mapping... This may take 30–60 seconds."):
                initial_state = {
                    "raw_profile_text": st.session_state.profile_input,
                    "resume_bytes": st.session_state.resume_bytes,
                    "resume_filename": st.session_state.resume_filename,
                    "raw_jd_texts": jds,
                    "status": "Starting pipeline...",
                }
                result_state = run_pipeline_phase(
                    graph, config,
                    input_data=initial_state,
                    phase_name="analysis_complete",
                )

            # Show any pipeline errors inline so nothing fails silently
            pipeline_error = st.session_state.get("last_error")
            if pipeline_error:
                st.error(f"❌ Pipeline error: {pipeline_error}")
            else:
                st.success("✅ Analysis complete! Check Tab 3 (Role Analysis) and Tab 4 (Skill Gap).")
            st.rerun()

    # ── Show existing JDs if loaded ───────────────────────────────
    if state.get("raw_jd_texts"):
        with st.expander(f"📝 {len(state['raw_jd_texts'])} JDs currently in pipeline"):
            for i, jd in enumerate(state["raw_jd_texts"]):
                st.text(f"--- JD {i + 1} ---")
                st.text(jd[:300] + ("..." if len(jd) > 300 else ""))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 3: ROLE ANALYSIS                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_role_analysis_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("🎯 Role Analysis")

    role = state.get("role_analysis")
    if not role:
        render_empty_state(
            "No role analysis yet.",
            "Run the full analysis from Tab 2 (Job Descriptions).",
        )
        return

    st.subheader(f"Target Role: {role.consensus_title}")
    st.write(f"**Seniority Level:** {role.seniority}")
    st.write(f"**Analyzed from:** {role.source_jd_count} job descriptions")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Required Skills:**")
        for skill in role.required_skills:
            st.markdown(f"- {skill}")
    with col2:
        st.markdown("**Optional / Bonus Skills:**")
        for skill in role.optional_skills:
            st.markdown(f"- {skill}")

    if role.tools_and_frameworks:
        with st.expander("🔧 Tools & Frameworks"):
            for tool in role.tools_and_frameworks:
                st.markdown(f"- {tool}")

    if role.project_expectations:
        with st.expander("📋 Common Project Expectations"):
            for exp in role.project_expectations:
                st.markdown(f"- {exp}")

    # ── Approval interrupt ────────────────────────────────────────
    if interrupt_node == "review_role":
        st.divider()
        render_approval_form(
            "role_analysis",
            "Does this target role and skill gap analysis align with your goals?",
            lambda resp: _handle_approval(graph, config, resp, "role_approved"),
            "role",
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 4: SKILL GAP                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_skill_gap_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("📊 Skill Gap Report")

    gap = state.get("gap_report")
    if not gap:
        render_empty_state(
            "No skill gap report yet.",
            "Run the full analysis from Tab 2.",
        )
        return

    st.metric("Relevance Score", f"{gap.relevance_score:.0%}")

    if state.get("gap_narrative"):
        with st.expander("📝 Gap Narrative", expanded=True):
            st.markdown(state["gap_narrative"])

    tab_missing, tab_partial, tab_strong = st.tabs(["❌ Missing", "⚡ Partial", "✅ Strong"])

    with tab_missing:
        if gap.missing_required:
            df = pd.DataFrame([s.model_dump() for s in gap.missing_required])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("No required skills are missing!")

    with tab_partial:
        if gap.partial:
            df = pd.DataFrame([s.model_dump() for s in gap.partial])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No partial-match skills identified.")

    with tab_strong:
        if gap.strong:
            df = pd.DataFrame([s.model_dump() for s in gap.strong])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No strong-match skills identified yet.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 5: LEARNING ROADMAP                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_roadmap_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("🗺️ Learning Roadmap")

    roadmap = state.get("learning_roadmap")
    if not roadmap:
        render_empty_state(
            "No learning roadmap yet.",
            "Approve the role analysis in Tab 3 to generate a roadmap.",
        )
        return

    st.subheader(roadmap.title)
    st.write(f"**Total Duration:** {roadmap.total_weeks} weeks")

    for week in roadmap.weeks:
        with st.expander(f"📅 Week {week.week_number}: {week.focus_topic}", expanded=False):
            st.write(f"**Estimated Hours:** {week.estimated_hours}")
            st.write(f"**Skills Covered:** {', '.join(week.skills_covered)}")

            if week.resources:
                st.markdown("**Resources:**")
                for res in week.resources:
                    st.markdown(f"- {res}")

            if week.tasks:
                st.markdown("**Tasks:**")
                for task in week.tasks:
                    st.markdown(f"- [ ] {task}")

    # ── Approval interrupt ────────────────────────────────────────
    if interrupt_node == "review_roadmap":
        st.divider()
        render_approval_form(
            "roadmap",
            "Does this learning plan look feasible for your schedule?",
            lambda resp: _handle_approval(graph, config, resp, "roadmap_approved"),
            "roadmap",
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 6: PROJECT IDEAS                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_project_ideas_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("💡 Project Ideas")

    ranked = state.get("ranked_projects")
    if not ranked:
        render_empty_state(
            "No project ideas generated yet.",
            "Approve the roadmap in Tab 5 to generate project ideas.",
        )
        return

    st.markdown("Projects are ranked by a composite score. Expand each to review.")

    for i, pair in enumerate(ranked):
        idea = pair["idea"]
        score = pair["score"]
        composite = score.get("composite_score", 0)

        with st.expander(f"🏆 #{i + 1}: {idea['title']}  (Score: {composite:.1f})", expanded=(i == 0)):
            st.write(idea.get("description", ""))
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Technologies:** {', '.join(idea.get('technologies', []))}")
                st.write(f"**Difficulty:** {idea.get('difficulty', 'unknown')}")
            with col2:
                st.write(f"**Resume Value:** {score.get('resume_value', 0):.1f}")
                st.write(f"**Agentic Fit:** {score.get('agentic_fit', 0):.1f}")
                st.write(f"**Buildability:** {score.get('buildability_4_6_weeks', 0):.1f}")

            if idea.get("architecture_overview"):
                st.markdown(f"**Architecture:** {idea['architecture_overview']}")

    # ── Project selection interrupt ───────────────────────────────
    if interrupt_node == "review_project":
        st.divider()

        def extra_content():
            options = {i: f"#{i + 1}: {p['idea']['title']}" for i, p in enumerate(ranked)}
            chosen = st.radio(
                "Select a project to build:",
                options=options.keys(),
                format_func=lambda x: options[x],
            )
            return {"chosen_index": chosen}

        render_approval_form(
            "project",
            "Choose a project to proceed with. The AI will critique it, build a 6-week plan, and generate portfolio artifacts.",
            lambda resp: _handle_approval(graph, config, resp, "project_selected"),
            "project",
            extra_content_callback=extra_content,
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 7: PROJECT SCORING                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_scoring_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("📈 Project Scoring Breakdown")

    ranked = state.get("ranked_projects")
    if not ranked:
        render_empty_state(
            "No scored projects yet.",
            "Generate project ideas first (Tab 6).",
        )
        return

    rows = []
    for i, pair in enumerate(ranked):
        score = dict(pair["score"])
        score["Rank"] = i + 1
        score["Title"] = pair["idea"]["title"]
        rows.append(score)

    df = pd.DataFrame(rows)
    display_cols = [
        "Rank", "Title", "composite_score",
        "resume_value", "agentic_fit", "buildability_4_6_weeks",
        "personal_relevance", "technical_depth",
        "differentiation", "recruiter_explainability",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Score breakdown chart ─────────────────────────────────────
    if len(rows) > 1:
        chart_cols = [c for c in display_cols if c not in ("Rank", "Title", "composite_score") and c in df.columns]
        chart_df = df.set_index("Title")[chart_cols]
        st.bar_chart(chart_df)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 8: 6-WEEK BUILD PLAN                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_build_plan_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("🏗️ 6-Week Build Plan")

    plan = state.get("build_plan")
    if not plan:
        render_empty_state(
            "No build plan generated yet.",
            "Select a project in Tab 6 to generate a build plan.",
        )
        return

    st.subheader(f"Build Plan: {plan.project_title}")

    critique = state.get("project_critique")
    if critique and critique.get("risks"):
        with st.expander("⚠️ Critique Risks", expanded=False):
            for risk in critique["risks"]:
                st.markdown(f"- {risk}")
        if critique.get("scope_warning"):
            st.warning(f"**Scope Warning:** {critique['scope_warning']}")

    for week in plan.weeks:
        with st.expander(f"📅 Week {week.week_number}: {week.deliverable}", expanded=True):
            st.markdown("**Goals:**")
            for goal in week.goals:
                st.markdown(f"- [ ] {goal}")
            st.write(f"**Deliverable:** {week.deliverable}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 9: PORTFOLIO OUTPUTS                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_portfolio_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("📁 Portfolio Outputs")

    portfolio = state.get("portfolio_outputs")
    if not portfolio:
        render_empty_state(
            "No portfolio outputs yet.",
            "Complete the build plan (Tab 8) to generate portfolio artifacts.",
        )
        return

    tab_bullets, tab_readme, tab_demo, tab_pitch = st.tabs([
        "📝 Resume Bullets",
        "📖 README Outline",
        "🎬 Demo Script",
        "🎤 Interview Pitches",
    ])

    with tab_bullets:
        st.markdown("### Resume Bullets")
        bullets_text = "\n".join(f"- {b}" for b in portfolio.resume_bullets)
        st.markdown(bullets_text)
        download_button("⬇️ Download Resume Bullets", bullets_text, "resume_bullets.md")

    with tab_readme:
        st.markdown("### README Outline")
        st.markdown(portfolio.readme_outline)
        download_button("⬇️ Download README", portfolio.readme_outline, "README_outline.md")

    with tab_demo:
        st.markdown("### Demo Script")
        st.markdown(portfolio.demo_script)
        download_button("⬇️ Download Demo Script", portfolio.demo_script, "demo_script.md")

    with tab_pitch:
        st.markdown("### 30-Second Elevator Pitch")
        st.markdown(portfolio.interview_explanation_30s)
        st.divider()
        st.markdown("### 2-Minute Technical Walkthrough")
        st.markdown(portfolio.interview_explanation_2m)

        combined = (
            f"# 30-Second Pitch\n\n{portfolio.interview_explanation_30s}\n\n"
            f"# 2-Minute Walkthrough\n\n{portfolio.interview_explanation_2m}"
        )
        download_button("⬇️ Download Interview Scripts", combined, "interview_scripts.md")

    # ── Portfolio approval interrupt ──────────────────────────────
    if interrupt_node == "review_portfolio":
        st.divider()
        render_approval_form(
            "portfolio",
            "⚠️ **Review all portfolio artifacts carefully before using them publicly.** "
            "Ensure no claims are exaggerated beyond what you actually built.",
            lambda resp: _handle_approval(graph, config, resp, "portfolio_approved"),
            "portfolio",
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 10: PROGRESS MEMORY                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_progress_tab(
    state: Dict[str, Any], interrupt_node: Optional[str], graph: Any, config: Any
) -> None:
    st.header("🧠 Progress Memory")

    from src.storage.sqlite_store import get_progress_logs

    logs = get_progress_logs(st.session_state.session_id)

    # ── Session summary ───────────────────────────────────────────
    from src.storage import export_session_summary
    summary = export_session_summary(st.session_state.session_id)
    if "error" not in summary:
        with st.expander("📊 Session Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Profile", summary.get("profile_name", "—"))
                st.metric("Target Role", summary.get("target_role", "—"))
            with col2:
                st.metric("Project Ideas", summary.get("project_ideas_count", 0))
                st.metric("Roadmap Weeks", summary.get("roadmap_weeks", 0))
            with col3:
                st.metric("Build Plan", "✅" if summary.get("has_build_plan") else "—")
                st.metric("Portfolio", "✅" if summary.get("has_portfolio") else "—")

    # ── Progress logs ─────────────────────────────────────────────
    if not logs:
        render_empty_state(
            "No weekly progress logs recorded yet.",
            "This will be active during your 6-week build.",
        )
        return

    st.subheader("Weekly Progress Logs")
    for log in logs:
        with st.expander(f"Week {log.week_number} — {log.date}"):
            st.markdown("**Tasks Completed:**")
            for t in log.tasks_completed:
                st.markdown(f"- ✅ {t}")
            st.markdown("**Blockers:**")
            for b in log.blockers:
                st.markdown(f"- 🚧 {b}")
            st.markdown("**Next Steps:**")
            for n in log.next_steps:
                st.markdown(f"- ➡️ {n}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SHARED HELPERS                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def _handle_approval(graph, config, response: dict, phase_name: str) -> None:
    """
    Resume the LangGraph workflow from an interrupt with the user's response,
    persist the resulting state, and trigger a Streamlit rerun.
    """
    with st.spinner("Processing your feedback and continuing the pipeline..."):
        run_pipeline_phase(
            graph, config,
            resume_data=response,
            phase_name=phase_name,
        )
    st.rerun()
