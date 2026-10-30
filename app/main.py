"""Streamlit User Interface and Observability Dashboard for ResearchPilot.

Provides a clean, functional interface that demonstrates the complete bounded agent
workflow: explicit planning, real tool use, evidence extraction and filtering,
provenance-checked synthesis, output quality gates, and runtime execution observability.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is available on sys.path regardless of how Streamlit is invoked
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.config import get_settings
from app.logging_config import configure_logging
from app.orchestrator import run_full_pipeline
from app.evaluation_recorder import record_streamlit_run
from app.schemas import (
    ClaimSupportStatus,
    ConfidenceLevel,
    ExecutionEvent,
    Finding,
    ResearchAnswer,
    ResearchAnswerStatus,
    ResearchContext,
    SourceDocument,
    WorkflowStage,
)
from app.ui_helpers import grouped_warnings, source_mix_chips, status_banner

# Initialize structured logging
logger = configure_logging(level="INFO", log_file="logs/researchpilot.jsonl", json_format=True)


# ==============================================================================
# UI Helper Functions (Pure & Testable)
# ==============================================================================


def validate_user_question(question: Optional[str]) -> tuple[bool, str]:
    """Validate research inquiry before dispatching to the orchestrator.

    Args:
        question: Raw user question string.

    Returns:
        (is_valid, error_message_or_empty)
    """
    if question is None:
        return False, "Research question cannot be empty."
    stripped = question.strip()
    if not stripped:
        return False, "Research question cannot be empty or whitespace only."
    if len(stripped) < 3:
        return False, "Research question must be at least 3 characters long."
    if len(stripped) > 1000:
        return False, "Research question cannot exceed 1000 characters."
    return True, ""


def clear_research_session(session_state: Any) -> None:
    """Clear the editable query and the currently displayed research result."""
    session_state.pop("research_context", None)
    session_state.pop("submitted_question", None)
    session_state["question_input"] = ""


def format_source_display_date(published_date: Optional[str]) -> str:
    """Format published date without inventing or hallucinating missing metadata."""
    if published_date and published_date.strip():
        return published_date.strip()
    return "Not specified in page metadata"


def build_source_comparison_data(
    finding: Finding,
    sources: list[SourceDocument],
) -> list[dict[str, str]]:
    """Map a finding's cited source IDs to real source excerpts for reviewer inspection."""
    sources_by_id = {s.source_id: s for s in sources}
    comparison_rows = []
    for sid in finding.source_ids:
        doc = sources_by_id.get(sid)
        if doc:
            snippet = doc.content[:400].strip()
            if len(doc.content) > 400:
                snippet += "..."
            comparison_rows.append(
                {
                    "source_id": sid,
                    "title": doc.title,
                    "domain": doc.domain,
                    "url": doc.url,
                    "excerpt": snippet,
                }
            )
        else:
            comparison_rows.append(
                {
                    "source_id": sid,
                    "title": "Unregistered Source",
                    "domain": "Unknown",
                    "url": "",
                    "excerpt": "No matching source document found in accepted evidence registry.",
                }
            )
    return comparison_rows


def build_trace_table_data(trace: list[ExecutionEvent]) -> list[dict[str, Any]]:
    """Transform audit execution events into tabular format for observability display."""
    rows = []
    for idx, evt in enumerate(trace):
        rows.append(
            {
                "#": idx + 1,
                "Stage": evt.stage.value if hasattr(evt.stage, "value") else str(evt.stage),
                "Event Type": evt.event_type,
                "Duration (ms)": f"{evt.duration_ms:.1f}" if evt.duration_ms is not None else "-",
                "Status": "✅ Success" if evt.success else "❌ Failed",
                "Error Category": evt.error_category.value if evt.error_category else "-",
                "Details": evt.message or "-",
            }
        )
    return rows


def calculate_trace_metrics(
    trace: list[ExecutionEvent],
    context: Any | None = None,
) -> dict[str, Any]:
    """Compute aggregate observability metrics from runtime trace events."""
    if context is not None and context.pipeline_started_at and context.pipeline_finished_at:
        total_duration_ms = (
            context.pipeline_finished_at - context.pipeline_started_at
        ).total_seconds() * 1000.0
        stage_wall_times_ms = context.stage_wall_times_ms
    else:
        # Legacy helper callers may provide synthetic events without pipeline
        # timing metadata. Actual dashboard runs always pass the context.
        total_duration_ms = sum(
            e.duration_ms for e in trace if e.duration_ms is not None
        )
        stage_wall_times_ms = {}
    tool_events = [e for e in trace if e.event_type in {"search_query_executed", "page_fetched"}]
    failed_events = [e for e in trace if not e.success]

    return {
        "total_duration_s": round(total_duration_ms / 1000.0, 2),
        "stage_wall_times_ms": stage_wall_times_ms,
        "total_events": len(trace),
        "tool_calls_count": len(tool_events),
        "failures_count": len(failed_events),
    }


# ==============================================================================
# Streamlit Rendering Components
# ==============================================================================


def _render_sidebar(settings: Any) -> bool:
    """Render sidebar with system configuration and controls."""
    with st.sidebar:
        st.markdown(
            "<style>[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] "
            "{padding-left: 0.35rem;}</style>",
            unsafe_allow_html=True,
        )
        st.header("⚙️ Configuration")
        st.markdown(
            f"**Model:** `{settings.groq_model}`\n\n"
            f"**Fallback Model:** `{settings.groq_synthesis_fallback_model or 'disabled'}`\n\n"
            f"**Search Provider:** `DuckDuckGo (ddgs) with Bing fallback`\n\n"
            f"**Query Budget:** `{settings.max_search_queries}`\n\n"
            f"**Page Budget:** `{settings.max_fetch_pages}`\n\n"
            f"**Fetch Workers:** `{settings.fetch_max_workers}`\n\n"
            f"**Search Cap:** `{settings.search_query_timeout_seconds}s`\n\n"
            f"**Deadline:** `{settings.overall_deadline_seconds}s`\n\n"
            f"**Early-stop:** `{settings.early_stop_consecutive_failures}` failures / "
            f"`{settings.early_stop_min_relevance}` relevance"
        )
        st.divider()

        st.subheader("Inspection Controls")
        verify_claims = st.checkbox(
            "Verify claim support with LLM",
            value=False,
            help="Optionally runs the claim-support verifier on key findings against cited excerpts.",
        )

        st.divider()
        if st.button("🗑️ Reset Session", use_container_width=True):
            clear_research_session(st.session_state)
            st.rerun()

        st.caption("ResearchPilot v1.0")
        return verify_claims


def _render_findings(
    findings: list[Finding],
    sources: list[SourceDocument],
    source_url_map: dict[str, str],
) -> None:
    """Render key findings with clickable citations and reviewer inspection drawers."""
    st.subheader("Key Findings & Provenance")
    if not findings:
        st.info("No factual findings synthesized.")
        return

    for idx, f in enumerate(findings):
        unsupported = f.support_status == ClaimSupportStatus.UNSUPPORTED
        finding_container = st.expander(
            f"⚪ Unsupported finding #{idx + 1}: {f.claim[:80]}",
            expanded=False,
        ) if unsupported else st.container()
        with finding_container:
            # Support status badge
            badge = ""
            if f.support_status == ClaimSupportStatus.SUPPORTED:
                badge = " `[Supported]`"
            elif f.support_status == ClaimSupportStatus.PARTIALLY_SUPPORTED:
                badge = " `[Partially Supported]`"
            elif f.support_status == ClaimSupportStatus.UNSUPPORTED:
                badge = " `[Ungrounded / Unsupported]`"

            st.markdown(f"**{idx + 1}. {html.escape(f.claim)}**{badge}")
            if f.explanation:
                st.write(f.explanation)

            # Clickable Citations
            if f.source_ids:
                citation_links = []
                for sid in f.source_ids:
                    real_url = source_url_map.get(sid)
                    if real_url:
                        citation_links.append(f"[{sid}]({real_url})")
                    else:
                        citation_links.append(f"`{sid}`")
                st.markdown(f"**Citations:** {' • '.join(citation_links)}")
                st.caption(" ".join(
                    f"`{sid}: {next((s.source_type.value + ' / ' + s.domain for s in sources if s.source_id == sid), 'unknown')}`"
                    for sid in f.source_ids
                ))

                # Reviewer comparison drawer
                with st.expander(f"🔎 Reviewer evidence comparison for Finding #{idx + 1}"):
                    comp_data = build_source_comparison_data(f, sources)
                    for item in comp_data:
                        st.markdown(
                            f"**{item['source_id']} — [{item['title']}]({item['url']})** ({item['domain']})"
                        )
                        st.caption(f"Supporting excerpt: \"{item['excerpt']}\"")
                        st.divider()
            else:
                st.caption("⚠️ No supporting citations linked to this finding.")
            st.write("")


def _render_research_plan(plan: Any) -> None:
    """Render structured research plan."""
    with st.expander("📋 Expandable Research Plan", expanded=False):
        st.markdown(f"**Objective:** {plan.objective}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Decomposed Subquestions:**")
            for sq in plan.subquestions:
                st.markdown(f"- {sq}")
        with col2:
            st.markdown("**Search Queries Generated:**")
            for q in plan.search_queries:
                st.markdown(f"- `{q}`")

        st.markdown(f"**Allowlisted Tools:** `{', '.join(plan.tools)}`")
        st.markdown("**Success Criteria:**")
        for sc in plan.success_criteria:
            st.markdown(f"- {sc}")

        if plan.plan_notes:
            st.caption(f"Planner Notes: {plan.plan_notes}")


def _render_sources_registry(sources: list[SourceDocument]) -> None:
    """Render accepted source registry showing real URLs, domains, and un-hallucinated dates."""
    with st.expander(f"📚 Sources Registry ({len(sources)} Accepted)", expanded=False):
        if not sources:
            st.write("No sources were accepted.")
            return

        for doc in sources:
            st.markdown(f"#### `{doc.source_id}` — [{doc.title}]({doc.url})")
            date_str = format_source_display_date(doc.published_date)
            st.markdown(
                f"- **Domain:** `{doc.domain}`\n"
                f"- **Source Type:** `{doc.source_type.value}`\n"
                f"- **Published Date:** {date_str}\n"
                f"- **Quality Score:** `{doc.quality_score:.2f}`\n"
                f"- **URL:** [{doc.url}]({doc.url})"
            )
            with st.expander(f"Preview extracted text ({len(doc.content)} chars)"):
                st.text(doc.content[:1500] + ("..." if len(doc.content) > 1500 else ""))
            st.divider()


def _render_coverage_and_rejections(
    context: ResearchContext,
) -> None:
    """Render subquestion evidence coverage and source rejection details."""
    with st.expander("⚖️ Evidence Coverage & Source Rejections", expanded=False):
        st.markdown(
            f"**Subquestion Coverage Ratio:** `{context.coverage.coverage_ratio:.0%}` "
            f"({'Adequate' if context.coverage.is_adequate else 'Inadequate'})"
        )

        st.write("**Coverage Breakdown:**")
        for sq, status in context.coverage.subquestion_coverage.items():
            icon = "✅" if status == "supported" else ("⚠️" if status == "partially_supported" else "❌")
            st.markdown(f"- {icon} **{sq}**: `{status}`")

        st.divider()
        st.markdown(f"**Rejected Sources ({len(context.rejected_sources)}):**")
        if context.rejected_sources:
            for r in context.rejected_sources:
                st.markdown(
                    f"- ❌ **[{r.title}]({r.url})** (`{r.domain}`) — "
                    f"**Reason:** *{r.rejection_reason or 'Failed quality threshold'}* "
                    f"(Quality score: {r.quality_score:.2f})"
                )
        else:
            st.caption("No candidate sources were rejected.")


def _render_trace_and_observability(
    trace: list[ExecutionEvent],
    context: ResearchContext | None = None,
) -> None:
    """Render execution audit trace with durations, retries, and failure categories."""
    with st.expander(f"🔍 Execution Trace & Observability ({len(trace)} Events)", expanded=False):
        metrics = calculate_trace_metrics(trace, context)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Latency", f"{metrics['total_duration_s']}s")
        c2.metric("Total Events", metrics["total_events"])
        c3.metric("Tool Calls", metrics["tool_calls_count"])
        c4.metric("Errors / Retries", metrics["failures_count"])
        if context is not None and context.error_breakdown:
            st.caption("Error breakdown")
            st.dataframe(
                [{"Error": key, "Count": value} for key, value in context.error_breakdown.items()],
                use_container_width=True,
                hide_index=True,
            )
        if metrics["stage_wall_times_ms"]:
            st.caption("Stage wall time")
            st.dataframe(
                [
                    {"Stage": stage, "Wall time (s)": round(duration / 1000.0, 2)}
                    for stage, duration in metrics["stage_wall_times_ms"].items()
                ],
                use_container_width=True,
                hide_index=True,
            )

        trace_data = build_trace_table_data(trace)
        st.dataframe(trace_data, use_container_width=True, hide_index=True)

        # JSON trace export
        trace_json = json.dumps([e.model_dump(mode="json") for e in trace], indent=2)
        st.download_button(
            label="📥 Download Trace JSON",
            data=trace_json,
            file_name="researchpilot_trace.json",
            mime="application/json",
        )
        if context is not None:
            result_json = json.dumps(context.model_dump(mode="json"), indent=2)
            st.download_button(
                label="📥 Download Result JSON",
                data=result_json,
                file_name="researchpilot_result.json",
                mime="application/json",
            )


def _render_results(context: ResearchContext) -> None:
    """Render full research deliverables and inspectable intermediary states."""
    answer = context.answer

    # 1. Unified status callout
    level, title, subtitle = status_banner(context)
    getattr(st, level)(f"**{title}** — {subtitle}")

    # 3. Quality Gate Warnings
    if context.status_reasons:
        with st.expander("Why this status?", expanded=False):
            for reason in context.status_reasons:
                st.markdown(f"- **{reason.code.value}:** {reason.message}")
    if context.quality_gate_warnings:
        with st.expander("⚠️ Quality Gate & Citation Warnings", expanded=True):
            for group, warnings in grouped_warnings(context).items():
                if warnings:
                    st.markdown(f"**{group}**")
                    for warning in warnings:
                        st.markdown(f"- {warning}")

    # 4. Metrics summary bar
    if answer:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Confidence", answer.confidence.value.upper())
        m2.metric("Evidence Coverage", f"{context.coverage.coverage_ratio:.0%}")
        m3.metric("Accepted Sources", len(context.sources))
        m4.metric("Rejected Sources", len(context.rejected_sources))
        total_latency = calculate_trace_metrics(context.trace, context)["total_duration_s"]
        m5.metric("Total Latency", f"{total_latency}s")
        st.caption("Entities: " + (context.entity_coverage and "  ".join(
            f"{entity} {'✓' if item.get('supported') else '✗'}"
            for entity, item in context.entity_coverage.items()
        ) or "No named entities"))
        st.caption("Source mix: " + " ".join(
            f"`{item['type']}: {item['count']}`" for item in source_mix_chips(context)
        ))
        if context.evaluation:
            st.caption(
                f"Claim grounding: {context.evaluation.grounded_finding_count}/"
                f"{context.evaluation.finding_count} findings; citation coverage: "
                f"{context.evaluation.citation_coverage_rate:.0%}."
            )

    st.divider()

    # 5. Short Answer
    if answer and answer.short_answer:
        st.subheader("Short Answer")
        for sentence in answer.short_answer.split(". "):
            marker = " ⚠ unverified" if any(
                sentence.strip() in item for item in answer.short_answer_ungrounded
            ) else ""
            st.markdown(sentence + ("." if not sentence.endswith(".") else "") + marker)

    # 6. Key Findings
    if answer:
        _render_findings(answer.key_findings, context.sources, answer.source_url_map)

    # 7. Assumptions, Limitations, Next Steps
    if answer:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### 📌 Assumptions")
            if answer.assumptions:
                for a in answer.assumptions:
                    st.markdown(f"- {a}")
            else:
                st.caption("No explicit assumptions recorded.")

        with col_b:
            st.markdown("#### ⚠️ Limitations & Gaps")
            if answer.limitations:
                for lim in answer.limitations:
                    st.markdown(f"- {lim}")
            else:
                st.caption("No limitations noted.")

        with col_c:
            st.markdown("#### 🔮 Next Steps")
            if answer.suggested_next_steps:
                for step in answer.suggested_next_steps:
                    st.markdown(f"- {step}")
            else:
                st.caption("No follow-up steps suggested.")

    st.divider()

    # 8. Expandable Sections
    _render_research_plan(context.plan)
    _render_sources_registry(context.sources)
    _render_coverage_and_rejections(context)
    _render_trace_and_observability(context.trace, context)


# ==============================================================================
# Main Streamlit Application Entrypoint
# ==============================================================================


def main() -> None:
    st.set_page_config(
        page_title="ResearchPilot — AI Research Agent",
        page_icon="🧭",
        layout="wide",
    )

    settings = get_settings()
    verify_claims = _render_sidebar(settings)

    st.title("🧭 ResearchPilot: Reliable AI Research Agent")
    st.caption(
        "A bounded, inspectable AI research agent implementing explicit planning, "
        "real web search, safe page extraction, evidence filtering, and verifiable citations."
    )

    # User Input Section
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = st.session_state.get("submitted_question", "")

    user_query = st.text_area(
        "Enter your research question:",
        key="question_input",
        placeholder="e.g. Compare Milvus and Qdrant for high-concurrency vector search. What are their memory requirements?",
        height=100,
        help="Enter a clear, specific technical or research question. Minimum 3 characters.",
    )

    col1, col2 = st.columns([1, 1])
    run_btn = col1.button("🚀 Run Research", type="primary", use_container_width=True)
    col2.button(
        "✖️ Clear Query",
        use_container_width=True,
        help="Clear the current question and remove the displayed result.",
        on_click=clear_research_session,
        args=(st.session_state,),
    )

    if run_btn:
        is_valid, err_msg = validate_user_question(user_query)
        if not is_valid:
            st.error(f"⚠️ Input Validation Error: {err_msg}")
            return

        # Check API key configuration
        if not settings.groq_api_key:
            st.error(
                "❌ Configuration Error: Missing `GROQ_API_KEY`. "
                "Please configure your Groq API key in your `.env` file before executing research."
            )
            return

        st.session_state["submitted_question"] = user_query.strip()
        started_at = datetime.now(timezone.utc)

        # Execute bounded research through the orchestrator as the single entry point
        with st.status("Executing research inquiry...", expanded=True) as status_box:
            status_container = st.empty()

            def progress_callback(stage: WorkflowStage, msg: str) -> None:
                stage_name = stage.value if hasattr(stage, "value") else str(stage)
                status_box.update(label=f"[{stage_name}] {msg}")
                status_container.info(f"**[{stage_name}]** {msg}")

            try:
                context = run_full_pipeline(
                    request=user_query.strip(),
                    settings=settings,
                    verify_claims=verify_claims,
                    progress_callback=progress_callback,
                )
                try:
                    log_path = record_streamlit_run(
                        context,
                        started_at=started_at,
                    )
                    logger.info("Streamlit evaluation run recorded at %s", log_path)
                except OSError as record_error:
                    logger.error(
                        "Could not persist Streamlit evaluation run: %s",
                        record_error,
                    )
                status_box.update(
                    label=f"Completed with status: {context.status.value}",
                    state="complete" if context.status == ResearchAnswerStatus.COMPLETED else "error",
                    expanded=False,
                )
                st.session_state["research_context"] = context
            except Exception as exc:
                logger.error("Unhandled orchestrator error in UI: %s", exc, exc_info=True)
                status_box.update(label="Research pipeline failed", state="error", expanded=False)
                st.error(f"Workflow execution failed: {type(exc).__name__}: {exc}")
                return

    # Render persisted research results if present in session state
    if "research_context" in st.session_state:
        _render_results(st.session_state["research_context"])


if __name__ == "__main__":
    main()
