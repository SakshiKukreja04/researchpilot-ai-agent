"""Unit tests for Streamlit UI helper functions."""

from app.main import (
    clear_research_session,
    build_source_comparison_data,
    build_trace_table_data,
    calculate_trace_metrics,
    format_source_display_date,
    validate_user_question,
)
from app.schemas import (
    ClaimSupportStatus,
    ErrorCategory,
    ExecutionEvent,
    ExtractionStatus,
    Finding,
    SourceDocument,
    WorkflowStage,
    ReasonCode,
    StatusReason,
    ResearchAnswerStatus,
)
from app.ui_helpers import reason_message, status_banner, aggregate_trace_errors
from types import SimpleNamespace
from pathlib import Path


def test_validate_user_question():
    # Invalid inputs
    assert validate_user_question(None)[0] is False
    assert validate_user_question("")[0] is False
    assert validate_user_question("   \t\n  ")[0] is False
    assert validate_user_question("ab")[0] is False
    assert validate_user_question("a" * 1001)[0] is False

    # Valid inputs
    is_valid, err = validate_user_question("What are vector database memory benchmarks?")
    assert is_valid is True
    assert err == ""


def test_clear_research_session_removes_query_and_result():
    state = {
        "question_input": "old question",
        "submitted_question": "old question",
        "research_context": object(),
    }

    clear_research_session(state)

    assert state["question_input"] == ""
    assert "submitted_question" not in state
    assert "research_context" not in state


def test_format_source_display_date():
    assert format_source_display_date(None) == "Not specified in page metadata"
    assert format_source_display_date("") == "Not specified in page metadata"
    assert format_source_display_date("   ") == "Not specified in page metadata"
    assert format_source_display_date("2026-04-12") == "2026-04-12"


def test_build_source_comparison_data():
    source_1 = SourceDocument(
        source_id="src_111122223333",
        title="Vector Database Guide",
        url="https://docs.example.com/guide",
        domain="docs.example.com",
        content="Substantive technical content explaining HNSW index RAM usage.",
        quality_score=0.95,
        extraction_status=ExtractionStatus.SUCCESS,
    )

    finding = Finding(
        claim="HNSW indexes require considerable RAM for vector storage.",
        explanation="Benchmark evidence shows high memory overhead.",
        source_ids=["src_111122223333", "src_999999999999"],
        support_status=ClaimSupportStatus.SUPPORTED,
    )

    rows = build_source_comparison_data(finding, [source_1])
    assert len(rows) == 2

    # Known source
    assert rows[0]["source_id"] == "src_111122223333"
    assert rows[0]["title"] == "Vector Database Guide"
    assert "HNSW index" in rows[0]["excerpt"]

    # Unknown source
    assert rows[1]["source_id"] == "src_999999999999"
    assert rows[1]["title"] == "Unregistered Source"


def test_build_trace_table_data():
    events = [
        ExecutionEvent(
            run_id="run_1",
            stage=WorkflowStage.RECEIVED,
            event_type="request_accepted",
            duration_ms=5.2,
            success=True,
            message="Accepted inquiry",
        ),
        ExecutionEvent(
            run_id="run_1",
            stage=WorkflowStage.SEARCHING,
            event_type="search_query_failed",
            duration_ms=520.0,
            success=False,
            error_category=ErrorCategory.TIMEOUT,
            message="Timed out",
        ),
    ]

    table_data = build_trace_table_data(events)
    assert len(table_data) == 2
    assert table_data[0]["Stage"] == "RECEIVED"
    assert table_data[0]["Status"] == "✅ Success"
    assert table_data[1]["Stage"] == "SEARCHING"
    assert table_data[1]["Status"] == "❌ Failed"
    assert table_data[1]["Error Category"] == "timeout"


def test_calculate_trace_metrics():
    events = [
        ExecutionEvent(
            run_id="run_1",
            stage=WorkflowStage.SEARCHING,
            event_type="search_query_executed",
            duration_ms=250.0,
            success=True,
        ),
        ExecutionEvent(
            run_id="run_1",
            stage=WorkflowStage.RETRIEVING,
            event_type="page_fetched",
            duration_ms=450.0,
            success=True,
        ),
        ExecutionEvent(
            run_id="run_1",
            stage=WorkflowStage.SEARCHING,
            event_type="search_query_failed",
            duration_ms=300.0,
            success=False,
            error_category=ErrorCategory.TIMEOUT,
        ),
    ]

    metrics = calculate_trace_metrics(events)
    assert metrics["total_duration_s"] == 1.0
    assert metrics["total_events"] == 3
    assert metrics["tool_calls_count"] == 2
    assert metrics["failures_count"] == 1


def _ui_context(status, coverage=1.0, reasons=None, entities=None):
    return SimpleNamespace(
        status=status,
        coverage=SimpleNamespace(coverage_ratio=coverage),
        status_reasons=reasons or [],
        entity_coverage=entities or {},
        error_breakdown={},
        answer=None,
        quality_gate_warnings=[],
    )


def test_status_banner_never_calls_missing_entity_evidence_green():
    context = _ui_context(
        ResearchAnswerStatus.COMPLETED,
        reasons=[StatusReason(code=ReasonCode.ENTITY_NO_EVIDENCE, message="No evidence for: Qdrant")],
    )
    level, title, _ = status_banner(context)
    assert level == "warning"
    assert "gaps" in title


def test_reason_mapping_and_error_breakdown():
    assert "deadline" in reason_message(ReasonCode.DEADLINE_EXHAUSTED, "deadline reached")
    event = ExecutionEvent(
        run_id="run_1",
        stage=WorkflowStage.SEARCHING,
        event_type="search_query_failed",
        success=False,
        error_category=ErrorCategory.TIMEOUT,
    )
    assert aggregate_trace_errors([event]) == {"timeout": 1}


def test_removed_legacy_footer_label():
    source = Path(__file__).parents[1].joinpath("app", "main.py").read_text(encoding="utf-8")
    assert "Antigravity" not in source
