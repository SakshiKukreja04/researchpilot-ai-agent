"""Regression tests for the three README sample-output scenarios."""

import json
from unittest.mock import MagicMock

from app.llm import GroqClientWrapper
from app.orchestrator import run_full_pipeline
from app.schemas import (
    ConfidenceLevel,
    ExtractionStatus,
    ResearchAnswerStatus,
    ResearchPlan,
    SearchResult,
    SourceDocument,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Compare embedded and server databases for an offline-first app",
        subquestions=["How do deployment models differ?"],
        search_queries=["SQLite versus PostgreSQL deployment model"],
        tools=["web_search", "fetch_and_extract"],
        success_criteria=["Deployment tradeoffs"],
    )


def _source() -> SourceDocument:
    return SourceDocument(
        source_id="src_abcdef12",
        url="https://docs.example.com/databases",
        title="Database deployment documentation",
        domain="docs.example.com",
        content=(
            "SQLite is an embedded database that runs without a separate server. "
            "PostgreSQL is a server-oriented database for centralized workloads."
        )
        * 20,
        quality_score=0.9,
        extraction_status=ExtractionStatus.SUCCESS,
    )


def _tools():
    return (
        MagicMock(
            return_value=[
                SearchResult(
                    title="Database deployment documentation",
                    url="https://docs.example.com/databases",
                    snippet="Embedded SQLite and server-oriented PostgreSQL.",
                    domain="docs.example.com",
                )
            ]
        ),
        MagicMock(return_value=_source()),
    )


def test_sample_output_one_successful_grounded_comparison():
    client = MagicMock(spec=GroqClientWrapper)
    client.model_name = "test-model"
    client.chat_completion.side_effect = [
        _plan().model_dump_json(),
        json.dumps(
            {
                "user_question": "Compare PostgreSQL and SQLite for a small offline-first research application.",
                "short_answer": "SQLite fits local-first use; PostgreSQL fits centralized concurrent use.",
                "key_findings": [
                    {
                        "claim": "SQLite is an embedded database that runs without a separate server.",
                        "explanation": "This reduces local deployment overhead.",
                        "source_ids": ["src_abcdef12"],
                    }
                ],
                "sources_used": ["src_abcdef12"],
                "confidence": "medium",
                "assumptions": [],
                "limitations": ["Workload-specific benchmarking is still required."],
                "suggested_next_steps": ["Benchmark the expected workload."],
                "status": "completed",
            }
        ),
    ]
    search, fetch = _tools()

    context = run_full_pipeline(
        "Compare PostgreSQL and SQLite for a small offline-first research application.",
        client=client,
        search_tool_fn=search,
        fetch_tool_fn=fetch,
    )

    assert context.status == ResearchAnswerStatus.COMPLETED
    assert context.answer is not None
    assert context.answer.confidence == ConfidenceLevel.MEDIUM
    assert context.answer.key_findings[0].source_ids == ["src_abcdef12"]
    assert context.evaluation is not None
    assert context.evaluation.citation_validity_rate == 1.0


def test_sample_output_two_preserves_synthesis_failure():
    client = MagicMock(spec=GroqClientWrapper)
    client.model_name = "test-model"
    client.chat_completion.side_effect = [_plan().model_dump_json(), RuntimeError("model unavailable")]
    search, fetch = _tools()

    context = run_full_pipeline(
        "Compare PostgreSQL and SQLite for a small offline-first research application.",
        client=client,
        search_tool_fn=search,
        fetch_tool_fn=fetch,
    )

    assert context.answer is not None
    assert context.answer.status == ResearchAnswerStatus.PARTIAL
    assert context.answer.confidence == ConfidenceLevel.LOW
    assert context.answer.key_findings == []
    assert any(event.event_type == "synthesis_failed" for event in context.trace)


def test_sample_output_three_rejects_whitespace_before_external_calls():
    search = MagicMock()
    fetch = MagicMock()

    context = run_full_pipeline("   ", search_tool_fn=search, fetch_tool_fn=fetch)

    assert context.status == ResearchAnswerStatus.FAILED
    assert context.answer is not None
    assert context.answer.status == ResearchAnswerStatus.FAILED
    assert "whitespace" in context.answer.short_answer.lower()
    search.assert_not_called()
    fetch.assert_not_called()
