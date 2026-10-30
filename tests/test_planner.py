"""Unit tests for the LLM-based research planner.

All tests run completely offline using mocked GroqClientWrapper instances.
Requires no live API key or internet access.
"""

import json
from unittest.mock import MagicMock

import pytest

from app.llm import (
    GroqClientWrapper,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.planner import (
    PlannerAuthError,
    PlannerInvalidInputError,
    PlannerProviderError,
    PlannerRateLimitError,
    PlannerTimeoutError,
    PlannerValidationError,
    create_fallback_plan,
    create_research_plan,
    extract_search_entities,
)
from app.schemas import ExecutionEvent, ResearchPlan, ResearchRequest, SearchResult


# ==============================================================================
# Helpers and Fixtures
# ==============================================================================


def _mock_plan_json(
    objective: str = "Compare open-source vector databases",
    subquestions: list[str] | None = None,
    search_queries: list[str] | None = None,
    tools: list[str] | None = None,
    success_criteria: list[str] | None = None,
    requires_fresh_evidence: bool = False,
    plan_notes: str | None = "Comparison intent focus",
) -> str:
    subquestions = subquestions or [
        "What are the indexing algorithms and querying capabilities?",
        "What are the horizontal scaling and clustering architectures?",
        "What are the open-source licenses and hosting models?",
    ]
    search_queries = search_queries or [
        "open source vector database comparison",
        "milvus vs qdrant vs weaviate benchmark",
        "vector database scaling architectures",
    ]
    tools = tools or ["web_search", "fetch_and_extract"]
    success_criteria = success_criteria or [
        "Identify filtering and indexing features for each candidate",
        "Compare memory footprint and performance benchmarks",
    ]
    data = {
        "objective": objective,
        "subquestions": subquestions,
        "search_queries": search_queries,
        "tools": tools,
        "success_criteria": success_criteria,
        "requires_fresh_evidence": requires_fresh_evidence,
        "plan_notes": plan_notes,
    }
    return json.dumps(data)


# ==============================================================================
# Valid Plan Generation Tests
# ==============================================================================


def test_comparison_question_produces_valid_plan():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    mock_client.chat_completion.return_value = _mock_plan_json()

    req = ResearchRequest(
        question="Compare three open-source vector databases for a startup building RAG products."
    )
    plan = create_research_plan(req, client=mock_client)

    assert isinstance(plan, ResearchPlan)
    assert len(plan.subquestions) >= 3
    assert len(plan.search_queries) >= 3
    assert set(plan.tools) <= {"web_search", "fetch_and_extract"}
    assert plan.requires_fresh_evidence is False


def test_company_research_produces_valid_plan():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    mock_client.chat_completion.return_value = _mock_plan_json(
        objective="Find 5 Indian B2B SaaS startups in HR tech",
        subquestions=[
            "What are 5 prominent Indian B2B HR tech startups?",
            "What specific HR problems does each startup solve?",
            "What is their market positioning and customer base?",
        ],
        search_queries=[
            "Indian B2B SaaS HR tech startups list",
            "top HR tech companies India 2026",
            "Indian payroll and HR software companies",
        ],
    )

    plan = create_research_plan(
        "Find five Indian B2B SaaS startups in HR tech and summarize their positioning.",
        client=mock_client,
    )

    assert isinstance(plan, ResearchPlan)
    assert "Indian B2B SaaS" in plan.objective
    assert len(plan.search_queries) <= 10
    assert any("Darwinbox official site" in query for query in plan.search_queries)
    assert any("Keka official site" in query for query in plan.search_queries)


def test_entity_extraction_rejects_generic_phrases_and_requires_two_results():
    results = [
        SearchResult(
            title="We Tested official site",
            url="https://example.com/one",
            snippet="We Tested several options.",
        ),
        SearchResult(
            title="Open-Source Vector DB official site",
            url="https://example.com/two",
            snippet="Open-Source Vector DB comparison.",
        ),
        SearchResult(
            title="PostgreSQL documentation",
            url="https://postgresql.org/docs",
            snippet="PostgreSQL supports extensions for vector search.",
        ),
        SearchResult(
            title="PostgreSQL for RAG",
            url="https://example.org/postgres",
            snippet="PostgreSQL is used in RAG systems.",
        ),
    ]

    entities = extract_search_entities(
        results,
        question="Compare PostgreSQL and SQLite for RAG products.",
    )

    assert "We Tested" not in entities
    assert "Open-Source Vector DB" not in entities
    assert "PostgreSQL" in entities


def test_time_sensitive_question_marks_requires_fresh_evidence():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    mock_client.chat_completion.return_value = _mock_plan_json(
        objective="Investigate latest AI model releases this week",
        requires_fresh_evidence=True,
    )

    plan = create_research_plan(
        "What are the latest AI announcements released this week?",
        client=mock_client,
    )

    assert plan.requires_fresh_evidence is True


def test_technical_explanation_does_not_require_freshness_by_default():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    mock_client.chat_completion.return_value = _mock_plan_json(
        objective="Explain the Raft consensus algorithm",
        requires_fresh_evidence=False,
    )

    plan = create_research_plan(
        "Explain how the Raft consensus algorithm works.",
        client=mock_client,
    )

    assert plan.requires_fresh_evidence is False


def test_selected_tools_are_strict_members_of_allowlist():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    mock_client.chat_completion.return_value = _mock_plan_json(
        tools=["web_search", "fetch_and_extract"]
    )

    plan = create_research_plan("Analyze search engines", client=mock_client)
    assert all(tool in {"web_search", "fetch_and_extract"} for tool in plan.tools)


# ==============================================================================
# Invalid Model Responses & Repair Tests
# ==============================================================================


def test_malformed_json_triggers_successful_repair():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    # First attempt returns broken JSON; second attempt returns valid plan JSON
    mock_client.chat_completion.side_effect = [
        "NOT JSON { broken",
        _mock_plan_json(objective="Repaired research plan"),
    ]

    plan = create_research_plan("Compare databases", client=mock_client)

    assert isinstance(plan, ResearchPlan)
    assert plan.objective == "Repaired research plan"
    assert mock_client.chat_completion.call_count == 2


def test_disallowed_tool_triggers_successful_repair():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    # First attempt returns an unauthorized tool; second returns allowlisted tools
    invalid_plan = _mock_plan_json(tools=["web_search", "execute_python_code"])
    valid_plan = _mock_plan_json(tools=["web_search", "fetch_and_extract"])

    mock_client.chat_completion.side_effect = [invalid_plan, valid_plan]

    plan = create_research_plan("Test tools", client=mock_client)
    assert "execute_python_code" not in plan.tools
    assert "web_search" in plan.tools
    assert mock_client.chat_completion.call_count == 2


def test_too_many_subquestions_triggers_repair():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    # 12 subquestions exceeds the max limit of 8
    excessive_subquestions = [f"Subquestion {i}" for i in range(12)]
    invalid_plan = _mock_plan_json(subquestions=excessive_subquestions)
    valid_plan = _mock_plan_json(subquestions=["Subquestion 1", "Subquestion 2", "Subquestion 3"])

    mock_client.chat_completion.side_effect = [invalid_plan, valid_plan]

    plan = create_research_plan("Test count", client=mock_client)
    assert len(plan.subquestions) == 3


def test_failed_repair_engages_fallback_when_enabled():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    # Both initial and repair attempt return broken JSON
    mock_client.chat_completion.side_effect = [
        "Broken 1",
        "Broken 2",
    ]

    trace_events: list[ExecutionEvent] = []
    plan = create_research_plan(
        "Compare vector databases for high concurrency search",
        client=mock_client,
        trace=trace_events,
        enable_fallback=True,
    )

    assert isinstance(plan, ResearchPlan)
    assert "[FALLBACK PLAN]" in (plan.plan_notes or "")
    assert mock_client.chat_completion.call_count == 2
    # Verify fallback event was recorded
    assert any(evt.event_type == "plan_fallback_used" for evt in trace_events)


def test_failed_repair_raises_validation_error_when_fallback_disabled():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    mock_client.chat_completion.side_effect = [
        "Broken 1",
        "Broken 2",
    ]

    with pytest.raises(PlannerValidationError):
        create_research_plan(
            "Compare vector databases",
            client=mock_client,
            enable_fallback=False,
        )


def test_repair_does_not_trigger_infinite_retry_loops():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "openai/gpt-oss-120b"
    mock_client.chat_completion.side_effect = [
        "Invalid JSON 1",
        "Invalid JSON 2",
        "Invalid JSON 3",
    ]

    create_research_plan(
        "Compare vector databases",
        client=mock_client,
        enable_fallback=True,
    )
    # Total calls must be strictly 2 (1 initial + 1 repair)
    assert mock_client.chat_completion.call_count == 2


# ==============================================================================
# Provider Failure Tests (Must NOT attempt model repair)
# ==============================================================================


def test_timeout_error_raises_planner_timeout_error():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.chat_completion.side_effect = LLMTimeoutError("Groq request timed out.")

    with pytest.raises(PlannerTimeoutError) as exc_info:
        create_research_plan("Compare vector databases", client=mock_client)
    assert "timed out" in str(exc_info.value).lower()
    # Ensure no retry was attempted on provider timeout
    assert mock_client.chat_completion.call_count == 1


def test_rate_limit_error_raises_planner_rate_limit_error():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.chat_completion.side_effect = LLMRateLimitError("Rate limit exceeded.")

    with pytest.raises(PlannerRateLimitError):
        create_research_plan("Compare vector databases", client=mock_client)
    assert mock_client.chat_completion.call_count == 1


def test_auth_error_raises_planner_auth_error():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.chat_completion.side_effect = LLMAuthError("Invalid API key.")

    with pytest.raises(PlannerAuthError):
        create_research_plan("Compare vector databases", client=mock_client)
    assert mock_client.chat_completion.call_count == 1


def test_connection_error_raises_planner_provider_error():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.chat_completion.side_effect = LLMConnectionError("Network drop.")

    with pytest.raises(PlannerProviderError):
        create_research_plan("Compare vector databases", client=mock_client)
    assert mock_client.chat_completion.call_count == 1


# ==============================================================================
# Deterministic Fallback Unit Tests
# ==============================================================================


def test_create_fallback_plan_comparison():
    plan = create_fallback_plan("Compare PostgreSQL and MongoDB for web apps")
    assert isinstance(plan, ResearchPlan)
    assert len(plan.subquestions) >= 3
    assert len(plan.search_queries) >= 3
    assert plan.tools == ["web_search", "fetch_and_extract"]
    assert "[FALLBACK PLAN]" in (plan.plan_notes or "")


def test_create_fallback_plan_freshness():
    plan = create_fallback_plan("What is the latest update on quantum computing in 2026?")
    assert isinstance(plan, ResearchPlan)
    assert plan.requires_fresh_evidence is True
    assert any("2026" in q or "latest" in q for q in plan.search_queries)


def test_invalid_input_questions_rejected():
    with pytest.raises(PlannerInvalidInputError):
        create_research_plan("")

    with pytest.raises(PlannerInvalidInputError):
        create_research_plan("   \n  ")

    with pytest.raises(PlannerInvalidInputError):
        create_research_plan(12345)  # type: ignore
