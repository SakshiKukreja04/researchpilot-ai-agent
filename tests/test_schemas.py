"""Unit tests for Pydantic schemas and contracts."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    ConfidenceLevel,
    ErrorCategory,
    ExecutionEvent,
    ExtractionStatus,
    Finding,
    ResearchAnswer,
    ResearchAnswerStatus,
    ResearchPlan,
    ResearchRequest,
    SearchResult,
    SourceDocument,
    WorkflowStage,
)


class TestResearchRequest:
    def test_valid_request(self):
        req = ResearchRequest(question="  What is the current state of quantum computing?  ")
        assert req.question == "What is the current state of quantum computing?"
        assert req.metadata is None

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError) as exc:
            ResearchRequest(question="")
        assert "question cannot be empty or whitespace only" in str(exc.value)

    def test_whitespace_question_raises(self):
        with pytest.raises(ValidationError) as exc:
            ResearchRequest(question="   \n\t  ")
        assert "question cannot be empty or whitespace only" in str(exc.value)

    def test_too_short_question_raises(self):
        with pytest.raises(ValidationError):
            ResearchRequest(question="Hi")

    def test_forbid_extra_fields(self):
        with pytest.raises(ValidationError):
            ResearchRequest(question="Valid question", unauthorized_field="malicious")


class TestResearchPlan:
    def test_valid_plan(self):
        plan = ResearchPlan(
            objective="Evaluate vector databases for high-concurrency search",
            subquestions=["What are the memory requirements?", "How does clustering work?"],
            search_queries=["vector db memory usage", "vector db horizontal clustering"],
            tools=["web_search", "fetch_and_extract"],
            success_criteria=["Find at least 2 independent benchmark reports"],
        )
        assert len(plan.subquestions) == 2
        assert "web_search" in plan.tools

    def test_disallowed_tools_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ResearchPlan(
                objective="Valid objective",
                subquestions=["Subquestion 1"],
                search_queries=["Query 1"],
                tools=["web_search", "execute_arbitrary_code"],
                success_criteria=["Criteria 1"],
            )
        assert "Disallowed tools requested" in str(exc.value)

    def test_empty_subquestions_rejected(self):
        with pytest.raises(ValidationError):
            ResearchPlan(
                objective="Valid objective",
                subquestions=[],
                search_queries=["Query 1"],
                tools=["web_search"],
                success_criteria=["Criteria 1"],
            )

    def test_whitespace_item_in_subquestions_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ResearchPlan(
                objective="Valid objective",
                subquestions=["Valid item", "   "],
                search_queries=["Query 1"],
                tools=["web_search"],
                success_criteria=["Criteria 1"],
            )
        assert "cannot be empty or whitespace" in str(exc.value)


class TestSearchResult:
    def test_valid_search_result(self):
        res = SearchResult(
            title="Vector Database Benchmark 2026",
            url="https://example.org/benchmarks/vector-db",
            snippet="A comprehensive performance evaluation of vector engines.",
            domain="example.org",
        )
        assert res.domain == "example.org"
        assert res.url == "https://example.org/benchmarks/vector-db"

    def test_invalid_url_scheme_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SearchResult(
                title="Local file",
                url="file:///etc/passwd",
                snippet="Sensitive file content",
            )
        assert "URL must have http or https scheme" in str(exc.value)

    def test_javascript_scheme_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SearchResult(
                title="XSS Attempt",
                url="javascript:alert(1)",
                snippet="Malicious script",
            )
        assert "URL must have http or https scheme" in str(exc.value)


class TestSourceDocument:
    def test_valid_source_document(self):
        doc = SourceDocument(
            source_id="S1",
            title="Official Documentation",
            url="https://docs.example.com/guide",
            domain="docs.example.com",
            content="Detailed architecture explanation...",
            quality_score=0.85,
            extraction_status=ExtractionStatus.SUCCESS,
        )
        assert doc.source_id == "S1"
        assert doc.quality_score == 0.85

    def test_invalid_source_id_format(self):
        with pytest.raises(ValidationError):
            SourceDocument(
                source_id="DOC-99",  # Must match ^S\d+$
                title="Doc",
                url="https://example.com",
                domain="example.com",
                content="Text",
                quality_score=0.5,
            )

    def test_quality_score_bounds(self):
        # Must be between 0.0 and 1.0
        with pytest.raises(ValidationError):
            SourceDocument(
                source_id="S1",
                title="Doc",
                url="https://example.com",
                domain="example.com",
                content="Text",
                quality_score=1.5,
            )

        with pytest.raises(ValidationError):
            SourceDocument(
                source_id="S1",
                title="Doc",
                url="https://example.com",
                domain="example.com",
                content="Text",
                quality_score=-0.1,
            )


class TestFindingAndResearchAnswer:
    def test_valid_finding(self):
        f = Finding(
            claim="Vector index HNSW scales logarithmically with node count.",
            explanation="Benchmarked under 10M vectors.",
            source_ids=["S1", "S2"],
        )
        assert f.source_ids == ["S1", "S2"]

    def test_valid_research_answer(self):
        ans = ResearchAnswer(
            user_question="Compare vector databases",
            short_answer="Vector DB A offers superior index latency while B is easier to operate.",
            key_findings=[
                Finding(claim="DB A latency is lower", source_ids=["S1"]),
            ],
            sources_used=["S1"],
            confidence=ConfidenceLevel.HIGH,
            assumptions=["Workload is 95% read-heavy."],
            limitations=["No multi-region benchmarks were performed."],
            suggested_next_steps=["Conduct internal load testing."],
            status=ResearchAnswerStatus.COMPLETED,
        )
        assert ans.confidence == ConfidenceLevel.HIGH
        assert ans.status == ResearchAnswerStatus.COMPLETED
        assert len(ans.key_findings) == 1

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValidationError):
            ResearchAnswer(
                user_question="Valid question",
                short_answer="Valid short answer",
                confidence="95%",  # Must be high, medium, or low
            )


class TestExecutionEvent:
    def test_execution_event_creation(self):
        evt = ExecutionEvent(
            run_id="run-1234",
            stage=WorkflowStage.SEARCHING,
            event_type="query_executed",
            duration_ms=250.5,
            success=True,
            error_category=None,
            message="Search returned 5 results",
        )
        assert evt.run_id == "run-1234"
        assert evt.stage == WorkflowStage.SEARCHING
        assert evt.timestamp is not None
        assert evt.duration_ms == 250.5
