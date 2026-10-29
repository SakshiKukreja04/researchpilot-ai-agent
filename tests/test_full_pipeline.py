"""Unit tests for the complete end-to-end research pipeline via orchestrator."""

from unittest.mock import MagicMock

import pytest

from app.llm import GroqClientWrapper
from app.orchestrator import run_full_pipeline
from app.schemas import (
    ConfidenceLevel,
    ExtractionStatus,
    ResearchAnswerStatus,
    ResearchContext,
    ResearchPlan,
    SearchResult,
    SourceDocument,
    WorkflowStage,
)


def _mock_plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Analyze vector databases for high-concurrency search",
        subquestions=[
            "What are the memory requirements?",
            "How does clustering work?",
        ],
        search_queries=[
            "vector db memory",
            "vector db clustering",
        ],
        tools=["web_search", "fetch_and_extract"],
        success_criteria=["Memory details", "Clustering details"],
    )


def _mock_doc(source_id: str, url: str, content: str) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        url=url,
        title="Vector Database Guide",
        domain="docs.example.com",
        content=content,
        quality_score=0.90,
        extraction_status=ExtractionStatus.SUCCESS,
    )


def test_full_pipeline_happy_path():
    plan_json = _mock_plan().model_dump_json()
    answer_json = {
        "user_question": "Compare vector databases for high-concurrency search",
        "short_answer": "Vector databases such as Milvus and Qdrant offer robust indexing and clustering architectures.",
        "key_findings": [
            {
                "claim": "HNSW index memory footprint scales roughly linearly with vectors.",
                "explanation": "Supported by benchmark documentation.",
                "source_ids": ["src_111122223333"],
            }
        ],
        "sources_used": ["src_111122223333"],
        "confidence": "high",
        "assumptions": ["Standard vector dimensions (128-1536d)."],
        "limitations": [],
        "suggested_next_steps": ["Run standalone memory profiler."],
        "status": "completed",
    }
    import json
    answer_str = json.dumps(answer_json)

    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "llama-3.3-70b-versatile"
    # First call: planner, Second call: synthesizer
    mock_client.chat_completion.side_effect = [plan_json, answer_str]

    mock_search = MagicMock(return_value=[
        SearchResult(
            title="Vector Memory Benchmarks",
            url="https://docs.example.com/memory",
            snippet="Memory requirements for HNSW under high concurrency.",
            domain="docs.example.com",
        ),
        SearchResult(
            title="Distributed Clustering Architecture",
            url="https://docs.example.com/clustering",
            snippet="Clustering and distributed nodes architecture.",
            domain="docs.example.com",
        ),
    ])

    mock_fetch = MagicMock(side_effect=[
        _mock_doc(
            "src_111122223333",
            "https://docs.example.com/memory",
            "Detailed benchmark of memory requirements for HNSW indexes and RAM footprint under high concurrency. Memory details." * 15,
        ),
        _mock_doc(
            "src_444455556666",
            "https://docs.example.com/clustering",
            "How distributed clustering architecture partitions, shards, and works with vector embeddings across clustered nodes. Clustering details." * 15,
        ),
    ])

    stages_recorded = []

    def on_progress(stage: WorkflowStage, msg: str):
        stages_recorded.append(stage)

    context = run_full_pipeline(
        "Compare vector databases for high-concurrency search",
        client=mock_client,
        search_tool_fn=mock_search,
        fetch_tool_fn=mock_fetch,
        progress_callback=on_progress,
    )

    assert isinstance(context, ResearchContext)
    assert context.status == ResearchAnswerStatus.COMPLETED
    assert context.answer is not None
    assert len(context.answer.key_findings) == 1
    assert context.answer.key_findings[0].source_ids == ["src_111122223333"]
    assert "src_111122223333" in context.answer.source_url_map
    assert context.answer.source_url_map["src_111122223333"] == "https://docs.example.com/memory"

    # Verify stage transitions were observed by progress callback
    assert WorkflowStage.RECEIVED in stages_recorded
    assert WorkflowStage.PLANNED in stages_recorded
    assert WorkflowStage.SEARCHING in stages_recorded
    assert WorkflowStage.RETRIEVING in stages_recorded
    assert WorkflowStage.FILTERING in stages_recorded
    assert WorkflowStage.SYNTHESIZING in stages_recorded
    assert WorkflowStage.VALIDATING in stages_recorded


def test_full_pipeline_empty_query():
    stages_recorded = []
    context = run_full_pipeline(
        "   ",
        progress_callback=lambda stage, msg: stages_recorded.append(stage),
    )
    assert context.status == ResearchAnswerStatus.FAILED
    assert context.answer is not None
    assert context.answer.status == ResearchAnswerStatus.FAILED
    assert "whitespace" in context.answer.short_answer


def test_full_pipeline_zero_search_results():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.model_name = "llama-3.3-70b-versatile"
    mock_client.chat_completion.return_value = _mock_plan().model_dump_json()

    mock_search = MagicMock(return_value=[])

    context = run_full_pipeline(
        "Obscure topic with no search results",
        client=mock_client,
        search_tool_fn=mock_search,
    )

    assert context.status == ResearchAnswerStatus.INSUFFICIENT_EVIDENCE
    assert context.answer is not None
    assert context.answer.status == ResearchAnswerStatus.INSUFFICIENT_EVIDENCE
    assert context.answer.confidence == ConfidenceLevel.LOW
    assert len(context.sources) == 0
