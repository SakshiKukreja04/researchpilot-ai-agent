from app.cache import TTLCache
from app.evaluator import evaluate_answer
from app.schemas import (
    ConfidenceLevel,
    Finding,
    ResearchAnswer,
    ResearchAnswerStatus,
    SourceDocument,
)


def _answer() -> ResearchAnswer:
    return ResearchAnswer(
        user_question="What is an HNSW index?",
        short_answer="An HNSW index supports approximate nearest-neighbor search.",
        key_findings=[
            Finding(
                claim="HNSW supports approximate nearest-neighbor search.",
                explanation="The cited documentation describes HNSW search.",
                source_ids=["src_abcdef12"],
            )
        ],
        sources_used=["src_abcdef12"],
        confidence=ConfidenceLevel.HIGH,
        assumptions=[],
        limitations=[],
        suggested_next_steps=[],
        status=ResearchAnswerStatus.COMPLETED,
    )


def _source() -> SourceDocument:
    return SourceDocument(
        source_id="src_abcdef12",
        url="https://example.com/hnsw",
        title="HNSW documentation",
        content="HNSW supports approximate nearest-neighbor search using a graph index.",
        domain="example.com",
        quality_score=0.9,
    )


def test_evaluator_reports_valid_citation_and_grounding():
    report = evaluate_answer(_answer(), [_source()])
    assert report.citation_validity_rate == 1.0
    assert report.grounding_rate == 1.0
    assert report.guardrail_triggered is False


def test_bounded_cache_replaces_expiring_capacity():
    cache = TTLCache[str](max_entries=1, ttl_seconds=60)
    cache.set("first", "value")
    cache.set("second", "value")
    assert cache.get("first") is None
    assert cache.get("second") == "value"
