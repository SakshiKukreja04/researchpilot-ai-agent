"""Unit tests for evidence scoring heuristics, filtering, and coverage assessment."""

import pytest

from app.evidence import (
    calculate_authority_score,
    calculate_completeness_score,
    calculate_freshness_score,
    calculate_relevance_score,
    process_evidence,
    score_source_document,
)
from app.schemas import ExtractionStatus, ResearchPlan, SourceDocument


def _make_source(
    source_id: str = "src_112233445566",
    title: str = "Vector Search Documentation",
    domain: str = "docs.example.org",
    url: str = "https://docs.example.org/guide",
    content: str = "Detailed architectural explanation of vector search indexes, HNSW graphs, and nearest neighbor latency benchmarks.",
    published_date: str | None = "2026-01-10",
    quality_score: float = 1.0,
    extraction_status: ExtractionStatus = ExtractionStatus.SUCCESS,
) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        title=title,
        domain=domain,
        url=url,
        content=content,
        published_date=published_date,
        quality_score=quality_score,
        extraction_status=extraction_status,
    )


# ==============================================================================
# Scoring Heuristics Tests
# ==============================================================================


def test_relevance_scoring():
    doc = _make_source(
        title="Vector Database Benchmarks",
        content="We benchmarked Milvus, Qdrant, and Weaviate under high-concurrency workloads.",
    )
    score_high = calculate_relevance_score(doc, "vector database benchmarks milvus qdrant")
    score_low = calculate_relevance_score(doc, "astronomy telescope optics planet rings")

    assert score_high > 0.6
    assert score_low < 0.2


def test_authority_scoring_tiers():
    # Tier High (.edu, docs.*, github.com)
    assert calculate_authority_score("docs.milvus.io") >= 0.90
    assert calculate_authority_score("mit.edu") >= 0.90
    assert calculate_authority_score("github.com") >= 0.90

    # Tier Medium (Tech journalism, forums)
    assert calculate_authority_score("techcrunch.com") == 0.75
    assert calculate_authority_score("stackoverflow.com") == 0.75

    # Standard commercial / unknown
    assert calculate_authority_score("example.com") == 0.50

    # Low authority
    assert calculate_authority_score("quora.com") == 0.25
    assert calculate_authority_score("blogspot.com") == 0.25


def test_freshness_scoring_neutral_for_unknown_dates():
    doc_undated = _make_source(published_date=None)
    doc_current = _make_source(published_date="2026-02-15")
    doc_old = _make_source(published_date="2020-05-01")

    # When time-sensitive is False
    assert calculate_freshness_score(doc_undated, requires_fresh_evidence=False) >= 0.7

    # When time-sensitive is True
    # Unknown date MUST be neutral (0.50), not penalized to 0.0
    assert calculate_freshness_score(doc_undated, requires_fresh_evidence=True) == 0.50
    assert calculate_freshness_score(doc_current, requires_fresh_evidence=True) == 1.0
    assert calculate_freshness_score(doc_old, requires_fresh_evidence=True) <= 0.30


def test_completeness_scoring():
    assert calculate_completeness_score("Stub text") == 0.10
    assert calculate_completeness_score("A" * 300) == 0.40
    assert calculate_completeness_score("A" * 1500) == 0.80
    assert calculate_completeness_score("A" * 3000) == 1.0


def test_overall_quality_score_formula():
    doc = _make_source(
        title="Vector Database Documentation",
        domain="docs.qdrant.tech",
        content="Complete overview of vector search indexing and clustering algorithms. " * 30,
        published_date="2026-01-01",
    )
    score = score_source_document(doc, "vector search indexing", requires_fresh_evidence=True)
    # High relevance (~0.8), high authority (0.95), fresh (1.0), complete (1.0)
    assert score >= 0.85


# ==============================================================================
# Filtering and Rejection Tests
# ==============================================================================


def test_process_evidence_filters_low_quality():
    plan = ResearchPlan(
        objective="Analyze vector databases",
        subquestions=["What are the scaling mechanisms?"],
        search_queries=["vector db scaling"],
        tools=["web_search", "fetch_and_extract"],
        success_criteria=["Find scaling benchmarks"],
    )

    good_doc = _make_source(
        source_id="src_111111111111",
        title="Vector Database Scaling Guide",
        domain="docs.example.com",
        content="Substantive guide to horizontal clustering and sharding algorithms. " * 20,
    )

    bad_doc = _make_source(
        source_id="src_222222222222",
        title="Cooking recipes",
        domain="quora.com",
        content="Irrelevant text about baking cookies.",
    )

    accepted, rejected, coverage = process_evidence(
        [good_doc, bad_doc],
        plan,
        question="Analyze vector databases",
        quality_threshold=0.55,
    )

    assert len(accepted) == 1
    assert accepted[0].source_id == "src_111111111111"
    assert len(rejected) == 1
    assert rejected[0].source_id == "src_222222222222"
    assert "below threshold" in (rejected[0].rejection_reason or "")


def test_process_evidence_rejects_failed_and_duplicate_extractions():
    plan = ResearchPlan(
        objective="Analyze vector databases",
        subquestions=["What are the scaling mechanisms?"],
        search_queries=["vector db scaling"],
        tools=["web_search", "fetch_and_extract"],
        success_criteria=["Find scaling benchmarks"],
    )

    failed_doc = _make_source(
        source_id="src_333333333333",
        content="",
        extraction_status=ExtractionStatus.FAILED,
    )

    doc_a = _make_source(
        source_id="src_444444444444",
        url="https://docs.a.com/page",
        domain="docs.a.com",
        content="Identical benchmark text content repeated across mirrors.",
    )

    doc_b = _make_source(
        source_id="src_555555555555",
        url="https://docs.b.com/mirror",
        domain="docs.b.com",
        content="Identical benchmark text content repeated across mirrors.",
    )

    accepted, rejected, coverage = process_evidence(
        [failed_doc, doc_a, doc_b],
        plan,
        question="Analyze vector databases",
        quality_threshold=0.30,
    )

    # doc_a accepted, doc_b rejected as duplicate, failed_doc rejected as empty
    assert any("Duplicate content" in (r.rejection_reason or "") for r in rejected)
    assert any("Extraction failed" in (r.rejection_reason or "") for r in rejected)


def test_domain_diversity_enforced():
    plan = ResearchPlan(
        objective="Analyze vector databases",
        subquestions=["What are the scaling mechanisms?"],
        search_queries=["vector db scaling"],
        tools=["web_search", "fetch_and_extract"],
        success_criteria=["Find scaling benchmarks"],
    )

    # 3 docs from the exact same domain
    docs = [
        _make_source(
            source_id=f"src_{i}{i}{i}{i}{i}{i}{i}{i}{i}{i}{i}{i}",
            url=f"https://docs.single-domain.com/p{i}",
            domain="docs.single-domain.com",
            content=f"Different unique content piece {i} about vector search scaling. " * 20,
        )
        for i in range(1, 4)
    ]

    accepted, rejected, coverage = process_evidence(
        docs,
        plan,
        question="Analyze vector databases",
        quality_threshold=0.50,
        max_sources_per_domain=2,
    )

    # Only 2 accepted from single-domain.com, 1 rejected due to diversity cap
    assert len(accepted) == 2
    assert len(rejected) == 1
    assert "diversity cap" in (rejected[0].rejection_reason or "")


# ==============================================================================
# Coverage Assessment Tests
# ==============================================================================


def test_coverage_assessment_evaluates_subquestions():
    plan = ResearchPlan(
        objective="Compare vector databases",
        subquestions=[
            "What is the memory footprint and HNSW indexing overhead?",
            "How does distributed sharding work?",
            "What are the quantum computing optical algorithms?",  # Completely unsupported
        ],
        search_queries=["vector db memory", "vector db sharding"],
        tools=["web_search", "fetch_and_extract"],
        success_criteria=["Memory benchmarks", "Sharding architecture"],
    )

    source1 = _make_source(
        source_id="src_111111111111",
        title="Vector Memory Footprint",
        content="Detailed analysis of HNSW memory footprint and indexing vector overhead in RAM.",
    )
    source2 = _make_source(
        source_id="src_222222222222",
        title="Distributed Sharding Architecture",
        content="How distributed sharding partitions vectors across clustered nodes.",
    )

    accepted, rejected, coverage = process_evidence(
        [source1, source2],
        plan,
        question="Compare vector databases",
    )

    assert coverage.supported_count >= 2
    assert coverage.unsupported_count == 1
    assert any("quantum" in sq for sq in coverage.unsupported_subquestions)
    assert coverage.coverage_ratio > 0.60
