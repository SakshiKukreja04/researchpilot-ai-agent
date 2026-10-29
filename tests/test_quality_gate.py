"""Unit tests for the quality gate and claim-support verification module."""

import json
from unittest.mock import MagicMock

import pytest

from app.llm import GroqClientWrapper, LLMTimeoutError
from app.quality_gate import (
    determine_rule_based_confidence,
    run_quality_gate,
    verify_claim_support,
)
from app.schemas import (
    ClaimSupportStatus,
    ClaimVerificationResult,
    ConfidenceLevel,
    CoverageReport,
    ExtractionStatus,
    Finding,
    ResearchAnswer,
    ResearchAnswerStatus,
    ResearchContext,
    ResearchPlan,
    ResearchRequest,
    SourceDocument,
)


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================


def _make_context(coverage_ratio: float = 0.85, num_sources: int = 2) -> ResearchContext:
    sources = [
        SourceDocument(
            source_id=f"src_{i}{i}{i}{i}{i}{i}{i}{i}{i}{i}{i}{i}",
            url=f"https://docs.example.com/p{i}",
            title=f"Source {i}",
            domain="docs.example.com",
            content=f"Substantive technical content for source {i}. " * 15,
            quality_score=0.80,
            extraction_status=ExtractionStatus.SUCCESS,
        )
        for i in range(1, num_sources + 1)
    ]
    cov = CoverageReport(
        subquestion_coverage={f"sq{i}": "supported" for i in range(1, 4)},
        supported_count=3,
        partially_supported_count=0,
        unsupported_count=0,
        coverage_ratio=coverage_ratio,
        is_adequate=coverage_ratio >= 0.60,
    )
    return ResearchContext(
        run_id="run_qg_test",
        request=ResearchRequest(question="Compare vector databases"),
        plan=ResearchPlan(
            objective="Compare vector databases",
            subquestions=["sq1", "sq2", "sq3"],
            search_queries=["q1"],
            tools=["web_search"],
            success_criteria=["c1"],
        ),
        sources=sources,
        rejected_sources=[],
        coverage=cov,
        trace=[],
        status=ResearchAnswerStatus.COMPLETED,
    )


def _make_answer(
    findings: list[Finding] | None = None,
    sources_used: list[str] | None = None,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    limitations: list[str] | None = None,
) -> ResearchAnswer:
    findings = findings or [
        Finding(
            claim="HNSW index requires RAM proportional to vector dimension.",
            explanation="Explanation text.",
            source_ids=["src_111111111111"],
        ),
    ]
    if sources_used is None:
        sources_used = ["src_111111111111"]
    return ResearchAnswer(
        user_question="Compare vector databases",
        short_answer="Short executive summary.",
        key_findings=findings,
        sources_used=sources_used,
        source_url_map={},
        confidence=confidence,
        assumptions=[],
        limitations=limitations or [],
        suggested_next_steps=[],
        status=ResearchAnswerStatus.COMPLETED,
    )


# ==============================================================================
# Citation Validation Tests
# ==============================================================================


def test_unknown_citation_ids_are_stripped_and_warned():
    context = _make_context(num_sources=2)  # Has src_111111111111 and src_222222222222

    # Finding references an invented citation 'src_hallucinated99'
    finding_with_hallucination = Finding(
        claim="Invented fact claim",
        source_ids=["src_111111111111", "src_hallucinated99"],
    )
    answer = _make_answer(
        findings=[finding_with_hallucination],
        sources_used=["src_111111111111", "src_hallucinated99"],
    )

    hardened_answer, warnings = run_quality_gate(answer, context, verify_claims=False)

    # Hallucinated ID removed from finding and sources_used
    finding = hardened_answer.key_findings[0]
    assert "src_hallucinated99" not in finding.source_ids
    assert "src_111111111111" in finding.source_ids
    assert "src_hallucinated99" not in hardened_answer.sources_used
    assert any("not found in retrieved evidence" in w for w in warnings)


def test_correct_source_url_mapping_enforced():
    context = _make_context(num_sources=2)
    answer = _make_answer(sources_used=["src_111111111111", "src_222222222222"])

    hardened_answer, _ = run_quality_gate(answer, context, verify_claims=False)

    assert hardened_answer.source_url_map["src_111111111111"] == "https://docs.example.com/p1"
    assert hardened_answer.source_url_map["src_222222222222"] == "https://docs.example.com/p2"


def test_sources_used_includes_valid_finding_citations():
    context = _make_context(num_sources=2)
    answer = _make_answer(
        findings=[
            Finding(
                claim="A source-backed claim",
                source_ids=["src_222222222222"],
            )
        ],
        sources_used=[],
    )

    hardened_answer, _ = run_quality_gate(answer, context, verify_claims=False)

    assert hardened_answer.sources_used == ["src_222222222222"]
    assert hardened_answer.source_url_map["src_222222222222"] == "https://docs.example.com/p2"


# ==============================================================================
# Deterministic Confidence Policy Tests
# ==============================================================================


def test_confidence_policy_downgrades_high_to_medium_on_low_coverage():
    # Context coverage is only 0.45 (< 0.70 required for HIGH)
    context = _make_context(coverage_ratio=0.45, num_sources=2)
    answer = _make_answer(confidence=ConfidenceLevel.HIGH)

    hardened_answer, warnings = run_quality_gate(answer, context, verify_claims=False)

    # Downgraded to medium deterministically
    assert hardened_answer.confidence == ConfidenceLevel.MEDIUM
    assert any("Confidence adjusted" in w for w in warnings)


def test_confidence_policy_downgrades_to_low_when_sources_empty():
    context = _make_context(coverage_ratio=0.0, num_sources=0)
    answer = _make_answer(confidence=ConfidenceLevel.HIGH)

    conf = determine_rule_based_confidence(context)
    assert conf == ConfidenceLevel.LOW


def test_low_coverage_appends_limitation_warning():
    context = _make_context(coverage_ratio=0.50, num_sources=1)
    answer = _make_answer(limitations=["Existing limitation"])

    hardened_answer, warnings = run_quality_gate(answer, context, verify_claims=False)

    assert any("Coverage note" in lim for lim in hardened_answer.limitations)
    assert any("Coverage note" in w for w in warnings)


# ==============================================================================
# Claim-Support Verification Tests
# ==============================================================================


def test_verify_claim_support_successful_supported():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.chat_completion.return_value = json.dumps({
        "claim": "Milvus uses HNSW indexing",
        "status": "supported",
        "explanation": "Excerpt explicitly mentions HNSW indexing.",
    })

    result = verify_claim_support(
        "Milvus uses HNSW indexing",
        ["Milvus supports HNSW and IVF_FLAT indexing algorithms."],
        mock_client,
    )

    assert isinstance(result, ClaimVerificationResult)
    assert result.status == ClaimSupportStatus.SUPPORTED
    assert "HNSW" in result.explanation


def test_verify_claim_support_unsupported_claim():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.chat_completion.return_value = json.dumps({
        "claim": "Milvus runs on quantum optical processors",
        "status": "unsupported",
        "explanation": "Excerpts do not mention optical quantum hardware.",
    })

    result = verify_claim_support(
        "Milvus runs on quantum optical processors",
        ["Milvus is designed for x86 and ARM CPU architectures."],
        mock_client,
    )

    assert result.status == ClaimSupportStatus.UNSUPPORTED


def test_verify_claim_support_handles_timeout_gracefully():
    mock_client = MagicMock(spec=GroqClientWrapper)
    mock_client.chat_completion.side_effect = LLMTimeoutError("Verifier call timed out.")

    result = verify_claim_support(
        "Some claim",
        ["Some excerpt"],
        mock_client,
    )

    # Returns UNVERIFIED without crashing
    assert result.status == ClaimSupportStatus.UNVERIFIED
    assert "unavailable" in result.explanation.lower()


def test_quality_gate_with_claim_verification_annotates_findings():
    context = _make_context(num_sources=1)
    mock_client = MagicMock(spec=GroqClientWrapper)
    # Verifier reports supported for the claim
    mock_client.chat_completion.return_value = json.dumps({
        "claim": "Claim text",
        "status": "supported",
        "explanation": "Supported by excerpt.",
    })

    answer = _make_answer(findings=[
        Finding(claim="Claim text", source_ids=["src_111111111111"]),
    ])

    hardened_answer, warnings = run_quality_gate(
        answer,
        context,
        client=mock_client,
        verify_claims=True,
    )

    assert hardened_answer.key_findings[0].support_status == ClaimSupportStatus.SUPPORTED
