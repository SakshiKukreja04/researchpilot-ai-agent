from app.evaluator import evaluate_answer
from app.schemas import (
    ConfidenceLevel,
    ExtractionStatus,
    Finding,
    ResearchAnswer,
    ResearchAnswerStatus,
    SourceDocument,
)


def test_short_answer_cannot_combine_unrelated_findings_into_broad_claim():
    sources = [
        SourceDocument(
            source_id="src_111111111111",
            title="Agent specialization",
            url="https://example.com/specialization",
            domain="example.com",
            content="Specialized agents can route tickets and retrieve knowledge.",
            quality_score=0.8,
            extraction_status=ExtractionStatus.SUCCESS,
        ),
        SourceDocument(
            source_id="src_222222222222",
            title="Coordination overhead",
            url="https://example.com/coordination",
            domain="example.com",
            content="Multiple agents require coordination and can increase operational complexity.",
            quality_score=0.8,
            extraction_status=ExtractionStatus.SUCCESS,
        ),
    ]
    answer = ResearchAnswer(
        user_question="Research multi-agent customer support",
        short_answer=(
            "Multi-agent systems provide faster resolution, higher accuracy, and better scalability."
        ),
        key_findings=[
            Finding(
                claim="Specialized agents can route tickets and retrieve knowledge.",
                explanation="The source describes task specialization.",
                source_ids=["src_111111111111"],
            ),
            Finding(
                claim="Multiple agents require coordination and can increase operational complexity.",
                explanation="The source describes coordination overhead.",
                source_ids=["src_222222222222"],
            ),
        ],
        sources_used=["src_111111111111", "src_222222222222"],
        confidence=ConfidenceLevel.MEDIUM,
        status=ResearchAnswerStatus.COMPLETED,
    )

    report = evaluate_answer(answer, sources)

    assert report.short_answer_ungrounded == [
        "Multi-agent systems provide faster resolution, higher accuracy, and better scalability."
    ]
    assert any("Ungrounded short-answer" in warning for warning in report.warnings)


def test_citation_validity_does_not_hide_uncited_findings():
    source = SourceDocument(
        source_id="src_111111111111",
        title="Vector database",
        url="https://example.com/vector-db",
        domain="example.com",
        content="A cited source about vector databases and retrieval.",
        quality_score=0.8,
        extraction_status=ExtractionStatus.SUCCESS,
    )
    answer = ResearchAnswer(
        user_question="Compare vector databases",
        short_answer="A vector database supports retrieval.",
        key_findings=[
            Finding(
                claim="A vector database supports retrieval.",
                source_ids=["src_111111111111"],
            ),
            Finding(claim="Licensing is permissive."),
        ],
        sources_used=["src_111111111111"],
        confidence=ConfidenceLevel.MEDIUM,
        status=ResearchAnswerStatus.COMPLETED,
    )

    report = evaluate_answer(answer, [source])

    assert report.citation_validity_rate == 1.0
    assert report.uncited_finding_count == 1
    assert report.citation_coverage_rate == 0.5
    assert any("1 finding(s) have no valid citations" in warning for warning in report.warnings)
