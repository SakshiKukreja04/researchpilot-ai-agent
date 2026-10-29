"""Deterministic evaluator and citation-grounding checks."""

from __future__ import annotations

import re

from app.schemas import AnswerEvaluation, ResearchAnswer, SourceDocument

_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "could", "from", "have", "into", "more", "most", "only", "that",
    "their", "these", "this", "those", "what", "which", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b[a-zA-Z0-9]{4,}\b", text.lower())
        if token not in _STOP_WORDS
    }


def evaluate_answer(
    answer: ResearchAnswer,
    sources: list[SourceDocument],
    low_confidence_threshold: float = 0.5,
) -> AnswerEvaluation:
    """Measure citation validity and lexical claim grounding.

    Lexical grounding is a conservative heuristic for review triage, not a
    factual entailment proof.
    """
    registry = {source.source_id: source for source in sources}
    citation_ids = [
        source_id
        for finding in answer.key_findings
        for source_id in finding.source_ids
    ]
    valid_ids = [source_id for source_id in citation_ids if source_id in registry]
    grounded = 0
    warnings: list[str] = []

    for finding in answer.key_findings:
        claim_tokens = _tokens(finding.claim)
        evidence_tokens = _tokens(
            " ".join(registry[source_id].content for source_id in finding.source_ids
                     if source_id in registry)
        )
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        if finding.source_ids and overlap >= 0.30:
            grounded += 1
        elif finding.source_ids:
            warnings.append(f"Low lexical support for claim: {finding.claim[:80]}")

    citation_rate = len(valid_ids) / len(citation_ids) if citation_ids else 0.0
    grounding_rate = (
        grounded / len(answer.key_findings) if answer.key_findings else 0.0
    )
    guardrail = answer.confidence.value == "low" or grounding_rate < low_confidence_threshold
    if guardrail:
        warnings.append("Low-confidence guardrail: present findings as provisional and request verification.")

    return AnswerEvaluation(
        citation_count=len(citation_ids),
        valid_citation_count=len(valid_ids),
        citation_validity_rate=round(citation_rate, 3),
        grounded_finding_count=grounded,
        finding_count=len(answer.key_findings),
        grounding_rate=round(grounding_rate, 3),
        warnings=warnings,
        guardrail_triggered=guardrail,
    )
