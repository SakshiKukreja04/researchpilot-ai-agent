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


def _answer_tokens(text: str) -> set[str]:
    """Tokenize prose while ignoring citation IDs appended by the synthesizer."""
    without_citations = re.sub(r"\([^)]*src_[a-f0-9]+[^)]*\)", " ", text, flags=re.IGNORECASE)
    without_citations = re.sub(r"\[[^]]*src_[a-f0-9]+[^]]*\]", " ", without_citations, flags=re.IGNORECASE)
    return _tokens(without_citations)


def _sentence_supported_by_one_finding(
    sentence: str,
    answer: ResearchAnswer,
    registry: dict[str, SourceDocument],
) -> bool:
    """Require one finding and its own cited excerpts to support a sentence.

    Comparing against the aggregate answer evidence is too permissive: unrelated
    findings can make an over-broad sentence appear lexically grounded.
    """
    sentence_tokens = _answer_tokens(sentence)
    if len(sentence_tokens) < 3:
        return True
    for finding in answer.key_findings:
        if finding.support_status in {"unsupported", "unverified"}:
            continue
        claim_tokens = _tokens(f"{finding.claim} {finding.explanation or ''}")
        evidence_tokens = _tokens(
            " ".join(
                registry[source_id].content
                for source_id in finding.source_ids
                if source_id in registry
            )
        )
        if not claim_tokens or not evidence_tokens:
            continue
        claim_overlap = len(sentence_tokens & claim_tokens) / len(sentence_tokens)
        evidence_overlap = len(sentence_tokens & evidence_tokens) / len(sentence_tokens)
        if claim_overlap >= 0.30 and evidence_overlap >= 0.20:
            return True
    return False


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
    cited_findings = sum(
        any(source_id in registry for source_id in finding.source_ids)
        for finding in answer.key_findings
    )
    uncited_finding_count = len(answer.key_findings) - cited_findings
    grounded = 0
    warnings: list[str] = []
    short_answer_ungrounded: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.short_answer.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        factual_sentence = len(_answer_tokens(sentence)) >= 3
        if factual_sentence and not _sentence_supported_by_one_finding(
            sentence, answer, registry
        ):
            short_answer_ungrounded.append(sentence)
            warnings.append(f"Ungrounded short-answer sentence: {sentence[:120]}")

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
    citation_coverage_rate = (
        cited_findings / len(answer.key_findings) if answer.key_findings else 0.0
    )
    if uncited_finding_count:
        warnings.append(
            f"{uncited_finding_count} finding(s) have no valid citations."
        )
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
        uncited_finding_count=uncited_finding_count,
        citation_coverage_rate=round(citation_coverage_rate, 3),
        grounded_finding_count=grounded,
        finding_count=len(answer.key_findings),
        grounding_rate=round(grounding_rate, 3),
        warnings=warnings,
        guardrail_triggered=guardrail,
        short_answer_ungrounded=short_answer_ungrounded,
    )
