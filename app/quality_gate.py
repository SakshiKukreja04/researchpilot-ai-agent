"""Output Quality Gate and Claim Verification for ResearchPilot.

Enforces deterministic citation checking, real URL registry mapping, rule-based
confidence policy, low-coverage warnings, and an optional bounded claim-support verifier.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional
from urllib.parse import urlparse

from app.llm import GroqClientWrapper
from app.evaluator import evaluate_answer
from app.schemas import (
    ClaimSupportStatus,
    ClaimVerificationResult,
    ConfidenceLevel,
    Finding,
    ResearchAnswer,
    ResearchAnswerStatus,
    ResearchContext,
    SourceType,
    ReasonCode,
    StatusReason,
)

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are a Strict Factual Claim Verifier for ResearchPilot.
Your task is to verify whether an individual factual claim is supported by the provided source excerpts.

Options for "status":
- "supported": The claim is directly and unambiguously supported by the excerpts.
- "partially_supported": The claim is partially supported, but contains details, exaggerations, or numbers not present in the excerpts.
- "unsupported": The claim is not supported, contradicted, or absent from the excerpts.

Respond strictly with valid JSON matching:
{
  "claim": "The exact claim",
  "status": "supported" | "partially_supported" | "unsupported",
  "explanation": "Brief 1-2 sentence evidence-based explanation"
}
"""


# ==============================================================================
# Claim Support Verifier
# ==============================================================================


def verify_claim_support(
    claim: str,
    source_excerpts: list[str],
    client: GroqClientWrapper,
) -> ClaimVerificationResult:
    """Verify whether a single claim is supported by its cited source excerpts.

    Treated strictly as a fallible heuristic. If verification times out or fails,
    returns an UNVERIFIED result without crashing the quality gate.
    """
    if not source_excerpts or not any(s.strip() for s in source_excerpts):
        return ClaimVerificationResult(
            claim=claim,
            status=ClaimSupportStatus.UNSUPPORTED,
            explanation="No source excerpts provided for verification.",
        )

    joined_excerpts = "\n\n".join(
        f"--- EXCERPT {i+1} ---\n{s[:1500].strip()}\n--- END EXCERPT ---"
        for i, s in enumerate(source_excerpts[:3])
    )

    user_prompt = (
        f"CLAIM TO VERIFY:\n{claim}\n\n"
        f"CITED SOURCE EXCERPTS:\n{joined_excerpts}\n\n"
        "Evaluate whether the claim is supported, partially_supported, or unsupported."
    )

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_text = client.chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=10.0,
        )
        data = json.loads(raw_text)
        raw_status = str(data.get("status", "unsupported")).lower()
        if raw_status in {"supported", "partially_supported", "unsupported"}:
            status = ClaimSupportStatus(raw_status)
        else:
            status = ClaimSupportStatus.UNVERIFIED

        return ClaimVerificationResult(
            claim=claim,
            status=status,
            explanation=str(data.get("explanation", "Verification completed.")),
        )

    except Exception as err:
        logger.warning("Claim verifier call failed or timed out: %s", err)
        return ClaimVerificationResult(
            claim=claim,
            status=ClaimSupportStatus.UNVERIFIED,
            explanation=f"Verifier unavailable ({type(err).__name__}).",
        )


# ==============================================================================
# Deterministic Confidence Policy
# ==============================================================================


def determine_rule_based_confidence(
    context: ResearchContext,
    unsupported_claims_count: int = 0,
    *,
    answer_status: ResearchAnswerStatus = ResearchAnswerStatus.COMPLETED,
    grounding_rate: float = 1.0,
    citation_coverage_rate: float = 1.0,
    has_ungrounded_short_answer: bool = False,
) -> ConfidenceLevel:
    """Compute rule-based confidence from observable evidence metrics.

    Rules:
    - HIGH: coverage >= 0.70, at least 2 accepted sources, avg quality >= 0.65,
      all findings cited and grounded, no ungrounded short-answer sentences, and
      0 unsupported claims. Partial answers are capped at MEDIUM.
    - MEDIUM: coverage >= 0.40, at least 1 accepted source, and no severe
      grounding or unsupported-claim failures.
    - LOW: failed/empty synthesis, coverage < 0.40, no sources, or multiple
      unsupported claims.
    """
    coverage_ratio = context.coverage.coverage_ratio
    sources_count = len(context.sources)

    if (
        answer_status in {ResearchAnswerStatus.FAILED, ResearchAnswerStatus.INSUFFICIENT_EVIDENCE}
        or sources_count == 0
        or coverage_ratio < 0.40
        or unsupported_claims_count > 1
        or grounding_rate < 0.50
        or citation_coverage_rate < 0.50
    ):
        return ConfidenceLevel.LOW

    avg_quality = (
        sum(s.quality_score for s in context.sources) / sources_count
        if sources_count > 0
        else 0.0
    )

    source_mix = [source.source_type for source in context.sources]
    primary_count = sum(
        source_type
        in {
            SourceType.OFFICIAL_DOCUMENTATION,
            SourceType.OFFICIAL_COMPANY,
            SourceType.ACADEMIC,
        }
        for source_type in source_mix
    )
    benchmark_claims = any(
        term in context.request.question.lower()
        for term in ("benchmark", "latency", "throughput", "performance")
    )
    source_quality_cap_applies = (
        primary_count < max(1, len(source_mix) // 2)
        or any(source_type == SourceType.VENDOR_BLOG for source_type in source_mix)
        or benchmark_claims
    )

    if (
        coverage_ratio >= 0.70
        and sources_count >= 2
        and avg_quality >= 0.65
        and unsupported_claims_count == 0
        and grounding_rate >= 0.80
        and citation_coverage_rate == 1.0
        and not has_ungrounded_short_answer
    ):
        if source_quality_cap_applies or answer_status == ResearchAnswerStatus.PARTIAL:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.HIGH

    return ConfidenceLevel.MEDIUM


def _requested_entity_count(question: str) -> Optional[int]:
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"\b(?:find|list|identify|name|top)\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        question.lower(),
    )
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else number_words[value]


def _named_entities(context: ResearchContext) -> list[str]:
    text = " ".join(
        [context.request.question, context.plan.objective, *context.plan.search_queries]
    )
    known = ("Milvus", "Weaviate", "Qdrant", "pgvector", "Pinecone", "PostgreSQL", "SQLite")
    found: list[str] = []
    for entity in known:
        if re.search(rf"\b{re.escape(entity)}\b", text, re.IGNORECASE):
            found.append(entity)
    generic = {
        "compare", "what", "find", "top", "best", "indian", "b2b", "saas",
        "HR", "RAG", "AI", "SaaS", "Indian",
    }
    question_text = context.request.question
    for candidate in re.findall(r"\b[A-Z][A-Za-z0-9.-]{2,}\b", question_text):
        if (
            candidate not in found
            and candidate.lower() not in {item.lower() for item in generic}
            and (
                re.search(r"\bcompare\b", context.request.question, re.IGNORECASE)
                or _requested_entity_count(context.request.question) is not None
            )
        ):
            found.append(candidate)
    return found[:10]


def _entity_coverage(context: ResearchContext) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for entity in _named_entities(context):
        aliases = {entity.lower()}
        if entity.lower() == "pgvector":
            aliases.add("pg vector")
        source_ids = [
            source.source_id
            for source in context.sources
            if any(
                alias in f"{source.title} {source.url} {source.content}".lower()
                for alias in aliases
            )
        ]
        result[entity] = {"supported": bool(source_ids), "source_ids": source_ids}
    return result


def _reason(code: ReasonCode, message: str) -> StatusReason:
    return StatusReason(code=code, message=message)


# ==============================================================================
# Quality Gate Validation Function
# ==============================================================================


def run_quality_gate(
    answer: ResearchAnswer,
    context: ResearchContext,
    client: Optional[GroqClientWrapper] = None,
    verify_claims: bool = False,
) -> tuple[ResearchAnswer, list[str]]:
    """Validate and harden a ResearchAnswer against evidence provenance and quality rules.

    Args:
        answer: Draft ResearchAnswer from the synthesizer.
        context: ResearchContext containing accepted sources and coverage.
        client: Optional GroqClientWrapper for claim verification.
        verify_claims: Whether to invoke the LLM claim-support verifier on key findings.

    Returns:
        (validated_and_hardened_answer, list_of_warnings)
    """
    warnings: list[str] = []
    status_reasons: list[StatusReason] = []
    known_source_registry = {doc.source_id: doc for doc in context.sources}

    # 1. Deterministic Citation Validation
    validated_findings: list[Finding] = []
    unsupported_count = 0

    for finding in answer.key_findings:
        finding_unsupported = False
        valid_sids = []
        for sid in finding.source_ids:
            if sid in known_source_registry:
                valid_sids.append(sid)
            else:
                warnings.append(
                    f"Citation '{sid}' in finding '{finding.claim[:40]}...' was not found in retrieved evidence and was removed."
                )

        finding_support = finding.support_status
        if re.search(r"\b(implied|inferred|not in (?:the )?sources)\b", finding.explanation or "", re.IGNORECASE):
            finding_support = ClaimSupportStatus.UNSUPPORTED
            warnings.append(f"Finding marked unsupported because its explanation is not source-grounded: {finding.claim[:80]}")
            finding_unsupported = True
        if not valid_sids:
            warnings.append(
                f"Finding '{finding.claim[:40]}...' has no valid citations; marked ungrounded."
            )
            finding_support = ClaimSupportStatus.UNSUPPORTED
            finding_unsupported = True

        # 2. Claim-Support Verification (Optional/Targeted)
        if verify_claims and client is not None and valid_sids:
            cited_excerpts = [known_source_registry[sid].content for sid in valid_sids]
            ver_res = verify_claim_support(finding.claim, cited_excerpts, client)
            finding_support = ver_res.status
            if ver_res.status == ClaimSupportStatus.UNSUPPORTED:
                warnings.append(
                    f"Claim verification failed for '{finding.claim[:40]}...': {ver_res.explanation}"
                )
                finding_unsupported = True
            elif ver_res.status == ClaimSupportStatus.PARTIALLY_SUPPORTED:
                warnings.append(
                    f"Claim only partially supported: '{finding.claim[:40]}...': {ver_res.explanation}"
                )

        if finding_support == ClaimSupportStatus.UNSUPPORTED:
            finding_unsupported = True
        if finding_unsupported:
            unsupported_count += 1

        validated_findings.append(
            finding.model_copy(
                update={"source_ids": valid_sids, "support_status": finding_support}
            )
        )

    # Validate sources_used list
    cited_source_ids = [
        sid
        for finding in validated_findings
        for sid in finding.source_ids
        if finding.support_status != ClaimSupportStatus.UNSUPPORTED
    ]
    valid_sources_used: list[str] = []
    for sid in [*answer.sources_used, *cited_source_ids]:
        if sid in known_source_registry and sid not in valid_sources_used:
            valid_sources_used.append(sid)
    if len(valid_sources_used) != len(answer.sources_used):
        dropped = set(answer.sources_used) - set(valid_sources_used)
        if dropped:
            warnings.append(f"Dropped unknown source IDs from sources_used: {dropped}")

    # 3. Source URL Registry Mapping
    trusted_url_map = {
        sid: known_source_registry[sid].url
        for sid in valid_sources_used
        if sid in known_source_registry
    }
    for f in validated_findings:
        for sid in f.source_ids:
            if sid in known_source_registry and sid not in trusted_url_map:
                trusted_url_map[sid] = known_source_registry[sid].url

    source_type_mix: dict[str, int] = {}
    official_domains = {
        "milvus.io",
        "weaviate.io",
        "qdrant.tech",
    }
    official_repositories = {
        ("milvus-io", "milvus"),
        ("weaviate", "weaviate"),
        ("qdrant", "qdrant"),
    }
    typed_sources = []
    for source in context.sources:
        updated_type = source.source_type
        parsed_url = urlparse(source.url)
        host = (parsed_url.hostname or "").lower().removeprefix("www.")
        path_parts = [part.lower() for part in parsed_url.path.split("/") if part]
        is_official_domain = any(
            host == domain or host.endswith(f".{domain}")
            for domain in official_domains
        )
        if (
            host == "github.com"
            and len(path_parts) >= 2
            and (path_parts[0], path_parts[1]) in official_repositories
        ):
            updated_type = SourceType.OFFICIAL_REPO
        elif is_official_domain and updated_type not in {
            SourceType.OFFICIAL_DOCUMENTATION,
            SourceType.OFFICIAL_REPO,
        }:
            updated_type = SourceType.OFFICIAL_COMPANY
        if updated_type != source.source_type:
            source = source.model_copy(update={"source_type": updated_type})
        typed_sources.append(source)
        key = source.source_type.value
        source_type_mix[key] = source_type_mix.get(key, 0) + 1
    object.__setattr__(context, "sources", typed_sources)
    if source_type_mix:
        warnings.append(
            "Source mix: "
            + ", ".join(f"{key}={value}" for key, value in sorted(source_type_mix.items()))
        )

    entity_coverage = _entity_coverage(context)
    object.__setattr__(context, "entity_coverage", entity_coverage)
    missing_entities = [
        entity for entity, item in entity_coverage.items() if not item["supported"]
    ]
    if missing_entities:
        status_reasons.append(
            _reason(
                ReasonCode.ENTITY_NO_EVIDENCE,
                f"No evidence for: {', '.join(missing_entities)}",
            )
        )
        warnings.append(f"No evidence for: {', '.join(missing_entities)}")
        object.__setattr__(context, "coverage", context.coverage.model_copy(
            update={
                "coverage_ratio": min(
                    context.coverage.coverage_ratio,
                    max(0.0, (len(entity_coverage) - len(missing_entities)) / max(1, len(entity_coverage))),
                ),
                "is_adequate": False,
            }
        ))

    requested_count = _requested_entity_count(context.request.question)
    entity_warning = False
    if requested_count is not None:
        distinct_entities = {entity.strip().lower() for entity in answer.entities if entity.strip()}
        if len(distinct_entities) < requested_count:
            entity_warning = True
            warnings.append(
                f"Entity-count check: requested {requested_count} items but the answer identified "
                f"{len(distinct_entities)} distinct entities."
            )
            status_reasons.append(
                _reason(
                    ReasonCode.ENTITY_COUNT_SHORT,
                    f"Requested {requested_count} entities but only {len(distinct_entities)} were identified.",
                )
            )

    if context.coverage.coverage_ratio < 0.60:
        status_reasons.append(
            _reason(
                ReasonCode.LOW_COVERAGE,
                f"Evidence coverage is {context.coverage.coverage_ratio:.0%}.",
            )
        )

    if unsupported_count:
        status_reasons.append(
            _reason(
                ReasonCode.UNSUPPORTED_FINDINGS,
                f"{unsupported_count} finding(s) were unsupported or lacked valid citations.",
            )
        )

    # Evaluate claims before assigning confidence so unsupported prose cannot
    # be hidden by high plan coverage or citation-ID validity alone.
    evaluation = evaluate_answer(answer, typed_sources)
    object.__setattr__(context, "evaluation", evaluation)

    # 4. Low-Coverage Warning & Limitations Adjustment
    updated_limitations = list(answer.limitations)
    if context.coverage.coverage_ratio < 0.60:
        cov_warning = (
            f"Coverage note: Evidence coverage is partial ({context.coverage.coverage_ratio:.0%}); "
            f"{context.coverage.unsupported_count} subquestion(s) lacked sufficient evidence."
        )
        if not any("coverage note" in lim.lower() for lim in updated_limitations):
            updated_limitations.append(cov_warning)
            warnings.append(cov_warning)

    if answer.status not in {
        ResearchAnswerStatus.FAILED,
        ResearchAnswerStatus.INSUFFICIENT_EVIDENCE,
    }:
        if not context.sources or context.coverage.coverage_ratio < 0.40:
            final_status = ResearchAnswerStatus.INSUFFICIENT_EVIDENCE
        elif (
            answer.status == ResearchAnswerStatus.PARTIAL
            or context.coverage.coverage_ratio < 0.60
            or entity_warning
            or missing_entities
        ):
            final_status = ResearchAnswerStatus.PARTIAL
        else:
            final_status = ResearchAnswerStatus.COMPLETED
    else:
        final_status = answer.status

    if final_status != answer.status:
        warnings.append(
            f"Status adjusted from '{answer.status.value}' to '{final_status.value}' "
            "by explicit evidence and synthesis thresholds."
        )

    # 5. Enforce confidence using the final status and claim-level evaluation.
    rule_confidence = determine_rule_based_confidence(
        context,
        unsupported_claims_count=unsupported_count,
        answer_status=final_status,
        grounding_rate=evaluation.grounding_rate,
        citation_coverage_rate=evaluation.citation_coverage_rate,
        has_ungrounded_short_answer=bool(evaluation.short_answer_ungrounded),
    )
    if answer.confidence != rule_confidence:
        warnings.append(
            f"Confidence adjusted from '{answer.confidence.value}' to rule-governed '{rule_confidence.value}'."
        )

    if evaluation and evaluation.short_answer_ungrounded:
        status_reasons.append(
            _reason(
                ReasonCode.UNGROUNDED_SHORT_ANSWER,
                f"{len(evaluation.short_answer_ungrounded)} short-answer sentence(s) lack finding support.",
            )
        )

    object.__setattr__(context, "status_reasons", status_reasons)
    hardened_answer = answer.model_copy(
        update={
            "key_findings": validated_findings,
            "sources_used": valid_sources_used,
            "source_url_map": trusted_url_map,
            "confidence": rule_confidence,
            "limitations": updated_limitations[:15],
            "status": final_status,
            "source_type_mix": source_type_mix,
            "status_reasons": status_reasons,
            "short_answer_ungrounded": evaluation.short_answer_ungrounded if evaluation else [],
        }
    )

    return hardened_answer, warnings
