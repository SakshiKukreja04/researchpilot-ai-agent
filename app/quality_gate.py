"""Output Quality Gate and Claim Verification for ResearchPilot.

Enforces deterministic citation checking, real URL registry mapping, rule-based
confidence policy, low-coverage warnings, and an optional bounded claim-support verifier.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.llm import GroqClientWrapper
from app.schemas import (
    ClaimSupportStatus,
    ClaimVerificationResult,
    ConfidenceLevel,
    Finding,
    ResearchAnswer,
    ResearchAnswerStatus,
    ResearchContext,
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
) -> ConfidenceLevel:
    """Compute rule-based confidence from observable evidence metrics.

    Rules:
    - HIGH: coverage >= 0.70, at least 2 accepted sources, avg quality >= 0.65,
      and 0 unsupported claims.
    - MEDIUM: coverage >= 0.40, at least 1 accepted source, and <= 1 unsupported claim.
    - LOW: coverage < 0.40, 0 accepted sources, or multiple unsupported claims.
    """
    coverage_ratio = context.coverage.coverage_ratio
    sources_count = len(context.sources)

    if sources_count == 0 or coverage_ratio < 0.40 or unsupported_claims_count > 1:
        return ConfidenceLevel.LOW

    avg_quality = (
        sum(s.quality_score for s in context.sources) / sources_count
        if sources_count > 0
        else 0.0
    )

    if (
        coverage_ratio >= 0.70
        and sources_count >= 2
        and avg_quality >= 0.65
        and unsupported_claims_count == 0
    ):
        return ConfidenceLevel.HIGH

    return ConfidenceLevel.MEDIUM


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
    known_source_registry = {doc.source_id: doc for doc in context.sources}

    # 1. Deterministic Citation Validation
    validated_findings: list[Finding] = []
    unsupported_count = 0

    for finding in answer.key_findings:
        valid_sids = []
        for sid in finding.source_ids:
            if sid in known_source_registry:
                valid_sids.append(sid)
            else:
                warnings.append(
                    f"Citation '{sid}' in finding '{finding.claim[:40]}...' was not found in retrieved evidence and was removed."
                )

        finding_support = finding.support_status
        if not valid_sids:
            warnings.append(
                f"Finding '{finding.claim[:40]}...' has no valid citations; marked ungrounded."
            )
            finding_support = ClaimSupportStatus.UNSUPPORTED
            unsupported_count += 1

        # 2. Claim-Support Verification (Optional/Targeted)
        if verify_claims and client is not None and valid_sids:
            cited_excerpts = [known_source_registry[sid].content for sid in valid_sids]
            ver_res = verify_claim_support(finding.claim, cited_excerpts, client)
            finding_support = ver_res.status
            if ver_res.status == ClaimSupportStatus.UNSUPPORTED:
                warnings.append(
                    f"Claim verification failed for '{finding.claim[:40]}...': {ver_res.explanation}"
                )
                unsupported_count += 1
            elif ver_res.status == ClaimSupportStatus.PARTIALLY_SUPPORTED:
                warnings.append(
                    f"Claim only partially supported: '{finding.claim[:40]}...': {ver_res.explanation}"
                )

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

    # 4. Enforce Deterministic Confidence Policy
    rule_confidence = (
        determine_rule_based_confidence(
            context,
            unsupported_claims_count=unsupported_count,
        )
        if answer.status == ResearchAnswerStatus.COMPLETED
        else ConfidenceLevel.LOW
    )
    if answer.confidence != rule_confidence:
        warnings.append(
            f"Confidence adjusted from '{answer.confidence.value}' to rule-governed '{rule_confidence.value}' "
            f"based on coverage ({context.coverage.coverage_ratio:.2f}) and evidence metrics."
        )

    # 5. Low-Coverage Warning & Limitations Adjustment
    updated_limitations = list(answer.limitations)
    if context.coverage.coverage_ratio < 0.60:
        cov_warning = (
            f"Coverage note: Evidence coverage is partial ({context.coverage.coverage_ratio:.0%}); "
            f"{context.coverage.unsupported_count} subquestion(s) lacked sufficient evidence."
        )
        if not any("coverage note" in lim.lower() for lim in updated_limitations):
            updated_limitations.append(cov_warning)
            warnings.append(cov_warning)

    hardened_answer = answer.model_copy(
        update={
            "key_findings": validated_findings,
            "sources_used": valid_sources_used,
            "source_url_map": trusted_url_map,
            "confidence": rule_confidence,
            "limitations": updated_limitations[:15],
            "status": answer.status,
        }
    )

    return hardened_answer, warnings
