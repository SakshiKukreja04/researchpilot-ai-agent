"""Pure presentation helpers for the Streamlit dashboard."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.schemas import ReasonCode, ResearchAnswerStatus, ResearchContext

REASON_LABELS = {
    ReasonCode.ENTITY_NO_EVIDENCE: "One or more named entities have no supporting source.",
    ReasonCode.LOW_COVERAGE: "Retrieved evidence does not cover enough of the research plan.",
    ReasonCode.SYNTHESIS_FALLBACK_USED: "The configured synthesis fallback model produced the answer.",
    ReasonCode.UNGROUNDED_SHORT_ANSWER: "A factual short-answer sentence lacks supporting findings.",
    ReasonCode.UNSUPPORTED_FINDINGS: "Some findings were not supported by cited evidence.",
    ReasonCode.VENDOR_SOURCES_DOMINATE: "Vendor sources dominate the accepted source set.",
    ReasonCode.BENCHMARK_CONFIDENCE_CAP: "Benchmark evidence limits confidence.",
    ReasonCode.DEADLINE_EXHAUSTED: "A stage deadline stopped additional retrieval.",
    ReasonCode.EARLY_STOP: "Retrieval stopped early because it was unproductive.",
    ReasonCode.ENTITY_COUNT_SHORT: "Fewer entities were supported than requested.",
    ReasonCode.FRESHNESS_WARNING: "Some evidence may be stale.",
}


def reason_message(code: ReasonCode | str, message: str | None = None) -> str:
    try:
        normalized = ReasonCode(code)
    except ValueError:
        return message or str(code)
    return message or REASON_LABELS.get(normalized, normalized.value)


def status_banner(context: ResearchContext) -> tuple[str, str, str]:
    reasons = {reason.code for reason in context.status_reasons}
    deadline = (
        ReasonCode.DEADLINE_EXHAUSTED in reasons
        or str(getattr(context, "reason_code", "")).startswith("deadline_exhausted")
    )
    if deadline:
        reason = next(
            (item.message for item in context.status_reasons if item.code == ReasonCode.DEADLINE_EXHAUSTED),
            "retrieval deadline reached",
        )
        return "warning", f"Stopped early: {reason}", "The pipeline returned the evidence collected before the deadline."
    if context.status == ResearchAnswerStatus.INSUFFICIENT_EVIDENCE:
        return "error", "Insufficient evidence", "No reliable evidence met the quality threshold."
    if context.status == ResearchAnswerStatus.PARTIAL:
        messages = [item.message for item in context.status_reasons[:2]]
        return "warning", "Partial result", " ".join(messages) or "Some evidence or quality requirements remain incomplete."
    if context.coverage.coverage_ratio < 1.0 or reasons & {
        ReasonCode.ENTITY_NO_EVIDENCE,
        ReasonCode.LOW_COVERAGE,
    }:
        return "warning", f"Completed with gaps (coverage {context.coverage.coverage_ratio:.0%})", "Some requested evidence is missing."
    return "success", "Research completed", "All subquestions supported by verifiable evidence."


def error_breakdown(context: ResearchContext) -> dict[str, int]:
    return dict(context.error_breakdown)


def source_mix_chips(context: ResearchContext) -> list[dict[str, Any]]:
    mix = context.answer.source_type_mix if context.answer else {}
    return [{"type": key, "count": value} for key, value in sorted(mix.items())]


def entity_coverage_line(context: ResearchContext) -> str:
    return "  ".join(
        f"{entity} {'✓' if data.get('supported') else '✗'}"
        for entity, data in context.entity_coverage.items()
    )


def grouped_warnings(context: ResearchContext) -> dict[str, list[str]]:
    groups = {"Coverage": [], "Grounding": [], "Sources": [], "Execution": []}
    for warning in context.quality_gate_warnings:
        lower = warning.lower()
        group = "Execution"
        if "coverage" in lower or "entity" in lower:
            group = "Coverage"
        elif "claim" in lower or "finding" in lower or "ungrounded" in lower:
            group = "Grounding"
        elif "source" in lower or "citation" in lower:
            group = "Sources"
        groups[group].append(warning)
    return groups


def aggregate_trace_errors(trace: list[Any]) -> dict[str, int]:
    counts = Counter()
    for event in trace:
        if not event.success:
            counts[event.error_category.value if event.error_category else "unknown"] += 1
    return dict(counts)
