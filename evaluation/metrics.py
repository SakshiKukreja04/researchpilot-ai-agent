"""Deterministic metrics for recorded ResearchPilot evaluation runs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def calculate_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calculate metrics using explicit denominators from serialized run records."""
    items = list(records)
    attempted = len(items)
    completed = sum(item.get("status") == "completed" for item in items)
    usable = sum(item.get("status") in {"completed", "partial"} for item in items)
    citation_total = 0
    citation_valid = 0
    for item in items:
        answer = item.get("answer") or {}
        known_ids = {
            source.get("source_id")
            for source in item.get("sources", [])
            if source.get("source_id")
        }
        for finding in answer.get("key_findings", []):
            for source_id in finding.get("source_ids", []):
                citation_total += 1
                citation_valid += source_id in known_ids

    return {
        "runs_attempted": attempted,
        "completed_runs": completed,
        "usable_runs": usable,
        "usable_run_rate": usable / attempted if attempted else None,
        "citation_references": citation_total,
        "valid_citation_references": citation_valid,
        "citation_validity_rate": (
            citation_valid / citation_total if citation_total else None
        ),
    }
