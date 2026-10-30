"""Deterministic handling for untrusted instructions embedded in research requests."""

from __future__ import annotations

import re


_INJECTION_MARKERS = re.compile(
    r"\b(?:ignore\s+(?:all\s+)?previous\s+instructions|"
    r"(?:print|reveal|show|disclose|provide)\b)",
    re.IGNORECASE,
)
_CONFIDENTIAL_TARGETS = re.compile(
    r"\b(?:system\s+prompt|developer\s+message|api\s+keys?|credentials?|secrets?)\b",
    re.IGNORECASE,
)


def sanitize_research_question(question: str) -> tuple[str, bool]:
    """Remove prompt-exfiltration clauses while preserving separate research requests."""
    clauses = re.split(
        r"(?<=[.!?])\s+|\s+\bthen\b\s+",
        question.strip(),
        flags=re.IGNORECASE,
    )
    kept: list[str] = []
    injection_detected = False
    for clause in clauses:
        if _INJECTION_MARKERS.search(clause) and _CONFIDENTIAL_TARGETS.search(clause):
            injection_detected = True
            continue
        cleaned_clause = re.sub(r"^\s*then\b[\s,;:-]*", "", clause, flags=re.IGNORECASE)
        if cleaned_clause.strip():
            kept.append(cleaned_clause.strip())

    return " ".join(kept), injection_detected
