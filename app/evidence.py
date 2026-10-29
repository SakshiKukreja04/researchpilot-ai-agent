"""Evidence processing, scoring heuristics, and coverage assessment for ResearchPilot.

Implements deterministic source-quality scoring:
    Q = 0.40*R + 0.30*A + 0.20*F + 0.10*C
where:
    R = Relevance to research inquiry and plan
    A = Domain authority and primary-source preference
    F = Freshness (with explicit neutral treatment of unknown dates)
    C = Content completeness and readability

Assesses coverage against plan subquestions and criteria, identifies evidence gaps,
and enforces domain diversity without fabricating claims.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

from app.schemas import (
    CoverageReport,
    ExtractionStatus,
    ResearchPlan,
    SourceDocument,
)

logger = logging.getLogger(__name__)

# Basic stop words to ignore during lexical relevance calculation
STOP_WORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves",
    }
)

# Domain authority tiers
AUTHORITY_TIER_HIGH = (
    ".edu",
    ".gov",
    "github.com",
    "arxiv.org",
    "ietf.org",
    "w3.org",
    "docs.",
    "developer.",
    "documentation.",
    "learn.microsoft.com",
    "cloud.google.com",
    "aws.amazon.com",
)

AUTHORITY_TIER_MEDIUM = (
    "acm.org",
    "ieee.org",
    "stackoverflow.com",
    "infoq.com",
    "techcrunch.com",
    "venturebeat.com",
    "zdnet.com",
    "theverge.com",
    "wired.com",
    "arstechnica.com",
    "medium.com",
    "dev.to",
)

LOW_AUTHORITY_DOMAINS = (
    "pinterest.",
    "quora.com",
    "blogspot.",
    "wordpress.com",
    "wixsite.com",
)


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercased meaningful keywords."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


# ==============================================================================
# Scoring Heuristics
# ==============================================================================


def calculate_relevance_score(doc: SourceDocument, query_text: str) -> float:
    """Calculate keyword relevance score between document and research inquiry (0.0 to 1.0)."""
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return 0.5

    title_tokens = _tokenize(doc.title)
    content_tokens = _tokenize(doc.content)

    title_matches = len(query_tokens & title_tokens)
    content_matches = len(query_tokens & content_tokens)

    # Title match has higher weight per term than content match
    score = (content_matches / len(query_tokens)) * 0.7 + min(1.0, title_matches * 0.3)
    return max(0.0, min(1.0, round(score, 3)))


def calculate_authority_score(domain: str) -> float:
    """Assign domain authority score based on domain category (0.0 to 1.0)."""
    dom = domain.lower().strip()
    if not dom:
        return 0.3

    # High authority tier (Primary docs, academic, government)
    if any(h in dom for h in AUTHORITY_TIER_HIGH):
        return 0.95

    # Medium authority tier (Tech journalism, engineering communities)
    if any(m in dom for m in AUTHORITY_TIER_MEDIUM):
        return 0.75

    # Low authority tier (Social aggregators, unverified user forums)
    if any(l in dom for l in LOW_AUTHORITY_DOMAINS):
        return 0.25

    # Standard commercial / organizational web domains
    return 0.50


def calculate_freshness_score(
    doc: SourceDocument, requires_fresh_evidence: bool = False
) -> float:
    """Calculate freshness score (0.0 to 1.0).

    Explicitly treats unknown publication dates as neutral (0.50) rather than
    penalizing them to 0.0.
    """
    if not requires_fresh_evidence:
        # For non-time-sensitive research, having a date is a slight bonus
        return 0.90 if doc.published_date else 0.75

    # Time-sensitive case
    date_str = (doc.published_date or "").strip()
    if not date_str:
        # Unknown date: neutral baseline
        return 0.50

    # Look for year in published_date string
    year_match = re.search(r"\b(20\d\d)\b", date_str)
    if year_match:
        year = int(year_match.group(1))
        if year >= 2025:
            return 1.0
        elif year == 2024:
            return 0.70
        elif year == 2023:
            return 0.40
        else:
            return 0.20

    return 0.50


def calculate_completeness_score(content: str) -> float:
    """Evaluate completeness and readability based on substantive text volume."""
    length = len(content.strip())
    if length < 150:
        return 0.10
    elif length < 500:
        return 0.40
    elif length < 2500:
        return 0.80
    else:
        return 1.0


def score_source_document(
    doc: SourceDocument,
    query_text: str,
    requires_fresh_evidence: bool = False,
    w_r: float = 0.40,
    w_a: float = 0.30,
    w_f: float = 0.20,
    w_c: float = 0.10,
) -> float:
    """Compute overall quality score Q = w_r*R + w_a*A + w_f*F + w_c*C."""
    r = calculate_relevance_score(doc, query_text)
    a = calculate_authority_score(doc.domain)
    f = calculate_freshness_score(doc, requires_fresh_evidence)
    c = calculate_completeness_score(doc.content)

    q = (w_r * r) + (w_a * a) + (w_f * f) + (w_c * c)
    return max(0.0, min(1.0, round(q, 3)))


# ==============================================================================
# Evidence Processing and Filtering
# ==============================================================================


def process_evidence(
    sources: list[SourceDocument],
    plan: ResearchPlan,
    question: str,
    quality_threshold: float = 0.55,
    max_sources_per_domain: int = 2,
) -> tuple[list[SourceDocument], list[SourceDocument], CoverageReport]:
    """Score, deduplicate, filter, and assess subquestion coverage for retrieved evidence.

    Args:
        sources: List of raw retrieved SourceDocuments.
        plan: Validated ResearchPlan.
        question: User research inquiry.
        quality_threshold: Minimum Q score required for acceptance (default 0.55).
        max_sources_per_domain: Domain diversity cap.

    Returns:
        (accepted_sources, rejected_sources, coverage_report)
    """
    relevance_contexts = [
        question,
        plan.objective,
        *plan.subquestions,
        *plan.search_queries,
    ]
    requires_fresh = getattr(plan, "requires_fresh_evidence", False)

    accepted: list[SourceDocument] = []
    rejected: list[SourceDocument] = []

    domain_counts: dict[str, int] = {}
    seen_content_hashes: set[str] = set()

    for doc in sources:
        # Check extraction status
        if doc.extraction_status != ExtractionStatus.SUCCESS or not doc.content.strip():
            rejected_doc = doc.model_copy(
                update={
                    "quality_score": 0.0,
                    "rejection_reason": "Extraction failed or empty content",
                }
            )
            rejected.append(rejected_doc)
            continue

        # Check near-duplicate content via normalized content snippet hash
        norm_snippet = re.sub(r"\s+", " ", doc.content[:1000].lower()).strip()
        snippet_hash = hash(norm_snippet)
        if snippet_hash in seen_content_hashes:
            rejected_doc = doc.model_copy(
                update={
                    "quality_score": 0.0,
                    "rejection_reason": "Duplicate content of existing source",
                }
            )
            rejected.append(rejected_doc)
            continue
        seen_content_hashes.add(snippet_hash)

        # Calculate quality score Q
        # Score against the best individual research criterion. A single
        # combined prompt dilutes relevance for sources covering one comparison
        # dimension, even when that source is useful evidence.
        q_score = max(
            score_source_document(
                doc,
                relevance_context,
                requires_fresh_evidence=requires_fresh,
            )
            for relevance_context in relevance_contexts
        )

        # Check quality threshold
        if q_score < quality_threshold:
            rejected_doc = doc.model_copy(
                update={
                    "quality_score": q_score,
                    "rejection_reason": f"Quality score {q_score:.2f} below threshold {quality_threshold:.2f}",
                }
            )
            rejected.append(rejected_doc)
            continue

        # Domain diversity enforcement
        current_domain_count = domain_counts.get(doc.domain, 0)
        if current_domain_count >= max_sources_per_domain:
            rejected_doc = doc.model_copy(
                update={
                    "quality_score": q_score,
                    "rejection_reason": f"Domain diversity cap reached for domain '{doc.domain}'",
                }
            )
            rejected.append(rejected_doc)
            continue

        domain_counts[doc.domain] = current_domain_count + 1
        accepted_doc = doc.model_copy(update={"quality_score": q_score})
        accepted.append(accepted_doc)

    # Sort accepted sources descending by quality score
    accepted.sort(key=lambda s: s.quality_score, reverse=True)

    # Assess coverage against plan subquestions
    coverage = assess_coverage(accepted, plan)

    return accepted, rejected, coverage


# ==============================================================================
# Coverage Assessment
# ==============================================================================


def assess_coverage(
    accepted_sources: list[SourceDocument],
    plan: ResearchPlan,
    min_coverage_threshold: float = 0.60,
) -> CoverageReport:
    """Assess whether accepted evidence adequately covers each subquestion in the plan."""
    subquestion_coverage: dict[str, str] = {}
    supported_count = 0
    partially_supported_count = 0
    unsupported_count = 0
    unsupported_list: list[str] = []

    combined_corpus = " ".join([f"{s.title} {s.content}" for s in accepted_sources]).lower()

    for sq in plan.subquestions:
        sq_tokens = _tokenize(sq)
        if not sq_tokens:
            subquestion_coverage[sq] = "supported"
            supported_count += 1
            continue

        matched_tokens = {t for t in sq_tokens if t in combined_corpus}
        match_ratio = len(matched_tokens) / len(sq_tokens)

        # Subquestion coverage heuristics
        if match_ratio >= 0.60 and len(accepted_sources) >= 1:
            subquestion_coverage[sq] = "supported"
            supported_count += 1
        elif match_ratio >= 0.30:
            subquestion_coverage[sq] = "partially_supported"
            partially_supported_count += 1
        else:
            subquestion_coverage[sq] = "unsupported"
            unsupported_count += 1
            unsupported_list.append(sq)

    total_sq = len(plan.subquestions)
    effective_support = supported_count + (0.5 * partially_supported_count)
    coverage_ratio = round(effective_support / max(1, total_sq), 3)

    is_adequate = (
        coverage_ratio >= min_coverage_threshold and len(accepted_sources) >= 2
    )

    # Identify missing criteria if comparison intent
    missing_criteria: list[str] = []
    if not is_adequate and unsupported_list:
        missing_criteria.append(
            f"Insufficient evidence for {len(unsupported_list)} subquestions: {unsupported_list[:2]}"
        )

    return CoverageReport(
        subquestion_coverage=subquestion_coverage,
        supported_count=supported_count,
        partially_supported_count=partially_supported_count,
        unsupported_count=unsupported_count,
        coverage_ratio=coverage_ratio,
        missing_criteria=missing_criteria,
        unsupported_subquestions=unsupported_list,
        is_adequate=is_adequate,
    )
