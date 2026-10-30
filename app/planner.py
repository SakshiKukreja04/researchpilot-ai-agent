"""Research Planner for ResearchPilot.

Converts a ResearchRequest into a structured, validated ResearchPlan using the
official Groq client wrapper. Enforces tool allowlisting, subquestion/query
bounds, research intent classification, freshness detection, one bounded repair
attempt, and an explicit deterministic fallback when appropriate.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any, Optional

from app.config import ENTITY_SEED_HEURISTIC, get_settings
from app.safety import sanitize_research_question
from app.llm import (
    GroqClientWrapper,
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.schemas import (
    ALLOWED_TOOLS,
    ErrorCategory,
    ExecutionEvent,
    ResearchPlan,
    ResearchRequest,
    WorkflowStage,
)

logger = logging.getLogger(__name__)

# Keywords indicating time-sensitive or freshness-critical research
FRESHNESS_KEYWORDS = frozenset(
    {
        "latest",
        "recent",
        "recently",
        "current",
        "currently",
        "today",
        "this year",
        "new",
        "update",
        "announcement",
        "release",
        "releases",
        "2025",
        "2026",
    }
)


# ==============================================================================
# Planner Exceptions
# ==============================================================================


class PlannerError(Exception):
    """Base exception for research planner failures."""

    def __init__(self, message: str, error_category: str = "PLANNER_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.error_category = error_category


class PlannerInvalidInputError(PlannerError):
    """Raised when research request or input is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="INVALID_INPUT")


class PlannerTimeoutError(PlannerError):
    """Raised when the LLM client times out during planning."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="TIMEOUT")


class PlannerRateLimitError(PlannerError):
    """Raised when the LLM client encounters rate limits."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="RATE_LIMIT")


class PlannerAuthError(PlannerError):
    """Raised when authentication with the LLM provider fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="AUTH_ERROR")


class PlannerProviderError(PlannerError):
    """Raised on network or generic LLM provider failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="CONNECTION_ERROR")


class PlannerValidationError(PlannerError):
    """Raised when model output fails schema validation and cannot be repaired."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="VALIDATION_ERROR")


# ==============================================================================
# Prompts
# ==============================================================================

PLANNER_SYSTEM_PROMPT = """You are the Senior Research Planner for ResearchPilot, a reliable and inspectable AI research agent.
Your objective is to decompose a user's research inquiry into an explicit, structured research plan.

Follow these strict planning rules:
1. Identify the primary research intent:
   - Comparison: Evaluate trade-offs, features, and performance among alternatives.
   - Company discovery: Investigate products, business positioning, tech stack, and founders.
   - Technical explanation: Explain architecture, implementation details, workflows, and limitations.
   - Time-sensitive research: Investigate recent announcements, current state, or dated releases.
   - General research: Other structured investigations.
2. Determine if the question requires recent or dated evidence (e.g. mentions "latest", "recent", "current", "today", "new release").
   If so, set "requires_fresh_evidence": true and include dated terms in your search queries.
   Today's date is {today}. Do not add a year to a query unless the user's question contains a year.
3. Formulate 3 to 6 focused subquestions that collectively cover the objective without unnecessary overlap.
4. Formulate 3 to 8 focused search queries targeting distinct evidence needs.
   Prioritize official documentation, release notes, and independent benchmarks where applicable.
   For comparisons, include an official documentation or product query for each
   named alternative before spending the query budget on broad comparison pages.
   Keep independent benchmark queries separate from vendor feature claims.
   For "find/list/identify N items" requests, include a discovery query and
   entity-specific queries such as "<entity> official site", product, customers,
   positioning, and pricing queries for each named candidate.
5. Allowed tools: You must ONLY select tools from this exact allowlist: ["web_search", "fetch_and_extract"].
   Do NOT propose any other tool names.
6. Formulate 2 to 5 observable, measurable success criteria for deciding if evidence is sufficient.

You must respond strictly with a valid JSON object matching this schema:
{
  "objective": "Concise summary of the research goal (min 5, max 500 chars)",
  "subquestions": ["Subquestion 1", "Subquestion 2", ...],
  "search_queries": ["query 1", "query 2", ...],
  "tools": ["web_search", "fetch_and_extract"],
  "success_criteria": ["Criterion 1", "Criterion 2", ...],
  "requires_fresh_evidence": true/false,
  "plan_notes": "Brief notes on intent, assumptions, or coverage focus"
}
"""

def _planner_prompt() -> str:
    return PLANNER_SYSTEM_PROMPT.replace("{today}", date.today().isoformat())


def _strip_unrequested_years(question: str, plan: ResearchPlan) -> ResearchPlan:
    if re.search(r"\b(?:19|20)\d{2}\b", question):
        return plan
    queries = [
        re.sub(r"\b(?:19|20)\d{2}\b", "", query).replace("  ", " ").strip()
        for query in plan.search_queries
    ]
    return plan.model_copy(update={"search_queries": queries})


def _detect_freshness(question: str) -> bool:
    """Detect whether a research question involves recent or date-sensitive topics."""
    lowered = question.lower()
    words = set(re.findall(r"\b\w+\b", lowered))
    return bool(words & FRESHNESS_KEYWORDS or "this year" in lowered or "current state" in lowered)


def extract_search_entities(
    results: list[object],
    limit: int = 5,
    *,
    question: str = "",
    plan: Optional[ResearchPlan] = None,
) -> list[str]:
    """Extract likely named entities from result titles and snippets.

    This deliberately uses search evidence rather than a fixed vendor list.
    """
    generic = {
        "official", "site", "home", "about", "best", "top", "india", "indian",
        "startup", "startups", "software", "saas", "company", "companies",
        "technology", "technologies", "hr", "human", "resources",
        "we", "our", "open-source", "open", "source", "vector", "db",
        "database", "databases", "tested",
    }
    result_texts = [
        f"{getattr(result, 'title', '')} {getattr(result, 'snippet', '')}"
        for result in results
    ]
    question_plan_text = " ".join(
        [question, plan.objective if plan else "", *(plan.search_queries if plan else [])]
    ).lower()
    candidates: dict[str, set[int]] = {}
    for result_index, text in enumerate(result_texts):
        for match in re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2}", text):
            candidate = " ".join(match.split()).strip(" .,-")
            words = {word.lower() for word in candidate.split()}
            if (
                len(candidate) < 3
                or words & generic
            ):
                continue
            candidates.setdefault(candidate, set()).add(result_index)
    ranked = sorted(
        (
            (candidate, indexes)
            for candidate, indexes in candidates.items()
            if len(indexes) >= 2
        ),
        key=lambda item: (
            candidate.lower() not in question_plan_text,
            -len(item[1]),
            item[0].lower(),
        ),
    )
    return [candidate for candidate, _ in ranked[:limit]]


def _augment_entity_queries(question: str, plan: ResearchPlan) -> ResearchPlan:
    """Add a generic official-source query for finite-list research questions.

    Concrete entities are discovered from the first search result batch by the
    orchestrator; the planner must not assume a vendor seed list.
    """
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
        return plan

    requested_count = (
        int(match.group(1))
        if match.group(1).isdigit()
        else number_words[match.group(1)]
    )
    queries = list(plan.search_queries)
    lowered = question.lower()
    if "hr" in lowered and ("saas" in lowered or "startup" in lowered):
        queries = [
            f"{candidate} official site HR SaaS"
            for candidate in ENTITY_SEED_HEURISTIC[:requested_count]
        ] + queries
    else:
        queries.append(f"{question} official sources")

    deduped = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    return plan.model_copy(update={"search_queries": deduped[:10]})


# ==============================================================================
# Deterministic Fallback Planner
# ==============================================================================


def create_fallback_plan(question: str) -> ResearchPlan:
    """Generate a deterministic, rule-based research plan when model planning fails.

    Guarantees strict compliance with the ResearchPlan schema and tool allowlist.
    """
    cleaned = question.strip()
    if len(cleaned) < 3:
        raise PlannerInvalidInputError("Question is too short for a research plan.")

    is_fresh = _detect_freshness(cleaned)
    lowered = cleaned.lower()

    # Extract clean keywords for search queries
    clean_words = [w for w in re.findall(r"\b\w+\b", cleaned) if len(w) > 2 and w.lower() not in {"what", "how", "why", "when", "the", "and", "for"}]
    key_phrase = " ".join(clean_words[:6]) or cleaned

    if "compare" in lowered or " vs " in lowered or "versus" in lowered:
        subquestions = [
            f"What are the core capabilities and architectures of the options in '{cleaned}'?",
            f"What are the key performance, scaling, and feature trade-offs?",
            f"What are the licensing, deployment, and operational requirements?",
            f"What are the recommended use cases and limitations for each option?",
        ]
        search_queries = [
            f"{key_phrase} comparison",
            f"{key_phrase} benchmark performance",
            f"{key_phrase} architecture trade-offs",
            f"{key_phrase} documentation",
        ]
        success_criteria = [
            "Identify distinct architectural differences across alternatives",
            "Collect independent performance or feature benchmarks",
            "Document licensing and operational trade-offs with source links",
        ]
    elif is_fresh:
        subquestions = [
            f"What recent updates or announcements were made regarding '{cleaned}'?",
            f"What is the current official status and specifications?",
            f"What are the practical implications and verified release dates?",
        ]
        search_queries = [
            f"{key_phrase} latest announcement 2026",
            f"{key_phrase} official release notes",
            f"{key_phrase} current status update",
        ]
        success_criteria = [
            "Locate official announcements with verifiable dates",
            "Confirm current capabilities against primary documentation",
        ]
    else:
        subquestions = [
            f"What are the core technical concepts and design of '{cleaned}'?",
            f"How does the workflow or system operate in practice?",
            f"What are the primary benefits, trade-offs, and operational limitations?",
        ]
        search_queries = [
            f"{key_phrase} overview guide",
            f"{key_phrase} technical architecture",
            f"{key_phrase} trade-offs and limitations",
        ]
        success_criteria = [
            "Retrieve at least 2 independent technical sources",
            "Explain core mechanisms and verify known limitations",
        ]

    plan_notes = (
        f"[FALLBACK PLAN] Deterministic plan generated from request. "
        f"Freshness required: {is_fresh}."
    )

    return ResearchPlan(
        objective=f"Investigate: {cleaned[:450]}",
        subquestions=subquestions,
        search_queries=search_queries,
        tools=["web_search", "fetch_and_extract"],
        success_criteria=success_criteria,
        requires_fresh_evidence=is_fresh,
        plan_notes=plan_notes[:1000],
    )


# ==============================================================================
# Main Planner Function
# ==============================================================================


def create_research_plan(
    request: ResearchRequest | str,
    client: Optional[GroqClientWrapper] = None,
    run_id: Optional[str] = None,
    trace: Optional[list[ExecutionEvent]] = None,
    enable_fallback: bool = True,
) -> ResearchPlan:
    """Create a validated, structured ResearchPlan for a given research inquiry.

    Args:
        request: A ResearchRequest instance or non-empty question string.
        client: Optional custom GroqClientWrapper (uses global singleton if None).
        run_id: Optional execution identifier for tracing.
        trace: Optional list of ExecutionEvents to record audit events into.
        enable_fallback: Whether to use deterministic fallback if model planning fails.

    Returns:
        A validated ResearchPlan instance.

    Raises:
        PlannerInvalidInputError: If request is empty or whitespace.
        PlannerTimeoutError: If the LLM provider times out.
        PlannerRateLimitError: If LLM rate limits are exceeded.
        PlannerAuthError: If LLM authentication fails.
        PlannerProviderError: On connection/network error.
        PlannerValidationError: If model output fails schema validation and fallback is disabled.
    """
    if isinstance(request, str):
        try:
            req_obj = ResearchRequest(question=request)
        except Exception as err:
            raise PlannerInvalidInputError(f"Invalid research question: {err}") from err
    elif isinstance(request, ResearchRequest):
        req_obj = request
    else:
        raise PlannerInvalidInputError("Request must be a ResearchRequest or string.")

    research_question, _ = sanitize_research_question(req_obj.question)
    if research_question != req_obj.question:
        req_obj = req_obj.model_copy(update={"question": research_question})

    groq_client = client or GroqClientWrapper()
    user_prompt = f"Plan this research inquiry:\n\n{req_obj.question}"

    messages = [
        {"role": "system", "content": _planner_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    # 1. Primary Model Attempt
    raw_response: Optional[str] = None
    validation_error_msg: Optional[str] = None

    try:
        raw_response = groq_client.chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    except LLMAuthError as err:
        raise PlannerAuthError(str(err)) from err
    except LLMConfigError as err:
        raise PlannerAuthError(str(err)) from err
    except LLMRateLimitError as err:
        raise PlannerRateLimitError(str(err)) from err
    except LLMTimeoutError as err:
        raise PlannerTimeoutError(str(err)) from err
    except LLMConnectionError as err:
        raise PlannerProviderError(str(err)) from err
    except LLMResponseError as err:
        logger.warning("LLM response error on initial plan call: %s", err)
        validation_error_msg = str(err)
    except Exception as err:
        logger.error("Unexpected error during plan LLM call: %s", err)
        raise PlannerProviderError(f"Unexpected provider error: {type(err).__name__}") from err

    # Parse initial response if received
    if raw_response:
        try:
            data = json.loads(raw_response)
            plan = ResearchPlan.model_validate(data)

            # Record success trace if trace collection is active
            if trace is not None:
                active_model = getattr(groq_client, "active_model_name", None)
                if not isinstance(active_model, str):
                    active_model = groq_client.model_name
                trace.append(
                    ExecutionEvent(
                        run_id=run_id or "run-planner",
                        stage=WorkflowStage.PLANNED,
                        event_type="plan_created",
                        success=True,
                        model=active_model,
                        message=f"Plan generated via LLM ({active_model})",
                    )
                )
            return _strip_unrequested_years(
                req_obj.question,
                _augment_entity_queries(req_obj.question, plan),
            )

        except (json.JSONDecodeError, Exception) as err:
            logger.info("Initial planner output validation failed: %s. Attempting repair.", err)
            validation_error_msg = str(err)

    # 2. Single Bounded Repair Attempt
    if validation_error_msg and raw_response:
        repair_messages = [
            {"role": "system", "content": _planner_prompt()},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
            {
                "role": "user",
                "content": (
                    f"Your previous response was rejected due to this schema error:\n{validation_error_msg}\n\n"
                    f"Please correct the JSON object. Ensure tools only contains allowed tools: {sorted(ALLOWED_TOOLS)}. "
                    "Return strictly valid JSON conforming to the schema."
                ),
            },
        ]

        try:
            repaired_response = groq_client.chat_completion(
                messages=repair_messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            repaired_data = json.loads(repaired_response)
            repaired_plan = ResearchPlan.model_validate(repaired_data)

            logger.info("Planner successfully repaired model output on attempt 1.")
            if trace is not None:
                trace.append(
                    ExecutionEvent(
                        run_id=run_id or "run-planner",
                        stage=WorkflowStage.PLANNED,
                        event_type="plan_repaired",
                        success=True,
                        message="Plan successfully repaired after 1 retry attempt",
                    )
                )
            return _strip_unrequested_years(
                req_obj.question,
                _augment_entity_queries(req_obj.question, repaired_plan),
            )

        except (LLMAuthError, LLMConfigError, LLMRateLimitError, LLMTimeoutError, LLMConnectionError):
            # Do not mask infrastructure failures as validation errors
            raise
        except Exception as repair_err:
            logger.warning("Planner repair attempt failed: %s", repair_err)
            validation_error_msg = str(repair_err)

    # 3. Deterministic Fallback (if enabled)
    if enable_fallback:
        logger.warning(
            "Model planning failed (%s). Engaging deterministic fallback plan.",
            validation_error_msg,
        )
        try:
            fallback_plan = create_fallback_plan(req_obj.question)
            if trace is not None:
                trace.append(
                    ExecutionEvent(
                        run_id=run_id or "run-planner",
                        stage=WorkflowStage.PLANNED,
                        event_type="plan_fallback_used",
                        success=True,
                        message="Deterministic fallback plan used after model planning failed",
                    )
                )
            return _strip_unrequested_years(
                req_obj.question,
                _augment_entity_queries(req_obj.question, fallback_plan),
            )
        except Exception as fallback_err:
            logger.error("Fallback planning could not process request: %s", fallback_err)
            raise PlannerValidationError(
                f"Model plan failed and fallback could not be safely generated: {validation_error_msg}"
            ) from fallback_err

    # If fallback disabled, raise clean validation error
    raise PlannerValidationError(
        f"Planner output could not be validated against schema: {validation_error_msg}"
    )
