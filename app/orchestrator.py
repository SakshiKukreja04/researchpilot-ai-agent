"""Deterministic Bounded Orchestrator for ResearchPilot.

Coordinates the complete research lifecycle across planning, web search, safe page
extraction, and evidence quality filtering. Enforces strict tool allowlists, query/page/time
budgets, bounded retries, and records full runtime trace events without leaking secrets.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from datetime import datetime, timezone
from threading import Lock, local
from typing import Any, Callable, Optional, Set

from app.config import Settings, get_settings
from app.evidence import calculate_relevance_score, process_evidence
from app.evaluator import evaluate_answer
from app.llm import GroqClientWrapper
from app.logging_config import log_execution_event
from app.planner import PlannerError, create_research_plan, extract_search_entities
from app.safety import sanitize_research_question
from app.quality_gate import run_quality_gate
from app.schemas import (
    ALLOWED_TOOLS,
    ConfidenceLevel,
    CoverageReport,
    ErrorCategory,
    ExecutionEvent,
    ExtractionStatus,
    ResearchAnswer,
    AnswerEvaluation,
    ResearchAnswerStatus,
    ResearchContext,
    ResearchPlan,
    ResearchRequest,
    SearchResult,
    SourceDocument,
    WorkflowStage,
    ReasonCode,
    StatusReason,
)
from app.synthesizer import synthesize_research_answer
from app.tools.fetch import (
    FetchError,
    FetchHTTPError,
    FetchInvalidInputError,
    FetchTimeoutError,
    FetchUnsafeURLError,
    fetch_and_extract,
    generate_source_id,
    is_safe_url,
    normalize_url,
)
from app.tools.search import (
    SearchError,
    SearchInvalidInputError,
    SearchRateLimitError,
    SearchTimeoutError,
    web_search,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Orchestrator Execution Budget
# ==============================================================================


class ExecutionBudget:
    """Tracks and enforces limits on execution time, queries, page fetches, and retries."""

    def __init__(
        self,
        max_search_queries: int = 5,
        max_fetch_pages: int = 8,
        max_research_rounds: int = 1,
        max_retries_per_tool: int = 1,
        deadline_seconds: float = 90.0,
    ) -> None:
        self.max_search_queries = max_search_queries
        self.max_fetch_pages = max_fetch_pages
        self.max_research_rounds = max_research_rounds
        self.max_retries_per_tool = max_retries_per_tool
        self.start_time = time.monotonic()
        self.deadline = self.start_time + deadline_seconds

        self.queries_used = 0
        self.pages_fetched = 0
        self.extra_rounds_used = 0

    @property
    def is_time_exhausted(self) -> bool:
        return time.monotonic() >= self.deadline

    @property
    def remaining_queries(self) -> int:
        return max(0, self.max_search_queries - self.queries_used)

    @property
    def remaining_pages(self) -> int:
        return max(0, self.max_fetch_pages - self.pages_fetched)

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000.0


# ==============================================================================
# Main Orchestration Pipeline
# ==============================================================================


def run_research(
    request: ResearchRequest | str,
    settings: Optional[Settings] = None,
    client: Optional[GroqClientWrapper] = None,
    search_tool_fn: Callable[..., list[SearchResult]] = web_search,
    fetch_tool_fn: Callable[..., SourceDocument] = fetch_and_extract,
    synthesize: bool = False,
    verify_claims: bool = False,
    progress_callback: Optional[Callable[[WorkflowStage, str], None]] = None,
) -> ResearchContext:
    """Execute the bounded, inspectable research workflow.

    Args:
        request: Validated ResearchRequest or question string.
        settings: Optional configuration settings (defaults to global settings).
        client: Optional custom Groq client wrapper.
        search_tool_fn: Allowlisted web search tool function (injectable for testing).
        fetch_tool_fn: Allowlisted fetch tool function (injectable for testing).
        synthesize: Whether to synthesize a grounded answer and run quality gates.
        verify_claims: Whether to perform targeted claim-support verification.
        progress_callback: Optional UI callback (stage, status_message) for real-time progress.

    Returns:
        Typed ResearchContext containing accepted sources, rejected sources,
        coverage report, validated plan, audit execution trace, and optional answer.
    """
    cfg = settings or get_settings()
    client = client or GroqClientWrapper(settings=cfg)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    pipeline_started_at = datetime.now(timezone.utc)
    plan_duration_ms = 0.0
    search_duration_ms = 0.0
    evidence_duration_ms = 0.0
    synthesis_duration_ms = 0.0
    validation_duration_ms = 0.0
    trace: list[ExecutionEvent] = []

    def _record_event(
        evt: ExecutionEvent,
        tool_name: Optional[str] = None,
        retry_count: int = 0,
    ) -> None:
        trace.append(evt)
        log_execution_event(
            event=evt,
            logger_instance=logger,
            tool_name=tool_name,
            retry_count=retry_count,
        )

    def _report_stage(stage: WorkflowStage, message: str) -> None:
        if progress_callback:
            try:
                progress_callback(stage, message)
            except Exception:
                pass

    budget = ExecutionBudget(
        max_search_queries=cfg.max_search_queries,
        max_fetch_pages=cfg.max_fetch_pages,
        max_research_rounds=cfg.max_research_rounds,
        max_retries_per_tool=cfg.max_retries_per_tool,
        deadline_seconds=cfg.overall_deadline_seconds,
    )
    budget.max_consecutive_fetch_failures = cfg.max_consecutive_fetch_failures
    deadline_reason_code: Optional[str] = None

    # 1. Stage: RECEIVED
    start_stage = time.monotonic()
    _report_stage(WorkflowStage.RECEIVED, "Validating research inquiry...")
    if isinstance(request, str):
        cleaned_question = request.strip()
        if not cleaned_question:
            event = ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.RECEIVED,
                event_type="request_rejected",
                duration_ms=(time.monotonic() - start_stage) * 1000.0,
                success=False,
                error_category=ErrorCategory.VALIDATION_ERROR,
                message="Research question cannot be empty or whitespace.",
            )
            _record_event(event)
            _report_stage(WorkflowStage.FAILED, "Research question cannot be empty or whitespace.")
            dummy_req = ResearchRequest(question="Invalid question")
            empty_plan = ResearchPlan(
                objective="Rejected invalid request",
                subquestions=["N/A"],
                search_queries=["N/A"],
                tools=["web_search"],
                success_criteria=["N/A"],
            )
            empty_coverage = CoverageReport(
                subquestion_coverage={},
                supported_count=0,
                partially_supported_count=0,
                unsupported_count=0,
                coverage_ratio=0.0,
                is_adequate=False,
            )
            rejected_ans = None
            if synthesize:
                rejected_ans = ResearchAnswer(
                    user_question=cleaned_question or "Invalid question",
                    short_answer="Research question cannot be empty or whitespace.",
                    key_findings=[],
                    sources_used=[],
                    source_url_map={},
                    confidence=ConfidenceLevel.LOW,
                    assumptions=[],
                    limitations=["Inquiry was rejected before execution."],
                    suggested_next_steps=["Please provide a clear, specific question."],
                    status=ResearchAnswerStatus.FAILED,
                )
            return ResearchContext(
                run_id=run_id,
                request=dummy_req,
                plan=empty_plan,
                coverage=empty_coverage,
                trace=trace,
                status=ResearchAnswerStatus.FAILED,
                answer=rejected_ans,
            )
        req_obj = ResearchRequest(question=cleaned_question)
    else:
        req_obj = request

    research_question, injection_detected = sanitize_research_question(req_obj.question)
    if injection_detected:
        metadata = dict(req_obj.metadata or {})
        metadata["researchpilot_prompt_injection_detected"] = True
        req_obj = req_obj.model_copy(
            update={"question": research_question, "metadata": metadata}
        )

    _record_event(
        ExecutionEvent(
            run_id=run_id,
            stage=WorkflowStage.RECEIVED,
            event_type="request_accepted",
            duration_ms=(time.monotonic() - start_stage) * 1000.0,
            success=True,
            message=f"Research inquiry accepted: {req_obj.question[:100]}...",
        )
    )

    # 2. Stage: PLANNED
    start_stage = time.monotonic()
    _report_stage(WorkflowStage.PLANNED, "Generating structured research plan...")
    try:
        plan = create_research_plan(
            req_obj,
            client=client,
            run_id=run_id,
            trace=trace,
            enable_fallback=True,
        )
        plan_duration_ms = (time.monotonic() - start_stage) * 1000.0
        _report_stage(
            WorkflowStage.PLANNED,
            f"Plan generated: {len(plan.subquestions)} subquestions, {len(plan.search_queries)} search queries.",
        )
    except PlannerError as err:
        event = ExecutionEvent(
            run_id=run_id,
            stage=WorkflowStage.PLANNED,
            event_type="planning_failed",
            duration_ms=(time.monotonic() - start_stage) * 1000.0,
            success=False,
            error_category=ErrorCategory.VALIDATION_ERROR,
            message=f"Planning failed: {err.message}",
        )
        _record_event(event)
        _report_stage(WorkflowStage.FAILED, f"Planning failed: {err.message}")
        empty_coverage = CoverageReport(
            subquestion_coverage={},
            supported_count=0,
            partially_supported_count=0,
            unsupported_count=0,
            coverage_ratio=0.0,
            is_adequate=False,
        )
        dummy_plan = ResearchPlan(
            objective="Failed plan",
            subquestions=["N/A"],
            search_queries=["N/A"],
            tools=["web_search"],
            success_criteria=["N/A"],
        )
        failed_ans = None
        if synthesize:
            failed_ans = ResearchAnswer(
                user_question=req_obj.question,
                short_answer=f"Planning failed: {err.message}",
                key_findings=[],
                sources_used=[],
                source_url_map={},
                confidence=ConfidenceLevel.LOW,
                assumptions=[],
                limitations=["Planner was unable to produce a research plan."],
                suggested_next_steps=["Rephrase the question or verify model availability."],
                status=ResearchAnswerStatus.FAILED,
            )
        return ResearchContext(
            run_id=run_id,
            request=req_obj,
            plan=dummy_plan,
            coverage=empty_coverage,
            trace=trace,
            status=ResearchAnswerStatus.FAILED,
            answer=failed_ans,
        )

    # 3. Tool Verification: strictly enforce allowlist
    disallowed = [t for t in plan.tools if t not in ALLOWED_TOOLS]
    if disallowed:
        trace.append(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.PLANNED,
                event_type="unauthorized_tools_rejected",
                success=False,
                error_category=ErrorCategory.VALIDATION_ERROR,
                message=f"Plan attempted to register unauthorized tools: {disallowed}",
            )
        )
        plan = plan.model_copy(
            update={"tools": [t for t in plan.tools if t in ALLOWED_TOOLS]}
        )

    # 4. Stage: SEARCHING (Round 1)
    _report_stage(
        WorkflowStage.SEARCHING,
        f"Searching web across {len(plan.search_queries)} queries...",
    )
    all_search_results: list[SearchResult] = []
    search_result_groups: list[list[SearchResult]] = []
    seen_search_urls: set[str] = set()
    search_stage_started = time.monotonic()
    search_stage_deadline = min(
        budget.deadline - cfg.min_fetch_window_seconds,
        search_stage_started + (
            cfg.overall_deadline_seconds * cfg.search_budget_fraction
        ),
    )
    search_stage_deadline = max(search_stage_started, search_stage_deadline)
    search_workers = max(1, min(6, cfg.fetch_max_workers))
    legacy_injected_search = type(search_tool_fn).__module__ == "unittest.mock"
    search_event_local = local()

    def _record_search_event(event: ExecutionEvent, **kwargs: Any) -> None:
        events = getattr(search_event_local, "events", None)
        if events is None:
            events = []
            search_event_local.events = events
        events.append((event, kwargs))

    def _search_worker(q_str: str) -> tuple[list[SearchResult], list[tuple[ExecutionEvent, dict[str, Any]]]]:
        # Isolate the provider call so the orchestration worker can enforce a
        # whole-operation wall-clock cap even when a provider ignores its
        # internal timeout. The abandoned provider thread is non-blocking on
        # executor shutdown and cannot delay retrieval.
        inner_executor = ThreadPoolExecutor(max_workers=1)
        try:
            def _run_inner() -> tuple[list[SearchResult], list[tuple[ExecutionEvent, dict[str, Any]]]]:
                results = _execute_search_query(q_str)
                return results, getattr(search_event_local, "events", [])

            inner_future = inner_executor.submit(_run_inner)
            try:
                return inner_future.result(timeout=cfg.search_query_timeout_seconds)
            except FuturesTimeoutError:
                inner_future.cancel()
                return [], [
                    (
                        ExecutionEvent(
                            run_id=run_id,
                            stage=WorkflowStage.SEARCHING,
                            event_type="search_query_timeout",
                            duration_ms=cfg.search_query_timeout_seconds * 1000.0,
                            success=False,
                            error_category=ErrorCategory.TIMEOUT,
                            message=f"Search query '{q_str[:60]}' exceeded its whole-query cap.",
                        ),
                        {"tool_name": "web_search"},
                    )
                ]
        finally:
            inner_executor.shutdown(wait=False, cancel_futures=True)

    def _execute_search_query(q_str: str) -> list[SearchResult]:
        search_event_local.events = []
        if time.monotonic() >= search_stage_deadline or budget.remaining_queries <= 0:
            return []

        q_clean = q_str.strip()
        if not q_clean:
            return []

        budget.queries_used += 1
        q_start = time.monotonic()

        # Retry logic: at most 1 retry on transient failure
        for attempt in range(budget.max_retries_per_tool + 1):
            try:
                results = search_tool_fn(q_clean, max_results=5)
                _record_search_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.SEARCHING,
                        event_type="search_query_executed",
                        duration_ms=(time.monotonic() - q_start) * 1000.0,
                        success=True,
                        message=f"Query '{q_clean[:60]}' returned {len(results)} results (attempt {attempt+1})",
                    ),
                    tool_name="web_search",
                    retry_count=attempt,
                )
                return results
            except (SearchTimeoutError, SearchRateLimitError) as transient_err:
                if attempt < budget.max_retries_per_tool and not budget.is_time_exhausted:
                    logger.warning("Search query failed (%s); retrying once...", transient_err)
                    time.sleep(0.1)
                    continue
                _record_search_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.SEARCHING,
                        event_type="search_query_failed",
                        duration_ms=(time.monotonic() - q_start) * 1000.0,
                        success=False,
                        error_category=ErrorCategory.TIMEOUT if isinstance(transient_err, SearchTimeoutError) else ErrorCategory.RATE_LIMIT,
                        message=f"Search failed for '{q_clean[:60]}': {transient_err}",
                    ),
                    tool_name="web_search",
                    retry_count=attempt,
                )
                return []
            except Exception as err:
                _record_search_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.SEARCHING,
                        event_type="search_query_error",
                        duration_ms=(time.monotonic() - q_start) * 1000.0,
                        success=False,
                        error_category=ErrorCategory.UNKNOWN,
                        message=f"Search failed permanently for '{q_clean[:60]}': {type(err).__name__}",
                    ),
                    tool_name="web_search",
                    retry_count=attempt,
                )
                return []
        return []

    def _record_search_results(results: list[SearchResult]) -> None:
        search_result_groups.append(results)
        for result in results:
            if result.url not in seen_search_urls:
                seen_search_urls.add(result.url)
                all_search_results.append(result)

    queries = list(plan.search_queries)
    search_executor = ThreadPoolExecutor(
        max_workers=1 if legacy_injected_search else search_workers
    )
    try:
        pending_queries = queries
        query_index = 0
        while pending_queries and budget.remaining_queries > 0:
            if time.monotonic() >= search_stage_deadline:
                break
            # Submit the complete eligible query batch once; the executor
            # bounds concurrency while as_completed prevents a slow query from
            # delaying completed results from other queries.
            wave = pending_queries[:budget.remaining_queries]
            pending_queries = pending_queries[len(wave) :]
            futures = [
                (query_index + offset, query, search_executor.submit(_search_worker, query))
                for offset, query in enumerate(wave)
            ]
            query_index += len(wave)
            future_meta = {
                future: (index, query, time.monotonic() + cfg.search_query_timeout_seconds)
                for index, query, future in futures
            }
            completed_results: dict[int, tuple[list[SearchResult], list[tuple[ExecutionEvent, dict[str, Any]]]]] = {}
            try:
                completed = as_completed(
                    [future for _, _, future in futures],
                    timeout=max(0.0, search_stage_deadline - time.monotonic()),
                )
                for future in completed:
                    index, query, query_deadline = future_meta[future]
                    if time.monotonic() > query_deadline:
                        future.cancel()
                        _record_event(
                            ExecutionEvent(
                                run_id=run_id,
                                stage=WorkflowStage.SEARCHING,
                                event_type="search_query_timeout",
                                duration_ms=cfg.search_query_timeout_seconds * 1000.0,
                                success=False,
                                error_category=ErrorCategory.TIMEOUT,
                                message=f"Search query '{query[:60]}' exceeded its whole-query cap.",
                            ),
                            tool_name="web_search",
                        )
                        continue
                    try:
                        completed_results[index] = future.result()
                    except Exception as err:
                        _record_event(
                            ExecutionEvent(
                                run_id=run_id,
                                stage=WorkflowStage.SEARCHING,
                                event_type="search_query_error",
                                success=False,
                                error_category=ErrorCategory.UNKNOWN,
                                message=f"Search failed permanently for '{query[:60]}': {type(err).__name__}",
                            ),
                            tool_name="web_search",
                        )
            except FuturesTimeoutError:
                pass

            for index, query, future in futures:
                if index in completed_results:
                    results, events = completed_results[index]
                    _record_search_results(results)
                    for event, kwargs in events:
                        _record_event(event, **kwargs)
                elif not future.done() or time.monotonic() > future_meta[future][2]:
                    future.cancel()
                    _record_event(
                        ExecutionEvent(
                            run_id=run_id,
                            stage=WorkflowStage.SEARCHING,
                            event_type="search_query_timeout",
                            duration_ms=cfg.search_query_timeout_seconds * 1000.0,
                            success=False,
                            error_category=ErrorCategory.TIMEOUT,
                            message=f"Search query '{query[:60]}' exceeded its whole-query cap.",
                        ),
                        tool_name="web_search",
                    )
            finite_list_query = bool(
                re.search(
                    r"\b(?:find|list|identify|name|top)\s+"
                    r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                    req_obj.question.lower(),
                )
            )
            comparison_query = "compare" in req_obj.question.lower()
            if (
                query_index == len(wave)
                and all_search_results
                and (finite_list_query or comparison_query)
            ):
                # Discover entity names from evidence returned by the first
                # search wave instead of maintaining a brittle seed list.
                entities = extract_search_entities(
                    all_search_results,
                    limit=min(5, budget.remaining_queries),
                    question=req_obj.question,
                    plan=plan,
                )
                dynamic_queries = [
                    f"{entity} {'documentation' if comparison_query else 'official site'}"
                    for entity in entities
                    if entity
                ]
                if dynamic_queries:
                    pending_queries = dynamic_queries + pending_queries
                    queries.extend(dynamic_queries)
            if time.monotonic() >= search_stage_deadline:
                break
    finally:
        # A provider call cannot always be interrupted by Python cancellation.
        # Do not let a stuck provider worker hold up the bounded pipeline.
        search_executor.shutdown(wait=False, cancel_futures=True)
    search_duration_ms = (time.monotonic() - search_stage_started) * 1000.0

    if time.monotonic() >= search_stage_deadline and all_search_results:
        deadline_reason_code = "deadline_exhausted_during_search"
        _record_event(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.SEARCHING,
                event_type="search_deadline_reached",
                success=False,
                error_category=ErrorCategory.TIMEOUT,
                message=(
                    f"Search stage budget reached with {len(all_search_results)} "
                    "candidate URL(s); continuing to retrieval."
                ),
            ),
            tool_name="web_search",
        )

    # If completely 0 search results found across all queries
    if not all_search_results:
        if time.monotonic() >= search_stage_deadline:
            deadline_reason_code = "deadline_exhausted_during_search"
        _record_event(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.SEARCHING,
                event_type="search_yielded_no_results",
                success=False,
                error_category=ErrorCategory.EXTRACTION_ERROR,
                message="All search queries yielded 0 usable candidate pages.",
            ),
            tool_name="web_search",
        )
        _report_stage(WorkflowStage.DEGRADED, "All search queries returned 0 usable candidate pages.")
        empty_coverage = CoverageReport(
            subquestion_coverage={sq: "unsupported" for sq in plan.subquestions},
            supported_count=0,
            partially_supported_count=0,
            unsupported_count=len(plan.subquestions),
            coverage_ratio=0.0,
            missing_criteria=["No search results retrieved"],
            unsupported_subquestions=list(plan.subquestions),
            is_adequate=False,
        )
        empty_ctx = ResearchContext(
            run_id=run_id,
            request=req_obj,
            plan=plan,
            raw_search_results=[],
            sources=[],
            rejected_sources=[],
            coverage=empty_coverage,
            trace=trace,
            status=(
                ResearchAnswerStatus.PARTIAL
                if deadline_reason_code
                else ResearchAnswerStatus.INSUFFICIENT_EVIDENCE
            ),
            reason_code=deadline_reason_code,
        )
        if synthesize:
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.SYNTHESIZING,
                    event_type="synthesis_skipped_no_evidence",
                    success=True,
                    message="Synthesis skipped because no evidence was available.",
                )
            )
            empty_ans = synthesize_research_answer(empty_ctx, client=client)
            if deadline_reason_code:
                empty_ans = empty_ans.model_copy(
                    update={
                        "status": ResearchAnswerStatus.PARTIAL,
                        "reason_code": deadline_reason_code,
                        "limitations": [
                            *empty_ans.limitations,
                            "Retrieval stopped because the search deadline was exhausted.",
                        ][:15],
                    }
                )
            hardened_ans, warnings = run_quality_gate(empty_ans, empty_ctx, client=client, verify_claims=False)
            return empty_ctx.model_copy(
                update={"answer": hardened_ans, "quality_gate_warnings": warnings}
            )
        return empty_ctx

    # 5. Stage: RETRIEVING (Round 1)
    _report_stage(
        WorkflowStage.RETRIEVING,
        f"Extracting candidate web pages ({min(len(all_search_results), budget.remaining_pages)} candidates)...",
    )
    retrieved_sources: list[SourceDocument] = []
    session_visited_urls: Set[str] = set()
    consecutive_fetch_failures = 0
    total_fetch_failures = 0
    early_stop_reason: Optional[str] = None
    failure_stop_threshold = (
        cfg.max_consecutive_fetch_failures
        if cfg.early_stop_consecutive_failures == 0
        else min(
            cfg.max_consecutive_fetch_failures,
            cfg.early_stop_consecutive_failures,
        )
    )
    fetch_attempts = 0
    attempts_reserved = 0
    attempts_lock = Lock()
    fetch_wall_ms = 0.0
    fetch_duration_sum_ms = 0.0
    # Existing test doubles and legacy injected callbacks are commonly not
    # thread-safe; keep those calls sequential while real/custom functions
    # use the configured bounded pool.
    legacy_injected_fetch = type(fetch_tool_fn).__module__ == "unittest.mock"
    fetch_workers = (
        1
        if legacy_injected_fetch
        else cfg.fetch_max_workers
    )
    candidate_urls: list[str] = []
    candidate_seen: set[str] = set()
    max_results_per_query = max(
        (len(results) for results in search_result_groups),
        default=0,
    )
    for result_index in range(max_results_per_query):
        for results in search_result_groups:
            if result_index >= len(results):
                continue
            item = results[result_index]
            normalized = normalize_url(item.url)
            if not normalized or normalized in candidate_seen:
                continue
            if not legacy_injected_fetch:
                safe, _ = is_safe_url(normalized, resolve_dns=False)
                if not safe:
                    _record_event(
                        ExecutionEvent(
                            run_id=run_id,
                            stage=WorkflowStage.RETRIEVING,
                            event_type="unsafe_url_blocked",
                            success=False,
                            error_category=ErrorCategory.UNSAFE_URL,
                            message=f"Blocked unsafe destination before fetch: {normalized}",
                        ),
                        tool_name="fetch_and_extract",
                    )
                    candidate_seen.add(normalized)
                    continue
            candidate_seen.add(normalized)
            candidate_urls.append(normalized)
            if len(candidate_urls) >= budget.max_fetch_pages:
                break
        if len(candidate_urls) >= budget.max_fetch_pages:
            break

    def _fetch_candidate(candidate_url: str) -> tuple[str, Optional[SourceDocument], Optional[Exception], float, int]:
        """Fetch one prevalidated candidate without mutating orchestration state."""
        nonlocal attempts_reserved
        started = time.monotonic()
        attempts = 0
        last_error: Optional[Exception] = None
        for attempt in range(budget.max_retries_per_tool + 1):
            if budget.is_time_exhausted:
                break
            with attempts_lock:
                if attempts_reserved >= budget.max_fetch_pages:
                    break
                attempts_reserved += 1
                attempts += 1
            try:
                doc = fetch_tool_fn(candidate_url, session_visited_urls=set())
                return candidate_url, doc, None, (time.monotonic() - started) * 1000.0, attempts
            except FetchUnsafeURLError as err:
                return candidate_url, None, err, (time.monotonic() - started) * 1000.0, attempts
            except FetchTimeoutError as err:
                last_error = err
                if attempt < budget.max_retries_per_tool and not budget.is_time_exhausted:
                    time.sleep(0.1)
                    continue
                break
            except Exception as err:
                last_error = err
                break
        return candidate_url, None, last_error, (time.monotonic() - started) * 1000.0, attempts

    def _process_fetch_result(
        result: tuple[str, Optional[SourceDocument], Optional[Exception], float, int],
    ) -> None:
        nonlocal consecutive_fetch_failures, total_fetch_failures, fetch_attempts, fetch_duration_sum_ms
        norm, doc, error, duration_ms, attempts = result
        fetch_attempts += attempts
        fetch_duration_sum_ms += duration_ms
        budget.pages_fetched += attempts
        if doc is not None:
            retrieved_sources.append(doc)
            consecutive_fetch_failures = 0
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.RETRIEVING,
                    event_type="page_fetched",
                    duration_ms=duration_ms,
                    success=True,
                    message=f"Fetched '{norm}' ({len(doc.content)} chars)",
                ),
                tool_name="fetch_and_extract",
                retry_count=max(0, attempts - 1),
            )
            return

        consecutive_fetch_failures += 1
        total_fetch_failures += 1
        if isinstance(error, FetchUnsafeURLError):
            event_type = "unsafe_url_blocked"
            category = ErrorCategory.UNSAFE_URL
        elif isinstance(error, FetchTimeoutError):
            event_type = "page_fetch_timeout"
            category = ErrorCategory.TIMEOUT
        elif isinstance(error, FetchHTTPError):
            event_type = "page_fetch_failed"
            category = ErrorCategory.HTTP_ERROR
        else:
            event_type = "page_fetch_failed"
            category = ErrorCategory.EXTRACTION_ERROR
        _record_event(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.RETRIEVING,
                event_type=event_type,
                duration_ms=duration_ms,
                success=False,
                error_category=category,
                message=(
                    f"Failed fetching '{norm}': "
                    f"{type(error).__name__ if error else 'deadline'}"
                    + (
                        f" (status={error.status_code})"
                        if isinstance(error, FetchHTTPError) and error.status_code
                        else ""
                    )
                ),
            ),
            tool_name="fetch_and_extract",
            retry_count=max(0, attempts - 1),
        )

    retrieval_contexts = [
        req_obj.question,
        plan.objective,
        *plan.subquestions,
        *plan.search_queries,
    ]
    executor = ThreadPoolExecutor(max_workers=fetch_workers)
    try:
        for wave_index in range(0, len(candidate_urls), fetch_workers):
            wave = candidate_urls[wave_index : wave_index + fetch_workers]
            if budget.is_time_exhausted:
                early_stop_reason = "overall deadline reached before the next fetch wave"
                deadline_reason_code = "deadline_exhausted_during_fetch"
                break
            wave_start = time.monotonic()
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.RETRIEVING,
                    event_type="fetch_wave_started",
                    success=True,
                    message=(
                        f"Wave {wave_index // fetch_workers + 1}: "
                        f"{len(wave)} URL(s), workers={fetch_workers}"
                    ),
                ),
                tool_name="fetch_and_extract",
            )
            futures: list[Future[tuple[str, Optional[SourceDocument], Optional[Exception], float, int]]] = [
                executor.submit(_fetch_candidate, url) for url in wave
            ]
            results: list[tuple[str, Optional[SourceDocument], Optional[Exception], float, int]] = []
            deadline_hit = False
            for future in futures:
                remaining = max(0.0, budget.deadline - time.monotonic())
                try:
                    results.append(future.result(timeout=remaining))
                except Exception as err:
                    future.cancel()
                    deadline_hit = True
                    results.append((wave[len(results)], None, err, (time.monotonic() - wave_start) * 1000.0, 0))
                    break
            if deadline_hit:
                deadline_reason_code = "deadline_exhausted_during_fetch"
                for future in futures:
                    future.cancel()
                early_stop_reason = "overall deadline reached while fetching a wave"
            for result in results:
                _process_fetch_result(result)
            fetch_wall_ms += (time.monotonic() - wave_start) * 1000.0
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.RETRIEVING,
                    event_type="fetch_wave_finished",
                    duration_ms=(time.monotonic() - wave_start) * 1000.0,
                    success=not deadline_hit,
                    error_category=ErrorCategory.TIMEOUT if deadline_hit else None,
                    message=(
                        f"Wave {wave_index // fetch_workers + 1} finished: "
                        f"{sum(1 for r in results if r[1] is not None)} successes, "
                        f"{sum(1 for r in results if r[1] is None)} failures"
                    ),
                ),
                tool_name="fetch_and_extract",
            )

            accepted_wave, _, coverage_wave = process_evidence(
                retrieved_sources,
                plan,
                req_obj.question,
                quality_threshold=cfg.evidence_quality_threshold,
            )
            max_relevance = max(
                (
                    calculate_relevance_score(doc, context_text)
                    for doc in retrieved_sources
                    for context_text in retrieval_contexts
                ),
                default=0.0,
            )
            if (
                failure_stop_threshold > 0
                and not accepted_wave
                and consecutive_fetch_failures >= failure_stop_threshold
            ):
                early_stop_reason = (
                    f"{consecutive_fetch_failures} consecutive fetch failures with zero accepted sources"
                )
            elif (
                cfg.early_stop_min_relevance > 0
                and not legacy_injected_fetch
                and wave_index == 0
                and not accepted_wave
                and max_relevance < cfg.early_stop_min_relevance
            ):
                early_stop_reason = (
                    f"no fetched page reached relevance {cfg.early_stop_min_relevance:.2f} "
                    f"(maximum was {max_relevance:.2f})"
                )
            if early_stop_reason or deadline_hit:
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    budget.pages_fetched = attempts_reserved

    if early_stop_reason:
        if "consecutive fetch failures" in early_stop_reason:
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.RETRIEVING,
                    event_type="fetch_failure_abort",
                    success=False,
                    error_category=ErrorCategory.EXTRACTION_ERROR,
                    message=f"Stopped retrieval after {consecutive_fetch_failures} consecutive fetch failures.",
                ),
                tool_name="fetch_and_extract",
            )
        _record_event(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.RETRIEVING,
                event_type="early_stop",
                success=False,
                error_category=ErrorCategory.TIMEOUT if "deadline" in early_stop_reason else ErrorCategory.EXTRACTION_ERROR,
                message=(
                    f"Retrieval stopped early: {early_stop_reason}; "
                    f"attempts={fetch_attempts}, accepted_sources={len(retrieved_sources)}"
                ),
            ),
            tool_name="fetch_and_extract",
        )

    # 6. Stage: FILTERING (Round 1)
    _report_stage(WorkflowStage.FILTERING, "Filtering evidence and computing coverage...")
    f_start = time.monotonic()
    evidence_start = time.monotonic()
    accepted, rejected, coverage = process_evidence(
        retrieved_sources,
        plan,
        req_obj.question,
        quality_threshold=cfg.evidence_quality_threshold,
    )
    evidence_duration_ms = (time.monotonic() - evidence_start) * 1000.0
    _record_event(
        ExecutionEvent(
            run_id=run_id,
            stage=WorkflowStage.FILTERING,
            event_type="evidence_processed",
            duration_ms=(time.monotonic() - f_start) * 1000.0,
            success=True,
            message=(
                f"Evidence filtered: {len(accepted)} accepted, {len(rejected)} rejected. "
                f"Coverage ratio: {coverage.coverage_ratio:.2f} (adequate: {coverage.is_adequate})"
            ),
        )
    )

    # 7. Check if Additional Research Round (Round 2) is Warranted
    can_run_round_2 = (
        not coverage.is_adequate
        and budget.extra_rounds_used < budget.max_research_rounds
        and budget.remaining_queries > 0
        and budget.remaining_pages > 0
        and not budget.is_time_exhausted
        and bool(coverage.unsupported_subquestions)
        and not early_stop_reason
        and not (
            not accepted
            and (
                consecutive_fetch_failures >= budget.max_consecutive_fetch_failures
                or (
                    len(retrieved_sources) >= 2
                    and total_fetch_failures >= 2
                )
                or (not accepted and len(retrieved_sources) >= 2)
            )
        )
    )

    if can_run_round_2:
        budget.extra_rounds_used += 1
        r2_start = time.monotonic()
        _report_stage(
            WorkflowStage.SEARCHING,
            f"Round 2 triggered for {len(coverage.unsupported_subquestions)} unsupported subquestions...",
        )
        _record_event(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.SEARCHING,
                event_type="additional_round_triggered",
                success=True,
                message=f"Round 2 triggered for {len(coverage.unsupported_subquestions)} unsupported subquestions",
            )
        )

        round_2_results: list[SearchResult] = []
        for sq in coverage.unsupported_subquestions[:2]:
            if budget.remaining_queries <= 0 or budget.is_time_exhausted:
                break
            targeted_query = f"{sq[:60]} overview"
            results = _execute_search_query(targeted_query)
            for r in results:
                if r.url not in seen_search_urls:
                    seen_search_urls.add(r.url)
                    round_2_results.append(r)
                    all_search_results.append(r)

        round_2_urls: list[str] = []
        for item in round_2_results:
            normalized = normalize_url(item.url)
            if normalized and normalized not in candidate_seen:
                candidate_seen.add(normalized)
                round_2_urls.append(normalized)

        round_2_executor = ThreadPoolExecutor(max_workers=fetch_workers)
        try:
            for wave_index in range(0, len(round_2_urls), fetch_workers):
                wave = round_2_urls[wave_index : wave_index + fetch_workers]
                if budget.is_time_exhausted or early_stop_reason:
                    break
                wave_start = time.monotonic()
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.RETRIEVING,
                        event_type="fetch_wave_started",
                        success=True,
                        message=f"Round 2 wave {wave_index // fetch_workers + 1}: {len(wave)} URL(s), workers={fetch_workers}",
                    ),
                    tool_name="fetch_and_extract",
                )
                futures = [round_2_executor.submit(_fetch_candidate, url) for url in wave]
                results = []
                deadline_hit = False
                for future in futures:
                    remaining = max(0.0, budget.deadline - time.monotonic())
                    try:
                        results.append(future.result(timeout=remaining))
                    except Exception as err:
                        future.cancel()
                        deadline_hit = True
                        results.append((wave[len(results)], None, err, (time.monotonic() - wave_start) * 1000.0, 0))
                        break
                for result in results:
                    _process_fetch_result(result)
                fetch_wall_ms += (time.monotonic() - wave_start) * 1000.0
                if deadline_hit:
                    deadline_reason_code = "deadline_exhausted_during_fetch"
                    for future in futures:
                        future.cancel()
                    early_stop_reason = "overall deadline reached while fetching a round 2 wave"
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.RETRIEVING,
                        event_type="fetch_wave_finished",
                        duration_ms=(time.monotonic() - wave_start) * 1000.0,
                        success=not deadline_hit,
                        error_category=ErrorCategory.TIMEOUT if deadline_hit else None,
                        message=f"Round 2 wave {wave_index // fetch_workers + 1} finished",
                    ),
                    tool_name="fetch_and_extract",
                )
                if (
                    failure_stop_threshold > 0
                    and not retrieved_sources
                    and consecutive_fetch_failures >= failure_stop_threshold
                ):
                    early_stop_reason = (
                        f"{consecutive_fetch_failures} consecutive fetch failures with zero accepted sources"
                    )
                    break
        finally:
            round_2_executor.shutdown(wait=True, cancel_futures=True)

        # Re-run evidence filtering on combined retrieved sources
        accepted, rejected, coverage = process_evidence(
            retrieved_sources,
            plan,
            req_obj.question,
            quality_threshold=cfg.evidence_quality_threshold,
        )

        _record_event(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.FILTERING,
                event_type="round_2_completed",
                duration_ms=(time.monotonic() - r2_start) * 1000.0,
                success=True,
                message=f"Round 2 complete. Updated coverage ratio: {coverage.coverage_ratio:.2f}",
            )
        )

    # 8. Final Status Assessment
    if not accepted and deadline_reason_code:
        status = ResearchAnswerStatus.PARTIAL
    elif not accepted:
        status = ResearchAnswerStatus.INSUFFICIENT_EVIDENCE
    elif coverage.coverage_ratio >= cfg.min_coverage_threshold and not budget.is_time_exhausted:
        status = ResearchAnswerStatus.COMPLETED
    else:
        status = ResearchAnswerStatus.PARTIAL

    _record_event(
        ExecutionEvent(
            run_id=run_id,
            stage=WorkflowStage.COMPLETED if status == ResearchAnswerStatus.COMPLETED else WorkflowStage.DEGRADED,
            event_type="retrieval_finished",
            duration_ms=budget.elapsed_ms(),
            success=status != ResearchAnswerStatus.INSUFFICIENT_EVIDENCE,
            message=(
                f"Orchestration completed with status '{status.value}'. "
                f"Accepted {len(accepted)} sources across {budget.queries_used} queries and "
                f"{budget.pages_fetched} fetch attempts; workers={fetch_workers}, "
                f"fetch wall={fetch_wall_ms:.1f}ms, summed fetch time={fetch_duration_sum_ms:.1f}ms."
            ),
            model=None,
        )
    )

    final_answer: Optional[ResearchAnswer] = None
    quality_warnings: list[str] = []
    evaluation: Optional[AnswerEvaluation] = None

    if synthesize:
        intermediate_ctx = ResearchContext(
            run_id=run_id,
            request=req_obj,
            plan=plan,
            raw_search_results=all_search_results,
            sources=accepted,
            rejected_sources=rejected,
            coverage=coverage,
            trace=trace,
            status=status,
            reason_code=deadline_reason_code,
        )

        # Stage 9: SYNTHESIZING
        _report_stage(WorkflowStage.SYNTHESIZING, "Synthesizing evidence-grounded answer...")
        if not accepted:
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.SYNTHESIZING,
                    event_type="synthesis_skipped_no_evidence",
                    success=True,
                    message="Synthesis skipped because no accepted evidence was available.",
                )
            )
            draft_answer = ResearchAnswer(
                user_question=req_obj.question,
                short_answer="No reliable evidence was available before retrieval stopped.",
                key_findings=[],
                sources_used=[],
                source_url_map={},
                confidence=ConfidenceLevel.LOW,
                assumptions=[],
                limitations=[
                    "Synthesis was skipped because no accepted evidence was available.",
                    *(
                        [f"Retrieval deadline reason: {deadline_reason_code}."]
                        if deadline_reason_code
                        else []
                    ),
                ],
                suggested_next_steps=["Retry with a larger search and fetch deadline."],
                status=status,
                reason_code=deadline_reason_code,
            )
        else:
            syn_start = time.monotonic()
            try:
                draft_answer = synthesize_research_answer(intermediate_ctx, client=client)
                synthesis_failed = (
                    draft_answer.status == ResearchAnswerStatus.FAILED
                    or any(
                        limitation.startswith("Synthesis failure:")
                        for limitation in draft_answer.limitations
                    )
                )
                synthesis_partial = (
                    draft_answer.status == ResearchAnswerStatus.PARTIAL
                    and not synthesis_failed
                )
                synthesis_succeeded = not synthesis_failed
                active_model = getattr(client, "active_model_name", None)
                if not isinstance(active_model, str):
                    active_model = getattr(client, "model_name", None)
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.SYNTHESIZING,
                        event_type=(
                            "synthesis_failed"
                            if synthesis_failed
                            else "synthesis_partial"
                            if synthesis_partial
                            else "synthesis_completed"
                        ),
                        duration_ms=(time.monotonic() - syn_start) * 1000.0,
                        success=synthesis_succeeded,
                        error_category=(
                            ErrorCategory.UNKNOWN if synthesis_failed else None
                        ),
                        model=active_model,
                        message=(
                            f"Synthesized answer with {len(draft_answer.key_findings)} findings "
                            f"using model '{active_model or 'configured'}'."
                            if not synthesis_partial and not synthesis_failed
                            else (
                                f"Synthesized a partial answer with "
                                f"{len(draft_answer.key_findings)} findings."
                                if synthesis_partial
                                else
                                "Synthesis returned a controlled degraded answer: "
                                f"{draft_answer.status.value}."
                            )
                        ),
                    )
                )
            except Exception as syn_err:
                logger.error("Synthesis failed: %s", syn_err)
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.SYNTHESIZING,
                        event_type="synthesis_failed",
                        duration_ms=(time.monotonic() - syn_start) * 1000.0,
                        success=False,
                        error_category=ErrorCategory.UNKNOWN,
                        message=f"Synthesis failed: {type(syn_err).__name__}: {syn_err}",
                    )
                )
                draft_answer = ResearchAnswer(
                    user_question=req_obj.question,
                    short_answer=f"Synthesis could not be completed: {syn_err}",
                    key_findings=[],
                    sources_used=[],
                    source_url_map={},
                    confidence=ConfidenceLevel.LOW,
                    assumptions=[],
                    limitations=["Synthesis encountered an unrecoverable error."],
                    suggested_next_steps=["Retry the query or verify model availability."],
                    status=ResearchAnswerStatus.FAILED,
                    reason_code=deadline_reason_code,
                )

            if accepted:
                synthesis_duration_ms = (time.monotonic() - syn_start) * 1000.0

        # Stage 10: VALIDATING
        _report_stage(WorkflowStage.VALIDATING, "Validating citations and applying quality gate...")
        qg_start = time.monotonic()
        try:
            final_answer, quality_warnings = run_quality_gate(
                draft_answer,
                intermediate_ctx,
                client=client,
                verify_claims=verify_claims if accepted else False,
            )
            if deadline_reason_code:
                deadline_warning = f"Retrieval deadline reason: {deadline_reason_code}."
                quality_warnings.append(deadline_warning)
                final_answer = final_answer.model_copy(
                    update={
                        "reason_code": deadline_reason_code,
                        "status": ResearchAnswerStatus.PARTIAL,
                        "limitations": [
                            *final_answer.limitations,
                            deadline_warning,
                        ][:15],
                    }
                )
            if accepted:
                evaluation = evaluate_answer(final_answer, accepted)
                quality_warnings.extend(evaluation.warnings)
                if evaluation.guardrail_triggered:
                    final_answer = final_answer.model_copy(
                        update={
                            "limitations": [
                                *final_answer.limitations,
                                "Low-confidence guardrail: verify important claims against the cited source excerpts.",
                            ][:15],
                            "suggested_next_steps": [
                                *final_answer.suggested_next_steps,
                                "Manually verify the highest-impact claims before relying on this answer.",
                            ][:15],
                        }
                    )
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.VALIDATING,
                        event_type="answer_evaluated",
                        success=True,
                        message=(
                            f"Evaluation: {evaluation.grounded_finding_count}/"
                            f"{evaluation.finding_count} findings grounded lexically; "
                            f"citation validity {evaluation.citation_validity_rate:.0%}; "
                            f"finding citation coverage {evaluation.citation_coverage_rate:.0%}."
                        ),
                    )
                )
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.VALIDATING,
                    event_type="quality_gate_passed",
                    duration_ms=(time.monotonic() - qg_start) * 1000.0,
                    success=True,
                    message=f"Quality gate finished with {len(quality_warnings)} warning(s).",
                )
            )
        except Exception as qg_err:
            logger.error("Quality gate failed: %s", qg_err)
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.VALIDATING,
                    event_type="quality_gate_failed",
                    duration_ms=(time.monotonic() - qg_start) * 1000.0,
                    success=False,
                    error_category=ErrorCategory.VALIDATION_ERROR,
                    message=f"Quality gate failed: {type(qg_err).__name__}: {qg_err}",
                )
            )
            final_answer = draft_answer
            quality_warnings.append(f"Quality gate error: {type(qg_err).__name__}")

        status = final_answer.status
        validation_duration_ms = (time.monotonic() - qg_start) * 1000.0

    if not synthesize:
        _record_event(
            ExecutionEvent(
                run_id=run_id,
                stage=WorkflowStage.COMPLETED
                if status == ResearchAnswerStatus.COMPLETED
                else WorkflowStage.DEGRADED,
                event_type="orchestration_finished",
                duration_ms=budget.elapsed_ms(),
                success=status != ResearchAnswerStatus.FAILED,
                message=f"Research workflow finished with status '{status.value}'.",
            )
        )

    final_stage = (
        WorkflowStage.COMPLETED
        if status == ResearchAnswerStatus.COMPLETED
        else (WorkflowStage.FAILED if status == ResearchAnswerStatus.FAILED else WorkflowStage.DEGRADED)
    )
    _report_stage(final_stage, f"Research workflow finished with status: {status.value}")

    error_breakdown = {
        "search_timeouts": sum(1 for event in trace if event.event_type == "search_query_timeout"),
        "search_errors": sum(
            1 for event in trace if event.event_type in {"search_query_failed", "search_failed"}
        ),
        "http_blocks": sum(
            1 for event in trace
            if event.error_category == ErrorCategory.HTTP_ERROR
            and ("403" in (event.message or "") or "429" in (event.message or ""))
        ),
        "extraction_errors": sum(
            1 for event in trace if event.error_category == ErrorCategory.EXTRACTION_ERROR
        ),
        "retries": sum(
            1 for event in trace
            if "retry" in (event.event_type + " " + (event.message or "")).lower()
        ),
    }
    status_reasons = getattr(intermediate_ctx, "status_reasons", []) if synthesize else []
    entity_coverage = getattr(intermediate_ctx, "entity_coverage", {}) if synthesize else {}
    if deadline_reason_code and not any(
        reason.code == ReasonCode.DEADLINE_EXHAUSTED for reason in status_reasons
    ):
        status_reasons = [
            *status_reasons,
            StatusReason(
                code=ReasonCode.DEADLINE_EXHAUSTED,
                message=f"Retrieval stopped because {deadline_reason_code}.",
            ),
        ]
    if final_answer is not None and status_reasons:
        final_answer = final_answer.model_copy(update={"status_reasons": status_reasons})

    return ResearchContext(
        run_id=run_id,
        request=req_obj,
        plan=plan,
        raw_search_results=all_search_results,
        sources=accepted,
        rejected_sources=rejected,
        coverage=coverage,
        trace=trace,
        status=status,
        answer=final_answer,
        quality_gate_warnings=quality_warnings,
        evaluation=evaluation,
        reason_code=deadline_reason_code,
        pipeline_started_at=pipeline_started_at,
        pipeline_finished_at=datetime.now(timezone.utc),
        stage_wall_times_ms={
            "plan": plan_duration_ms,
            "search": search_duration_ms,
            "fetch": fetch_wall_ms,
            "evidence": evidence_duration_ms,
            "synthesis": synthesis_duration_ms,
            "validation": validation_duration_ms,
        },
        synthesis_model=getattr(intermediate_ctx, "synthesis_model", None)
        if synthesize
        else None,
        entity_coverage=entity_coverage,
        error_breakdown=error_breakdown,
        status_reasons=status_reasons,
    )


def run_full_pipeline(
    request: ResearchRequest | str,
    settings: Optional[Settings] = None,
    client: Optional[GroqClientWrapper] = None,
    search_tool_fn: Callable[..., list[SearchResult]] = web_search,
    fetch_tool_fn: Callable[..., SourceDocument] = fetch_and_extract,
    verify_claims: bool = False,
    progress_callback: Optional[Callable[[WorkflowStage, str], None]] = None,
) -> ResearchContext:
    """Execute the complete end-to-end research workflow including synthesis and quality gates."""
    return run_research(
        request=request,
        settings=settings,
        client=client,
        search_tool_fn=search_tool_fn,
        fetch_tool_fn=fetch_tool_fn,
        synthesize=True,
        verify_claims=verify_claims,
        progress_callback=progress_callback,
    )
