"""Deterministic Bounded Orchestrator for ResearchPilot.

Coordinates the complete research lifecycle across planning, web search, safe page
extraction, and evidence quality filtering. Enforces strict tool allowlists, query/page/time
budgets, bounded retries, and records full runtime trace events without leaking secrets.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional, Set

from app.config import Settings, get_settings
from app.evidence import process_evidence
from app.evaluator import evaluate_answer
from app.llm import GroqClientWrapper
from app.logging_config import log_execution_event
from app.planner import PlannerError, create_research_plan
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
)
from app.synthesizer import synthesize_research_answer
from app.tools.fetch import (
    FetchError,
    FetchInvalidInputError,
    FetchTimeoutError,
    FetchUnsafeURLError,
    fetch_and_extract,
    generate_source_id,
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
    run_id = f"run_{uuid.uuid4().hex[:12]}"
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
    seen_search_urls: set[str] = set()

    def _execute_search_query(q_str: str) -> list[SearchResult]:
        if budget.is_time_exhausted or budget.remaining_queries <= 0:
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
                _record_event(
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
                _record_event(
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
                _record_event(
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

    for query in plan.search_queries:
        if budget.remaining_queries <= 0 or budget.is_time_exhausted:
            break
        results = _execute_search_query(query)
        for r in results:
            if r.url not in seen_search_urls:
                seen_search_urls.add(r.url)
                all_search_results.append(r)

    # If completely 0 search results found across all queries
    if not all_search_results:
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
            status=ResearchAnswerStatus.INSUFFICIENT_EVIDENCE,
        )
        if synthesize:
            empty_ans = synthesize_research_answer(empty_ctx, client=client)
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

    def _fetch_candidate(candidate_url: str) -> Optional[SourceDocument]:
        if budget.is_time_exhausted or budget.remaining_pages <= 0:
            return None

        norm = normalize_url(candidate_url)
        if norm in session_visited_urls:
            return None

        budget.pages_fetched += 1
        f_start = time.monotonic()

        for attempt in range(budget.max_retries_per_tool + 1):
            try:
                doc = fetch_tool_fn(
                    norm,
                    session_visited_urls=session_visited_urls,
                )
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.RETRIEVING,
                        event_type="page_fetched",
                        duration_ms=(time.monotonic() - f_start) * 1000.0,
                        success=True,
                        message=f"Fetched '{norm}' ({len(doc.content)} chars)",
                    ),
                    tool_name="fetch_and_extract",
                    retry_count=attempt,
                )
                return doc
            except FetchUnsafeURLError as unsafe_err:
                # NEVER retry unsafe URLs!
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.RETRIEVING,
                        event_type="unsafe_url_blocked",
                        duration_ms=(time.monotonic() - f_start) * 1000.0,
                        success=False,
                        error_category=ErrorCategory.UNSAFE_URL,
                        message=f"Blocked unsafe destination: {unsafe_err}",
                    ),
                    tool_name="fetch_and_extract",
                    retry_count=attempt,
                )
                return None
            except FetchTimeoutError as timeout_err:
                if attempt < budget.max_retries_per_tool and not budget.is_time_exhausted:
                    logger.warning("Page fetch timed out; retrying once...")
                    time.sleep(0.1)
                    continue
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.RETRIEVING,
                        event_type="page_fetch_timeout",
                        duration_ms=(time.monotonic() - f_start) * 1000.0,
                        success=False,
                        error_category=ErrorCategory.TIMEOUT,
                        message=f"Timeout fetching '{norm}': {timeout_err}",
                    ),
                    tool_name="fetch_and_extract",
                    retry_count=attempt,
                )
                return None
            except Exception as err:
                _record_event(
                    ExecutionEvent(
                        run_id=run_id,
                        stage=WorkflowStage.RETRIEVING,
                        event_type="page_fetch_failed",
                        duration_ms=(time.monotonic() - f_start) * 1000.0,
                        success=False,
                        error_category=ErrorCategory.EXTRACTION_ERROR,
                        message=f"Failed fetching '{norm}': {type(err).__name__}",
                    ),
                    tool_name="fetch_and_extract",
                    retry_count=attempt,
                )
                return None
        return None

    # Pick candidate URLs up to page budget
    for item in all_search_results:
        if budget.remaining_pages <= 0 or budget.is_time_exhausted:
            break
        doc = _fetch_candidate(item.url)
        if doc is not None:
            retrieved_sources.append(doc)

    # 6. Stage: FILTERING (Round 1)
    _report_stage(WorkflowStage.FILTERING, "Filtering evidence and computing coverage...")
    f_start = time.monotonic()
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

        for item in round_2_results:
            if budget.remaining_pages <= 0 or budget.is_time_exhausted:
                break
            doc = _fetch_candidate(item.url)
            if doc is not None:
                retrieved_sources.append(doc)

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
    if not accepted:
        status = ResearchAnswerStatus.INSUFFICIENT_EVIDENCE
    elif coverage.coverage_ratio >= cfg.min_coverage_threshold and not budget.is_time_exhausted:
        status = ResearchAnswerStatus.COMPLETED
    else:
        status = ResearchAnswerStatus.PARTIAL

    _record_event(
        ExecutionEvent(
            run_id=run_id,
            stage=WorkflowStage.COMPLETED if status == ResearchAnswerStatus.COMPLETED else WorkflowStage.DEGRADED,
            event_type="orchestration_finished",
            duration_ms=budget.elapsed_ms(),
            success=status != ResearchAnswerStatus.INSUFFICIENT_EVIDENCE,
            message=(
                f"Orchestration completed with status '{status.value}'. "
                f"Accepted {len(accepted)} sources across {budget.queries_used} queries and {budget.pages_fetched} page fetches."
            ),
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
        )

        # Stage 9: SYNTHESIZING
        _report_stage(WorkflowStage.SYNTHESIZING, "Synthesizing evidence-grounded answer...")
        syn_start = time.monotonic()
        try:
            draft_answer = synthesize_research_answer(intermediate_ctx, client=client)
            synthesis_succeeded = draft_answer.status in {
                ResearchAnswerStatus.COMPLETED,
                ResearchAnswerStatus.INSUFFICIENT_EVIDENCE,
            }
            _record_event(
                ExecutionEvent(
                    run_id=run_id,
                    stage=WorkflowStage.SYNTHESIZING,
                    event_type=(
                        "synthesis_completed"
                        if synthesis_succeeded
                        else "synthesis_failed"
                    ),
                    duration_ms=(time.monotonic() - syn_start) * 1000.0,
                    success=synthesis_succeeded,
                    error_category=(
                        None if synthesis_succeeded else ErrorCategory.UNKNOWN
                    ),
                    message=(
                        f"Synthesized answer with {len(draft_answer.key_findings)} findings."
                        if synthesis_succeeded
                        else (
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
            )

        # Stage 10: VALIDATING
        _report_stage(WorkflowStage.VALIDATING, "Validating citations and applying quality gate...")
        qg_start = time.monotonic()
        try:
            final_answer, quality_warnings = run_quality_gate(
                draft_answer,
                intermediate_ctx,
                client=client,
                verify_claims=verify_claims,
            )
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
                        f"citation validity {evaluation.citation_validity_rate:.0%}."
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

    final_stage = (
        WorkflowStage.COMPLETED
        if status == ResearchAnswerStatus.COMPLETED
        else (WorkflowStage.FAILED if status == ResearchAnswerStatus.FAILED else WorkflowStage.DEGRADED)
    )
    _report_stage(final_stage, f"Research workflow finished with status: {status.value}")

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
