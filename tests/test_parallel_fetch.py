"""Deterministic regression tests for bounded wave-based page fetching."""

from __future__ import annotations

import threading
import time

from app.config import Settings
from app.orchestrator import run_research
from app.schemas import ExtractionStatus, ResearchAnswerStatus, SearchResult, SourceDocument


def _doc(url: str, index: int) -> SourceDocument:
    return SourceDocument(
        source_id=f"src_{index:012x}",
        url=url,
        title=f"Vector database guide {index}",
        domain=f"docs{index}.example.com",
        content=(
            "Vector database memory, clustering, licensing, and RAG architecture "
            f"details for a startup research comparison, source variant {index}. "
        )
        * 20,
        quality_score=1.0,
        extraction_status=ExtractionStatus.SUCCESS,
    )


def _run(fetch, urls: list[str], **settings_overrides):
    results = [SearchResult(title=f"Page {i}", url=url, snippet="vector database") for i, url in enumerate(urls)]
    plan_json = (
        '{"objective":"Compare vector databases","subquestions":["memory","clustering"],'
        '"search_queries":["vector database"],"tools":["web_search","fetch_and_extract"],'
        '"success_criteria":["memory"],"requires_fresh_evidence":false}'
    )

    def search(_query, **_kwargs):
        return results

    class Client:
        model_name = "test-model"

        def chat_completion(self, **_kwargs):
            return plan_json

    return run_research(
        "Compare vector databases for RAG",
        settings=Settings(
            max_research_rounds=0,
            early_stop_min_relevance=0,
            **settings_overrides,
        ),
        client=Client(),
        search_tool_fn=search,
        fetch_tool_fn=fetch,
    )


def test_results_keep_original_candidate_order():
    def fetch(url, **_kwargs):
        time.sleep(0.03 if url.endswith("/1") else 0.001)
        return _doc(url, int(url.rsplit("/", 1)[-1]))

    context = _run(fetch, [f"https://docs.example.com/{i}" for i in range(3)])
    assert [source.url for source in context.sources] == [
        "https://docs.example.com/0",
        "https://docs.example.com/1",
        "https://docs.example.com/2",
    ]


def test_worker_limit_and_speedup():
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fetch(url, **_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return _doc(url, int(url.rsplit("/", 1)[-1]))

    started = time.monotonic()
    _run(fetch, [f"https://docs.example.com/{i}" for i in range(6)], fetch_max_workers=3)
    elapsed = time.monotonic() - started
    assert maximum <= 3
    assert elapsed < 0.26


def test_fetch_budget_and_duplicate_normalization():
    calls: list[str] = []

    def fetch(url, **_kwargs):
        calls.append(url)
        return _doc(url, len(calls))

    context = _run(
        fetch,
        [
            "https://docs.example.com/a?utm_source=x",
            "https://docs.example.com/a",
            "https://docs.example.com/b",
            "https://docs.example.com/c",
        ],
        max_fetch_pages=2,
        fetch_max_workers=2,
    )
    assert len(calls) <= 2
    assert len({url for url in calls}) == len(calls)
    assert len(context.trace) > 0


def test_retries_cannot_exceed_total_fetch_attempt_budget():
    calls: list[str] = []

    def fetch(url, **_kwargs):
        calls.append(url)
        raise RuntimeError("temporary failure")

    _run(
        fetch,
        [f"https://docs.example.com/{i}" for i in range(4)],
        max_fetch_pages=2,
        max_retries_per_tool=1,
        fetch_max_workers=2,
    )
    assert len(calls) <= 2


def test_failure_isolated_and_unsafe_candidate_never_called():
    calls: list[str] = []

    def fetch(url, **_kwargs):
        calls.append(url)
        if url.endswith("/bad"):
            raise RuntimeError("offline failure")
        return _doc(url, 1)

    context = _run(
        fetch,
        ["http://127.0.0.1/admin", "https://docs.example.com/bad", "https://docs.example.com/good"],
        fetch_max_workers=2,
    )
    assert "http://127.0.0.1/admin" not in calls
    assert "https://docs.example.com/good" in calls
    assert any(event.event_type == "fetch_wave_started" for event in context.trace)


def test_early_stop_returns_insufficient_evidence_and_warning():
    def fetch(_url, **_kwargs):
        raise RuntimeError("unavailable")

    context = _run(
        fetch,
        [f"https://docs.example.com/{i}" for i in range(6)],
        fetch_max_workers=3,
        early_stop_consecutive_failures=3,
    )
    assert context.status == ResearchAnswerStatus.INSUFFICIENT_EVIDENCE
    assert any(event.event_type == "early_stop" for event in context.trace)


def test_fetch_workers_one_is_supported():
    def fetch(url, **_kwargs):
        return _doc(url, int(url.rsplit("/", 1)[-1]))

    context = _run(
        fetch,
        ["https://docs.example.com/0", "https://docs.example.com/1"],
        fetch_max_workers=1,
    )
    assert context.status in {
        ResearchAnswerStatus.COMPLETED,
        ResearchAnswerStatus.PARTIAL,
    }
