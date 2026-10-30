r"""Offline latency benchmark for bounded parallel page retrieval.

Run with:
    .venv\Scripts\python.exe evaluation\latency_benchmark.py
    .venv\Scripts\python.exe evaluation\latency_benchmark.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, get_settings
from app.orchestrator import run_full_pipeline, run_research
from app.schemas import ExtractionStatus, SearchResult, SourceDocument


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "results" / "latency_benchmark.json"


def _doc(url: str, index: int) -> SourceDocument:
    return SourceDocument(
        source_id=f"src_{index:012x}",
        url=url,
        title=f"Vector database benchmark source {index}",
        domain=f"docs{index}.example.com",
        content=(
            f"Vector database architecture, memory, clustering, licensing, and RAG "
            f"details for source {index}. "
        )
        * 25,
        quality_score=1.0,
        extraction_status=ExtractionStatus.SUCCESS,
    )


def run_offline(workers: int) -> dict[str, object]:
    urls = [f"https://docs.example.com/{index}" for index in range(6)]
    results = [
        SearchResult(title=f"Source {index}", url=url, snippet="vector database")
        for index, url in enumerate(urls)
    ]

    def search(_query: str, **_kwargs: object) -> list[SearchResult]:
        return results

    def fetch(url: str, **_kwargs: object) -> SourceDocument:
        time.sleep(0.2)
        return _doc(url, int(url.rsplit("/", 1)[-1]))

    class Client:
        model_name = "benchmark-model"

        def chat_completion(self, **_kwargs: object) -> str:
            return (
                '{"objective":"Compare vector databases","subquestions":["memory","clustering"],'
                '"search_queries":["vector database"],"tools":["web_search","fetch_and_extract"],'
                '"success_criteria":["memory"],"requires_fresh_evidence":false}'
            )

    settings = Settings(
        fetch_max_workers=workers,
        max_fetch_pages=6,
        max_research_rounds=0,
        max_consecutive_fetch_failures=0,
        early_stop_consecutive_failures=0,
        early_stop_min_relevance=0,
    )
    started = time.perf_counter()
    context = run_research(
        "Compare vector databases for a startup building RAG products",
        settings=settings,
        client=Client(),
        search_tool_fn=search,
        fetch_tool_fn=fetch,
    )
    elapsed = time.perf_counter() - started
    attempts = sum(
        1
        for event in context.trace
        if event.event_type in {"page_fetched", "page_fetch_failed", "page_fetch_timeout"}
    )
    return {
        "workers": workers,
        "wall_time_seconds": round(elapsed, 4),
        "fetch_attempts": attempts,
        "accepted_sources": len(context.sources),
        "status": context.status.value,
    }


def run_live() -> list[dict[str, object]]:
    settings = get_settings(reload=True)
    if not settings.groq_api_key:
        print("--live skipped: GROQ_API_KEY is not configured.")
        return []
    cases_path = ROOT / "evaluation" / "test_cases.json"
    cases = {case["id"]: case for case in json.loads(cases_path.read_text(encoding="utf-8"))}
    records: list[dict[str, object]] = []
    for case_id in ("E1", "E10"):
        for workers in (1, 3):
            run_settings = Settings(**{**settings.__dict__, "fetch_max_workers": workers})
            started = time.perf_counter()
            context = run_full_pipeline(cases[case_id]["question"], settings=run_settings)
            records.append(
                {
                    "case_id": case_id,
                    "workers": workers,
                    "wall_time_seconds": round(time.perf_counter() - started, 4),
                    "status": context.status.value,
                    "accepted_sources": len(context.sources),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run E1 and E10 against live providers.")
    args = parser.parse_args()
    offline = [run_offline(workers) for workers in (1, 2, 3, 4)]
    baseline = float(offline[0]["wall_time_seconds"])
    for row in offline:
        row["speedup_vs_1"] = round(baseline / float(row["wall_time_seconds"]), 2)
    payload: dict[str, object] = {"offline": offline}
    if args.live:
        payload["live"] = run_live()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("workers | wall time (s) | speedup vs 1 | fetch attempts | accepted sources")
    for row in offline:
        print(
            f"{row['workers']:>7} | {row['wall_time_seconds']:>14} | "
            f"{row['speedup_vs_1']:>13} | {row['fetch_attempts']:>14} | "
            f"{row['accepted_sources']:>16}"
        )
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
