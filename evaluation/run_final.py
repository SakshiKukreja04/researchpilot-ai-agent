"""Run the final evaluation suite against the real ResearchPilot pipeline.

This runner intentionally does not inject clients or tool functions. It keeps
each completed case on disk so an interrupted live run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.orchestrator import run_full_pipeline


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"suite": path.stem, "version": "unknown", "cases": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"{path} must contain a 'cases' list")
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _error_type(event: dict[str, Any]) -> str:
    if event.get("success", True):
        return ""
    stage = str(event.get("stage", "")).upper()
    event_type = str(event.get("event_type", "")).lower()
    category = str(event.get("error_category", "")).lower()
    if stage == "SEARCHING" or "search" in event_type:
        return "search"
    if stage == "RETRIEVING" or "fetch" in event_type or category in {
        "http_error",
        "unsafe_url",
        "extraction_error",
    }:
        return "fetch"
    if stage in {"PLANNED", "SYNTHESIZING"} or "synthesis" in event_type:
        return "llm"
    return "other"


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(record.get("observed_status", "unknown")) for record in records)
    walls = [
        float(record["wall_time_seconds"])
        for record in records
        if isinstance(record.get("wall_time_seconds"), (int, float))
    ]
    return {
        "count_by_status": dict(sorted(statuses.items())),
        "count_meeting_expectation": sum(
            bool(record.get("meets_expectation")) for record in records
        ),
        "median_wall_time_seconds": round(statistics.median(walls), 3) if walls else 0.0,
        "min_wall_time_seconds": round(min(walls), 3) if walls else 0.0,
        "max_wall_time_seconds": round(max(walls), 3) if walls else 0.0,
    }


def _metadata(suite: dict[str, Any], settings: Any) -> dict[str, Any]:
    return {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "groq_model": settings.groq_model,
        "fallback_model": settings.groq_synthesis_fallback_model,
        "search_backend": f"DDGS with {settings.search_fallback_backend or 'no'} fallback",
        "settings": {
            "overall_deadline_seconds": settings.overall_deadline_seconds,
            "max_search_queries": settings.max_search_queries,
            "max_fetch_pages": settings.max_fetch_pages,
            "fetch_max_workers": settings.fetch_max_workers,
            "search_query_timeout_seconds": settings.search_query_timeout_seconds,
            "early_stop_consecutive_failures": settings.early_stop_consecutive_failures,
            "early_stop_min_relevance": settings.early_stop_min_relevance,
        },
        "suite": suite.get("suite", "ResearchPilot evaluation"),
        "suite_version": suite.get("version", "unknown"),
    }


def _record_case(case: dict[str, Any], context: Any, started: datetime, wall: float) -> dict[str, Any]:
    answer = context.answer
    evaluation = context.evaluation
    trace = [event.model_dump(mode="json") for event in context.trace]
    allowed = list(case.get("allowed_statuses", []))
    observed = context.status.value
    models = sorted(
        {
            value
            for value in [context.synthesis_model, *[event.get("model") for event in trace]]
            if isinstance(value, str) and value
        }
    )
    errors = Counter(
        kind for event in trace if (kind := _error_type(event))
    )
    return {
        "id": case["id"],
        "category": case.get("category", "unspecified"),
        "question": case["question"],
        "expected_behavior": case.get("expected_behavior", case.get("expected", "")),
        "allowed_statuses": allowed,
        "observed_status": observed,
        "meets_expectation": observed in allowed,
        "manual_review": bool(case.get("manual_review", False)),
        "wall_time_seconds": round(wall, 3),
        "coverage": context.coverage.coverage_ratio,
        "confidence": answer.confidence.value if answer else None,
        "accepted_sources": len(context.sources),
        "rejected_sources": len(context.rejected_sources),
        "findings_count": len(answer.key_findings) if answer else 0,
        "citation_validity": evaluation.citation_validity_rate if evaluation else None,
        "short_answer": answer.short_answer if answer else None,
        "limitations": answer.limitations if answer else [],
        "warnings": context.quality_gate_warnings,
        "error_categories": dict(sorted(errors.items())),
        "models_used": models,
        "event_count": len(trace),
        "run_id": context.run_id,
        "start_timestamp": started.isoformat(),
        "end_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _exception_record(case: dict[str, Any], started: datetime, wall: float, error: Exception) -> dict[str, Any]:
    allowed = list(case.get("allowed_statuses", []))
    return {
        "id": case["id"],
        "category": case.get("category", "unspecified"),
        "question": case["question"],
        "expected_behavior": case.get("expected_behavior", case.get("expected", "")),
        "allowed_statuses": allowed,
        "observed_status": "failed",
        "meets_expectation": "failed" in allowed,
        "manual_review": bool(case.get("manual_review", False)),
        "wall_time_seconds": round(wall, 3),
        "coverage": 0.0,
        "confidence": None,
        "accepted_sources": 0,
        "rejected_sources": 0,
        "findings_count": 0,
        "citation_validity": None,
        "short_answer": None,
        "limitations": [],
        "warnings": [f"Runner exception: {type(error).__name__}: {error}"],
        "error_categories": {"other": 1},
        "models_used": [],
        "event_count": 0,
        "run_id": None,
        "start_timestamp": started.isoformat(),
        "end_timestamp": datetime.now(timezone.utc).isoformat(),
        "error": f"{type(error).__name__}: {error}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final evaluation cases live.")
    parser.add_argument("--cases", default="evaluation/final_test_cases.json")
    parser.add_argument("--output", default="evaluation/results/final_results.json")
    parser.add_argument("--only", help="Comma-separated case IDs, for example E1,EC3.")
    parser.add_argument("--delay", type=float, default=15.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be non-negative")

    suite = _load_suite(_resolve(args.cases))
    cases = suite["cases"]
    only = {item.strip() for item in args.only.split(",") if item.strip()} if args.only else None
    output_path = _resolve(args.output)
    trace_dir = output_path.parent / "final_traces"
    records_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        records_by_id = {
            record["id"]: record
            for record in existing.get("records", [])
            if isinstance(record, dict) and record.get("id")
        }

    settings = get_settings(reload=True)
    selected = [case for case in cases if not only or case.get("id") in only]
    for index, case in enumerate(selected):
        case_id = str(case["id"])
        if args.resume and case_id in records_by_id:
            print(f"{case_id}: skipped (already recorded)")
            continue
        if index and args.delay:
            time.sleep(args.delay)
        started = datetime.now(timezone.utc)
        timer = time.monotonic()
        try:
            # Deliberately use the real default client, search, and fetch tools.
            context = run_full_pipeline(case["question"])
            record = _record_case(case, context, started, time.monotonic() - timer)
            trace_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                trace_dir / f"{case_id}.json",
                {"id": case_id, "trace": [event.model_dump(mode="json") for event in context.trace]},
            )
        except Exception as error:
            record = _exception_record(case, started, time.monotonic() - timer, error)
        records_by_id[case_id] = record
        records = [records_by_id[str(case["id"])] for case in cases if str(case["id"]) in records_by_id]
        payload = {
            **_metadata(suite, settings),
            "summary": _summary(records),
            "records": records,
        }
        _atomic_write(output_path, payload)
        print(
            f"{case_id}: status={record['observed_status']} "
            f"wall={record['wall_time_seconds']:.2f}s coverage={record['coverage']:.2f} "
            f"confidence={record['confidence'] or '-'} sources={record['accepted_sources']} "
            f"warnings={len(record['warnings'])} "
            f"{'PASS' if record['meets_expectation'] else 'FAIL'}",
            flush=True,
        )

    records = [records_by_id[str(case["id"])] for case in cases if str(case["id"]) in records_by_id]
    print("\nFinal evaluation summary")
    print("ID   Status                 Wall(s)  Coverage  Sources  Result")
    print("-" * 70)
    for record in records:
        print(
            f"{record['id']:<4} {record['observed_status']:<22} "
            f"{record['wall_time_seconds']:>7.2f}  {record['coverage']:>8.2f} "
            f"{record['accepted_sources']:>7}  "
            f"{'PASS' if record['meets_expectation'] else 'FAIL'}"
        )
    print(json.dumps(_summary(records), indent=2))


if __name__ == "__main__":
    main()
