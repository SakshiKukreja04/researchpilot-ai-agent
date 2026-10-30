"""Run deterministic or live ResearchPilot evaluation cases.

Usage from PowerShell:
    .venv/Scripts/python.exe evaluation/run_eval.py --mode deterministic
    .venv/Scripts/python.exe evaluation/run_eval.py --mode live
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.orchestrator import run_full_pipeline
from app.schemas import ExtractionStatus, ResearchPlan, SearchResult, SourceDocument
from evaluation.metrics import calculate_metrics


def load_cases() -> list[dict[str, Any]]:
    return json.loads((ROOT / "evaluation" / "test_cases.json").read_text(encoding="utf-8"))


EDGE_CASES = {
    "EC1": {
        "id": "EC1",
        "question": "   ",
        "mode": "deterministic",
        "expected": "Whitespace-only input is rejected before external calls.",
    }
}


def run_deterministic_case(case: dict[str, Any]):
    """Run the full pipeline with deterministic stand-ins for external services."""
    if not case["question"].strip():
        return run_full_pipeline(case["question"])

    plan = ResearchPlan(
        objective=f"Investigate: {case['question'][:300]}",
        subquestions=["What does the available evidence say?", "What limitations remain?"],
        search_queries=[case["question"][:200]],
        tools=["web_search", "fetch_and_extract"],
        success_criteria=["At least one source-backed finding", "Limitations are disclosed"],
    )
    answer = {
        "user_question": case["question"],
        "short_answer": "The deterministic fixture provides evidence for pipeline validation only.",
        "key_findings": [
            {
                "claim": "A deterministic source was retrieved for this pipeline run.",
                "explanation": "This is a test fixture, not a live research conclusion.",
                "source_ids": ["src_111122223333"],
            }
        ],
        "sources_used": ["src_111122223333"],
        "confidence": "medium",
        "assumptions": ["The fixture stands in for external search and fetch services."],
        "limitations": ["This deterministic run does not measure live source quality."],
        "suggested_next_steps": ["Run the same case in live mode with a valid Groq key."],
        "status": "completed",
    }
    client = MagicMock()
    client.model_name = "deterministic-fixture"
    client.chat_completion.side_effect = [plan.model_dump_json(), json.dumps(answer)]
    search = MagicMock(return_value=[
        SearchResult(
            title="Deterministic Research Source",
            url="https://docs.example.com/research",
            snippet="Evidence fixture for the orchestrator.",
            domain="docs.example.com",
        )
    ])
    fetch = MagicMock(return_value=SourceDocument(
        source_id="src_111122223333",
        title="Deterministic Research Source",
        url="https://docs.example.com/research",
        domain="docs.example.com",
        content=(
            "This deterministic source contains enough text for the evidence "
            "pipeline and is explicitly a test fixture. "
        ) * 30,
        quality_score=0.9,
        extraction_status=ExtractionStatus.SUCCESS,
    ))
    return run_full_pipeline(
        case["question"],
        client=client,
        search_tool_fn=search,
        fetch_tool_fn=fetch,
    )


def run_cases(mode: str, only: set[str] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cases = load_cases() + list(EDGE_CASES.values())
    for case in cases:
        if only and case["id"] not in only:
            continue
        if mode == "live" and case["mode"] != "live":
            continue
        started = datetime.now(timezone.utc)
        try:
            context = (
                run_deterministic_case(case)
                if mode == "deterministic"
                else run_full_pipeline(case["question"])
            )
            record: dict[str, Any] = {
                "case_id": case["id"],
                "question": case["question"],
                "expected": case["expected"],
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": context.status.value,
                "run_id": context.run_id,
                "sources": [source.model_dump(mode="json") for source in context.sources],
                "rejected_sources": [
                    source.model_dump(mode="json") for source in context.rejected_sources
                ],
                "coverage": context.coverage.model_dump(mode="json"),
                "answer": context.answer.model_dump(mode="json") if context.answer else None,
                "trace": [event.model_dump(mode="json") for event in context.trace],
                "warnings": context.quality_gate_warnings,
                "error": None,
            }
        except Exception as error:
            record = {
                "case_id": case["id"],
                "question": case["question"],
                "expected": case["expected"],
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "runner_error",
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deterministic", "live", "all"), default="all")
    parser.add_argument(
        "--only",
        help="Comma-separated case IDs, including built-in edge case EC1.",
    )
    parser.add_argument("--output", default="evaluation/results/latest.json")
    args = parser.parse_args()

    only = {item.strip() for item in args.only.split(",")} if args.only else None
    records = run_cases(args.mode, only=only)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "metrics": calculate_metrics(records),
        "records": records,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
