"""Persist Streamlit research runs as a local JSON evaluation artifact."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas import ResearchContext


def evaluation_log_path() -> Path:
    """Return the configured local evaluation log path."""
    configured = os.getenv("STREAMLIT_EVALUATION_LOG")
    if configured:
        return Path(configured)
    return Path("evaluation") / "results" / "streamlit_runs.json"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    return []


def record_streamlit_run(
    context: ResearchContext,
    *,
    started_at: datetime,
    finished_at: datetime | None = None,
    error: str | None = None,
    path: Path | None = None,
) -> Path:
    """Append one dashboard run while preserving prior evaluation records."""
    output_path = path or evaluation_log_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finished = finished_at or datetime.now(timezone.utc)
    record = {
        "case_id": f"STREAMLIT-{context.run_id}",
        "question": context.request.question,
        "started_at": started_at.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_ms": round((finished - started_at).total_seconds() * 1000, 2),
        "stage_wall_times_ms": context.stage_wall_times_ms,
        "status": context.status.value,
        "run_id": context.run_id,
        "expected": "Review final status, evidence coverage, citations, warnings, and trace.",
        "sources": [source.model_dump(mode="json") for source in context.sources],
        "rejected_sources": [
            source.model_dump(mode="json") for source in context.rejected_sources
        ],
        "coverage": context.coverage.model_dump(mode="json"),
        "entity_coverage": context.entity_coverage,
        "status_reasons": [reason.model_dump(mode="json") for reason in context.status_reasons],
        "error_breakdown": context.error_breakdown,
        "source_type_mix": context.answer.source_type_mix if context.answer else {},
        "run_metadata": {
            "pipeline_started_at": context.pipeline_started_at.isoformat()
            if context.pipeline_started_at else None,
            "pipeline_finished_at": context.pipeline_finished_at.isoformat()
            if context.pipeline_finished_at else None,
            "synthesis_model": context.synthesis_model,
        },
        "answer": context.answer.model_dump(mode="json") if context.answer else None,
        "evaluation": context.evaluation.model_dump(mode="json")
        if context.evaluation
        else None,
        "trace": [event.model_dump(mode="json") for event in context.trace],
        "warnings": context.quality_gate_warnings,
        "error": error,
    }
    records = _read_records(output_path)
    records.append(record)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "streamlit",
        "records": records,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path
