import json
from datetime import datetime, timezone

from app.evaluation_recorder import record_streamlit_run
from app.schemas import (
    CoverageReport,
    ResearchAnswerStatus,
    ResearchContext,
    ResearchPlan,
    ResearchRequest,
)


def _context(question: str, run_id: str) -> ResearchContext:
    return ResearchContext(
        run_id=run_id,
        request=ResearchRequest(question=question),
        plan=ResearchPlan(
            objective="Test objective",
            subquestions=["What does the evidence say?"],
            search_queries=["test evidence"],
            tools=["web_search", "fetch_and_extract"],
            success_criteria=["A source-backed answer"],
        ),
        coverage=CoverageReport(
            subquestion_coverage={"What does the evidence say?": "supported"},
            supported_count=1,
            coverage_ratio=1.0,
            is_adequate=True,
        ),
        status=ResearchAnswerStatus.COMPLETED,
    )


def test_record_streamlit_run_appends_complete_json_record(tmp_path):
    output = tmp_path / "results" / "streamlit_runs.json"
    started = datetime(2026, 10, 29, 17, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 10, 29, 17, 0, 1, tzinfo=timezone.utc)

    record_streamlit_run(
        _context("First dashboard question", "run_one"),
        started_at=started,
        finished_at=finished,
        path=output,
    )
    record_streamlit_run(
        _context("Second dashboard question", "run_two"),
        started_at=started,
        finished_at=finished,
        path=output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source"] == "streamlit"
    assert len(payload["records"]) == 2
    assert payload["records"][0]["question"] == "First dashboard question"
    assert payload["records"][1]["status"] == "completed"
    assert payload["records"][0]["duration_ms"] == 1000.0
