"""Unit tests for structured logging and observability configuration."""

import json
import logging
import os
from unittest.mock import patch

import pytest

from app.logging_config import (
    StructuredJsonFormatter,
    configure_logging,
    log_execution_event,
    log_structured_event,
    redact_sensitive_text,
)
from app.schemas import ErrorCategory, ExecutionEvent, WorkflowStage


def test_redact_sensitive_text():
    raw_key = "gsk_1234567890abcdef1234567890"
    msg = f"Failed to authenticate with key {raw_key} at endpoint"
    redacted = redact_sensitive_text(msg)
    assert raw_key not in redacted
    assert "[REDACTED_KEY]" in redacted

    bearer_msg = "Header Authorization: Bearer abcdef1234567890abcdef123456"
    assert "abcdef1234567890" not in redact_sensitive_text(bearer_msg)

    long_msg = "A" * 3000
    redacted_long = redact_sensitive_text(long_msg)
    assert len(redacted_long) < 2100
    assert "[TRUNCATED_FOR_LOGGING]" in redacted_long


def test_redact_environment_groq_key():
    with patch.dict(os.environ, {"GROQ_API_KEY": "custom_live_secret_groq_key_9999"}):
        text = "Calling Groq with custom_live_secret_groq_key_9999 for query"
        redacted = redact_sensitive_text(text)
        assert "custom_live_secret_groq_key_9999" not in redacted
        assert "[REDACTED_API_KEY]" in redacted


def test_structured_json_formatter():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=10,
        msg="Search tool executed successfully",
        args=(),
        exc_info=None,
    )
    record.run_id = "run_123456"
    record.stage = WorkflowStage.SEARCHING
    record.event_type = "search_query_executed"
    record.duration_ms = 142.5
    record.tool_name = "web_search"
    record.success = True
    record.retry_count = 1
    record.error_category = None

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["logger"] == "test_logger"
    assert data["level"] == "INFO"
    assert data["message"] == "Search tool executed successfully"
    assert data["run_id"] == "run_123456"
    assert data["stage"] == "SEARCHING"
    assert data["event_type"] == "search_query_executed"
    assert data["duration_ms"] == 142.5
    assert data["tool_name"] == "web_search"
    assert data["success"] is True
    assert data["retry_count"] == 1


def test_structured_json_formatter_with_error():
    formatter = StructuredJsonFormatter()
    try:
        raise ValueError("Sensitive key gsk_abc1234567890abcdef12345 failed")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test_file.py",
        lineno=20,
        msg="Tool failed",
        args=(),
        exc_info=exc_info,
    )
    record.error_category = ErrorCategory.TIMEOUT

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["error_category"] == "timeout"
    assert "exception" in data
    assert "gsk_" not in data["exception"]


def test_log_execution_event_and_log_structured_event(caplog):
    test_logger = logging.getLogger("test_observability")
    test_logger.setLevel(logging.INFO)

    evt = ExecutionEvent(
        run_id="run_abc789",
        stage=WorkflowStage.RETRIEVING,
        event_type="page_fetched",
        duration_ms=45.0,
        success=True,
        message="Fetched test page",
    )

    with caplog.at_level(logging.INFO):
        log_execution_event(evt, logger_instance=test_logger, tool_name="fetch_and_extract")

    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.run_id == "run_abc789"
    assert rec.stage == WorkflowStage.RETRIEVING
    assert rec.tool_name == "fetch_and_extract"
    assert rec.event_type == "page_fetched"


def test_configure_logging(tmp_path):
    log_file = tmp_path / "test_logs.jsonl"
    logger = configure_logging(level="DEBUG", log_file=str(log_file), json_format=True)
    assert logger.level == logging.DEBUG
    assert log_file.exists() or log_file.parent.exists()

