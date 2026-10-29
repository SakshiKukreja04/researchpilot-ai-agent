"""Structured Logging and Observability Configuration for ResearchPilot.

Provides JSON-formatted event logging with explicit run_id, stage, event_type,
duration, tool_name, success, retry_count, and error_category fields. Enforces
automatic redaction of API keys, tokens, and overly long prompts or pages.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.schemas import ErrorCategory, ExecutionEvent, WorkflowStage

# Sensitive patterns to redact (e.g., Groq API keys, Bearer tokens)
_SENSITIVE_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?", re.IGNORECASE),
]

_MAX_LOG_MESSAGE_LENGTH = 2000


def redact_sensitive_text(text: str) -> str:
    """Redact API keys, tokens, and sensitive credentials from log text."""
    if not text:
        return text

    # Redact specific configured Groq key if present in environment
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key and len(env_key) > 8:
        text = text.replace(env_key, "[REDACTED_API_KEY]")

    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED_KEY]", text)

    if len(text) > _MAX_LOG_MESSAGE_LENGTH:
        text = text[:_MAX_LOG_MESSAGE_LENGTH] + "... [TRUNCATED_FOR_LOGGING]"

    return text


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects with observability metadata."""

    def format(self, record: logging.LogRecord) -> str:
        raw_msg = record.getMessage()
        sanitized_msg = redact_sensitive_text(raw_msg)

        # Base structured event payload
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_msg,
        }

        # Extract structured observability fields from LogRecord extra
        run_id = getattr(record, "run_id", None)
        if run_id:
            log_entry["run_id"] = str(run_id)

        stage = getattr(record, "stage", None)
        if stage is not None:
            log_entry["stage"] = stage.value if isinstance(stage, WorkflowStage) else str(stage)

        event_type = getattr(record, "event_type", None)
        if event_type:
            log_entry["event_type"] = str(event_type)

        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms is not None:
            try:
                log_entry["duration_ms"] = round(float(duration_ms), 2)
            except (ValueError, TypeError):
                log_entry["duration_ms"] = duration_ms

        tool_name = getattr(record, "tool_name", None)
        if tool_name:
            log_entry["tool_name"] = str(tool_name)

        success = getattr(record, "success", None)
        if success is not None:
            log_entry["success"] = bool(success)

        retry_count = getattr(record, "retry_count", None)
        if retry_count is not None:
            try:
                log_entry["retry_count"] = int(retry_count)
            except (ValueError, TypeError):
                log_entry["retry_count"] = 0

        error_category = getattr(record, "error_category", None)
        if error_category is not None:
            log_entry["error_category"] = (
                error_category.value
                if isinstance(error_category, ErrorCategory)
                else str(error_category)
            )

        # Include exception summary without sensitive frames
        if record.exc_info and record.exc_info[0]:
            exc_type = record.exc_info[0].__name__
            exc_msg = redact_sensitive_text(str(record.exc_info[1]))
            log_entry["exception"] = f"{exc_type}: {exc_msg}"

        return json.dumps(log_entry, default=str)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
) -> logging.Logger:
    """Configure application-wide structured logging for ResearchPilot.

    Args:
        level: Minimum log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to write JSONL logs (e.g. 'logs/researchpilot.jsonl').
        json_format: Whether to format output as single-line JSON.

    Returns:
        The configured root or app logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if called multiple times (e.g. Streamlit reloads)
    for handler in list(root_logger.handlers):
        if getattr(handler, "_researchpilot_managed", False):
            root_logger.removeHandler(handler)

    formatter = (
        StructuredJsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    console_handler._researchpilot_managed = True  # type: ignore[attr-defined]
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(StructuredJsonFormatter())
        file_handler._researchpilot_managed = True  # type: ignore[attr-defined]
        root_logger.addHandler(file_handler)

    return root_logger


def log_structured_event(
    logger: logging.Logger,
    level: int,
    event_type: str,
    run_id: str,
    stage: WorkflowStage | str,
    message: str,
    tool_name: Optional[str] = None,
    duration_ms: Optional[float] = None,
    success: bool = True,
    retry_count: int = 0,
    error_category: Optional[ErrorCategory | str] = None,
    **extra_fields: Any,
) -> None:
    """Log an event with explicit observability fields attached."""
    extra = {
        "event_type": event_type,
        "run_id": run_id,
        "stage": stage,
        "tool_name": tool_name,
        "duration_ms": duration_ms,
        "success": success,
        "retry_count": retry_count,
        "error_category": error_category,
        **extra_fields,
    }
    logger.log(level, message, extra=extra)


def log_execution_event(
    event: ExecutionEvent,
    logger_instance: Optional[logging.Logger] = None,
    tool_name: Optional[str] = None,
    retry_count: int = 0,
) -> None:
    """Helper to log an ExecutionEvent instance using structured formatting."""
    log = logger_instance or logging.getLogger("app.orchestrator")
    level = logging.INFO if event.success else logging.WARNING
    log_structured_event(
        logger=log,
        level=level,
        event_type=event.event_type,
        run_id=event.run_id,
        stage=event.stage,
        message=event.message or event.event_type,
        tool_name=tool_name,
        duration_ms=event.duration_ms,
        success=event.success,
        retry_count=retry_count,
        error_category=event.error_category,
    )

