"""Configuration management for ResearchPilot.

Centralizes environment variables, execution budgets, model selection,
and safety limits. Secrets are never logged or exposed in representations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load local .env if present
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE)
else:
    load_dotenv()


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Application settings with safe defaults and hidden secrets."""

    # LLM Settings
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_seconds: float = 30.0
    groq_max_output_tokens: int = 2048

    # Application Environment
    app_env: str = "development"
    log_level: str = "INFO"

    # Orchestrator Budgets & Limits
    max_search_queries: int = 5
    max_fetch_pages: int = 8
    max_research_rounds: int = 1
    max_retries_per_tool: int = 1
    overall_deadline_seconds: float = 90.0
    max_extracted_chars_per_source: int = 8000
    evidence_quality_threshold: float = 0.55
    min_coverage_threshold: float = 0.60
    cache_enabled: bool = True
    cache_ttl_seconds: float = 300.0
    cache_max_entries: int = 128

    def require_groq_api_key(self) -> str:
        """Return the Groq API key or raise an actionable, safe ConfigurationError."""
        if not self.groq_api_key or not self.groq_api_key.strip():
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Please set it in your local .env file or environment variables. "
                "See .env.example for guidance."
            )
        stripped = self.groq_api_key.strip()
        if "your_groq_api_key" in stripped.lower() or "replace_with" in stripped.lower():
            raise ConfigurationError(
                "GROQ_API_KEY appears to contain placeholder text. "
                "Please replace it with a valid API key from https://console.groq.com/keys in your .env file."
            )
        return stripped

    def __repr__(self) -> str:
        """Safe representation that masks the API key."""
        key_status = "SET (masked)" if self.groq_api_key else "NOT SET"
        return (
            f"Settings("
            f"groq_api_key='{key_status}', "
            f"groq_model='{self.groq_model}', "
            f"app_env='{self.app_env}', "
            f"log_level='{self.log_level}', "
            f"groq_timeout_seconds={self.groq_timeout_seconds}, "
            f"max_search_queries={self.max_search_queries}, "
            f"max_fetch_pages={self.max_fetch_pages})"
        )


def load_settings_from_env() -> Settings:
    """Load settings from current environment variables."""
    raw_api_key = os.getenv("GROQ_API_KEY")
    api_key = raw_api_key.strip() if raw_api_key else None

    # Parse numerical limits with safe fallbacks
    def _parse_float(name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def _parse_int(name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def _parse_bool(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return val.strip().lower() in {"1", "true", "yes", "on"}

    return Settings(
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        groq_timeout_seconds=_parse_float("GROQ_TIMEOUT_SECONDS", 30.0),
        groq_max_output_tokens=_parse_int("GROQ_MAX_OUTPUT_TOKENS", 2048),
        app_env=os.getenv("APP_ENV", "development").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        max_search_queries=_parse_int("MAX_SEARCH_QUERIES", 5),
        max_fetch_pages=_parse_int("MAX_FETCH_PAGES", 8),
        max_research_rounds=_parse_int("MAX_RESEARCH_ROUNDS", 1),
        max_retries_per_tool=_parse_int("MAX_RETRIES_PER_TOOL", 1),
        overall_deadline_seconds=_parse_float("OVERALL_DEADLINE_SECONDS", 90.0),
        max_extracted_chars_per_source=_parse_int("MAX_EXTRACTED_CHARS_PER_SOURCE", 8000),
        cache_enabled=_parse_bool("CACHE_ENABLED", True),
        cache_ttl_seconds=_parse_float("CACHE_TTL_SECONDS", 300.0),
        cache_max_entries=_parse_int("CACHE_MAX_ENTRIES", 128),
    )


# Cached global settings instance
_SETTINGS_INSTANCE: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Return the cached settings singleton, optionally reloading from env."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None or reload:
        _SETTINGS_INSTANCE = load_settings_from_env()
    return _SETTINGS_INSTANCE
