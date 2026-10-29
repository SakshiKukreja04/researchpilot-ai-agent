"""Groq LLM Client Wrapper for ResearchPilot.

Provides a robust, mockable, and typed interface around the official Groq SDK.
Enforces timeouts, token limits, and safe error normalization without leaking secrets.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import ConfigurationError, Settings, get_settings

try:
    from groq import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        Groq,
        GroqError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore
    GroqError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ==============================================================================
# Domain Exceptions (Typed, Safe, Secret-Free)
# ==============================================================================


class LLMError(Exception):
    """Base exception for all ResearchPilot LLM operations."""


class LLMConfigError(LLMError):
    """Raised when LLM configuration (such as GROQ_API_KEY) is missing or invalid."""


class LLMAuthError(LLMError):
    """Raised when authentication with the LLM provider fails."""


class LLMRateLimitError(LLMError):
    """Raised when LLM provider rate limits are encountered."""


class LLMTimeoutError(LLMError):
    """Raised when a request to the LLM provider times out."""


class LLMConnectionError(LLMError):
    """Raised when a network connection error prevents reaching the LLM provider."""


class LLMResponseError(LLMError):
    """Raised when the LLM returns an invalid, malformed, or empty response."""


# ==============================================================================
# Groq Client Wrapper
# ==============================================================================


class GroqClientWrapper:
    """Encapsulates Groq SDK calls with error translation, timeouts, and safe mocking."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    @property
    def model_name(self) -> str:
        """Configured model name."""
        return self._settings.groq_model

    def _get_client(self) -> Any:
        """Lazily initialize and return the Groq SDK client."""
        if self._client is not None:
            return self._client

        try:
            api_key = self._settings.require_groq_api_key()
        except ConfigurationError as err:
            raise LLMConfigError(str(err)) from err

        if Groq is None:
            raise LLMConfigError(
                "The 'groq' Python package is not installed. Run 'pip install -r requirements.txt'."
            )

        self._client = Groq(
            api_key=api_key,
            timeout=self._settings.groq_timeout_seconds,
        )
        return self._client

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> str:
        """Execute a chat completion request with safe error translation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (default 0.2 for deterministic research).
            max_tokens: Maximum tokens in completion (defaults to configured budget).
            timeout: Optional override for request timeout.
            response_format: Optional format specification, e.g. {"type": "json_object"}.

        Returns:
            The completion content string.

        Raises:
            LLMAuthError, LLMRateLimitError, LLMTimeoutError,
            LLMConnectionError, LLMResponseError, LLMConfigError.
        """
        client = self._get_client()
        token_limit = max_tokens or self._settings.groq_max_output_tokens

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": token_limit,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            response = client.chat.completions.create(**kwargs)
        except AuthenticationError as err:
            logger.error("Groq authentication failure. Verify GROQ_API_KEY.")
            raise LLMAuthError(
                "Failed to authenticate with Groq API. Please check your GROQ_API_KEY."
            ) from err
        except RateLimitError as err:
            logger.warning("Groq rate limit exceeded.")
            raise LLMRateLimitError("Groq API rate limit exceeded. Please back off and retry.") from err
        except APITimeoutError as err:
            logger.warning("Groq request timed out.")
            raise LLMTimeoutError(
                f"Groq API call timed out after {timeout or self._settings.groq_timeout_seconds}s."
            ) from err
        except APIConnectionError as err:
            logger.warning("Groq connection error.")
            raise LLMConnectionError(
                "Could not connect to Groq API. Please check network connectivity."
            ) from err
        except GroqError as err:
            logger.error("Groq generic API error.")
            raise LLMResponseError(f"Groq API returned an error: {type(err).__name__}") from err
        except Exception as err:
            logger.error("Unexpected error during Groq API call: %s", type(err).__name__)
            raise LLMResponseError(
                f"Unexpected error communicating with LLM: {type(err).__name__}"
            ) from err

        if not response or not getattr(response, "choices", None):
            raise LLMResponseError("Groq returned an empty response with no choices.")

        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if content is None:
            raise LLMResponseError("Groq returned a null message content.")

        return content.strip()

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Simplified one-shot completion helper."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: Type[T],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> T:
        """Request JSON output and validate against a target Pydantic schema.

        Uses JSON object mode, parses the result, and validates against the schema.
        """
        raw_text = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as err:
            raise LLMResponseError(
                f"LLM output could not be parsed as JSON: {err.msg}"
            ) from err

        try:
            return schema.model_validate(data)
        except ValidationError as err:
            raise LLMResponseError(
                f"LLM JSON output failed schema validation for {schema.__name__}: {err.errors()}"
            ) from err


def run_smoke_test() -> bool:
    """Execute a minimal, live connectivity test against Groq.

    Returns True on success, or raises an informative error.
    """
    settings = get_settings(reload=True)
    print(f"[*] Testing Groq connectivity...")
    print(f"[*] Configured model: {settings.groq_model}")

    try:
        wrapper = GroqClientWrapper(settings=settings)
        response = wrapper.complete(
            prompt="Respond strictly with the single word 'PONG'.",
            system_prompt="You are a ping-pong diagnostic agent.",
            temperature=0.0,
            max_tokens=10,
        )
        print(f"[+] Groq response received: {response}")
        if "PONG" in response.upper():
            print("[+] Smoke test SUCCESSFUL: Groq API responded as expected.")
            return True
        else:
            print(f"[!] Warning: Received unexpected response text: {response}")
            return True
    except LLMConfigError as e:
        print(f"[-] Configuration error: {e}")
        return False
    except LLMAuthError as e:
        print(f"[-] Authentication failed: {e}")
        return False
    except LLMError as e:
        print(f"[-] Groq API error: {e}")
        return False


if __name__ == "__main__":
    import sys

    success = run_smoke_test()
    sys.exit(0 if success else 1)
