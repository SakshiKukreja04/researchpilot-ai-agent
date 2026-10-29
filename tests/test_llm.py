"""Unit tests for Groq LLM client wrapper and configuration.

All tests in this module run offline with mocked Groq SDK clients and require
no live internet connection or valid API credentials.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.config import ConfigurationError, Settings
from app.llm import (
    GroqClientWrapper,
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

try:
    from groq import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )
except ImportError:
    AuthenticationError = type("AuthenticationError", (Exception,), {})
    RateLimitError = type("RateLimitError", (Exception,), {})
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    APIConnectionError = type("APIConnectionError", (Exception,), {})


class SampleSchema(BaseModel):
    summary: str
    item_count: int


# ==============================================================================
# Configuration & Secret Masking Tests
# ==============================================================================


def test_missing_api_key_raises_actionable_error():
    settings = Settings(groq_api_key=None)
    with pytest.raises(ConfigurationError) as exc_info:
        settings.require_groq_api_key()
    assert "GROQ_API_KEY is not set" in str(exc_info.value)
    assert ".env.example" in str(exc_info.value)


def test_placeholder_api_key_raises_actionable_error():
    settings = Settings(groq_api_key="gsk_your_groq_api_key_here")
    with pytest.raises(ConfigurationError) as exc_info:
        settings.require_groq_api_key()
    assert "placeholder text" in str(exc_info.value)


def test_wrapper_raises_llm_config_error_when_key_missing():
    settings = Settings(groq_api_key=None)
    wrapper = GroqClientWrapper(settings=settings)
    with pytest.raises(LLMConfigError) as exc_info:
        wrapper.complete("Hello")
    assert "GROQ_API_KEY is not set" in str(exc_info.value)


def test_api_key_is_masked_in_repr():
    secret_key = "gsk_super_secret_production_key_12345"
    settings = Settings(groq_api_key=secret_key)
    rep = repr(settings)
    assert secret_key not in rep
    assert "SET (masked)" in rep


def test_custom_model_configuration():
    custom_model = "llama-3.1-8b-instant"
    settings = Settings(groq_api_key="gsk_test", groq_model=custom_model)
    wrapper = GroqClientWrapper(settings=settings)
    assert wrapper.model_name == custom_model


# ==============================================================================
# Mocked Chat Completion Tests
# ==============================================================================


def test_successful_chat_completion():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Paris is the capital of France."))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy", groq_model="llama-3.3-70b-versatile"),
        client=mock_client,
    )

    result = wrapper.complete(prompt="What is the capital of France?")
    assert result == "Paris is the capital of France."

    # Verify call parameters
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["temperature"] == 0.2
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["content"] == "What is the capital of France?"


def test_empty_or_null_choices_raises_llm_response_error():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client.chat.completions.create.return_value = mock_response

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy"),
        client=mock_client,
    )

    with pytest.raises(LLMResponseError) as exc_info:
        wrapper.complete("Test")
    assert "empty response with no choices" in str(exc_info.value)


# ==============================================================================
# Error Normalization Tests
# ==============================================================================


def test_authentication_error_mapped_to_llm_auth_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key", response=MagicMock(status_code=401), body=None
    )

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_invalid"),
        client=mock_client,
    )

    with pytest.raises(LLMAuthError) as exc_info:
        wrapper.complete("Hello")
    assert "Failed to authenticate" in str(exc_info.value)


def test_rate_limit_mapped_to_llm_rate_limit_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RateLimitError(
        message="Rate limit reached", response=MagicMock(status_code=429), body=None
    )

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy"),
        client=mock_client,
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        wrapper.complete("Hello")
    assert "rate limit exceeded" in str(exc_info.value)


def test_timeout_mapped_to_llm_timeout_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(
        request=MagicMock()
    )

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy"),
        client=mock_client,
    )

    with pytest.raises(LLMTimeoutError) as exc_info:
        wrapper.complete("Hello")
    assert "timed out" in str(exc_info.value)


def test_connection_error_mapped_to_llm_connection_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=MagicMock()
    )

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy"),
        client=mock_client,
    )

    with pytest.raises(LLMConnectionError) as exc_info:
        wrapper.complete("Hello")
    assert "Could not connect to Groq API" in str(exc_info.value)


# ==============================================================================
# Structured Completion Tests
# ==============================================================================


def test_structured_completion_valid():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"summary": "Found 3 databases", "item_count": 3}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy"),
        client=mock_client,
    )

    parsed = wrapper.complete_structured(
        messages=[{"role": "user", "content": "Analyze options"}],
        schema=SampleSchema,
    )
    assert isinstance(parsed, SampleSchema)
    assert parsed.summary == "Found 3 databases"
    assert parsed.item_count == 3


def test_structured_completion_invalid_json():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Not a JSON string"))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy"),
        client=mock_client,
    )

    with pytest.raises(LLMResponseError) as exc_info:
        wrapper.complete_structured(
            messages=[{"role": "user", "content": "Analyze options"}],
            schema=SampleSchema,
        )
    assert "could not be parsed as JSON" in str(exc_info.value)


def test_structured_completion_schema_mismatch():
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Missing required field 'item_count'
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"summary": "Only summary"}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    wrapper = GroqClientWrapper(
        settings=Settings(groq_api_key="gsk_dummy"),
        client=mock_client,
    )

    with pytest.raises(LLMResponseError) as exc_info:
        wrapper.complete_structured(
            messages=[{"role": "user", "content": "Analyze options"}],
            schema=SampleSchema,
        )
    assert "failed schema validation" in str(exc_info.value)
