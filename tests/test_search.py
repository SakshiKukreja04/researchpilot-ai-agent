"""Unit tests for the web_search tool.

All tests run completely offline with mocked search provider interactions.
"""

from unittest.mock import MagicMock

import pytest

from app.schemas import SearchResult
from app.tools.search import (
    SearchInvalidInputError,
    SearchProviderError,
    SearchRateLimitError,
    SearchTimeoutError,
    normalize_search_url,
    web_search,
)

try:
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
except ImportError:
    DDGSException = type("DDGSException", (Exception,), {})
    RatelimitException = type("RatelimitException", (Exception,), {})
    TimeoutException = type("TimeoutException", (Exception,), {})


# ==============================================================================
# Query Validation Tests
# ==============================================================================


def test_empty_query_raises_invalid_input():
    with pytest.raises(SearchInvalidInputError):
        web_search("")


def test_whitespace_query_raises_invalid_input():
    with pytest.raises(SearchInvalidInputError):
        web_search("   \n\t  ")


def test_non_string_query_raises_invalid_input():
    with pytest.raises(SearchInvalidInputError):
        web_search(None)  # type: ignore


def test_overly_long_query_raises_invalid_input():
    with pytest.raises(SearchInvalidInputError):
        web_search("a" * 501)


# ==============================================================================
# URL Normalization Tests
# ==============================================================================


def test_normalize_search_url_strips_tracking_params():
    url = "https://example.com/article?utm_source=twitter&utm_medium=social&page=2&id=42&gclid=123"
    normalized = normalize_search_url(url)
    assert "utm_source" not in normalized
    assert "gclid" not in normalized
    assert "id=42" in normalized
    assert "page=2" in normalized
    assert normalized.startswith("https://example.com/article?")


def test_normalize_search_url_lowercases_host_and_strips_fragment():
    url = "HTTPS://Docs.Example.COM/guide#section-3"
    normalized = normalize_search_url(url)
    assert normalized == "https://docs.example.com/guide"


# ==============================================================================
# Result Processing & Deduplication Tests
# ==============================================================================


def test_successful_search_returns_normalized_results():
    mock_client = MagicMock()
    mock_client.text.return_value = [
        {
            "href": "https://example.com/page1?utm_source=newsletter",
            "title": "Page 1 Title",
            "body": "Page 1 summary text describing architecture.",
            "date": "2026-03-15",
        },
        {
            "href": "https://example.com/page2",
            "title": "Page 2 Title",
            "body": "Page 2 overview.",
        },
    ]

    results = web_search("vector database benchmarks", max_results=5, client=mock_client)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].url == "https://example.com/page1"
    assert results[0].domain == "example.com"
    assert results[0].published_date == "2026-03-15"
    assert results[1].published_date is None  # Never invented when missing


def test_duplicate_normalized_urls_are_deduplicated():
    mock_client = MagicMock()
    mock_client.text.return_value = [
        {
            "href": "https://example.com/post?utm_source=google",
            "title": "Post Title Short",
            "body": "Short snippet.",
        },
        {
            "href": "https://example.com/post?utm_source=bing",
            "title": "Post Title Longer",
            "body": "Much longer and more comprehensive snippet describing the findings.",
        },
    ]

    results = web_search("test query", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0].url == "https://example.com/post"
    # Kept the richer snippet
    assert "Much longer" in results[0].snippet


def test_empty_provider_results_returns_empty_list():
    mock_client = MagicMock()
    mock_client.text.return_value = []

    results = web_search("nonexistent query 12345", max_results=5, client=mock_client)
    assert results == []


def test_no_results_found_exception_returns_empty_list():
    mock_client = MagicMock()
    mock_client.text.side_effect = DDGSException("No results found.")

    results = web_search("obscure rare query", max_results=5, client=mock_client)
    assert results == []


def test_malformed_and_unsafe_records_are_skipped():
    mock_client = MagicMock()
    mock_client.text.return_value = [
        {"href": "javascript:alert(1)", "title": "XSS", "body": "Bad script"},
        {"href": "ftp://files.example.com/archive", "title": "FTP", "body": "FTP URL"},
        {"href": None, "title": "No URL", "body": "Missing URL"},
        "not-even-a-dict",
        {"href": "https://valid.com/page", "title": "Valid Page", "body": "Real content"},
    ]

    results = web_search("test query", max_results=5, client=mock_client)
    assert len(results) == 1
    assert results[0].url == "https://valid.com/page"


# ==============================================================================
# Error Handling Tests
# ==============================================================================


def test_timeout_exception_raises_search_timeout_error():
    mock_client = MagicMock()
    mock_client.text.side_effect = TimeoutException("Connection timed out.")

    with pytest.raises(SearchTimeoutError) as exc_info:
        web_search("timeout query", client=mock_client)
    assert "timed out" in str(exc_info.value).lower()
    assert exc_info.value.error_category == "SEARCH_TIMEOUT"


def test_rate_limit_exception_raises_search_rate_limit_error():
    mock_client = MagicMock()
    mock_client.text.side_effect = RatelimitException("Too many requests.")

    with pytest.raises(SearchRateLimitError) as exc_info:
        web_search("rate limit query", client=mock_client)
    assert "rate limit" in str(exc_info.value).lower()
    assert exc_info.value.error_category == "RATE_LIMITED"


def test_unexpected_provider_exception_raises_search_provider_error():
    mock_client = MagicMock()
    mock_client.text.side_effect = RuntimeError("Fatal network error")

    with pytest.raises(SearchProviderError) as exc_info:
        web_search("failing query", client=mock_client)
    assert "unexpected" in str(exc_info.value).lower()
    assert exc_info.value.error_category == "SEARCH_PROVIDER_ERROR"
