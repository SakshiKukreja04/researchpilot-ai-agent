"""Web search tool for ResearchPilot using ddgs.

Discovers candidate web pages, normalizes results into SearchResult schemas,
deduplicates URLs safely, and handles provider errors cleanly without fabricating results.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.schemas import SearchResult
from app.cache import TTLCache
from app.config import get_settings

try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
except ImportError:  # pragma: no cover
    DDGS = None  # type: ignore
    DDGSException = Exception  # type: ignore
    RatelimitException = Exception  # type: ignore
    TimeoutException = Exception  # type: ignore

logger = logging.getLogger(__name__)
_SEARCH_CACHE: TTLCache[list[SearchResult]] = TTLCache()

# Standard marketing and analytics tracking query parameters to safely strip
TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "_hsenc",
        "_hsmi",
        "ref",
        "source",
    }
)


# ==============================================================================
# Search Exceptions
# ==============================================================================


class SearchError(Exception):
    """Base exception for search tool failures."""

    def __init__(self, message: str, error_category: str = "SEARCH_PROVIDER_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.error_category = error_category


class SearchInvalidInputError(SearchError):
    """Raised when query or search parameters are invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="INVALID_INPUT")


class SearchTimeoutError(SearchError):
    """Raised when the search provider times out."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="SEARCH_TIMEOUT")


class SearchRateLimitError(SearchError):
    """Raised when the search provider rate-limits requests."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="RATE_LIMITED")


class SearchProviderError(SearchError):
    """Raised when the search provider fails with an unexpected exception."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="SEARCH_PROVIDER_ERROR")


# ==============================================================================
# URL Normalization
# ==============================================================================


def normalize_search_url(raw_url: str) -> str:
    """Normalize a URL for deduplication while preserving its functional target.

    - Forces scheme and hostname to lowercase.
    - Strips URL fragments (#section).
    - Removes known harmless marketing/tracking parameters (utm_*, gclid, etc.).
    - Sorts remaining functional query parameters deterministically.
    - Strips redundant trailing slashes on root domains.
    """
    cleaned = raw_url.strip()
    if not cleaned:
        return ""

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Filter out harmless tracking query parameters
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_items = [
        (k, v) for k, v in query_items if k.lower() not in TRACKING_PARAMS
    ]
    # Sort deterministically
    filtered_items.sort(key=lambda pair: pair[0])
    new_query = urlencode(filtered_items)

    path = parsed.path
    if path == "/" and not new_query:
        path = ""

    return urlunparse((scheme, netloc, path, "", new_query, ""))


def extract_domain(url: str) -> Optional[str]:
    """Extract clean domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower() or None
    except Exception:
        return None


def normalize_search_query(query: str) -> str:
    """Normalize model-generated queries for providers with limited punctuation support."""
    normalized = unicodedata.normalize("NFKC", query)
    normalized = normalized.replace("\u2010", "-").replace("\u2011", "-")
    normalized = normalized.replace("\u2012", "-").replace("\u2013", "-")
    normalized = normalized.replace("\u2014", "-").replace("\u2212", "-")
    normalized = re.sub(r"[^\w\s:/.-]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:500]


def _simplify_search_query(query: str) -> str:
    """Create one less-specific fallback without changing the query budget."""
    simplified = re.sub(r"\([^)]*\)", " ", query)
    simplified = re.sub(
        r"\b(latency|benchmark|benchmarks|detailed|solutions|e\.g\.|"
        r"research|application|specification|specifications)\b",
        " ",
        simplified,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", simplified).strip()


# ==============================================================================
# Search Tool Implementation
# ==============================================================================


def web_search(
    query: str,
    max_results: int = 5,
    timeout: float = 10.0,
    client: Optional[Any] = None,
) -> list[SearchResult]:
    """Perform a web search using the official ddgs library and return validated results.

    Args:
        query: Non-empty search string.
        max_results: Upper bound on returned results (clamped between 1 and 20).
        timeout: Search request timeout in seconds.
        client: Optional custom DDGS instance for testing/mocking.

    Returns:
        List of validated, deduplicated SearchResult objects.

    Raises:
        SearchInvalidInputError: If query is empty or whitespace.
        SearchTimeoutError: If the search request times out.
        SearchRateLimitError: If provider rate-limits the request.
        SearchProviderError: On unexpected provider error.
    """
    if not isinstance(query, str):
        raise SearchInvalidInputError("Search query must be a string.")

    raw_query = query.strip()
    if not raw_query:
        raise SearchInvalidInputError("Search query cannot be empty or whitespace only.")

    if len(raw_query) > 500:
        raise SearchInvalidInputError("Search query exceeds maximum length of 500 characters.")

    cleaned_query = normalize_search_query(raw_query)
    if not cleaned_query:
        raise SearchInvalidInputError("Search query contains no searchable text.")

    clamped_max_results = max(1, min(max_results, 20))
    settings = get_settings()
    cache_key = f"{cleaned_query}|{clamped_max_results}"
    if settings.cache_enabled and client is None:
        cached = _SEARCH_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

    if client is not None:
        ddgs_client = client
    else:
        if DDGS is None:
            raise SearchProviderError("The 'ddgs' package is not installed.")
        ddgs_client = DDGS(timeout=timeout)

    def _provider_search(search_query: str, provider: Any = ddgs_client) -> Any:
        kwargs: dict[str, Any] = {
            "max_results": clamped_max_results * 2,
        }
        if provider is not ddgs_client:
            kwargs["backend"] = settings.search_fallback_backend
        return provider.text(search_query, **kwargs)

    def _fallback_search(search_query: str) -> Any:
        if client is not None or not settings.search_fallback_backend:
            return []
        if DDGS is None:
            return []
        fallback_client = DDGS(timeout=timeout)
        try:
            logger.info(
                "Trying fallback search backend '%s' for query: %s",
                settings.search_fallback_backend,
                search_query,
            )
            return _provider_search(search_query, fallback_client)
        finally:
            close = getattr(fallback_client, "close", None)
            if callable(close):
                close()

    raw_results: Any = []
    primary_error: Optional[SearchError] = None
    try:
        # Request slightly more raw results to account for deduplication and filtering.
        raw_results = _provider_search(cleaned_query)
    except TimeoutException as err:
        logger.warning("Search timed out for query: %s", cleaned_query)
        primary_error = SearchTimeoutError(f"Search provider timed out after {timeout}s.")
    except RatelimitException as err:
        logger.warning("Search provider rate-limited for query: %s", cleaned_query)
        primary_error = SearchRateLimitError("Search provider rate limit exceeded. Please back off.")
    except DDGSException as err:
        err_msg = str(err).lower()
        if "no results found" in err_msg:
            logger.info("Search returned 0 results for query: %s", cleaned_query)
            raw_results = []
        else:
            logger.error("DDGSException during search: %s", err)
            primary_error = SearchProviderError(f"Search provider error: {err}")
    except Exception as err:
        logger.error("Unexpected error during search: %s", type(err).__name__)
        primary_error = SearchProviderError(
            f"Unexpected search error: {type(err).__name__}"
        )

    if primary_error is not None:
        try:
            raw_results = _fallback_search(cleaned_query)
        except Exception as fallback_err:
            logger.warning(
                "Fallback search backend failed: %s", type(fallback_err).__name__
            )
            raise primary_error from fallback_err
        if not raw_results:
            raise primary_error

    if not raw_results:
        try:
            raw_results = _fallback_search(cleaned_query)
        except Exception as fallback_err:
            logger.info(
                "Fallback search for empty result failed: %s",
                type(fallback_err).__name__,
            )

    if not raw_results:
        fallback_query = _simplify_search_query(cleaned_query)
        if fallback_query and fallback_query != cleaned_query:
            logger.info("Retrying empty search with simplified query: %s", fallback_query)
            try:
                raw_results = _provider_search(fallback_query)
            except Exception as err:
                logger.info(
                    "Simplified search also failed for query '%s': %s",
                    cleaned_query,
                    type(err).__name__,
                )
                return []
        if not raw_results:
            return []

    # Normalize, validate, and deduplicate
    seen_normalized_urls: dict[str, SearchResult] = {}

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        raw_url = item.get("href") or item.get("url")
        raw_title = item.get("title") or ""
        raw_snippet = item.get("body") or item.get("snippet") or ""
        raw_date = item.get("date") or item.get("published_date")

        if not raw_url or not isinstance(raw_url, str):
            continue

        # Scheme check
        parsed = urlparse(raw_url.strip())
        if parsed.scheme.lower() not in {"http", "https"}:
            continue

        normalized_url = normalize_search_url(raw_url)
        if not normalized_url:
            continue

        domain = extract_domain(normalized_url)

        # Do not invent a publication date if absent
        published_date = str(raw_date).strip() if raw_date else None

        try:
            candidate = SearchResult(
                title=str(raw_title).strip() or "Untitled",
                url=normalized_url,
                snippet=str(raw_snippet).strip(),
                domain=domain,
                published_date=published_date,
            )
        except Exception as err:
            logger.debug("Skipping malformed search result (%s): %s", raw_url, err)
            continue

        # Deduplicate while keeping the result with richer content
        if normalized_url not in seen_normalized_urls:
            seen_normalized_urls[normalized_url] = candidate
        else:
            existing = seen_normalized_urls[normalized_url]
            if len(candidate.snippet) > len(existing.snippet):
                seen_normalized_urls[normalized_url] = candidate

        if len(seen_normalized_urls) >= clamped_max_results:
            break

    results = list(seen_normalized_urls.values())
    if settings.cache_enabled and client is None:
        _SEARCH_CACHE.max_entries = settings.cache_max_entries
        _SEARCH_CACHE.ttl_seconds = settings.cache_ttl_seconds
        _SEARCH_CACHE.set(cache_key, results)
    return results
