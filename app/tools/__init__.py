"""Research tools for ResearchPilot.

Provides independent, deterministic, and safe implementations of:
- web_search (via ddgs)
- fetch_and_extract (via httpx and trafilatura)
"""

from app.tools.fetch import (
    FetchEmptyExtractionError,
    FetchError,
    FetchExtractionError,
    FetchHTTPError,
    FetchInvalidInputError,
    FetchResponseTooLargeError,
    FetchTimeoutError,
    FetchUnsafeURLError,
    FetchUnsupportedContentTypeError,
    fetch_and_extract,
    is_safe_url,
    normalize_url,
)
from app.tools.search import (
    SearchError,
    SearchInvalidInputError,
    SearchProviderError,
    SearchRateLimitError,
    SearchTimeoutError,
    web_search,
)

__all__ = [
    "web_search",
    "fetch_and_extract",
    "normalize_url",
    "is_safe_url",
    "SearchError",
    "SearchInvalidInputError",
    "SearchTimeoutError",
    "SearchRateLimitError",
    "SearchProviderError",
    "FetchError",
    "FetchInvalidInputError",
    "FetchUnsafeURLError",
    "FetchTimeoutError",
    "FetchHTTPError",
    "FetchUnsupportedContentTypeError",
    "FetchResponseTooLargeError",
    "FetchExtractionError",
    "FetchEmptyExtractionError",
]
