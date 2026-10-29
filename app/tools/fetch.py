"""Safe page retrieval and content extraction tool for ResearchPilot.

Uses httpx for controlled, bounded HTTP retrieval and trafilatura for main-content
extraction. Enforces strict URL scheme validation, pre-flight IP/SSRF resolution,
manual redirect validation, response-size bounds during streaming, and deterministic
source ID generation.

Security Notice (DNS Rebinding):
Pre-flight IP resolution validates the resolved address before connecting with httpx.
However, because standard HTTP clients perform their own internal DNS lookup, a
hostile DNS server using DNS rebinding could return a safe IP during pre-flight
and a local IP during the subsequent connection (TOCTOU). This tool implements
the strongest practical application-level defense available with standard httpx.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
from typing import Optional, Set
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx

from app.config import get_settings
from app.cache import TTLCache
from app.schemas import ExtractionStatus, SourceDocument
from app.tools.search import TRACKING_PARAMS

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None  # type: ignore

logger = logging.getLogger(__name__)
_FETCH_CACHE: TTLCache[SourceDocument] = TTLCache()

# Accepted content-type prefixes for text/HTML research documents
SUPPORTED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/xml",
    "text/xml",
)


# ==============================================================================
# Fetch Exceptions
# ==============================================================================


class FetchError(Exception):
    """Base exception for page fetch and extraction failures."""

    def __init__(self, message: str, error_category: str = "HTTP_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.error_category = error_category


class FetchInvalidInputError(FetchError):
    """Raised when URL or fetch arguments are invalid or duplicate."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="INVALID_INPUT")


class FetchUnsafeURLError(FetchError):
    """Raised when URL targets localhost, private IP, link-local, or unsafe scheme."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="UNSAFE_URL")


class FetchTimeoutError(FetchError):
    """Raised when HTTP connection or read times out."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="HTTP_TIMEOUT")


class FetchHTTPError(FetchError):
    """Raised on non-success HTTP status codes or connection errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="HTTP_ERROR")


class FetchUnsupportedContentTypeError(FetchError):
    """Raised when server returns binary or unsupported media types."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="UNSUPPORTED_CONTENT_TYPE")


class FetchResponseTooLargeError(FetchError):
    """Raised when response exceeds maximum permitted byte count while streaming."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="RESPONSE_TOO_LARGE")


class FetchExtractionError(FetchError):
    """Raised when content extraction library encounters an error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="EXTRACTION_FAILED")


class FetchEmptyExtractionError(FetchError):
    """Raised when a page contains no extractable main text content."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_category="EMPTY_EXTRACTION")


# ==============================================================================
# URL Normalization & Deterministic Source IDs
# ==============================================================================


def normalize_url(raw_url: str) -> str:
    """Normalize a URL for deduplication and consistent source ID generation."""
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
    filtered_items.sort(key=lambda pair: pair[0])
    new_query = urlencode(filtered_items)

    path = parsed.path
    if path == "/" and not new_query:
        path = ""

    return urlunparse((scheme, netloc, path, "", new_query, ""))


def generate_source_id(normalized_url: str) -> str:
    """Derive a stable, deterministic source identifier from the normalized URL.

    Format: src_<12-hex-sha256>, e.g. 'src_a1b2c3d4e5f6'.
    """
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:12]
    return f"src_{digest}"


# ==============================================================================
# SSRF & IP Validation
# ==============================================================================


def _is_ip_safe(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IP address is safe for public outbound research requests."""
    # Check for IPv4 mapped IPv6 addresses (e.g., ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return _is_ip_safe(ip.ipv4_mapped)

    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    return True


def is_safe_url(url: str, resolve_dns: bool = True) -> tuple[bool, str]:
    """Validate a URL against SSRF, dangerous schemes, and private destination addresses.

    Returns:
        (is_safe, error_reason)
    """
    if not isinstance(url, str) or not url.strip():
        return False, "URL is empty or not a string."

    parsed = urlparse(url.strip())

    # 1. Check scheme
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, f"Unsupported scheme '{parsed.scheme}'. Only http and https are allowed."

    # 2. Reject credentials in URL authority component (user:pass@host)
    if parsed.username or parsed.password:
        return False, "URLs containing embedded user credentials are not permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname."

    hostname_lower = hostname.lower().strip()

    # 3. Reject common local hostname patterns
    if hostname_lower in {"localhost", "localhost.localdomain"} or hostname_lower.endswith(
        (".localhost", ".local", ".internal", ".lan")
    ):
        return False, f"Blocked local or reserved hostname: '{hostname_lower}'."

    # 4. Check if hostname is an IP literal
    try:
        # Strip brackets if IPv6
        ip_str = hostname_lower.strip("[]")
        ip_obj = ipaddress.ip_address(ip_str)
        if not _is_ip_safe(ip_obj):
            return False, f"Blocked unsafe IP address target: '{ip_obj}'."
        return True, ""
    except ValueError:
        # Hostname is a domain name, proceed to DNS check
        pass

    # 5. Resolve hostname and inspect all resolved addresses
    if resolve_dns:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        try:
            addr_info = socket.getaddrinfo(
                hostname_lower, port, proto=socket.IPPROTO_TCP
            )
        except socket.gaierror as err:
            return False, f"DNS resolution failed for hostname '{hostname_lower}': {err}."
        except Exception as err:
            return False, f"DNS lookup failed for '{hostname_lower}': {type(err).__name__}."

        if not addr_info:
            return False, f"No IP addresses resolved for hostname '{hostname_lower}'."

        for entry in addr_info:
            sockaddr = entry[4]
            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                if not _is_ip_safe(ip_obj):
                    return False, f"Resolved address '{ip_obj}' is a private or unsafe destination."
            except ValueError:
                return False, f"Invalid resolved address format: '{ip_str}'."

    return True, ""


# ==============================================================================
# Page Fetch and Content Extraction
# ==============================================================================


def fetch_and_extract(
    url: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 10.0,
    max_redirects: int = 3,
    max_response_bytes: int = 2 * 1024 * 1024,  # 2MB
    max_extracted_chars: Optional[int] = None,
    session_visited_urls: Optional[Set[str]] = None,
    client: Optional[httpx.Client] = None,
) -> SourceDocument:
    """Retrieve webpage HTML safely and extract main content using trafilatura.

    Args:
        url: Validated HTTP/HTTPS web address.
        connect_timeout: Timeout for initial socket connection in seconds.
        read_timeout: Timeout for streaming reads in seconds.
        max_redirects: Maximum permitted HTTP redirect hops.
        max_response_bytes: Maximum response size read before aborting.
        max_extracted_chars: Character limit for main text (defaults to settings).
        session_visited_urls: Optional set for per-run duplicate prevention.
        client: Optional custom httpx.Client instance for testing/mocking.

    Returns:
        Validated SourceDocument schema instance.

    Raises:
        FetchInvalidInputError: If URL is empty or already visited in this session.
        FetchUnsafeURLError: If destination is private, loopback, or scheme is unsafe.
        FetchTimeoutError: On connection or read timeouts.
        FetchHTTPError: On non-success HTTP status codes or connection drops.
        FetchUnsupportedContentTypeError: If response is non-HTML/text.
        FetchResponseTooLargeError: If response exceeds maximum byte budget.
        FetchEmptyExtractionError: If page has no readable text.
        FetchExtractionError: On parsing or extraction failures.
    """
    if not isinstance(url, str) or not url.strip():
        raise FetchInvalidInputError("URL must be a non-empty string.")

    normalized = normalize_url(url)
    if not normalized:
        raise FetchInvalidInputError("URL could not be normalized.")

    # Deduplication check for this session/run
    if session_visited_urls is not None:
        if normalized in session_visited_urls:
            raise FetchInvalidInputError(
                f"URL has already been fetched in this session: {normalized}"
            )

    # 1. Pre-flight URL & SSRF Validation
    safe, reason = is_safe_url(normalized, resolve_dns=True)
    if not safe:
        raise FetchUnsafeURLError(f"Access to URL rejected: {reason}")

    settings = get_settings()
    char_limit = max_extracted_chars or settings.max_extracted_chars_per_source
    if settings.cache_enabled and session_visited_urls is None and client is None:
        _FETCH_CACHE.max_entries = settings.cache_max_entries
        _FETCH_CACHE.ttl_seconds = settings.cache_ttl_seconds
        cached = _FETCH_CACHE.get(normalized)
        if cached is not None:
            return cached

    # 2. HTTP Retrieval with Manual Redirect Loop & Streaming
    current_url = normalized
    visited_redirects: set[str] = {current_url}
    redirect_count = 0
    final_response = None
    accumulated_bytes = bytearray()
    content_type_header = ""

    # Setup or use provided HTTP client
    own_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=False,
        timeout=httpx.Timeout(
            timeout=connect_timeout + read_timeout,
            connect=connect_timeout,
            read=read_timeout,
        ),
        headers={"User-Agent": "ResearchPilot/0.1 (+https://github.com/example/researchpilot)"},
    )

    try:
        while True:
            try:
                # Stream response to strictly enforce byte limit without loading huge files
                with http_client.stream("GET", current_url) as response:
                    # Check for redirects
                    if response.is_redirect:
                        redirect_count += 1
                        if redirect_count > max_redirects:
                            raise FetchHTTPError(
                                f"Too many redirects: exceeded limit of {max_redirects}."
                            )

                        location = response.headers.get("location")
                        if not location:
                            raise FetchHTTPError("Redirect response missing Location header.")

                        next_url = normalize_url(urljoin(current_url, location))
                        if next_url in visited_redirects:
                            raise FetchHTTPError(f"Redirect loop detected: '{next_url}'.")

                        # Revalidate redirect target against SSRF rules!
                        target_safe, target_reason = is_safe_url(next_url, resolve_dns=True)
                        if not target_safe:
                            raise FetchUnsafeURLError(
                                f"Redirect target '{next_url}' rejected: {target_reason}"
                            )

                        visited_redirects.add(next_url)
                        current_url = next_url
                        continue

                    # Check HTTP status
                    if response.status_code != 200:
                        raise FetchHTTPError(
                            f"HTTP {response.status_code} received from server for '{current_url}'."
                        )

                    # Check Content-Type header
                    content_type_header = response.headers.get("content-type", "").lower()
                    if not any(
                        ct in content_type_header for ct in SUPPORTED_CONTENT_TYPES
                    ):
                        raise FetchUnsupportedContentTypeError(
                            f"Unsupported content type '{content_type_header}'. "
                            f"Expected text or HTML."
                        )

                    # Stream chunks and enforce byte limit
                    for chunk in response.iter_bytes(chunk_size=8192):
                        accumulated_bytes.extend(chunk)
                        if len(accumulated_bytes) > max_response_bytes:
                            raise FetchResponseTooLargeError(
                                f"Response size exceeded limit of {max_response_bytes} bytes."
                            )

                    # Successfully fetched body
                    break

            except httpx.TimeoutException as err:
                raise FetchTimeoutError(f"HTTP request timed out: {err}") from err
            except httpx.ConnectError as err:
                raise FetchHTTPError(f"Failed to connect to host: {err}") from err
            except httpx.HTTPError as err:
                raise FetchHTTPError(f"HTTP client error: {err}") from err

    finally:
        if own_client:
            http_client.close()

    # 3. Content Extraction with Trafilatura
    if not accumulated_bytes:
        raise FetchEmptyExtractionError("Retrieved page content was completely empty.")

    if trafilatura is None:
        raise FetchExtractionError("The 'trafilatura' package is not installed.")

    # Detect character encoding safely
    try:
        # Simple charset extraction from Content-Type header
        encoding_match = re.search(r"charset=([\w-]+)", content_type_header)
        encoding = encoding_match.group(1) if encoding_match else "utf-8"
        html_text = accumulated_bytes.decode(encoding, errors="replace")
    except Exception:
        html_text = accumulated_bytes.decode("utf-8", errors="replace")

    try:
        extracted_text = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
    except Exception as err:
        raise FetchExtractionError(f"Trafilatura extraction failed: {err}") from err

    if not extracted_text or not extracted_text.strip():
        raise FetchEmptyExtractionError(
            f"No readable main content could be extracted from '{current_url}'."
        )

    # Extract metadata (title, date)
    try:
        meta = trafilatura.extract_metadata(html_text)
    except Exception:
        meta = None

    raw_title = getattr(meta, "title", None) if meta else None
    title = (raw_title.strip() if raw_title else None) or "Untitled Document"

    raw_date = getattr(meta, "date", None) if meta else None
    published_date = str(raw_date).strip() if raw_date else None

    # Cap extracted text length
    cleaned_content = extracted_text.strip()
    if len(cleaned_content) > char_limit:
        cleaned_content = cleaned_content[:char_limit] + "..."

    domain = urlparse(current_url).netloc.lower()
    source_id = generate_source_id(normalized)

    # Register as visited in this session
    if session_visited_urls is not None:
        session_visited_urls.add(normalized)

    document = SourceDocument(
        source_id=source_id,
        title=title[:300],
        url=current_url,
        domain=domain,
        content=cleaned_content,
        published_date=published_date,
        quality_score=1.0,  # Default baseline before scoring in Phase 4
        extraction_status=ExtractionStatus.SUCCESS,
    )
    if settings.cache_enabled and session_visited_urls is None and client is None:
        _FETCH_CACHE.set(normalized, document)
    return document
