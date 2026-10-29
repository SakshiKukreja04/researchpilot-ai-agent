from unittest.mock import MagicMock, patch

from app.schemas import SearchResult
from app.tools.search import web_search


def test_search_uses_fallback_backend_after_primary_failure():
    primary = MagicMock()
    primary.text.side_effect = RuntimeError("primary unavailable")
    fallback = MagicMock()
    fallback.text.return_value = [
        {
            "href": "https://example.com/vector-databases",
            "title": "Vector database guide",
            "body": "Open-source vector database comparison.",
        }
    ]

    with patch("app.tools.search.DDGS", side_effect=[primary, fallback]):
        results = web_search("open source vector databases")

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].url == "https://example.com/vector-databases"
    fallback.text.assert_called_once()
    assert fallback.text.call_args.kwargs["backend"] == "bing"


def test_search_fallback_does_not_apply_to_injected_test_client():
    client = MagicMock()
    client.text.side_effect = RuntimeError("provider unavailable")

    with patch("app.tools.search.DDGS") as ddgs:
        try:
            web_search("test query", client=client)
        except Exception:
            pass

    ddgs.assert_not_called()
