"""Tests for the web pack: search/fetch/http + the SSRF URL guard. Offline."""

from __future__ import annotations

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import REDACTED, ToolSecurityError
from himmy.toolkit._net import guard_url
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.web import HttpResponse, register_web_pack

# A literal public IP avoids any DNS lookup, keeping these tests fully offline.
PUBLIC = "http://93.184.216.34/"

_DDG_HTML = """
<div class="result">
  <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Title A</a>
  <a class="result__snippet">Snippet A</a>
</div>
<div class="result">
  <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fb">Title B</a>
  <a class="result__snippet">Snippet B</a>
</div>
"""

_PAGE_HTML = (
    "<html><head><title>Hello Page</title></head>"
    "<body><script>var x=1;</script><p>Hello world body.</p></body></html>"
)


class FakeFetcher:
    """A canned :class:`~himmy.connectors.fetcher.Fetcher` (no network)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, url: str) -> str:
        return self._text

    def get_bytes(self, url: str) -> bytes:
        return self._text.encode("utf-8")


def _registry(html: str, caller=None, ddg_html: str = "") -> ToolRegistry:
    registry = ToolRegistry()
    register_web_pack(
        registry,
        ToolkitConfig(),
        fetcher=FakeFetcher(html),
        http_caller=caller,
        ddg_post=lambda q: ddg_html,
    )
    return registry


# ------------------------------------------------------------------ guard_url


def test_guard_url_allows_public_ip() -> None:
    assert guard_url(PUBLIC) == PUBLIC


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://localhost/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "http://user:pass@93.184.216.34/",
    ],
)
def test_guard_url_rejects_unsafe(url: str) -> None:
    with pytest.raises(ToolSecurityError):
        guard_url(url)


# ----------------------------------------------------------------- web_search


def test_web_search_parses_ddg_results() -> None:
    handler = _registry(_PAGE_HTML, ddg_html=_DDG_HTML).handler_for("web_search")
    out = handler({"query": "hello", "max_results": 5})
    assert out["query"] == "hello"
    assert out["results"][0]["title"] == "Title A"
    assert out["results"][0]["url"] == "https://example.com/a"
    assert out["results"][0]["snippet"] == "Snippet A"
    assert len(out["results"]) == 2


# ------------------------------------------------------------------ web_fetch


def test_web_fetch_extracts_title_and_text() -> None:
    handler = _registry(_PAGE_HTML).handler_for("web_fetch")
    out = handler({"url": PUBLIC})
    assert out["title"] == "Hello Page"
    assert "Hello world body." in out["text"]
    assert "var x=1" not in out["text"]  # script content stripped


def test_web_fetch_rejects_private_host() -> None:
    handler = _registry(_PAGE_HTML).handler_for("web_fetch")
    with pytest.raises(ToolSecurityError):
        handler({"url": "http://127.0.0.1/"})


# --------------------------------------------------------------- http_request


def test_http_request_redacts_secret_headers() -> None:
    def caller(method, url, *, headers, json_body, timeout):
        return HttpResponse(
            status_code=200,
            headers={"Authorization": "shh", "Content-Type": "text/plain"},
            text="ok",
        )

    handler = _registry(_PAGE_HTML, caller=caller).handler_for("http_request")
    out = handler({"url": PUBLIC, "method": "GET"})
    assert out["status_code"] == 200
    assert out["headers"]["Authorization"] == REDACTED
    assert out["headers"]["Content-Type"] == "text/plain"
    assert out["text"] == "ok"
