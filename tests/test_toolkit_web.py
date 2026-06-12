"""Tests for the web pack: search/fetch/http + the SSRF URL guard. Offline."""

from __future__ import annotations

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import REDACTED, ToolSecurityError
from himmy.toolkit._net import guard_url
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.web import (
    HttpResponse,
    HttpStatusError,
    TavilyBackend,
    register_web_pack,
)

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


def _web_fetch_with_transport(handler, **config):
    """Register web_fetch backed by the real HttpxFetcher over a mock transport.

    This exercises the production redirect-following path (the FakeFetcher used
    elsewhere never sees a 3xx), so the SSRF redirect guard is actually tested.
    """
    import httpx

    from himmy.connectors.fetcher import HttpxFetcher

    registry = ToolRegistry()
    cfg = ToolkitConfig(**config)
    fetcher = HttpxFetcher(
        transport=httpx.MockTransport(handler),
        retries=0,
        guard=lambda url: guard_url(
            url,
            allow_private=cfg.allow_private_hosts,
            allow_hosts=set(cfg.egress_allow_hosts) or None,
        ),
        max_bytes=cfg.http_max_bytes,
    )
    register_web_pack(registry, cfg, fetcher=fetcher, ddg_post=lambda q: "")
    return registry.handler_for("web_fetch")


def test_web_fetch_refuses_redirect_to_private_ip() -> None:
    """A 302 to an internal address (IMDS / loopback / RFC1918) is refused.

    Without the per-hop guard, a public URL that 302s to 169.254.169.254 would
    sail past the one-shot URL check straight into the cloud metadata endpoint.
    """
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(
                302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(200, text="SECRET CLOUD CREDS")  # must never be reached

    fetch = _web_fetch_with_transport(handler)
    with pytest.raises(ToolSecurityError):
        fetch({"url": PUBLIC})


def test_web_fetch_redirect_to_public_host_is_followed() -> None:
    """A redirect to another public host is still allowed (no false positive)."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            # Redirect to another path on the same public IP (literal → no DNS).
            return httpx.Response(302, headers={"Location": "http://93.184.216.34/p"})
        return httpx.Response(200, text="<html><body><p>landed</p></body></html>")

    fetch = _web_fetch_with_transport(handler)
    out = fetch({"url": PUBLIC})
    assert "landed" in out["text"]


def test_web_fetch_rejects_oversize_body() -> None:
    """A response larger than the configured cap is refused (DoS guard)."""
    import httpx

    from himmy.connectors.fetcher import ResponseTooLarge

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 50_000)

    fetch = _web_fetch_with_transport(handler, http_max_bytes=1024)
    with pytest.raises(ResponseTooLarge):
        fetch({"url": PUBLIC})


# --------------------------------------------------------------- http_request


def test_http_request_redacts_secret_headers() -> None:
    def caller(method, url, *, headers, json_body, timeout, max_bytes=None):
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


def test_http_request_error_status_raises_typed_error() -> None:
    body = "<html>Server exploded: " + "x" * 1000 + "</html>"

    def caller(method, url, *, headers, json_body, timeout, max_bytes=None):
        return HttpResponse(status_code=500, headers={}, text=body)

    handler = _registry(_PAGE_HTML, caller=caller).handler_for("http_request")
    with pytest.raises(HttpStatusError) as excinfo:
        handler({"url": PUBLIC, "method": "GET"})
    err = excinfo.value
    assert err.status_code == 500  # raw status stays inspectable
    assert "500" in str(err)
    assert "Internal Server Error" in str(err)  # reason phrase
    assert "Server exploded" in str(err)  # body snippet surfaces...
    assert len(err.body_snippet) <= 300  # ...but truncated, never the full page


def test_http_request_4xx_also_raises() -> None:
    def caller(method, url, *, headers, json_body, timeout, max_bytes=None):
        return HttpResponse(status_code=404, headers={}, text="nope")

    handler = _registry(_PAGE_HTML, caller=caller).handler_for("http_request")
    with pytest.raises(HttpStatusError) as excinfo:
        handler({"url": PUBLIC, "method": "GET"})
    assert excinfo.value.status_code == 404


def test_http_request_streams_and_caps_oversize_body() -> None:
    """A multi-GB http_request body is rejected mid-download, never buffered whole.

    Routes through the real ``_httpx_caller`` over a mock transport so the streamed
    cap is exercised (the previous code did ``resp.text`` — a full read — then sliced
    afterwards, which never bounded memory). The cap raises before the body lands.
    """
    import httpx

    from himmy.connectors.fetcher import ResponseTooLarge
    from himmy.toolkit.web import _httpx_caller

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 50_000)

    real_client = httpx.Client

    def _client(**kwargs: object) -> httpx.Client:
        kwargs.pop("timeout", None)
        kwargs.pop("follow_redirects", None)
        return real_client(transport=httpx.MockTransport(handler))

    registry = ToolRegistry()
    register_web_pack(
        registry,
        ToolkitConfig(http_max_bytes=1024),
        http_caller=_httpx_caller,
        ddg_post=lambda q: "",
    )
    handler_fn = registry.handler_for("http_request")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx, "Client", _client)
        with pytest.raises(ResponseTooLarge):
            handler_fn({"url": PUBLIC, "method": "GET"})


def test_httpx_caller_caps_body_during_stream() -> None:
    """The default caller itself enforces the byte cap incrementally (unit-level)."""
    import httpx

    from himmy.connectors.fetcher import ResponseTooLarge
    from himmy.toolkit.web import _httpx_caller

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"y" * 5000)

    real_client = httpx.Client

    def _client(**kwargs: object) -> httpx.Client:
        kwargs.pop("timeout", None)
        kwargs.pop("follow_redirects", None)
        return real_client(transport=httpx.MockTransport(handler))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx, "Client", _client)
        with pytest.raises(ResponseTooLarge):
            _httpx_caller(
                "GET",
                PUBLIC,
                headers=None,
                json_body=None,
                timeout=5.0,
                max_bytes=1024,
            )
        # A body within the cap returns normally with the decoded text.
        resp = _httpx_caller(
            "GET",
            PUBLIC,
            headers=None,
            json_body=None,
            timeout=5.0,
            max_bytes=1_000_000,
        )
    assert resp.status_code == 200
    assert resp.text == "y" * 5000


def test_search_backend_error_status_raises_typed_error() -> None:
    def caller(method, url, *, headers, json_body, timeout, max_bytes=None):
        return HttpResponse(status_code=429, headers={}, text="slow down")

    backend = TavilyBackend("key", caller, timeout=5.0)
    with pytest.raises(HttpStatusError) as excinfo:
        backend.search("hello", 3)
    assert excinfo.value.status_code == 429
    assert "slow down" in str(excinfo.value)


def test_default_ddg_search_throttle_surfaces_as_error() -> None:
    """The keyless DuckDuckGo path must not return a throttle page as results."""
    import httpx

    from himmy.toolkit.web import ddg_post_html

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="rate limited, try later")

    # Patch httpx.Client so ddg_post_html uses a mock transport (no network).
    real_client = httpx.Client

    def _client(**kwargs: object) -> httpx.Client:
        kwargs.pop("timeout", None)
        kwargs.pop("headers", None)
        kwargs.pop("follow_redirects", None)
        return real_client(transport=httpx.MockTransport(handler))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx, "Client", _client)
        with pytest.raises(HttpStatusError) as excinfo:
            ddg_post_html("hello")
    assert excinfo.value.status_code == 503
    assert "rate limited" in str(excinfo.value)
