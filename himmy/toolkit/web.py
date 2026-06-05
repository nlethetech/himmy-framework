"""Web pack: ``web_search``, ``web_fetch``, and ``http_request``.

These give an agent the single most-requested capability it otherwise lacks —
reaching the open web — while staying offline-testable and SSRF-safe:

* every model-supplied URL passes :func:`himmy.toolkit._net.guard_url` (no private
  hosts, http/https only, no embedded creds), redirects are not followed, and bodies
  are size- and time-bounded;
* network I/O goes through injectable seams (:class:`~himmy.connectors.fetcher.Fetcher`
  for GET, an ``HttpCaller`` for full requests) so tests pass fakes and never hit the
  network;
* ``web_search`` is backend-pluggable: a keyless DuckDuckGo HTML scrape by default,
  or Tavily/Brave when an API key is configured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, quote_plus, urlsplit

from himmy.connectors.fetcher import Fetcher, HttpxFetcher
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError, redact_mapping
from himmy.toolkit._net import guard_url
from himmy.toolkit.config import ToolkitConfig

# DuckDuckGo's HTML endpoint challenges non-browser agents, so search requests
# present a common browser User-Agent (the connectors' polite bot UA gets a 202).
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# --------------------------------------------------------------------- HTTP seam


@dataclass(frozen=True)
class HttpResponse:
    """A minimal, transport-agnostic HTTP response (the injectable seam's output)."""

    status_code: int
    headers: dict[str, str]
    text: str


@runtime_checkable
class HttpCaller(Protocol):
    """Performs a single HTTP request and returns an :class:`HttpResponse`."""

    def __call__(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None,
        json_body: Any | None,
        timeout: float,
    ) -> HttpResponse: ...


def _httpx_caller(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None,
    json_body: Any | None,
    timeout: float,
) -> HttpResponse:
    """Default :class:`HttpCaller` backed by ``httpx`` (redirects disabled)."""
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        resp = client.request(method, url, headers=headers, json=json_body)
        return HttpResponse(
            status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items()},
            text=resp.text,
        )


# ------------------------------------------------------------------ HTML helpers


class _TextExtractor(HTMLParser):
    """Collect visible text, dropping ``<script>``/``<style>`` content + tags."""

    _SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0
        self.title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title and self.title is None:
            self.title = text
        self._chunks.append(text)

    def text(self) -> str:
        """Return the collected text as whitespace-collapsed paragraphs."""
        return "\n".join(self._chunks)


def html_to_text(html: str) -> tuple[str | None, str]:
    """Extract ``(title, text)`` from HTML. Uses BeautifulSoup when installed."""
    try:  # optional `toolkit` extra — nicer extraction when present
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        text = "\n".join(
            line for line in (soup.get_text("\n")).splitlines() if line.strip()
        )
        return title, text
    except Exception:
        parser = _TextExtractor()
        parser.feed(html)
        return parser.title, parser.text()


# ------------------------------------------------------------- search backends


@runtime_checkable
class SearchBackend(Protocol):
    """Returns web results as ``[{title, url, snippet}]`` for a query."""

    def search(self, query: str, max_results: int) -> list[dict[str, str]]: ...


class _DdgResultParser(HTMLParser):
    """Parse DuckDuckGo HTML results (``result__a`` links + ``result__snippet``)."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._mode: str | None = None
        self._href = ""
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        attr = dict(attrs)
        cls = attr.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            self._mode = "title"
            self._href = attr.get("href", "") or ""
            self._buf = []
        elif "result__snippet" in cls:
            self._mode = "snippet"
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._mode == "title" and tag == "a":
            self.results.append(
                {"title": "".join(self._buf).strip(), "url": _ddg_clean(self._href), "snippet": ""}
            )
            self._mode = None
        elif self._mode == "snippet" and tag in ("a", "td", "div"):
            if self.results:
                self.results[-1]["snippet"] = "".join(self._buf).strip()
            self._mode = None


def _ddg_clean(href: str) -> str:
    """Unwrap a DuckDuckGo redirect link (``/l/?uddg=...``) to the real URL."""
    parts = urlsplit(href)
    if "/l/" in parts.path:
        uddg = parse_qs(parts.query).get("uddg")
        if uddg:
            return uddg[0]
    if href.startswith("//"):
        return "https:" + href
    return href


#: A function that POSTs a query to DuckDuckGo and returns the results HTML.
DdgPost = Callable[[str], str]


def ddg_post_html(query: str, timeout: float = 20.0) -> str:
    """Default DuckDuckGo fetch: POST the query (form-encoded) and return the HTML."""
    import httpx

    with httpx.Client(
        timeout=timeout, headers={"User-Agent": _BROWSER_UA}, follow_redirects=True
    ) as client:
        resp = client.post("https://html.duckduckgo.com/html/", data={"q": query})
        return resp.text


class DuckDuckGoBackend:
    """Keyless web search via DuckDuckGo's HTML endpoint (POST, parsed offline)."""

    def __init__(self, post_html: DdgPost) -> None:
        self._post_html = post_html

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        parser = _DdgResultParser()
        parser.feed(self._post_html(query))
        return parser.results[:max_results]


class TavilyBackend:
    """Web search via the Tavily API (requires an API key)."""

    def __init__(self, api_key: str, caller: HttpCaller, timeout: float) -> None:
        self._key = api_key
        self._caller = caller
        self._timeout = timeout

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        resp = self._caller(
            "POST",
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json_body={
                "api_key": self._key,
                "query": query,
                "max_results": max_results,
            },
            timeout=self._timeout,
        )
        import json

        data = json.loads(resp.text)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in data.get("results", [])[:max_results]
        ]


class BraveBackend:
    """Web search via the Brave Search API (requires an API key)."""

    def __init__(self, api_key: str, caller: HttpCaller, timeout: float) -> None:
        self._key = api_key
        self._caller = caller
        self._timeout = timeout

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        resp = self._caller(
            "GET",
            f"https://api.search.brave.com/res/v1/web/search?q={quote_plus(query)}",
            headers={"X-Subscription-Token": self._key, "Accept": "application/json"},
            json_body=None,
            timeout=self._timeout,
        )
        import json

        data = json.loads(resp.text)
        web = data.get("web", {}).get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("description", ""),
            }
            for r in web[:max_results]
        ]


def build_search_backend(
    config: ToolkitConfig, caller: HttpCaller, ddg_post: DdgPost
) -> SearchBackend:
    """Select a :class:`SearchBackend` from the toolkit config."""
    backend = config.search_backend
    if backend == "duckduckgo":
        return DuckDuckGoBackend(ddg_post)
    if backend == "tavily":
        if not config.search_api_key:
            raise ToolSecurityError("tavily backend needs HIMMY_SEARCH_API_KEY")
        return TavilyBackend(config.search_api_key, caller, config.http_timeout)
    if backend == "brave":
        if not config.search_api_key:
            raise ToolSecurityError("brave backend needs HIMMY_SEARCH_API_KEY")
        return BraveBackend(config.search_api_key, caller, config.http_timeout)
    raise ToolSecurityError(f"unknown search backend {backend!r}")


# ------------------------------------------------------------------- schemas

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_FETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "An http(s) URL to fetch."},
        "max_chars": {"type": "integer", "minimum": 100, "default": 8000},
    },
    "required": ["url"],
    "additionalProperties": False,
}

_HTTP_SCHEMA = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
        "url": {"type": "string"},
        "headers": {"type": "object"},
        "json_body": {},
    },
    "required": ["url"],
    "additionalProperties": False,
}


def register_web_pack(
    registry: ToolRegistry,
    config: ToolkitConfig,
    *,
    fetcher: Fetcher | None = None,
    http_caller: HttpCaller | None = None,
    ddg_post: DdgPost | None = None,
) -> None:
    """Register ``web_search``, ``web_fetch``, and ``http_request`` onto ``registry``."""
    fetcher = fetcher or HttpxFetcher(timeout=config.http_timeout)
    caller = http_caller or _httpx_caller
    post = ddg_post or (lambda q: ddg_post_html(q, config.http_timeout))
    backend = build_search_backend(config, caller, post)

    def web_search(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        max_results = max(1, min(int(args.get("max_results", 5)), 20))
        return {"query": query, "results": backend.search(query, max_results)}

    def web_fetch(args: dict[str, Any]) -> dict[str, Any]:
        url = str(args["url"])
        max_chars = int(args.get("max_chars", 8000))
        guard_url(url, allow_private=config.allow_private_hosts)
        html = fetcher.get_text(url)
        title, text = html_to_text(html)
        return {
            "url": url,
            "title": title,
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
        }

    def http_request(args: dict[str, Any]) -> dict[str, Any]:
        url = str(args["url"])
        method = str(args.get("method", "GET")).upper()
        headers = args.get("headers")
        json_body = args.get("json_body")
        guard_url(url, allow_private=config.allow_private_hosts)
        resp = caller(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout=config.http_timeout,
        )
        body = resp.text[: config.http_max_bytes]
        return {
            "status_code": resp.status_code,
            "headers": redact_mapping(resp.headers),
            "text": body,
            "truncated": len(resp.text) > len(body),
        }

    register_local_tool(
        registry,
        name="web_search",
        handler=web_search,
        description="Search the web; returns a list of {title, url, snippet}.",
        args_json_schema=_SEARCH_SCHEMA,
        metadata={"pack": "web"},
    )
    register_local_tool(
        registry,
        name="web_fetch",
        handler=web_fetch,
        description="Fetch an http(s) URL and return its readable text + title.",
        args_json_schema=_FETCH_SCHEMA,
        metadata={"pack": "web"},
    )
    register_local_tool(
        registry,
        name="http_request",
        handler=http_request,
        description="Make an HTTP request (GET/POST/...) to an http(s) URL.",
        args_json_schema=_HTTP_SCHEMA,
        sensitive_arg_names=["headers"],
        metadata={"pack": "web"},
    )


__all__ = [
    "register_web_pack",
    "SearchBackend",
    "ddg_post_html",
    "DuckDuckGoBackend",
    "TavilyBackend",
    "BraveBackend",
    "build_search_backend",
    "html_to_text",
    "HttpResponse",
    "HttpCaller",
]
