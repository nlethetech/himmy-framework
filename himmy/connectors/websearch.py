"""Web search connector on the SDK: a single guarded ``web_search`` tool.

Agents can already *fetch* a known URL (the ``web_fetch`` tool + the SSRF-guarded
fetcher) but cannot *discover* one — there is no way to turn a natural-language query
into a ranked list of result links. This connector adds that missing capability as one
**outbound** tool, ``web_search(query) -> results[]`` (each result a
``{title, url, snippet}``), inheriting the connector SDK's security posture (secrets-layer
credentials, SSRF-guarded egress pinned to the search API host, secret-safe errors,
capability detection, structured audit events).

The search backend is **pluggable**. Two providers are supported and the connector
selects one automatically from configuration / the available API key:

* **Tavily** (``https://api.tavily.com/search``) — key :data:`TAVILY_API_KEY_SECRET`; the
  key rides in the JSON body (Tavily's documented auth), never in a log.
* **Brave** (``https://api.search.brave.com/res/v1/web/search``) — key
  :data:`BRAVE_API_KEY_SECRET`; the key rides in the ``X-Subscription-Token`` header.

Selection: an explicit ``backend=`` (or ``HIMMY_WEBSEARCH_BACKEND``) wins; otherwise the
first backend whose key resolves through the secrets layer is used (Tavily preferred, then
Brave). With NO key for either, the connector is **unavailable** — capability-detected, so
discovery / ``register_tools`` skip it cleanly rather than registering a tool that always
errors.

Safety rails specific to search:

* every call goes through the connector's **SSRF-guarded pinned transport**, pinned to the
  selected provider's API host (egress allow-list) with redirects OFF, so an authenticated
  request can never be bounced to an internal host;
* the requested result count is **clamped** to :data:`_MAX_RESULTS` so a model cannot ask
  for an unbounded page, and the parsed result list is truncated to the same cap;
* the response body is **size-capped** (:data:`_MAX_RESPONSE_BYTES`) by the fetcher so a
  hostile / runaway provider response cannot exhaust memory, and each result's snippet is
  truncated to :data:`_MAX_SNIPPET_CHARS` to bound what lands in the model context;
* the API key is read from the **secrets layer** at call time, never from a constructor arg
  the model could set, never logged, and is scrubbed from any surfaced error.

Search is a read-only, side-effect-free query, so the tool is registered ``read_only`` and
needs no idempotency key or approval gate.

Nothing here imports ``httpx`` at module load — capability detection probes lazily — so
importing this module is free in an air-gapped install.

LIVE-PENDING: the request shapes, auth handling (body key for Tavily / header token for
Brave), result parsing, and the size/count caps are contract-tested against a mocked
transport for BOTH backends (see ``tests/connectors/test_websearch.py``); verifying against
the REAL Tavily / Brave APIs needs a live API key and cannot run on this machine.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from himmy.connectors.sdk import (
    Capability,
    ConnectorContext,
    ConnectorError,
    OutboundToolConnector,
    RateLimiter,
    capability,
    redact_args,
    safe_error,
)
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit._net import build_pinned_transport

if TYPE_CHECKING:  # pragma: no cover - typing only (httpx stays a lazy import)
    import httpx

# --------------------------------------------------------------------- backends
#: Tavily search endpoint + its single egress host. The API key travels in the JSON
#: body (Tavily's documented scheme), so it is NOT in the URL or a log.
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_API_HOST = "api.tavily.com"

#: Brave Web Search endpoint + its single egress host. The API key travels in the
#: ``X-Subscription-Token`` request header.
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_API_HOST = "api.search.brave.com"

#: Secret NAMES (not values) for each backend's API key, resolved via the secrets layer,
#: NEVER the model. These are the env/secret variable names (S105 false positive).
TAVILY_API_KEY_SECRET = "HIMMY_TAVILY_API_KEY"  # noqa: S105
BRAVE_API_KEY_SECRET = "HIMMY_BRAVE_API_KEY"  # noqa: S105

#: Env var an operator can set to force a backend (overrides auto-detection).
WEBSEARCH_BACKEND_ENV = "HIMMY_WEBSEARCH_BACKEND"

#: Backend preference order when auto-selecting by available key.
_BACKEND_ORDER = ("tavily", "brave")
_BACKEND_HOSTS = {"tavily": TAVILY_API_HOST, "brave": BRAVE_API_HOST}
_BACKEND_SECRETS = {"tavily": TAVILY_API_KEY_SECRET, "brave": BRAVE_API_KEY_SECRET}

# --------------------------------------------------------------------- caps
#: Hard ceiling on results returned for one query (a DoS / context guard). The model's
#: requested count is clamped to this and the parsed list is truncated to it.
_MAX_RESULTS = 10
#: Default result count when the model does not ask for one.
_DEFAULT_RESULTS = 5
#: Ceiling on a search response body (a provider could otherwise stream forever). The
#: fetcher enforces this while streaming, so an unbounded body is refused early.
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
#: Ceiling on a single result's snippet length surfaced to the model.
_MAX_SNIPPET_CHARS = 600
#: Ceiling on the query length accepted (a sane bound; providers reject huge queries).
_MAX_QUERY_CHARS = 400
#: Per-request timeout for a search call (seconds).
_DEFAULT_TIMEOUT = 20.0


class WebSearchError(ConnectorError):
    """A web-search API call failed. Carries the HTTP status so a caller can branch.

    The message is constructed to be secret-safe (only the status, the backend name, and
    the provider's own token-free error text — never a request header or the raw body that
    could echo the key). ``status`` is the HTTP status code (0 for a transport-level
    failure), so a handler can distinguish 401 / 429 from a generic error.
    """

    def __init__(self, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def _select_backend(explicit: str | None) -> str | None:
    """Pick a backend: an explicit choice (validated) else the first with a key.

    ``explicit`` (a constructor arg or ``HIMMY_WEBSEARCH_BACKEND``) wins when it names a
    known backend AND that backend's key resolves; an unknown explicit name raises so a
    typo is loud rather than silently falling back. With no explicit choice, the first
    backend in :data:`_BACKEND_ORDER` whose key resolves through the secrets layer is
    used. Returns ``None`` when no backend has a key (the connector is then unavailable).
    """
    from himmy.config.secrets import get_secret

    if explicit:
        choice = explicit.strip().lower()
        if choice not in _BACKEND_ORDER:
            raise ConnectorError(
                f"unknown web-search backend {explicit!r} "
                f"(supported: {', '.join(_BACKEND_ORDER)})"
            )
        return choice
    for name in _BACKEND_ORDER:
        if get_secret(_BACKEND_SECRETS[name]):
            return name
    return None


class WebSearchClient:
    """A tiny client over a single search provider: one POST/GET, parse, cap.

    All HTTP rides on the SSRF-guarded **pinned transport** (the DNS-rebinding-safe one the
    rest of the connector layer uses) with redirects OFF, so a 3xx can never bounce an
    authenticated request to an internal host. The API key is passed by the caller (resolved
    from the secrets layer) and placed where the provider wants it — the JSON body for
    Tavily, the ``X-Subscription-Token`` header for Brave — never logged. ``transport`` is a
    test seam for injecting an ``httpx`` mock.
    """

    def __init__(
        self,
        backend: str,
        api_key: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        allow_private_hosts: bool = False,
        egress_allow_hosts: Collection[str] | None = None,
        transport: Any = None,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
    ) -> None:
        if backend not in _BACKEND_ORDER:
            raise ConnectorError(f"unknown web-search backend {backend!r}")
        self._backend = backend
        self._key = api_key
        self._timeout = timeout
        self._allow_private = allow_private_hosts
        # Default the egress allow-list to the selected backend's host (defence in depth):
        # even if a caller widens it, the pinned transport still vets the resolved IP.
        self._allow_hosts = (
            set(egress_allow_hosts)
            if egress_allow_hosts
            else {_BACKEND_HOSTS[backend]}
        )
        self._transport = transport
        self._max_bytes = max_response_bytes

    @property
    def backend(self) -> str:
        return self._backend

    def _client(self) -> httpx.Client:
        import httpx

        transport = self._transport
        if transport is None:
            transport = build_pinned_transport(
                allow_private=self._allow_private, allow_hosts=self._allow_hosts
            )
        return httpx.Client(
            timeout=self._timeout,
            follow_redirects=False,  # never auto-follow a 3xx with the key attached
            transport=transport,
        )

    def _read_capped(self, resp: httpx.Response) -> bytes:
        """Read the response body, refusing one over ``max_response_bytes``.

        Reading incrementally means a never-ending / multi-GB body is rejected as soon as
        it crosses the cap, not after the whole thing lands in memory.
        """
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > self._max_bytes:
                resp.close()
                raise WebSearchError(
                    f"{self._backend} search response exceeded the "
                    f"{self._max_bytes}-byte safety cap",
                    status=resp.status_code,
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _raise_for_status(self, resp: httpx.Response) -> None:
        """Map a non-2xx response to a secret-safe :class:`WebSearchError`.

        Surfaces ONLY the status and the backend name — never a request header or the raw
        body verbatim, so a key a proxy might echo cannot ride out in an error.
        """
        if resp.status_code < 400:
            return
        raise WebSearchError(
            f"{self._backend} search failed: HTTP {resp.status_code}",
            status=resp.status_code,
        )

    def search(self, query: str, *, count: int) -> list[dict[str, str]]:
        """Run one search and return up to ``count`` ``{title,url,snippet}`` dicts."""
        if self._backend == "tavily":
            raw = self._search_tavily(query, count)
            return _parse_tavily(raw, count)
        raw = self._search_brave(query, count)
        return _parse_brave(raw, count)

    # -- Tavily: key in the JSON body, POST -------------------------------
    def _search_tavily(self, query: str, count: int) -> Any:
        body = {
            "api_key": self._key,
            "query": query,
            "max_results": count,
            "search_depth": "basic",
        }
        with self._client() as client:
            with client.stream(
                "POST",
                TAVILY_SEARCH_URL,
                content=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ) as resp:
                payload = self._read_capped(resp)
                self._raise_for_status(resp)
        return _load_json(payload, self._backend)

    # -- Brave: key in the X-Subscription-Token header, GET ---------------
    def _search_brave(self, query: str, count: int) -> Any:
        url = f"{BRAVE_SEARCH_URL}?{urlencode({'q': query, 'count': count})}"
        with self._client() as client:
            with client.stream(
                "GET",
                url,
                headers={
                    "X-Subscription-Token": self._key,
                    "Accept": "application/json",
                },
            ) as resp:
                payload = self._read_capped(resp)
                self._raise_for_status(resp)
        return _load_json(payload, self._backend)


def _load_json(payload: bytes, backend: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8", "replace"))
    except (ValueError, TypeError):
        raise WebSearchError(f"{backend} search returned a non-JSON body") from None


# --------------------------------------------------------------- result parsing
def _clean_snippet(text: Any) -> str:
    """Coerce a snippet to a bounded, single-line-ish string for the model."""
    s = str(text or "").strip()
    if len(s) > _MAX_SNIPPET_CHARS:
        s = s[:_MAX_SNIPPET_CHARS].rstrip() + "…"
    return s


def _parse_tavily(raw: Any, count: int) -> list[dict[str, str]]:
    """Project Tavily's ``results[]`` into ``{title,url,snippet}`` (capped to ``count``)."""
    results = (raw or {}).get("results") if isinstance(raw, Mapping) else None
    out: list[dict[str, str]] = []
    for item in results or []:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        out.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": url,
                "snippet": _clean_snippet(item.get("content")),
            }
        )
        if len(out) >= count:
            break
    return out


def _parse_brave(raw: Any, count: int) -> list[dict[str, str]]:
    """Project Brave's ``web.results[]`` into ``{title,url,snippet}`` (capped to ``count``)."""
    web = (raw or {}).get("web") if isinstance(raw, Mapping) else None
    results = web.get("results") if isinstance(web, Mapping) else None
    out: list[dict[str, str]] = []
    for item in results or []:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        out.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": url,
                "snippet": _clean_snippet(item.get("description")),
            }
        )
        if len(out) >= count:
            break
    return out


# --------------------------------------------------------------- tool schema
_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The natural-language search query (required).",
        },
        "count": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_RESULTS,
            "description": (
                f"How many results to return (1-{_MAX_RESULTS}; "
                f"default {_DEFAULT_RESULTS}). Clamped to {_MAX_RESULTS}."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


class WebSearchConnector(OutboundToolConnector):
    """Exposes a single ``web_search`` tool over a pluggable search backend.

    Security contract (inherited from :class:`OutboundToolConnector` + enforced here):

    * the backend's API key is read from the **secrets layer**
      (:data:`TAVILY_API_KEY_SECRET` / :data:`BRAVE_API_KEY_SECRET`) at call time, never
      from a constructor arg the model could set, never logged;
    * the connector is **unavailable** (skipped cleanly by discovery / ``register_tools``)
      when no backend key resolves — capability-gated;
    * every request goes through the SSRF-guarded **pinned transport**, pinned to the
      selected provider's API host (egress allow-list);
    * the requested result count is **clamped** and the parsed list **truncated** to a hard
      cap, and the response body is **size-capped** while streaming;
    * ``web_search`` is read-only and side-effect-free — registered ``read_only``, no
      approval gate, no idempotency key.
    """

    name = "websearch"

    def __init__(
        self,
        *,
        backend: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        client_factory: Callable[[str, str], WebSearchClient] | None = None,
        context: ConnectorContext | None = None,
        **kwargs: Any,
    ) -> None:
        """Wire the (optional explicit) backend choice + the search client seam.

        ``backend`` forces a provider (``"tavily"`` / ``"brave"``); when ``None`` the
        connector auto-selects the first backend whose key resolves (env override:
        ``HIMMY_WEBSEARCH_BACKEND``). ``client_factory`` is the injectable seam tests use to
        supply a :class:`WebSearchClient` over a mock transport.
        """
        # A sane per-connector self-throttle so a runaway agent can't burn the search quota.
        if context is None:
            context = ConnectorContext(rate_limiter=RateLimiter(rate=5, window=1.0))
        super().__init__(context=context, **kwargs)
        self._explicit_backend = backend
        self._timeout = timeout
        self._client_factory = client_factory

    def _capability_secret(self) -> str:
        """The key NAME whose presence makes this connector available.

        Resolves the backend the same way a call would, so capability reflects exactly the
        provider that would run: an explicit/forced backend needs ITS key; otherwise the
        connector is available when EITHER backend's key is present (auto-selection picks the
        first that resolves). With no key anywhere, returns the preferred backend's name so
        the unavailable ``reason`` is informative.
        """
        from himmy.config.secrets import get_secret

        chosen = self._explicit_backend or os.environ.get(WEBSEARCH_BACKEND_ENV)
        if chosen:
            normalized = chosen.strip().lower()
            # An unknown forced backend has no resolvable key → reported as missing, so the
            # connector is (correctly) unavailable rather than silently auto-selecting.
            return _BACKEND_SECRETS.get(normalized, f"{WEBSEARCH_BACKEND_ENV}:{chosen}")
        for name in _BACKEND_ORDER:
            if get_secret(_BACKEND_SECRETS[name]):
                return _BACKEND_SECRETS[name]
        return _BACKEND_SECRETS[_BACKEND_ORDER[0]]

    def capability(self) -> Capability:
        """Available only when the selected backend's key resolves via the secrets layer."""
        return capability(requires_secrets=(self._capability_secret(),))

    def _resolve_backend(self) -> str:
        """Select the backend for a call (explicit/env override else first key present)."""
        explicit = self._explicit_backend or os.environ.get(WEBSEARCH_BACKEND_ENV)
        backend = _select_backend(explicit)
        if backend is None:
            raise ConnectorError(
                "no web-search backend is configured "
                f"(set {TAVILY_API_KEY_SECRET} or {BRAVE_API_KEY_SECRET})"
            )
        return backend

    def _build_client(self, backend: str) -> WebSearchClient:
        key = self.secret(_BACKEND_SECRETS[backend])  # raises (named) if absent
        if self._client_factory is not None:
            return self._client_factory(backend, key or "")
        return WebSearchClient(
            backend,
            key or "",
            timeout=self._timeout,
            allow_private_hosts=self._allow_private,
            egress_allow_hosts=self._allow_hosts or {_BACKEND_HOSTS[backend]},
        )

    def _fail(self, backend: str, exc: BaseException) -> ConnectorError:
        """Audit + normalize ``exc`` into a secret-safe :class:`ConnectorError`.

        The key is scrubbed from the surfaced text belt-and-braces, on top of
        :func:`safe_error` only trusting a connector-authored message — so even if a
        third-party library echoed the key into an exception, it never rides out.
        """
        key = self.secret(_BACKEND_SECRETS[backend], required=False) or ""
        detail = safe_error(exc, secrets=[key])
        self._audit("web_search", "deny", actor={"subject": backend}, detail=detail)
        return ConnectorError(f"web_search failed: {detail}")

    def register(self, registry: ToolRegistry) -> list[str]:
        import asyncio

        async def web_search(args: dict[str, Any]) -> dict[str, Any]:
            query = str(args.get("query") or "").strip()
            if not query:
                raise ToolSecurityError("web_search requires a non-empty `query`")
            if len(query) > _MAX_QUERY_CHARS:
                query = query[:_MAX_QUERY_CHARS]
            count = _clamp_count(args.get("count"))
            backend = self._resolve_backend()

            def _call() -> list[dict[str, str]]:
                client = self._build_client(backend)
                self._throttle()
                return client.search(query, count=count)

            try:
                results = await asyncio.to_thread(_call)
            except ToolSecurityError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize to a secret-safe error
                raise self._fail(backend, exc) from None
            self._audit(
                "web_search",
                "allow",
                actor={"subject": backend},
                detail=f"{dict(redact_args(args))}",
            )
            return {
                "backend": backend,
                "query": query,
                "count": len(results),
                "results": results,
            }

        register_local_tool(
            registry,
            name="web_search",
            read_only=True,
            handler=web_search,
            description=(
                "Search the web for a natural-language query; returns a ranked list of "
                "{title, url, snippet} results. Use this to discover URLs, then web_fetch "
                "to read a page."
            ),
            args_json_schema=_SEARCH_SCHEMA,
            requires_approval=False,  # a read-only query never needs approval
            sensitive_arg_names=[],
            metadata={"connector": "websearch"},
        )
        return ["web_search"]


def _clamp_count(raw: Any) -> int:
    """Coerce + clamp the requested result count into ``[1, _MAX_RESULTS]``.

    A missing/invalid value falls back to :data:`_DEFAULT_RESULTS`; anything above the cap
    is clamped down so a model cannot request an unbounded page.
    """
    if raw is None:
        return _DEFAULT_RESULTS
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_RESULTS
    return max(1, min(_MAX_RESULTS, count))


# ----------------------------------------------------- registry self-registration
def _make_outbound() -> WebSearchConnector:
    """Build the default web-search connector from the environment.

    Reads an optional forced backend (``HIMMY_WEBSEARCH_BACKEND``); the API key is resolved
    lazily through the secrets layer at call time, so a deployment that sets neither key
    gets a connector that is simply unavailable (skipped cleanly by discovery).
    """
    return WebSearchConnector(backend=os.environ.get(WEBSEARCH_BACKEND_ENV) or None)


def register_websearch_connectors() -> None:
    """Register the web-search outbound connector on the process-wide registry.

    Safe to call more than once.
    """
    from himmy.connectors.sdk import register_connector

    register_connector("websearch", _make_outbound)


__all__ = [
    "TAVILY_SEARCH_URL",
    "TAVILY_API_HOST",
    "TAVILY_API_KEY_SECRET",
    "BRAVE_SEARCH_URL",
    "BRAVE_API_HOST",
    "BRAVE_API_KEY_SECRET",
    "WEBSEARCH_BACKEND_ENV",
    "WebSearchClient",
    "WebSearchConnector",
    "WebSearchError",
    "register_websearch_connectors",
]
