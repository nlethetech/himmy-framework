"""Contract tests for the web-search connector — fully offline (mocked transport).

Tavily and Brave are EXTERNAL services: end-to-end verification against the real APIs
needs a live API key and cannot run on this machine. These tests therefore CONTRACT-test
the connector against a mocked ``httpx`` transport and a synthetic key, for BOTH backends:
they assert the exact request (method/URL/query/body) each backend makes; that the API key
is carried where the provider wants it (Tavily JSON body, Brave ``X-Subscription-Token``
header) but is redacted-in-logs/errors; that each backend's result schema is parsed into a
uniform ``{title, url, snippet}``; that the requested result count is clamped and the parsed
list truncated to the hard cap; that an oversized response body is refused; that backend
selection follows config / the available key (default-deny capability when no key); and that
401 / non-JSON responses become clean, key-free errors. What still needs a real key to
verify is marked ``live-pending`` in the module docstring.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.connectors.sdk import AuditSink, ConnectorContext, ConnectorError
from himmy.connectors.websearch import (
    BRAVE_API_KEY_SECRET,
    TAVILY_API_KEY_SECRET,
    WEBSEARCH_BACKEND_ENV,
    WebSearchClient,
    WebSearchConnector,
    WebSearchError,
    _clamp_count,
    _parse_brave,
    _parse_tavily,
)
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from tests.conftest import run_async


class _MemSecrets:
    """In-memory secret provider for tests (installed via configure_secrets)."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class _RecordingAudit:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)


_TAVILY_KEY = "tvly-SUPER_SECRET_key_value"  # a fake key (test only)
_BRAVE_KEY = "BSA-SUPER_SECRET_key_value"  # a fake key (test only)


def _mock_transport(app: Any) -> Any:
    import httpx

    return httpx.MockTransport(app)


def _connector(
    app: Any,
    *,
    backend: str | None = None,
    audit: _RecordingAudit | None = None,
) -> WebSearchConnector:
    ctx = ConnectorContext(audit=AuditSink(audit.record)) if audit else None
    return WebSearchConnector(
        backend=backend,
        client_factory=lambda b, key: WebSearchClient(
            b, key, transport=_mock_transport(app)
        ),
        context=ctx,
    )


def _registered(conn: WebSearchConnector) -> ToolRegistry:
    registry = ToolRegistry()
    conn.register_tools(registry)
    return registry


# ------------------------------------------------------------- tool registration


def test_registers_single_web_search_tool() -> None:
    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        conn = WebSearchConnector()
        registry = ToolRegistry()
        names = conn.register_tools(registry)
        assert names == ["web_search"]
        definition = registry.get("web_search")
        assert definition.read_only is True
        assert definition.requires_approval is False
        assert definition.metadata.get("connector") == "websearch"
    finally:
        configure_secrets(None)


def test_capability_gated_on_any_key() -> None:
    """With no backend key the connector is unavailable and registers no tools."""
    configure_secrets(_MemSecrets({}))
    try:
        conn = WebSearchConnector()
        assert conn.available() is False
        registry = ToolRegistry()
        assert conn.register_tools(registry) == []  # skipped cleanly, no crash
    finally:
        configure_secrets(None)


def test_capability_ok_with_brave_key_only() -> None:
    """Auto-selection finds Brave when only its key is present."""
    configure_secrets(_MemSecrets({BRAVE_API_KEY_SECRET: _BRAVE_KEY}))
    try:
        conn = WebSearchConnector()
        assert conn.available() is True
        assert conn._resolve_backend() == "brave"
    finally:
        configure_secrets(None)


def test_explicit_backend_requires_its_own_key() -> None:
    """Forcing tavily while only a brave key is present is unavailable (no silent swap)."""
    configure_secrets(_MemSecrets({BRAVE_API_KEY_SECRET: _BRAVE_KEY}))
    try:
        conn = WebSearchConnector(backend="tavily")
        assert conn.available() is False
    finally:
        configure_secrets(None)


def test_unknown_backend_is_loud() -> None:
    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        conn = WebSearchConnector(backend="duckduckgo")
        # Unavailable (unknown backend has no resolvable key)...
        assert conn.available() is False
        # ...and an explicit call raises a clear error rather than silently picking one.
        with pytest.raises(ConnectorError, match="unknown web-search backend"):
            conn._resolve_backend()
    finally:
        configure_secrets(None)


def test_env_var_forces_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WEBSEARCH_BACKEND_ENV, "brave")
    configure_secrets(
        _MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY, BRAVE_API_KEY_SECRET: _BRAVE_KEY})
    )
    try:
        conn = WebSearchConnector()
        assert conn._resolve_backend() == "brave"  # env override beats tavily preference
    finally:
        configure_secrets(None)


def test_auto_prefers_tavily_when_both_keys_present() -> None:
    configure_secrets(
        _MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY, BRAVE_API_KEY_SECRET: _BRAVE_KEY})
    )
    try:
        conn = WebSearchConnector()
        assert conn._resolve_backend() == "tavily"
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- Tavily backend


def test_tavily_posts_key_in_body_and_parses_results() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth_header"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Result One",
                        "url": "https://example.com/1",
                        "content": "first snippet",
                    },
                    {
                        "title": "Result Two",
                        "url": "https://example.com/2",
                        "content": "second snippet",
                    },
                    # A malformed entry (no url) must be skipped, not crash.
                    {"title": "no url", "content": "x"},
                ]
            },
        )

    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily"))
        out = run_async(
            registry.handler_for("web_search")({"query": "python testing", "count": 5})
        )
        assert seen["method"] == "POST"
        assert seen["url"] == "https://api.tavily.com/search"
        # No bearer/Authorization header — Tavily auth is the body `api_key`.
        assert seen["auth_header"] is None
        assert seen["body"]["api_key"] == _TAVILY_KEY
        assert seen["body"]["query"] == "python testing"
        assert seen["body"]["max_results"] == 5
        assert out["backend"] == "tavily"
        assert out["count"] == 2  # the no-url entry was dropped
        assert out["results"][0] == {
            "title": "Result One",
            "url": "https://example.com/1",
            "snippet": "first snippet",
        }
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- Brave backend


def test_brave_sends_key_in_header_and_parses_results() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("X-Subscription-Token")
        seen["accept"] = request.headers.get("Accept")
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave One",
                            "url": "https://example.org/a",
                            "description": "brave snippet a",
                        },
                        {
                            "title": "Brave Two",
                            "url": "https://example.org/b",
                            "description": "brave snippet b",
                        },
                    ]
                }
            },
        )

    configure_secrets(_MemSecrets({BRAVE_API_KEY_SECRET: _BRAVE_KEY}))
    try:
        registry = _registered(_connector(app, backend="brave"))
        out = run_async(
            registry.handler_for("web_search")({"query": "nepal news", "count": 3})
        )
        assert seen["method"] == "GET"
        assert seen["url"].startswith(
            "https://api.search.brave.com/res/v1/web/search?"
        )
        assert "q=nepal+news" in seen["url"]
        assert "count=3" in seen["url"]
        # The key rides the subscription-token header, never the URL.
        assert seen["token"] == _BRAVE_KEY
        assert _BRAVE_KEY not in seen["url"]
        assert seen["accept"] == "application/json"
        assert out["backend"] == "brave"
        assert out["count"] == 2
        assert out["results"][1] == {
            "title": "Brave Two",
            "url": "https://example.org/b",
            "snippet": "brave snippet b",
        }
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- caps + safety


def test_count_is_clamped_to_max() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"results": []})

    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily"))
        run_async(
            registry.handler_for("web_search")({"query": "q", "count": 9999})
        )
        assert seen["body"]["max_results"] == 10  # clamped to _MAX_RESULTS
    finally:
        configure_secrets(None)


def test_result_list_truncated_to_requested_count() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": f"r{i}", "url": f"https://e/{i}", "content": "c"}
                    for i in range(10)
                ]
            },
        )

    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily"))
        out = run_async(
            registry.handler_for("web_search")({"query": "q", "count": 2})
        )
        assert out["count"] == 2  # only the first two are returned
    finally:
        configure_secrets(None)


def test_snippet_is_length_capped() -> None:
    import httpx

    long_text = "x" * 5000

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "t", "url": "https://e/1", "content": long_text}
                ]
            },
        )

    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily"))
        out = run_async(registry.handler_for("web_search")({"query": "q"}))
        snippet = out["results"][0]["snippet"]
        assert len(snippet) <= 601  # 600 chars + the truncation ellipsis
        assert snippet.endswith("…")
    finally:
        configure_secrets(None)


def test_oversized_response_body_is_refused() -> None:
    import httpx

    huge = json.dumps(
        {"results": [{"title": "t", "url": "https://e/1", "content": "x" * (3 * 1024 * 1024)}]}
    ).encode()

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=huge)

    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily"))
        # The body crosses the 2 MiB cap → normalized to a clean ConnectorError.
        with pytest.raises(ConnectorError, match="web_search failed"):
            run_async(registry.handler_for("web_search")({"query": "q"}))
    finally:
        configure_secrets(None)


def test_empty_query_is_rejected() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(200, json={"results": []})

    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily"))
        with pytest.raises(ToolSecurityError, match="query"):
            run_async(registry.handler_for("web_search")({"query": "   "}))
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- error handling


def test_http_error_is_clean_and_key_free() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    audit = _RecordingAudit()
    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily", audit=audit))
        with pytest.raises(ConnectorError) as excinfo:
            run_async(registry.handler_for("web_search")({"query": "q"}))
        message = str(excinfo.value)
        assert "401" in message
        assert _TAVILY_KEY not in message  # the key never rides out in an error
        # A deny event was audited, also key-free.
        deny = [e for e in audit.events if e.outcome == "deny"]
        assert deny and all(_TAVILY_KEY not in e.detail for e in deny)
    finally:
        configure_secrets(None)


def test_non_json_body_is_clean_error() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily"))
        with pytest.raises(ConnectorError, match="web_search failed"):
            run_async(registry.handler_for("web_search")({"query": "q"}))
    finally:
        configure_secrets(None)


def test_successful_call_audits_allow_with_redacted_args() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"results": [{"title": "t", "url": "https://e/1", "content": "c"}]}
        )

    audit = _RecordingAudit()
    configure_secrets(_MemSecrets({TAVILY_API_KEY_SECRET: _TAVILY_KEY}))
    try:
        registry = _registered(_connector(app, backend="tavily", audit=audit))
        run_async(registry.handler_for("web_search")({"query": "q"}))
        allows = [e for e in audit.events if e.outcome == "allow" and e.action == "web_search"]
        assert allows
        assert _TAVILY_KEY not in allows[0].detail
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- pure parsers


def test_clamp_count_bounds() -> None:
    assert _clamp_count(None) == 5
    assert _clamp_count(0) == 1
    assert _clamp_count(-3) == 1
    assert _clamp_count(7) == 7
    assert _clamp_count(99) == 10
    assert _clamp_count("not a number") == 5


def test_parse_tavily_handles_missing_sections() -> None:
    assert _parse_tavily({}, 5) == []
    assert _parse_tavily({"results": None}, 5) == []
    assert _parse_tavily(None, 5) == []


def test_parse_brave_handles_missing_sections() -> None:
    assert _parse_brave({}, 5) == []
    assert _parse_brave({"web": {}}, 5) == []
    assert _parse_brave({"web": {"results": None}}, 5) == []


def test_client_rejects_unknown_backend() -> None:
    with pytest.raises(ConnectorError, match="unknown web-search backend"):
        WebSearchClient("yahoo", "k")


def test_client_search_raises_typed_error_with_status() -> None:
    """The low-level client surfaces a WebSearchError carrying the HTTP status."""
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client = WebSearchClient("tavily", _TAVILY_KEY, transport=_mock_transport(app))
    with pytest.raises(WebSearchError) as excinfo:
        client.search("q", count=3)
    assert excinfo.value.status == 429
    assert _TAVILY_KEY not in str(excinfo.value)
