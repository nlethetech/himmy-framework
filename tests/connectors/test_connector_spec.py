"""Tests for the declarative connector spec (himmy.connectors.spec).

A YAML-shaped ``ConnectorSpec`` becomes a live, secure outbound connector: credentials
from the secrets layer, SSRF-guarded URLs, idempotent side effects, capability gating.
This is also the CONTRACT-test pattern for a real external-service connector (GitHub /
Slack): the transport is mocked, and we assert the exact request — URL, method, auth
header, body — plus the SSRF/allowlist/idempotency enforcement. A connector built this
way needs a LIVE token only to verify against the real service; the contract is proven
here offline.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.connectors.sdk import ConnectorError
from himmy.connectors.spec import ConnectorSpec, register_connector_specs
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from tests.conftest import run_async


class _MemSecrets:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class _RecordingFetcher:
    """A Fetcher that records GET URLs + per-request headers and returns canned JSON.

    Header-aware on purpose: the prior version took only ``get_text(url)``, which
    structurally COULD NOT observe the ``Authorization`` header — exactly what masked
    the bug where authenticated declarative GET tools were sent unauthenticated. The
    contract tests assert ``headers`` here so a regression is caught offline.
    """

    def __init__(self, text: str = "{}") -> None:
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self._text = text

    def get_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        self.urls.append(url)
        self.headers.append(dict(headers or {}))
        return self._text

    def get_bytes(self, url: str, *, headers: dict[str, str] | None = None) -> bytes:
        self.urls.append(url)
        self.headers.append(dict(headers or {}))
        return self._text.encode()


class _HeaderBlindFetcher:
    """A minimal Fetcher whose get_text takes only the URL (a header-unaware double).

    Proves :func:`get_json` degrades cleanly: an authenticated GET through a fetcher
    that cannot carry headers still fetches (the headers are simply not applied) rather
    than crashing on an unexpected ``headers`` kwarg.
    """

    def __init__(self, text: str = "{}") -> None:
        self.urls: list[str] = []
        self._text = text

    def get_text(self, url: str) -> str:
        self.urls.append(url)
        return self._text

    def get_bytes(self, url: str) -> bytes:
        self.urls.append(url)
        return self._text.encode()


def _github_like_spec() -> ConnectorSpec:
    return ConnectorSpec(
        name="github",
        description="GitHub REST API.",
        base_url="https://api.github.com",
        auth={"type": "bearer", "secret": "GITHUB_TOKEN"},
        egress_allow_hosts=["api.github.com"],
        rate_limit={"rate": 5, "per_seconds": 60},
        tools=[
            {
                "name": "github_get_issue",
                "method": "GET",
                "path": "/repos/{owner}/{repo}/issues/{number}",
            },
            {
                "name": "github_create_issue",
                "method": "POST",
                "path": "/repos/{owner}/{repo}/issues",
                "body": ["title", "body"],
                "requires_approval": True,
                "idempotency_arg": "title",
            },
        ],
    )


def test_spec_get_tool_builds_guarded_url_and_calls_fetcher() -> None:
    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "ghp_x"}))
    try:
        fetcher = _RecordingFetcher('{"number": 7, "title": "bug"}')
        connector = _github_like_spec().build(fetcher=fetcher)
        registry = ToolRegistry()
        names = connector.register_tools(registry)
        assert "github_get_issue" in names
        out = run_async(
            registry.handler_for("github_get_issue")(
                {"owner": "nlethetech", "repo": "himmy", "number": "7"}
            )
        )
        assert out["ok"] is True
        assert out["data"]["number"] == 7
        assert fetcher.urls == [
            "https://api.github.com/repos/nlethetech/himmy/issues/7"
        ]
        # The GET path MUST carry the auth header from the secrets layer — a read tool
        # under auth:{bearer} that is sent unauthenticated breaks every API that
        # authenticates reads (Slack, private GitHub, search). This is the regression
        # the old header-blind fetcher could not see.
        assert fetcher.headers == [{"Authorization": "Bearer ghp_x"}]
    finally:
        configure_secrets(None)


def test_spec_get_tool_without_auth_sends_no_auth_header() -> None:
    """A connector with no auth sends a bare GET — no spurious Authorization header."""
    fetcher = _RecordingFetcher('{"ok": true}')
    spec = ConnectorSpec(
        name="public",
        base_url="https://api.github.com",
        egress_allow_hosts=["api.github.com"],
        tools=[{"name": "ping", "method": "GET", "path": "/zen"}],
    )
    registry = ToolRegistry()
    spec.build(fetcher=fetcher).register_tools(registry)
    run_async(registry.handler_for("ping")({}))
    assert fetcher.headers == [{}]


def test_spec_get_tool_with_header_blind_fetcher_still_fetches() -> None:
    """An authenticated GET through a header-unaware fetcher degrades, doesn't crash."""
    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "ghp_x"}))
    try:
        fetcher = _HeaderBlindFetcher('{"number": 1}')
        connector = _github_like_spec().build(fetcher=fetcher)
        registry = ToolRegistry()
        connector.register_tools(registry)
        out = run_async(
            registry.handler_for("github_get_issue")(
                {"owner": "o", "repo": "r", "number": "1"}
            )
        )
        assert out["ok"] is True
        assert fetcher.urls == ["https://api.github.com/repos/o/r/issues/1"]
    finally:
        configure_secrets(None)


def test_spec_read_tool_is_read_only_and_write_tool_is_approval_gated() -> None:
    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "ghp_x"}))
    try:
        registry = ToolRegistry()
        _github_like_spec().build(fetcher=_RecordingFetcher()).register_tools(registry)
        assert registry.get("github_get_issue").read_only is True
        assert registry.get("github_get_issue").requires_approval is False
        assert registry.get("github_create_issue").read_only is False
        assert registry.get("github_create_issue").requires_approval is True
    finally:
        configure_secrets(None)


def test_spec_post_sends_auth_header_and_body_via_mock_transport() -> None:
    """Contract test: assert the real request a POST tool would make (mocked transport)."""
    import httpx

    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "ghp_secret"}))
    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = request.read().decode()
        return httpx.Response(201, json={"id": 1, "number": 99})

    try:
        connector = _github_like_spec().build(fetcher=_RecordingFetcher())
        registry = ToolRegistry()
        connector.register_tools(registry)
        # Pin the POST path's transport to the mock app (no real socket, no real DNS).
        import himmy.connectors.spec as spec_mod

        orig = spec_mod.build_pinned_transport
        spec_mod.build_pinned_transport = lambda **_k: httpx.MockTransport(app)  # type: ignore[assignment]
        try:
            out = run_async(
                registry.handler_for("github_create_issue")(
                    {
                        "owner": "nlethetech",
                        "repo": "himmy",
                        "title": "New issue",
                        "body": "details",
                    }
                )
            )
        finally:
            spec_mod.build_pinned_transport = orig
        assert out["ok"] is True
        assert seen["method"] == "POST"
        assert seen["url"] == "https://api.github.com/repos/nlethetech/himmy/issues"
        assert seen["auth"] == "Bearer ghp_secret"  # from the secrets layer
        assert '"title":"New issue"' in seen["body"]
        assert '"body":"details"' in seen["body"]
    finally:
        configure_secrets(None)


def test_spec_get_sends_auth_header_on_the_wire_via_mock_transport() -> None:
    """End-to-end contract: a GET tool's request actually carries Authorization.

    Uses a real :class:`HttpxFetcher` over an ``httpx.MockTransport`` (the same setup
    that empirically confirmed the bug) so the assertion is on the request that hits the
    wire — not on a recording double. Before the fix this header was absent because the
    GET path discarded the connector's auth headers.
    """
    import httpx

    from himmy.connectors.fetcher import HttpxFetcher

    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "ghp_wire"}))
    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"number": 7})

    try:
        # allow_private so the fetcher's guard/pinning is a passthrough for the mock.
        fetcher = HttpxFetcher(transport=httpx.MockTransport(app), retries=0)
        connector = _github_like_spec().build(fetcher=fetcher)
        registry = ToolRegistry()
        connector.register_tools(registry)
        out = run_async(
            registry.handler_for("github_get_issue")(
                {"owner": "nlethetech", "repo": "himmy", "number": "7"}
            )
        )
        assert out["ok"] is True
        assert seen["method"] == "GET"
        assert seen["url"] == "https://api.github.com/repos/nlethetech/himmy/issues/7"
        assert seen["auth"] == "Bearer ghp_wire"  # the fix: auth reaches the wire
    finally:
        configure_secrets(None)


def test_spec_post_idempotent_arg_dedupes_duplicate_call() -> None:
    """A POST with an idempotency_arg fires its effect once for a repeated key."""
    import httpx

    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "ghp_x"}))
    hits = {"n": 0}

    def app(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(201, json={"id": hits["n"]})

    try:
        connector = _github_like_spec().build(fetcher=_RecordingFetcher())
        registry = ToolRegistry()
        connector.register_tools(registry)
        import himmy.connectors.spec as spec_mod

        orig = spec_mod.build_pinned_transport
        spec_mod.build_pinned_transport = lambda **_k: httpx.MockTransport(app)  # type: ignore[assignment]
        try:
            args = {
                "owner": "o",
                "repo": "r",
                "title": "same-title",
                "body": "b",
            }
            first = run_async(registry.handler_for("github_create_issue")(dict(args)))
            second = run_async(registry.handler_for("github_create_issue")(dict(args)))
        finally:
            spec_mod.build_pinned_transport = orig
        assert first == second  # cached
        assert hits["n"] == 1  # the effect fired exactly once
    finally:
        configure_secrets(None)


def test_spec_url_building_rejects_off_allowlist_host() -> None:
    """A base_url whose host is not on the egress allow-list is refused at guard time."""
    spec = ConnectorSpec(
        name="evil",
        base_url="https://api.evil.test",
        egress_allow_hosts=["api.github.com"],
        tools=[{"name": "t", "method": "GET", "path": "/x"}],
    )
    connector = spec.build(fetcher=_RecordingFetcher())
    registry = ToolRegistry()
    connector.register_tools(registry)
    with pytest.raises(ToolSecurityError, match="egress allow-list"):
        run_async(registry.handler_for("t")({}))


def test_spec_path_arg_cannot_escape_to_another_host() -> None:
    """A model-supplied path arg is percent-encoded so it can't inject a host/path."""
    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "x"}))
    try:
        fetcher = _RecordingFetcher()
        connector = _github_like_spec().build(fetcher=fetcher)
        registry = ToolRegistry()
        connector.register_tools(registry)
        run_async(
            registry.handler_for("github_get_issue")(
                {"owner": "../../evil.com", "repo": "r", "number": "1"}
            )
        )
        # The traversal-looking owner is percent-encoded (its slashes became %2F), so
        # it stays a single opaque path segment and can't escape to another host/path.
        url = fetcher.urls[0]
        assert url.startswith("https://api.github.com/repos/")
        assert "..%2F..%2Fevil.com" in url  # slashes encoded, not raw separators
        assert "/../" not in url
    finally:
        configure_secrets(None)


def test_register_connector_specs_skips_connector_with_missing_credential() -> None:
    """A connector whose secret is absent is skipped; a configured one still registers."""
    configure_secrets(_MemSecrets({"OK_TOKEN": "v"}))
    try:
        ok = ConnectorSpec(
            name="ok",
            base_url="https://api.github.com",
            auth={"type": "bearer", "secret": "OK_TOKEN"},
            tools=[{"name": "ok_tool", "method": "GET", "path": "/x"}],
        )
        missing = ConnectorSpec(
            name="missing",
            base_url="https://api.github.com",
            auth={"type": "bearer", "secret": "ABSENT_TOKEN"},
            tools=[{"name": "missing_tool", "method": "GET", "path": "/y"}],
        )
        registry = ToolRegistry()
        names = register_connector_specs(
            registry,
            [ok, missing],
            fetchers=[_RecordingFetcher(), _RecordingFetcher()],
        )
        assert names == ["ok_tool"]
        assert registry.get("missing_tool") is None
    finally:
        configure_secrets(None)


def test_spec_rejects_unknown_fields() -> None:
    """extra='forbid' keeps a typo'd YAML key from silently doing nothing."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ConnectorSpec(
            name="x",
            base_url="https://api.github.com",
            tooools=[],  # typo
        )


def test_spec_missing_path_arg_is_a_clean_error() -> None:
    configure_secrets(_MemSecrets({"GITHUB_TOKEN": "x"}))
    try:
        connector = _github_like_spec().build(fetcher=_RecordingFetcher())
        registry = ToolRegistry()
        connector.register_tools(registry)
        with pytest.raises(ConnectorError, match="path arg"):
            run_async(
                registry.handler_for("github_get_issue")({"owner": "o", "repo": "r"})
            )
    finally:
        configure_secrets(None)
