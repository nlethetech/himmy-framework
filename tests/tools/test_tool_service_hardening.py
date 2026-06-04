"""Additional ToolService hardening tests (HTTP/auth/retry/timeout/redaction/events).

Complements ``tests/tools/test_tool_service.py`` by locking in production behavior
the base suite does not yet pin: the HEADER / PREENCODED_BASIC / NONE auth modes,
header-arg secret redaction in events, retry exhaustion + non-retryable codes not
being retried, the HTTP timeout path via the definition-level timeout, args-schema
INVALID_REQUEST for required path args, and event payloads for failure paths.

Offline-only: all HTTP is driven through ``httpx.MockTransport`` (no network); the
pydantic-ai toolset binding is exercised only behind a skipif when the extra is
absent.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from opensims.core.events import EventType
from opensims.services.storage.service import StorageService
from opensims.services.tools import (
    HttpAuthConfig,
    HttpAuthMode,
    HttpToolConfig,
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_http_tool,
    register_local_tool,
)
from opensims.services.tools.security import REDACTED
from tests.conftest import run_async

try:  # optional provider extra — gates the pydantic-ai binding test
    import pydantic_ai  # noqa: F401

    _HAS_PYDANTIC_AI = True
except Exception:  # pragma: no cover - import guard
    _HAS_PYDANTIC_AI = False


def _http_service(transport: httpx.MockTransport) -> ToolService:
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    return ToolService(ToolRegistry(), http_client=client)


# ----------------------------------------------------------------- auth modes
def test_http_header_auth_uses_custom_header(monkeypatch) -> None:
    """HEADER mode writes the secret to the configured header name verbatim."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.setenv("APIKEY", "k-123")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(
                mode=HttpAuthMode.HEADER, env_var="APIKEY", header_name="X-API-Key"
            ),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["api_key"] == "k-123"


def test_http_preencoded_basic_passes_through(monkeypatch) -> None:
    """PREENCODED_BASIC sends the env value verbatim (already base64), unencoded here."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.setenv("CREDS", "QWxhZGRpbjpvcGVuIHNlc2FtZQ==")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(mode=HttpAuthMode.PREENCODED_BASIC, env_var="CREDS"),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["auth"] == "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="


def test_http_auth_none_sends_no_authorization(monkeypatch) -> None:
    """NONE auth (the default) attaches no Authorization header."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/x"),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["auth"] is None


def test_http_auth_env_unset_sends_no_header(monkeypatch) -> None:
    """A configured auth mode with an unset env var attaches no header (no crash)."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.delenv("UNSET_TOKEN", raising=False)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(mode=HttpAuthMode.BEARER, env_var="UNSET_TOKEN"),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert seen["auth"] is None


# ----------------------------------------------------------------- SSRF (host pin)
def test_http_post_assembles_body_args(monkeypatch) -> None:
    """A POST routes declared body_arg_names into the JSON body, not the path."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.content.decode() if request.content else ""
        return httpx.Response(201, json={"created": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="create",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            method="POST",
            path_template="/orders",
            body_arg_names=["sku", "qty"],
        ),
    )
    result = run_async(
        svc.execute(ToolInvocation(tool_name="create", args={"sku": "A1", "qty": 2}))
    )
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert seen["method"] == "POST"
    assert '"sku"' in seen["body"] and '"qty"' in seen["body"]


def test_http_backslash_in_path_arg_rejected(monkeypatch) -> None:
    """A backslash in a path arg is rejected as INVALID_REQUEST (traversal variant)."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/c/{cid}"),
    )
    result = run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"cid": "a\\..\\b"}))
    )
    run_async(svc.aclose())
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_crlf_in_path_arg_rejected(monkeypatch) -> None:
    """A CRLF-bearing path arg (header/log injection) is rejected as INVALID_REQUEST."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/c/{cid}"),
    )
    result = run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"cid": "x\r\nHost: evil"}))
    )
    run_async(svc.aclose())
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_missing_path_arg_is_invalid_request(monkeypatch) -> None:
    """A referenced path placeholder with no supplied arg fails as INVALID_REQUEST."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/c/{cid}"),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t", args={})))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


# ----------------------------------------------------------------- timeout
def test_http_definition_timeout_fires_on_slow_response(monkeypatch) -> None:
    """A handler slower than the definition timeout is cancelled and reported TIMEOUT."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5.0)
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="slow",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/x"),
        timeout_seconds=0.05,
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="slow")))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT


# ----------------------------------------------------------------- retry semantics
def test_retry_exhausted_returns_last_failure() -> None:
    """A handler that always fails transiently exhausts attempts and returns failed."""
    registry = ToolRegistry()
    calls = {"n": 0}

    async def always_timeout(args: dict) -> dict:
        calls["n"] += 1
        raise TimeoutError()

    register_local_tool(
        registry,
        name="dead",
        handler=always_timeout,
        retry_hints={"max_attempts": 3, "base_delay_seconds": 0.0},
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="dead")))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT
    assert calls["n"] == 3  # all attempts consumed


def test_non_retryable_error_is_not_retried() -> None:
    """A non-retryable failure (EXECUTION_ERROR) is returned after a single attempt."""
    registry = ToolRegistry()
    calls = {"n": 0}

    async def boom(args: dict) -> dict:
        calls["n"] += 1
        raise ValueError("hard failure")

    register_local_tool(
        registry,
        name="boom",
        handler=boom,
        retry_hints={"max_attempts": 5, "base_delay_seconds": 0.0},
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="boom")))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.EXECUTION_ERROR
    assert calls["n"] == 1  # not retried despite max_attempts=5


def test_retry_hints_max_retries_alias() -> None:
    """retry_hints={'max_retries': N} means N+1 total attempts."""
    registry = ToolRegistry()
    calls = {"n": 0}

    async def flaky(args: dict) -> dict:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError()
        return {"n": calls["n"]}

    register_local_tool(
        registry,
        name="flaky",
        handler=flaky,
        retry_hints={"max_retries": 2, "base_delay_seconds": 0.0},  # => 3 attempts
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="flaky")))
    assert result.outcome == "success"
    assert calls["n"] == 3


# ----------------------------------------------------------------- redaction in events
def test_http_header_secret_redacted_in_event(monkeypatch) -> None:
    """A header-arg whose name hints a secret is redacted on the TOOL_CALLED event."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    sink = StorageService()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    svc = ToolService(ToolRegistry(), event_sink=sink, http_client=client)
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            header_arg_names=["x_api_token"],
        ),
    )
    run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"x_api_token": "super-secret"}))
    )
    run_async(svc.aclose())
    events = run_async(sink.list_events())
    called = next(e for e in events if e.event_type == EventType.TOOL_CALLED)
    # "token" is a sensitive hint substring -> redacted by default.
    assert called.payload["args"]["x_api_token"] == REDACTED
    assert "super-secret" not in str(called.payload)


def test_failed_call_emits_tool_failed_with_code(monkeypatch) -> None:
    """A failed dispatch emits TOOL_FAILED carrying the normalized error code."""
    sink = StorageService()
    svc = ToolService(ToolRegistry(), event_sink=sink)
    result = run_async(svc.execute(ToolInvocation(tool_name="nope")))
    assert result.error_code == ToolErrorCode.NOT_FOUND
    events = run_async(sink.list_events())
    failed = next(e for e in events if e.event_type == EventType.TOOL_FAILED)
    assert failed.payload["error_code"] == ToolErrorCode.NOT_FOUND.value
    assert failed.payload["tool_outcome"] == "failed"


def test_denied_approval_event_marks_outcome_denied() -> None:
    """An approval-blocked call emits TOOL_FAILED with outcome 'denied'."""
    registry = ToolRegistry()
    register_local_tool(
        registry, name="danger", handler=lambda a: {"ok": True}, requires_approval=True
    )
    sink = StorageService()
    svc = ToolService(registry, event_sink=sink)
    run_async(svc.execute(ToolInvocation(tool_name="danger")))
    events = run_async(sink.list_events())
    failed = next(e for e in events if e.event_type == EventType.TOOL_FAILED)
    assert failed.payload["error_code"] == ToolErrorCode.POLICY_BLOCKED.value
    assert failed.payload["tool_outcome"] == "denied"


# ----------------------------------------------------------------- input validation
def test_required_arg_missing_is_invalid_request_before_dispatch() -> None:
    """A required arg absent from invocation fails INVALID_REQUEST (handler not called)."""
    registry = ToolRegistry()
    called = {"hit": False}

    def handler(args: dict) -> dict:
        called["hit"] = True
        return {"ok": True}

    register_local_tool(
        registry,
        name="needsx",
        handler=handler,
        args_json_schema={
            "type": "object",
            "required": ["x"],
            "properties": {"x": {"type": "string"}},
        },
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="needsx", args={})))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST
    assert called["hit"] is False


def test_additional_properties_false_rejected_on_input() -> None:
    """additionalProperties:false on the input schema rejects unexpected args."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="strict",
        handler=lambda a: {"ok": True},
        args_json_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
        },
    )
    svc = ToolService(registry)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="strict", args={"a": 1, "rogue": 2}))
    )
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


# ----------------------------------------------------------------- post-hook + validation order
def test_post_hook_runs_before_output_validation_and_can_pass() -> None:
    """A post-hook reshaping into a schema-valid value passes output validation."""

    async def reshape(result, definition):
        # Original handler returns {'sum': 7}; reshape to a still-valid value.
        return {"sum": 100}

    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="adder",
        handler=lambda a: {"sum": a.get("a", 0) + a.get("b", 0)},
        output_json_schema={
            "type": "object",
            "properties": {"sum": {"type": "integer"}},
        },
    )
    svc = ToolService(registry, post_execution_hook=reshape)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": 3, "b": 4}))
    )
    assert result.outcome == "success"
    assert result.result == {"sum": 100}


# ----------------------------------------------------------------- registry params
def test_register_local_tool_sets_timeout_and_retry() -> None:
    """register_local_tool surfaces timeout_seconds + retry_hints onto the definition."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="cfg",
        handler=lambda a: {"ok": True},
        timeout_seconds=2.5,
        retry_hints={"max_attempts": 4},
        sensitive_arg_names=["pw"],
    )
    d = registry.get("cfg")
    assert d is not None
    assert d.timeout_seconds == 2.5
    assert d.retry_hints == {"max_attempts": 4}
    assert d.sensitive_arg_names == ["pw"]


def test_register_http_tool_definition_timeout_overrides_config() -> None:
    """Definition-level timeout_seconds takes precedence over http_config.timeout_seconds."""
    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="h",
        http_config=HttpToolConfig(
            base_url_env_var="BASE", path_template="/x", timeout_seconds=9.0
        ),
        timeout_seconds=1.0,
    )
    svc = ToolService(registry)
    d = registry.get("h")
    assert d is not None
    # The service resolves definition > http_config > default.
    assert svc._resolve_timeout(d) == 1.0


# ----------------------------------------------------------------- pydantic-ai toolset (gated)
def test_toolset_binding_raises_clear_error_without_extra() -> None:
    """Without pydantic-ai, the toolset binding raises an actionable OpenSimsError."""
    if _HAS_PYDANTIC_AI:
        pytest.skip("pydantic-ai installed; the absent-extra path is not exercised")
    from opensims.core.errors import OpenSimsError
    from opensims.services.tools.runtime_adapter import ToolServiceToolset

    registry = ToolRegistry()
    register_local_tool(registry, name="x", handler=lambda a: {"ok": True})
    toolset = ToolServiceToolset(ToolService(registry))
    with pytest.raises(OpenSimsError):
        toolset.as_pydantic_ai_toolset()


@pytest.mark.skipif(not _HAS_PYDANTIC_AI, reason="requires the providers extra")
def test_toolset_binding_builds_tools_with_extra() -> (
    None
):  # pragma: no cover - extra only
    """With pydantic-ai installed, the toolset yields one Tool per definition."""
    from opensims.services.tools.runtime_adapter import ToolServiceToolset

    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="adder",
        handler=lambda a: {"sum": a.get("a", 0) + a.get("b", 0)},
        args_json_schema={
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    )
    toolset = ToolServiceToolset(ToolService(registry))
    tools = toolset.as_pydantic_ai_toolset()
    assert len(tools) == 1
