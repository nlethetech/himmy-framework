"""Tests for the tools kernel: local dispatch, policy hooks, validation, binding."""

from __future__ import annotations

import asyncio

import httpx

from himmy.core.events import EventType
from himmy.services.storage.service import StorageService
from himmy.services.tools import (
    HttpAuthConfig,
    HttpAuthMode,
    HttpToolConfig,
    ToolErrorCode,
    ToolInvocation,
    ToolPolicyDecision,
    ToolRegistry,
    ToolService,
    register_http_tool,
    register_local_tool,
)
from himmy.services.tools.security import REDACTED
from tests.conftest import run_async


def _registry_with_adder() -> ToolRegistry:
    registry = ToolRegistry()

    def add(args: dict) -> dict:
        return {"sum": args.get("a", 0) + args.get("b", 0)}

    register_local_tool(
        registry,
        name="adder",
        handler=add,
        description="adds two numbers",
        args_json_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        output_json_schema={
            "type": "object",
            "properties": {"sum": {"type": "integer"}},
        },
    )
    return registry


def test_local_tool_executes_and_validates_output() -> None:
    """A local tool runs and its output passes schema validation."""
    svc = ToolService(_registry_with_adder())
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": 2, "b": 3}))
    )
    assert result.outcome == "success"
    assert result.result == {"sum": 5}
    assert result.error_code is None


def test_async_local_tool_is_awaited() -> None:
    """A coroutine handler is awaited transparently."""
    registry = ToolRegistry()

    async def greet(args: dict) -> dict:
        return {"msg": f"hi {args.get('name', '')}"}

    register_local_tool(registry, name="greet", handler=greet)
    svc = ToolService(registry)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="greet", args={"name": "Bob"}))
    )
    assert result.outcome == "success"
    assert result.result == {"msg": "hi Bob"}


def test_unknown_tool_returns_not_found() -> None:
    """Executing an unregistered tool fails with NOT_FOUND."""
    svc = ToolService(ToolRegistry())
    result = run_async(svc.execute(ToolInvocation(tool_name="missing")))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.NOT_FOUND


def test_pre_hook_can_deny() -> None:
    """A pre-execution hook that denies yields outcome 'denied' / POLICY_BLOCKED."""

    async def deny_hook(invocation, definition) -> ToolPolicyDecision:
        return ToolPolicyDecision(allow=False, reason="not allowed")

    svc = ToolService(_registry_with_adder(), pre_execution_hook=deny_hook)
    result = run_async(svc.execute(ToolInvocation(tool_name="adder", args={})))
    assert result.outcome == "denied"
    assert result.error_code == ToolErrorCode.POLICY_BLOCKED
    assert result.error_message == "not allowed"


def test_pre_hook_can_transform_args() -> None:
    """A pre-hook may rewrite args before dispatch."""

    async def transform_hook(invocation, definition) -> ToolPolicyDecision:
        return ToolPolicyDecision(allow=True, transformed_args={"a": 10, "b": 10})

    svc = ToolService(_registry_with_adder(), pre_execution_hook=transform_hook)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": 1, "b": 1}))
    )
    assert result.result == {"sum": 20}


def test_output_validation_failure() -> None:
    """A handler returning the wrong type fails OUTPUT_VALIDATION."""
    registry = ToolRegistry()

    def bad(args: dict) -> dict:
        return {"sum": "not-an-int"}

    register_local_tool(
        registry,
        name="bad",
        handler=bad,
        output_json_schema={
            "type": "object",
            "properties": {"sum": {"type": "integer"}},
        },
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="bad")))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.OUTPUT_VALIDATION


def test_events_emitted_to_sink() -> None:
    """A successful call emits TOOL_CALLED then TOOL_COMPLETED."""
    sink = StorageService()
    svc = ToolService(_registry_with_adder(), event_sink=sink)
    run_async(svc.execute(ToolInvocation(tool_name="adder", args={"a": 1, "b": 1})))
    events = run_async(sink.list_events())
    types = {e.event_type for e in events}
    assert EventType.TOOL_CALLED in types
    assert EventType.TOOL_COMPLETED in types


def test_bound_tools_round_trip_through_execute() -> None:
    """bound_tools yields BoundTools whose handler routes back through execute."""
    svc = ToolService(_registry_with_adder())
    bound = svc.bound_tools(["adder"])
    assert len(bound) == 1
    tool = bound[0]
    assert tool.name == "adder"
    ret = run_async(tool.handler({"a": 4, "b": 6}))
    assert ret.tool_name == "adder"
    assert ret.content == {"sum": 10}
    assert ret.outcome == "success"


# ----------------------------------------------------------------- input validation
def test_invalid_input_args_rejected() -> None:
    """Args violating args_json_schema fail with INVALID_REQUEST before dispatch."""
    svc = ToolService(_registry_with_adder())
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": "not-int", "b": 1}))
    )
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_transformed_args_revalidated() -> None:
    """A pre-hook that produces invalid args is caught by re-validation."""

    async def bad_transform(invocation, definition) -> ToolPolicyDecision:
        return ToolPolicyDecision(allow=True, transformed_args={"a": "x", "b": "y"})

    svc = ToolService(_registry_with_adder(), pre_execution_hook=bad_transform)
    result = run_async(svc.execute(ToolInvocation(tool_name="adder", args={})))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


# ----------------------------------------------------------------- approval gating
def test_requires_approval_blocks_without_approval() -> None:
    """A tool needing approval is POLICY_BLOCKED unless metadata.approved is set."""
    registry = ToolRegistry()
    register_local_tool(
        registry, name="danger", handler=lambda a: {"ok": True}, requires_approval=True
    )
    svc = ToolService(registry)
    blocked = run_async(svc.execute(ToolInvocation(tool_name="danger")))
    assert blocked.outcome == "denied"
    assert blocked.error_code == ToolErrorCode.POLICY_BLOCKED

    approved = run_async(
        svc.execute(ToolInvocation(tool_name="danger", metadata={"approved": True}))
    )
    assert approved.outcome == "success"


# ----------------------------------------------------------------- timeout
def test_timeout_enforced_on_local_path() -> None:
    """A slow local handler is cancelled and reported as TIMEOUT."""
    registry = ToolRegistry()

    async def slow(args: dict) -> dict:
        await asyncio.sleep(5.0)
        return {"done": True}

    register_local_tool(registry, name="slow", handler=slow, timeout_seconds=0.05)
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="slow")))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT


# ----------------------------------------------------------------- retry
def test_retry_succeeds_after_transient_failures() -> None:
    """A handler that fails transiently then succeeds is retried per retry_hints."""
    registry = ToolRegistry()
    calls = {"n": 0}

    async def flaky(args: dict) -> dict:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError()  # mapped to TIMEOUT (retryable)
        return {"attempt": calls["n"]}

    register_local_tool(
        registry,
        name="flaky",
        handler=flaky,
        retry_hints={"max_attempts": 3, "base_delay_seconds": 0.0},
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="flaky")))
    assert result.outcome == "success"
    assert calls["n"] == 3


# ----------------------------------------------------------------- post-hook
def test_post_hook_transforms_result() -> None:
    """A post-hook return value replaces the result."""

    async def redact(result, definition):
        return {"sum": -1}

    svc = ToolService(_registry_with_adder(), post_execution_hook=redact)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": 1, "b": 2}))
    )
    assert result.result == {"sum": -1}


def test_post_hook_output_revalidated() -> None:
    """A post-hook that breaks the output schema fails OUTPUT_VALIDATION (not success)."""

    async def break_schema(result, definition):
        return {"sum": "not-an-int"}

    svc = ToolService(_registry_with_adder(), post_execution_hook=break_schema)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": 1, "b": 2}))
    )
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.OUTPUT_VALIDATION


def test_post_hook_exception_surfaced() -> None:
    """A raising post-hook surfaces as an EXECUTION_ERROR, not a silent swallow."""

    async def boom(result, definition):
        raise ValueError("redaction broke")

    svc = ToolService(_registry_with_adder(), post_execution_hook=boom)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": 1, "b": 2}))
    )
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.EXECUTION_ERROR


# ----------------------------------------------------------------- secret redaction
def test_sensitive_args_redacted_in_events() -> None:
    """Sensitive arg values never appear verbatim on the TOOL_CALLED event."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="login",
        handler=lambda a: {"ok": True},
        sensitive_arg_names=["password"],
        args_json_schema={
            "type": "object",
            "properties": {
                "user": {"type": "string"},
                "password": {"type": "string"},
            },
        },
    )
    sink = StorageService()
    svc = ToolService(registry, event_sink=sink)
    run_async(
        svc.execute(
            ToolInvocation(
                tool_name="login", args={"user": "bob", "password": "hunter2"}
            )
        )
    )
    events = run_async(sink.list_events())
    called = next(e for e in events if e.event_type == EventType.TOOL_CALLED)
    assert called.payload["args"]["password"] == REDACTED
    assert called.payload["args"]["user"] == "bob"
    # The raw secret must not appear anywhere in the emitted payload.
    assert "hunter2" not in str(called.payload)


# ----------------------------------------------------------------- HTTP connector
def _http_service(transport: httpx.MockTransport) -> ToolService:
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    return ToolService(ToolRegistry(), http_client=client)


def test_http_tool_get_assembles_path_and_query(monkeypatch) -> None:
    """An HTTP GET resolves the path placeholder + query args and returns JSON."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"id": "123", "status": "ok"})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="get_order",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL",
            method="GET",
            path_template="/customers/{customer_id}/orders",
            query_arg_names=["limit"],
        ),
    )
    result = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="get_order", args={"customer_id": "123", "limit": 5}
            )
        )
    )
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert result.result == {"id": "123", "status": "ok"}
    assert seen["url"] == "https://api.example.com/customers/123/orders?limit=5"


def test_http_path_traversal_is_rejected(monkeypatch) -> None:
    """A traversal payload in a path arg is rejected as INVALID_REQUEST."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="get_order",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL",
            path_template="/customers/{customer_id}/orders",
        ),
    )
    result = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="get_order",
                args={"customer_id": "../../admin/secrets"},
            )
        )
    )
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_query_injection_in_path_is_encoded(monkeypatch) -> None:
    """A '?'-bearing path arg is rejected (cannot inject a query string)."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="get_order",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL",
            path_template="/customers/{customer_id}",
        ),
    )
    result = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="get_order", args={"customer_id": "123?admin=true"}
            )
        )
    )
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_special_chars_in_path_are_url_encoded(monkeypatch) -> None:
    """Spaces and unicode in a path arg are percent-encoded, not rejected."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="lookup",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL", path_template="/items/{q}"
        ),
    )
    result = run_async(
        svc.execute(ToolInvocation(tool_name="lookup", args={"q": "a b&c"}))
    )
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert "a%20b%26c" in seen["url"]


def test_http_error_code_mapping(monkeypatch) -> None:
    """429/5xx/4xx upstream statuses map to the documented tool error codes."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")

    def make_svc(status: int) -> ToolService:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"err": status})

        svc = _http_service(httpx.MockTransport(handler))
        register_http_tool(
            svc.registry,
            name="t",
            http_config=HttpToolConfig(
                base_url_env_var="ORDERS_BASE_URL", path_template="/x"
            ),
        )
        return svc

    for status, code in (
        (429, ToolErrorCode.RATE_LIMITED),
        (503, ToolErrorCode.PROVIDER_UNAVAILABLE),
        (404, ToolErrorCode.INVALID_REQUEST),
    ):
        svc = make_svc(status)
        result = run_async(svc.execute(ToolInvocation(tool_name="t")))
        run_async(svc.aclose())
        assert result.error_code == code, status


def test_http_non_json_falls_back_to_text(monkeypatch) -> None:
    """A non-JSON 200 response falls back to a {'text': ...} payload."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="plain body")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL", path_template="/x"
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert result.result == {"text": "plain body"}


def test_http_missing_base_url_env(monkeypatch) -> None:
    """An unset base-url env var fails as INVALID_REQUEST."""
    monkeypatch.delenv("MISSING_BASE_URL", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="MISSING_BASE_URL", path_template="/x"
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_disallowed_method_rejected(monkeypatch) -> None:
    """A non-allow-listed HTTP method is rejected before any request is sent."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL", method="TRACE", path_template="/x"
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_basic_auth_is_base64_encoded(monkeypatch) -> None:
    """BASIC mode base64-encodes raw user:pass from the env var."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("ORDERS_CREDS", "bob:hunter2")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL",
            path_template="/x",
            auth=HttpAuthConfig(mode=HttpAuthMode.BASIC, env_var="ORDERS_CREDS"),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    # base64("bob:hunter2") == "Ym9iOmh1bnRlcjI="
    assert seen["auth"] == "Basic Ym9iOmh1bnRlcjI="


def test_http_bearer_auth_header(monkeypatch) -> None:
    """BEARER mode sends Authorization: Bearer <secret>."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("ORDERS_TOKEN", "abc123")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL",
            path_template="/x",
            auth=HttpAuthConfig(mode=HttpAuthMode.BEARER, env_var="ORDERS_TOKEN"),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["auth"] == "Bearer abc123"


def test_http_timeout_maps_to_timeout_code(monkeypatch) -> None:
    """An httpx timeout maps to the TIMEOUT error code."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL", path_template="/x"
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert result.error_code == ToolErrorCode.TIMEOUT


def test_http_retry_on_transient_5xx(monkeypatch) -> None:
    """A 503 then 200 succeeds when retry_hints permit a second attempt."""
    monkeypatch.setenv("ORDERS_BASE_URL", "https://api.example.com")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"err": "down"})
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="ORDERS_BASE_URL", path_template="/x"
        ),
        retry_hints={"max_attempts": 2, "base_delay_seconds": 0.0},
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert calls["n"] == 2


def test_shared_client_is_reused_and_closed() -> None:
    """The lazily-built httpx client is shared across calls and closable."""
    svc = ToolService(ToolRegistry())
    client1 = run_async(svc._get_http_client())
    client2 = run_async(svc._get_http_client())
    assert client1 is client2
    run_async(svc.aclose())
    # A new client is built after close.
    client3 = run_async(svc._get_http_client())
    assert client3 is not client1
    run_async(svc.aclose())
