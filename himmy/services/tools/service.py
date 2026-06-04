"""Tools kernel: the dispatcher that runs policy hooks, calls tools, and emits events.

The service enforces the documented tool-execution guarantees end-to-end:

* incoming args are validated against ``args_json_schema`` (and again after a
  pre-hook transforms them);
* ``requires_approval`` is gated before the user pre-hook (``POLICY_BLOCKED``);
* a per-tool timeout bounds BOTH the local and HTTP paths (``TIMEOUT``);
* retryable failures are retried with backoff per the tool's ``retry_hints``;
* the HTTP connector is SSRF/path-traversal hardened (encoded path args, host
  pinning, method allow-list, no cross-host redirects, a shared pooled client);
* secrets are redacted from emitted events and never echoed in error messages;
* output is validated AFTER the post-execution hook so a reshaping hook cannot
  bypass the schema check.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import os
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from himmy.core.events import EventType, RunEvent
from himmy.services.tools.models import (
    HttpAuthMode,
    ToolBackendKind,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionResult,
    ToolInvocation,
    ToolPolicyDecision,
)
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import (
    ALLOWED_HTTP_METHODS,
    ToolSecurityError,
    assemble_url,
    build_safe_path,
    redact_mapping,
)
from himmy.services.tools.validation import validate_against_schema

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.core.events import EventSink
    from himmy.services.inference.models import BoundTool

PreExecutionHook = Callable[
    [ToolInvocation, ToolDefinition], Awaitable[ToolPolicyDecision]
]
PostExecutionHook = Callable[[ToolExecutionResult, ToolDefinition], Awaitable[Any]]

#: Error codes worth retrying inside the tool service (transient failures).
RETRYABLE_TOOL_CODES: frozenset[ToolErrorCode] = frozenset(
    {
        ToolErrorCode.RATE_LIMITED,
        ToolErrorCode.TIMEOUT,
        ToolErrorCode.PROVIDER_UNAVAILABLE,
    }
)

#: Default per-tool execution timeout when neither the definition nor the HTTP
#: config specifies one.
DEFAULT_TOOL_TIMEOUT_SECONDS: float = 30.0


def _validate_against_schema(value: Any, schema: dict[str, Any]) -> str | None:
    """Backwards-compatible alias delegating to the validation module.

    Kept so existing imports keep working; new code should call
    :func:`himmy.services.tools.validation.validate_against_schema`.
    """
    return validate_against_schema(value, schema)


class ToolService:
    """Dispatches tool calls with policy hooks, output validation, and events.

    Two backends are supported: ``LOCAL`` (a Python callable, sync or async) and
    ``HTTP`` (a declarative REST connector executed via a shared, lazily-built
    ``httpx`` client). Every call emits ``TOOL_CALLED`` then either
    ``TOOL_COMPLETED`` or ``TOOL_FAILED`` through the optional event sink.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        pre_execution_hook: PreExecutionHook | None = None,
        post_execution_hook: PostExecutionHook | None = None,
        event_sink: EventSink | None = None,
        default_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        http_client: Any = None,
        http_max_connections: int = 100,
        http_max_keepalive_connections: int = 20,
    ) -> None:
        self._registry = registry
        self._pre_hook = pre_execution_hook
        self._post_hook = post_execution_hook
        self.event_sink = event_sink
        self._default_timeout_seconds = default_timeout_seconds
        # Shared httpx client, built lazily on first HTTP call (or injected for tests).
        self._http_client = http_client
        self._http_client_owned = http_client is None
        self._http_max_connections = http_max_connections
        self._http_max_keepalive = http_max_keepalive_connections
        self._http_lock = asyncio.Lock()

    @property
    def registry(self) -> ToolRegistry:
        """The underlying tool catalog this service dispatches against."""
        return self._registry

    async def _emit(self, event: RunEvent) -> None:
        """Best-effort emit; a broken sink must never fail a tool call."""
        if self.event_sink is None:
            return
        try:
            await self.event_sink.append_event(event)
        except Exception:  # pragma: no cover - defensive
            pass

    async def _get_http_client(self) -> Any:
        """Return the shared httpx client, building it lazily (thread-safe)."""
        if self._http_client is not None:
            return self._http_client
        async with self._http_lock:
            if self._http_client is None:  # re-check under the lock
                import httpx

                limits = httpx.Limits(
                    max_connections=self._http_max_connections,
                    max_keepalive_connections=self._http_max_keepalive,
                )
                self._http_client = httpx.AsyncClient(
                    follow_redirects=False, limits=limits
                )
                self._http_client_owned = True
        return self._http_client

    async def aclose(self) -> None:
        """Close the shared httpx client if this service owns it (idempotent)."""
        client = self._http_client
        if client is not None and self._http_client_owned:
            self._http_client = None
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - defensive
                pass

    # Alias for callers that expect a sync-style name.
    close = aclose

    async def execute(self, invocation: ToolInvocation) -> ToolExecutionResult:
        """Execute one tool invocation end-to-end and return its result."""
        start = time.perf_counter()
        definition = self._registry.get(invocation.tool_name)

        # Emit TOOL_CALLED with secrets redacted (sensitive args never leak).
        sensitive_names = (
            definition.sensitive_arg_names if definition is not None else []
        )
        await self._emit(
            RunEvent(
                event_type=EventType.TOOL_CALLED,
                tool_call_id=invocation.tool_call_id,
                payload={
                    "tool_name": invocation.tool_name,
                    "args": redact_mapping(invocation.args, extra_keys=sensitive_names),
                },
            )
        )

        if definition is None:
            return await self._fail(
                invocation,
                start,
                ToolErrorCode.NOT_FOUND,
                f"tool {invocation.tool_name!r} is not registered",
            )

        # --- approval gate (before the user pre-hook) ----------------------
        if definition.requires_approval and not invocation.metadata.get("approved"):
            return await self._fail(
                invocation,
                start,
                ToolErrorCode.POLICY_BLOCKED,
                f"tool {definition.name!r} requires approval (no approval granted)",
                outcome="denied",
            )

        # --- input validation against args_json_schema ---------------------
        args = dict(invocation.args)
        schema = definition.args_json_schema
        if schema and schema.get("type") == "object":
            err = validate_against_schema(args, schema)
            if err is not None:
                return await self._fail(
                    invocation,
                    start,
                    ToolErrorCode.INVALID_REQUEST,
                    f"args failed schema validation: {err}",
                )

        # --- pre-execution policy hook -------------------------------------
        if self._pre_hook is not None:
            try:
                decision = await self._pre_hook(invocation, definition)
            except Exception as exc:  # pragma: no cover - defensive
                return await self._fail(
                    invocation, start, ToolErrorCode.EXECUTION_ERROR, str(exc)
                )
            if not decision.allow:
                return await self._fail(
                    invocation,
                    start,
                    ToolErrorCode.POLICY_BLOCKED,
                    decision.reason or "blocked by policy",
                    outcome="denied",
                )
            if decision.transformed_args is not None:
                args = dict(decision.transformed_args)
                # Re-validate transformed args so a pre-hook cannot smuggle in
                # an invalid payload.
                if schema and schema.get("type") == "object":
                    err = validate_against_schema(args, schema)
                    if err is not None:
                        return await self._fail(
                            invocation,
                            start,
                            ToolErrorCode.INVALID_REQUEST,
                            f"transformed args failed schema validation: {err}",
                        )

        # --- dispatch (with per-tool timeout + bounded retry) --------------
        timeout = self._resolve_timeout(definition)
        retry = _RetryPolicy.from_hints(definition.retry_hints)
        last_error: _ToolDispatchError | None = None
        raw: Any = None
        dispatched = False
        for attempt in range(retry.max_attempts):
            try:
                raw = await asyncio.wait_for(
                    self._dispatch(definition, args), timeout=timeout
                )
                dispatched = True
                break
            except TimeoutError:
                last_error = _ToolDispatchError(
                    ToolErrorCode.TIMEOUT,
                    f"tool {definition.name!r} exceeded {timeout:.1f}s timeout",
                )
            except _ToolDispatchError as exc:
                last_error = exc
            except Exception as exc:  # pragma: no cover - defensive
                last_error = _ToolDispatchError(ToolErrorCode.EXECUTION_ERROR, str(exc))
            # Retry only on transient codes with attempts remaining.
            if (
                last_error is not None
                and last_error.code in RETRYABLE_TOOL_CODES
                and attempt < retry.max_attempts - 1
            ):
                await asyncio.sleep(retry.delay_for(attempt))
                continue
            break

        if not dispatched:
            assert last_error is not None
            return await self._fail(
                invocation, start, last_error.code, last_error.message
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        result = ToolExecutionResult(
            tool_call_id=invocation.tool_call_id,
            tool_name=invocation.tool_name,
            outcome="success",
            result=raw,
            latency_ms=latency_ms,
        )

        # --- post-execution hook (may transform the result) ---------------
        if self._post_hook is not None:
            try:
                transformed = await self._post_hook(result, definition)
                if transformed is not None:
                    result.result = transformed
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                return await self._fail(
                    invocation,
                    start,
                    ToolErrorCode.EXECUTION_ERROR,
                    f"post-execution hook raised: {exc}",
                )

        # --- output validation (AFTER the post-hook) ----------------------
        if definition.output_json_schema is not None:
            err = validate_against_schema(result.result, definition.output_json_schema)
            if err is not None:
                return await self._fail(
                    invocation,
                    start,
                    ToolErrorCode.OUTPUT_VALIDATION,
                    f"output failed schema validation: {err}",
                )

        result.latency_ms = (time.perf_counter() - start) * 1000.0
        await self._emit(
            RunEvent(
                event_type=EventType.TOOL_COMPLETED,
                tool_call_id=invocation.tool_call_id,
                latency_ms=result.latency_ms,
                payload={
                    "tool_name": invocation.tool_name,
                    "tool_outcome": "success",
                },
            )
        )
        return result

    def _resolve_timeout(self, definition: ToolDefinition) -> float:
        """Resolve the effective per-tool timeout (def > http_config > default)."""
        if definition.timeout_seconds:
            return definition.timeout_seconds
        if (
            definition.http_config is not None
            and definition.http_config.timeout_seconds
        ):
            return definition.http_config.timeout_seconds
        return self._default_timeout_seconds

    async def _dispatch(self, definition: ToolDefinition, args: dict[str, Any]) -> Any:
        """Route to the LOCAL or HTTP backend."""
        if definition.kind is ToolBackendKind.LOCAL:
            return await self._dispatch_local(definition, args)
        return await self._dispatch_http(definition, args)

    async def _dispatch_local(
        self, definition: ToolDefinition, args: dict[str, Any]
    ) -> Any:
        """Invoke a local handler, awaiting it when it is a coroutine function."""
        handler = self._registry.handler_for(definition.name)
        if handler is None:
            raise _ToolDispatchError(
                ToolErrorCode.EXECUTION_ERROR,
                f"no local handler registered for {definition.name!r}",
            )
        outcome = handler(args)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return outcome

    async def _dispatch_http(
        self, definition: ToolDefinition, args: dict[str, Any]
    ) -> Any:
        """Execute a declarative HTTP connector via the shared httpx client."""
        cfg = definition.http_config
        if cfg is None:
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"HTTP tool {definition.name!r} has no http_config",
            )

        method = cfg.method.upper()
        if method not in ALLOWED_HTTP_METHODS:
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"HTTP method {method!r} is not allowed",
            )

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a core dep
            raise _ToolDispatchError(
                ToolErrorCode.PROVIDER_UNAVAILABLE,
                f"httpx is required for HTTP tools: {exc}",
            ) from exc

        base_url = os.environ.get(cfg.base_url_env_var)
        if not base_url:
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"env var {cfg.base_url_env_var!r} is not set",
            )

        # Resolve & encode path placeholders, then pin the final host.
        try:
            path = build_safe_path(cfg.path_template, args)
        except KeyError as exc:
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"missing path argument {exc.args[0]!r}",
            ) from exc
        except ToolSecurityError as exc:
            raise _ToolDispatchError(ToolErrorCode.INVALID_REQUEST, str(exc)) from exc

        try:
            url = assemble_url(base_url, path)
        except ToolSecurityError as exc:
            raise _ToolDispatchError(ToolErrorCode.INVALID_REQUEST, str(exc)) from exc

        params = {k: args[k] for k in cfg.query_arg_names if k in args}
        body = {k: args[k] for k in cfg.body_arg_names if k in args}
        headers = {k: str(args[k]) for k in cfg.header_arg_names if k in args}
        headers.update(self._build_auth_headers(cfg.auth))

        timeout = self._resolve_timeout(definition)
        client = await self._get_http_client()
        try:
            response = await client.request(
                method,
                url,
                params=params or None,
                json=body or None,
                headers=headers or None,
                timeout=timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            # Never echo header/secret values; report the URL path only.
            raise _ToolDispatchError(
                ToolErrorCode.TIMEOUT, f"request timed out: {type(exc).__name__}"
            ) from exc
        except httpx.HTTPError as exc:
            raise _ToolDispatchError(
                ToolErrorCode.PROVIDER_UNAVAILABLE,
                f"request failed: {type(exc).__name__}",
            ) from exc

        if response.status_code == 429:
            raise _ToolDispatchError(
                ToolErrorCode.RATE_LIMITED, f"upstream returned {response.status_code}"
            )
        if response.status_code >= 500:
            raise _ToolDispatchError(
                ToolErrorCode.PROVIDER_UNAVAILABLE,
                f"upstream returned {response.status_code}",
            )
        if response.status_code >= 400:
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"upstream returned {response.status_code}",
            )
        try:
            return response.json()
        except Exception:
            return {"text": response.text}

    @staticmethod
    def _build_auth_headers(auth: Any) -> dict[str, str]:
        """Resolve env-backed auth into request headers (empty when NONE/unset).

        ``BASIC`` reads ``user:pass`` from the env var and base64-encodes it here;
        ``PREENCODED_BASIC`` passes an already-encoded credential through verbatim.
        """
        if auth.mode is HttpAuthMode.NONE or not auth.env_var:
            return {}
        secret = os.environ.get(auth.env_var)
        if not secret:
            return {}
        if auth.mode is HttpAuthMode.BEARER:
            return {"Authorization": f"Bearer {secret}"}
        if auth.mode is HttpAuthMode.BASIC:
            encoded = base64.b64encode(secret.encode("utf-8")).decode("ascii")
            return {"Authorization": f"Basic {encoded}"}
        if auth.mode is HttpAuthMode.PREENCODED_BASIC:
            return {"Authorization": f"Basic {secret}"}
        if auth.mode is HttpAuthMode.HEADER:
            header_name = auth.header_name or "Authorization"
            return {header_name: secret}
        return {}

    async def _fail(
        self,
        invocation: ToolInvocation,
        start: float,
        code: ToolErrorCode,
        message: str,
        *,
        outcome: str = "failed",
    ) -> ToolExecutionResult:
        """Build, emit, and return a failed/denied ``ToolExecutionResult``."""
        latency_ms = (time.perf_counter() - start) * 1000.0
        result = ToolExecutionResult(
            tool_call_id=invocation.tool_call_id,
            tool_name=invocation.tool_name,
            outcome=outcome,
            error_code=code,
            error_message=message,
            latency_ms=latency_ms,
        )
        await self._emit(
            RunEvent(
                event_type=EventType.TOOL_FAILED,
                tool_call_id=invocation.tool_call_id,
                latency_ms=latency_ms,
                error=message,
                payload={
                    "tool_name": invocation.tool_name,
                    "error_code": code.value,
                    "tool_outcome": outcome,
                },
            )
        )
        return result

    def bound_tools(self, names: list[str] | None = None) -> list[BoundTool]:
        """Bind selected tools as inference ``BoundTool``s for the offline path.

        Each returned ``BoundTool`` carries the tool's schemas and a handler that
        routes back through :meth:`execute`, returning a ``ToolReturnRecord``. The
        inference kernel is imported lazily so this module imports without it.
        """
        # Imported lazily: the inference kernel may be built in parallel and is
        # not on the core import path for this kernel.
        from himmy.services.inference.models import BoundTool, ToolReturnRecord

        selected = (
            self._registry.list()
            if names is None
            else [d for n in names if (d := self._registry.get(n)) is not None]
        )

        bound: list[BoundTool] = []
        for definition in selected:

            def _make_handler(
                tool_name: str,
            ) -> Callable[[dict[str, Any]], Awaitable[Any]]:
                async def _handler(args: dict[str, Any]) -> ToolReturnRecord:
                    invocation = ToolInvocation(tool_name=tool_name, args=args)
                    res = await self.execute(invocation)
                    return ToolReturnRecord(
                        tool_call_id=res.tool_call_id,
                        tool_name=res.tool_name,
                        content=res.result,
                        outcome=res.outcome,
                        metadata={
                            "error_code": res.error_code.value
                            if res.error_code
                            else None,
                            "error_message": res.error_message,
                            "latency_ms": res.latency_ms,
                        },
                    )

                return _handler

            bound.append(
                BoundTool(
                    name=definition.name,
                    description=definition.description,
                    args_json_schema=definition.args_json_schema,
                    output_json_schema=definition.output_json_schema,
                    handler=_make_handler(definition.name),
                )
            )
        return bound


class _RetryPolicy:
    """A bounded retry policy derived from a tool's ``retry_hints`` dict."""

    def __init__(
        self,
        *,
        max_attempts: int = 1,
        base_delay_seconds: float = 0.0,
        max_delay_seconds: float = 5.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.base_delay_seconds = max(0.0, base_delay_seconds)
        self.max_delay_seconds = max(0.0, max_delay_seconds)
        self.backoff_multiplier = max(1.0, backoff_multiplier)

    @classmethod
    def from_hints(cls, hints: dict[str, Any] | None) -> _RetryPolicy:
        """Build a policy from a tool's ``retry_hints`` (all keys optional).

        Recognized keys: ``max_attempts`` (or ``max_retries`` = attempts-1),
        ``base_delay_seconds``/``backoff_seconds``, ``max_delay_seconds``,
        ``backoff_multiplier``.
        """
        if not hints:
            return cls()

        def _num(key: str, default: float) -> float:
            value = hints.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        if "max_attempts" in hints:
            max_attempts = int(_num("max_attempts", 1))
        elif "max_retries" in hints:
            max_attempts = int(_num("max_retries", 0)) + 1
        else:
            max_attempts = 1
        base_delay = _num("base_delay_seconds", _num("backoff_seconds", 0.0))
        return cls(
            max_attempts=max_attempts,
            base_delay_seconds=base_delay,
            max_delay_seconds=_num("max_delay_seconds", 5.0),
            backoff_multiplier=_num("backoff_multiplier", 2.0),
        )

    def delay_for(self, attempt: int) -> float:
        """Exponential backoff delay (seconds) for a 0-based attempt index."""
        if self.base_delay_seconds <= 0.0:
            return 0.0
        delay = self.base_delay_seconds * (self.backoff_multiplier**attempt)
        return min(delay, self.max_delay_seconds)


class _ToolDispatchError(Exception):
    """Internal control-flow error carrying a normalized tool error code."""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
