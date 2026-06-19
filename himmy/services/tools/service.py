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
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from himmy.config.secrets import get_secret
from himmy.core.events import EventType, RunEvent
from himmy.services.tools.models import (
    HttpAuthMode,
    HttpPaginationMode,
    HttpToolConfig,
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

logger = logging.getLogger("himmy.services.tools")

#: Methods that are safe to repeat (read-only): a blind retry of these can't mutate
#: state. Anything else is non-idempotent — never retried unless an idempotency key is
#: supplied so the upstream can dedupe it.
_SAFE_HTTP_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.core.events import EventSink
    from himmy.services.inference.models import BoundTool, ToolExecutor

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


@runtime_checkable
class ToolIdempotencyStore(Protocol):
    """Records completed tool executions by idempotency key (exactly-once seam).

    The dedup point for resume-style paths (HITL approve/resume today, any future
    replay path): when :meth:`ToolService.execute` is given a store and the
    invocation carries ``metadata["idempotency_key"]``, a previously recorded
    result is replayed instead of executing the tool a second time, and new
    results are recorded via :meth:`put` — which should persist durably BEFORE
    returning, so a crash right after a state-mutating tool ran cannot lose the
    record and re-execute on retry.
    """

    def get(self, key: str) -> ToolExecutionResult | None:
        """Return the recorded result for ``key``, or None if never executed."""
        ...

    def put(self, key: str, result: ToolExecutionResult) -> None:
        """Durably record ``result`` under ``key``."""
        ...


@runtime_checkable
class ToolReputationLike(Protocol):
    """The minimal sync reputation surface ``bound_tools`` reorders against (P1).

    Anything exposing a synchronous ``score_for(name) -> float`` (a cached snapshot, so
    the per-turn binding pays no I/O) and ``is_unreliable(name) -> bool`` satisfies the
    reorder hook. ``himmy.services.learning.ToolReputationProvider`` conforms structurally
    without an import dependency (keeping the tools kernel independent of learning).
    """

    def score_for(self, tool_name: str) -> float:  # pragma: no cover - structural typing
        ...

    def is_unreliable(
        self, tool_name: str
    ) -> bool:  # pragma: no cover - structural typing
        ...


def _validate_against_schema(value: Any, schema: dict[str, Any]) -> str | None:
    """Backwards-compatible alias delegating to the validation module.

    Kept so existing imports keep working; new code should call
    :func:`himmy.services.tools.validation.validate_against_schema`.
    """
    return validate_against_schema(value, schema)


def _dig(payload: Any, dotted_path: str) -> Any:
    """Walk a dotted path into a nested JSON ``payload`` (empty path → payload itself).

    Used by HTTP pagination to pull the records list / next-cursor out of a page body.
    Returns ``None`` when any segment is missing or the shape doesn't match, so a
    malformed/hostile page can't crash the connector — it simply yields no records.
    """
    if not dotted_path:
        return payload
    current = payload
    for segment in dotted_path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return None
    return current


def _next_link(link_header: str) -> str | None:
    """Parse an RFC 5988 ``Link`` header and return the ``rel="next"`` URL, if any."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        target = segments[0].strip().strip("<>").strip()
        if not target:
            continue
        for attr in segments[1:]:
            key, _, value = attr.strip().partition("=")
            if key.strip().lower() == "rel" and value.strip().strip('"') == "next":
                return target
    return None


def _link_target_allowed(
    link: str, origin_url: str, allow_hosts: tuple[str, ...]
) -> bool:
    """Default-deny gate for a pagination ``Link`` rel="next" target.

    A cross-host ``Link`` is a credential-exfiltration vector: the bearer/auth header
    is reused on every page, so a hostile upstream returning
    ``Link: <https://evil.example/x>; rel="next"`` would otherwise pull the secret
    off-host. We follow the target ONLY when it is same-origin with the original
    request (identical scheme + host + port) or — when an explicit egress allow-list
    is configured — its host matches that list. With the default empty allow-list this
    collapses to same-origin only, so the secret never leaves the API it was minted for.
    The downstream SSRF guard still runs on whatever we allow through here.
    """
    target = urlsplit(link)
    if not target.scheme or not target.netloc:
        return False  # relative / malformed → not a host we can vet; refuse
    if allow_hosts:
        from himmy.toolkit._net import _host_allowed

        host = target.hostname
        return host is not None and _host_allowed(host, allow_hosts)
    origin = urlsplit(origin_url)
    return (
        target.scheme.lower() == origin.scheme.lower()
        and (target.hostname or "").lower() == (origin.hostname or "").lower()
        and target.port == origin.port
    )


def _scalar_types(prop: dict[str, Any]) -> set[str]:
    """The set of JSON-schema scalar type names a property declares (if any).

    Reads ``type`` (a single string or a list union). Only the scalar targets we can
    losslessly coerce a string into — ``integer``/``number``/``boolean`` — are
    returned; everything else is dropped so coercion never fires for them.
    """
    declared = prop.get("type")
    names: set[str]
    if isinstance(declared, str):
        names = {declared}
    elif isinstance(declared, list):
        names = {t for t in declared if isinstance(t, str)}
    else:
        return set()
    return names & {"integer", "number", "boolean"}


def _coerce_scalar(value: str, targets: set[str]) -> tuple[bool, Any]:
    """Losslessly coerce a stringified scalar to a declared scalar type.

    Returns ``(True, coerced)`` only when ``value`` round-trips EXACTLY to a target
    the schema declares (so no information is lost and an ambiguous/lossy value is
    left untouched). Integer is tried before number so ``"5"`` becomes ``5`` (int)
    when ``integer`` is allowed, and ``"3.14"`` becomes a float only under ``number``.
    Booleans accept just the canonical ``"true"``/``"false"`` tokens. Returns
    ``(False, value)`` when no lossless coercion applies.
    """
    if "boolean" in targets and value in ("true", "false"):
        return True, value == "true"
    if "integer" in targets:
        try:
            parsed = int(value)
        except ValueError:
            parsed = None
        # Lossless only when the canonical str of the parsed int equals the input
        # (rejects "5.0", " 5", "0x5", "+5", leading zeros like "05", etc.).
        if parsed is not None and str(parsed) == value:
            return True, parsed
    if "number" in targets:
        try:
            parsed_f = float(value)
        except ValueError:
            parsed_f = None
        # Reject non-finite (inf/nan) and require an exact float round-trip so we
        # never silently reshape the model's literal (e.g. "1e3" -> 1000.0).
        if (
            parsed_f is not None
            and parsed_f == parsed_f  # not NaN
            and parsed_f not in (float("inf"), float("-inf"))
            and repr(parsed_f) == value
        ):
            return True, parsed_f
    return False, value


def _coerce_lenient_args(
    args: dict[str, Any], schema: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Tolerate small-model arg fuzz before strict validation (Tier 1.2).

    Three minimal, safe normalizations against an object schema: (1) drop keys the
    model hallucinated that the schema forbids (``additionalProperties: false``); (2)
    drop ``null``-valued *optional* keys (a model emitting ``"date": null`` for an
    absent optional should behave like omitting it, so the handler's default applies);
    and (3) LOSSLESSLY coerce a stringified scalar (``"5"`` -> ``5``, ``"3.14"`` ->
    ``3.14``, ``"true"`` -> ``True``) ONLY when the schema declares that scalar type
    and the conversion round-trips exactly. This closes the gap where the local
    Ollama/Claude-CLI path (small models that stringify everything) was strictly worse
    than cloud providers, whose strict validator hard-fails on a stringified scalar.

    Required fields keep their values; coercion never widens an ambiguous/lossy string.
    Returns ``(cleaned_args, coerced_keys)`` — ``coerced_keys`` names every key whose
    value was scalar-coerced, recorded on the result metadata for measurement.
    """
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    extras_allowed = schema.get("additionalProperties", True) is not False
    cleaned: dict[str, Any] = {}
    coerced_keys: list[str] = []
    for key, value in args.items():
        prop = properties.get(key)
        if prop is None:
            if extras_allowed:
                cleaned[key] = value  # schema permits extras — keep as-is
            continue  # otherwise strip the unknown key
        if value is None and key not in required:
            prop_type = prop.get("type")
            allows_null = prop_type == "null" or (
                isinstance(prop_type, list) and "null" in prop_type
            )
            if not allows_null:
                continue  # drop the null optional → treated as omitted
        # Lossless scalar coercion: only a bare string against a declared scalar type.
        if isinstance(value, str) and isinstance(prop, dict):
            targets = _scalar_types(prop)
            if targets:
                did, coerced = _coerce_scalar(value, targets)
                if did:
                    cleaned[key] = coerced
                    coerced_keys.append(key)
                    continue
        cleaned[key] = value
    return cleaned, coerced_keys


def _schema_hint(schema: dict[str, Any]) -> str:
    """A compact, model-facing description of an object schema's args (Tier 1.4)."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not properties:
        return "this tool takes no arguments."
    parts = []
    for name, spec in properties.items():
        ptype = spec.get("type", "any")
        if isinstance(ptype, list):
            ptype = "|".join(ptype)
        parts.append(f"{name} ({ptype}){'*' if name in required else ''}")
    return "expected args: " + ", ".join(parts) + " (* = required)."


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
        lenient_args: bool = True,
        reputation_provider: ToolReputationLike | None = None,
        event_workspace_id: str | None = None,
    ) -> None:
        self._registry = registry
        self._pre_hook = pre_execution_hook
        self._post_hook = post_execution_hook
        self.event_sink = event_sink
        self._default_timeout_seconds = default_timeout_seconds
        # P1 tenancy: the owning workspace stamped onto every tool event this service
        # emits, so the self-learning reputation miner can scope its read to ONE tenant
        # on a shared event store (``list_events(workspace_id=...)``). ``None`` (the
        # default, and every non-self-learning / CLI path) leaves events unscoped —
        # byte-identical to before. Set only on the per-run server wiring that knows the
        # run's workspace (``build_runtime_for_spec(subject=...)``).
        self._event_workspace_id = event_workspace_id
        # P1 self-learning (opt-in): a sync reputation snapshot used to STABLE-sort flaky
        # tools after reliable ones in ``bound_tools`` (and annotate the worst). ``None``
        # (the default) makes ``bound_tools`` byte-for-byte identical to before — no
        # reordering, no annotation. Injected only for ``self_learning`` specs.
        self._reputation_provider = reputation_provider
        # Tolerate small-model arg fuzz: drop hallucinated unknown keys (when the
        # schema forbids extras) and null-valued optionals (treated as omitted) BEFORE
        # validation. Keeps required/typed checks strict; off → strict pass-through.
        self._lenient_args = lenient_args
        # Shared httpx client, built lazily on first HTTP call (or injected for tests).
        # When a client is injected (tests pass a MockTransport-backed one) it is used
        # for EVERY HTTP tool. Otherwise a per-egress-policy pinned client is built and
        # cached, so each tool's SSRF allow-list / private-host policy is enforced at the
        # transport (DNS-pinned) layer, not just at URL-guard time.
        self._http_client = http_client
        self._http_client_owned = http_client is None
        self._http_max_connections = http_max_connections
        self._http_max_keepalive = http_max_keepalive_connections
        self._http_lock = asyncio.Lock()
        self._pinned_clients: dict[tuple[bool, tuple[str, ...]], Any] = {}

    @property
    def registry(self) -> ToolRegistry:
        """The underlying tool catalog this service dispatches against."""
        return self._registry

    async def _emit(self, event: RunEvent) -> None:
        """Best-effort emit; a broken sink must never fail a tool call.

        Stamps the owning ``workspace_id`` (P1 tenancy) onto the event when the service
        is workspace-scoped and the event does not already carry one, so the durable
        backends denormalise it and the self-learning reputation miner can read it back
        scoped to ONE tenant. A no-op when ``event_workspace_id`` is None (the default).
        """
        if self.event_sink is None:
            return
        if self._event_workspace_id is not None and event.workspace_id is None:
            event.workspace_id = self._event_workspace_id
        try:
            await self.event_sink.append_event(event)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "failed to emit %s tool event", event.event_type.value, exc_info=True
            )

    async def _get_http_client(
        self,
        *,
        allow_private: bool = False,
        allow_hosts: tuple[str, ...] = (),
    ) -> Any:
        """Return an httpx client for the given egress policy, building it lazily.

        An *injected* client (the test seam) is used verbatim for every tool. Otherwise
        a DNS-pinned ``AsyncClient`` is built per ``(allow_private, allow_hosts)`` policy
        and cached — so the SSRF allow-list and the private-host opt-out are enforced at
        the transport layer (closing the DNS-rebinding window), not only at URL-guard
        time. ``follow_redirects`` is always off so a 3xx can't bounce to an internal
        host.
        """
        if self._http_client is not None:
            return self._http_client
        key = (allow_private, allow_hosts)
        cached = self._pinned_clients.get(key)
        if cached is not None:
            return cached
        async with self._http_lock:
            cached = self._pinned_clients.get(key)  # re-check under the lock
            if cached is None:
                import httpx

                from himmy.toolkit._net import build_async_pinned_transport

                limits = httpx.Limits(
                    max_connections=self._http_max_connections,
                    max_keepalive_connections=self._http_max_keepalive,
                )
                cached = httpx.AsyncClient(
                    follow_redirects=False,
                    limits=limits,
                    transport=build_async_pinned_transport(
                        allow_private=allow_private,
                        allow_hosts=allow_hosts or None,
                    ),
                )
                self._http_client_owned = True
                self._pinned_clients[key] = cached
        return cached

    async def aclose(self) -> None:
        """Close every owned httpx client (shared + per-egress pinned); idempotent."""
        clients: list[Any] = []
        if self._http_client is not None and self._http_client_owned:
            clients.append(self._http_client)
            self._http_client = None
        clients.extend(self._pinned_clients.values())
        self._pinned_clients = {}
        for client in clients:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - defensive
                pass

    # Alias for callers that expect a sync-style name.
    close = aclose

    async def execute(
        self,
        invocation: ToolInvocation,
        *,
        idempotency_store: ToolIdempotencyStore | None = None,
    ) -> ToolExecutionResult:
        """Execute one tool invocation end-to-end and return its result.

        When ``idempotency_store`` is provided AND the invocation carries a string
        ``metadata["idempotency_key"]``, execution is at-most-once per key: a
        previously recorded result is replayed verbatim (no handler call, no
        duplicate tool events, ``metadata["idempotent_replay"] = True``), and a
        fresh result is recorded via ``put`` before being returned — including
        failures, since a post-dispatch failure (timeout, upstream error) may
        already have mutated state and must not be silently re-attempted by a
        resume retry. ``denied`` outcomes are never recorded, so a later
        *approved* execution of the same key still runs. Invocations without a
        key — or calls without a store — behave exactly as before; the seam is
        fully opt-in.
        """
        key = invocation.metadata.get("idempotency_key")
        if idempotency_store is None or not isinstance(key, str) or not key:
            return await self._execute_invocation(invocation)
        cached = idempotency_store.get(key)
        if cached is not None:
            replay = cached.model_copy(deep=True)
            replay.metadata = {**replay.metadata, "idempotent_replay": True}
            return replay
        result = await self._execute_invocation(invocation)
        if result.outcome != "denied":
            idempotency_store.put(key, result)
        return result

    async def _execute_invocation(
        self, invocation: ToolInvocation
    ) -> ToolExecutionResult:
        """The uncached execution pipeline behind :meth:`execute`."""
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
        coerced_keys: list[str] = []
        if schema and schema.get("type") == "object":
            if self._lenient_args:
                args, coerced_keys = _coerce_lenient_args(args, schema)
            err = validate_against_schema(args, schema)
            if err is not None:
                # Self-correction (Tier 1.4): hand the model the schema so it can fix
                # its args on the next turn, not just the error.
                return await self._fail(
                    invocation,
                    start,
                    ToolErrorCode.INVALID_REQUEST,
                    f"args failed schema validation: {err}. {_schema_hint(schema)}",
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
            # Retry only on transient codes with attempts remaining — and only when the
            # failure is safe to repeat. A non-idempotent HTTP mutation (a POST/PUT/etc.
            # with no idempotency key) is NOT retried on TIMEOUT / PROVIDER_UNAVAILABLE,
            # since the request may already have landed and a blind resend would
            # double-act; a 429 (RATE_LIMITED) is always safe — the upstream rejected it
            # before doing any work.
            if (
                last_error is not None
                and last_error.code in RETRYABLE_TOOL_CODES
                and attempt < retry.max_attempts - 1
                and self._is_retry_safe(definition, last_error.code, args)
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
            # Record which args were losslessly scalar-coerced (small-model fuzz
            # repair) so the gap-closing can be measured; absent when none, so the
            # default-path metadata is unchanged.
            metadata=({"coerced_args": coerced_keys} if coerced_keys else {}),
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

    @staticmethod
    def _is_retry_safe(
        definition: ToolDefinition, code: ToolErrorCode, args: dict[str, Any]
    ) -> bool:
        """Whether a transient ``code`` may be retried for THIS invocation.

        LOCAL tools and read-only / idempotent HTTP calls are always safe to repeat. A
        side-effecting HTTP method (POST/PUT/PATCH/DELETE) with NO idempotency key may
        already have landed when a TIMEOUT or PROVIDER_UNAVAILABLE fires, so those are
        not retried — only RATE_LIMITED (429, rejected before any work) is.

        The idempotency exemption is keyed off whether the key was actually supplied
        for *this* call — not merely whether ``idempotency_arg`` is configured. The
        ``Idempotency-Key`` header is only attached when ``args`` carry that key (see
        :meth:`_dispatch_http`), so a configured-but-omitted key means the upstream
        gets NO dedup header and a blind resend could double-act. Treating such a call
        as retry-safe (the old behavior) silently re-fires the mutation; we don't.
        """
        cfg = definition.http_config
        if cfg is None:
            return True
        if cfg.method.upper() in _SAFE_HTTP_METHODS:
            return True
        if cfg.idempotency_arg and args.get(cfg.idempotency_arg):
            return True  # the Idempotency-Key lets the upstream dedupe a resend
        return code is ToolErrorCode.RATE_LIMITED

    async def _dispatch(self, definition: ToolDefinition, args: dict[str, Any]) -> Any:
        """Route to the LOCAL or HTTP backend."""
        if definition.kind is ToolBackendKind.LOCAL:
            return await self._dispatch_local(definition, args)
        return await self._dispatch_http(definition, args)

    async def _dispatch_local(
        self, definition: ToolDefinition, args: dict[str, Any]
    ) -> Any:
        """Invoke a local handler off the event loop when it is synchronous.

        A coroutine handler is awaited directly. A plain sync handler — which may do
        blocking network/file/SQLite IO (the web/file/data packs are sync) — is run
        in a worker thread via :func:`asyncio.to_thread`, so one slow tool can't
        freeze the single event loop shared by every concurrent run, stream, and the
        scheduler. Offloading also lets the surrounding ``wait_for`` timeout actually
        fire: the loop stays responsive and abandons the worker thread on timeout.
        """
        handler = self._registry.handler_for(definition.name)
        if handler is None:
            raise _ToolDispatchError(
                ToolErrorCode.EXECUTION_ERROR,
                f"no local handler registered for {definition.name!r}",
            )
        if inspect.iscoroutinefunction(handler):
            return await handler(args)
        outcome = await asyncio.to_thread(handler, args)
        # A sync handler may still return an awaitable (e.g. a callable that returns
        # a coroutine); preserve the original await-the-result contract.
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return outcome

    async def _dispatch_http(
        self, definition: ToolDefinition, args: dict[str, Any]
    ) -> Any:
        """Execute a declarative HTTP/REST connector, guarded end-to-end.

        Hardening on every call: the method must be on the allow-list; the base URL is
        resolved through the secrets layer; path args are percent-encoded and re-checked
        against the base host (no host pivot, no traversal); the final URL is SSRF-guarded
        (public-only, plus an optional egress allow-list) and dialed through a DNS-pinned
        transport (no rebinding); auth is read from the secrets layer (never logged);
        redirects are refused; a non-GET is NOT blind-retried here (it carries an
        ``Idempotency-Key`` when configured); and, when pagination is on, pages are
        followed up to a hard ``max_pages`` cap and the records are concatenated.
        """
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

        base_url = ""
        if cfg.base_url_env_var:
            base_url = get_secret(cfg.base_url_env_var) or ""
        if not base_url:
            base_url = cfg.base_url
        if not base_url:
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"no base URL for HTTP tool: set `base_url` or the "
                f"{cfg.base_url_env_var!r} secret",
            )

        # Resolve & encode path placeholders, then re-pin to the base host.
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

        # SSRF guard the resolved URL: scheme / no-embedded-creds / host-allow-list are
        # checked eagerly here; the DNS→public-IP check is done at CONNECT time by the
        # pinned transport (so the same name is resolved once, and the IP that is vetted
        # is the IP that is dialed — closing the rebinding window). When the operator
        # injects their own client (the test seam), we run the cheap checks and trust
        # their transport. ``resolve=False`` avoids a redundant/offline DNS lookup here.
        from himmy.toolkit._net import guard_url

        allow_hosts = tuple(cfg.egress_allow_hosts) or None
        try:
            guard_url(
                url,
                allow_private=cfg.allow_private_hosts,
                allow_hosts=allow_hosts,
                resolve=False,
            )
        except ToolSecurityError as exc:
            raise _ToolDispatchError(ToolErrorCode.INVALID_REQUEST, str(exc)) from exc

        body = {k: args[k] for k in cfg.body_arg_names if k in args}
        headers = {k: str(args[k]) for k in cfg.header_arg_names if k in args}
        auth_headers, auth_params = self._build_auth(cfg.auth)
        headers.update(auth_headers)

        # An idempotency key turns a side-effecting retry into a safe no-op upstream.
        if cfg.idempotency_arg and args.get(cfg.idempotency_arg):
            headers["Idempotency-Key"] = str(args[cfg.idempotency_arg])

        base_params: dict[str, Any] = dict(cfg.static_query)
        base_params.update({k: args[k] for k in cfg.query_arg_names if k in args})
        base_params.update(auth_params)

        timeout = self._resolve_timeout(definition)
        client = await self._get_http_client(
            allow_private=cfg.allow_private_hosts,
            allow_hosts=tuple(cfg.egress_allow_hosts),
        )

        if cfg.pagination.mode is HttpPaginationMode.NONE:
            return await self._http_single(
                client, method, url, base_params, body, headers, timeout
            )
        return await self._http_paginated(
            client, cfg, method, url, base_params, body, headers, timeout
        )

    async def _http_single(
        self,
        client: Any,
        method: str,
        url: str,
        params: dict[str, Any],
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> Any:
        """Issue ONE request, normalize transport/HTTP errors, return parsed JSON."""
        response = await self._http_request(
            client, method, url, params, body, headers, timeout
        )
        try:
            return response.json()
        except Exception:
            return {"text": response.text}

    async def _http_request(
        self,
        client: Any,
        method: str,
        url: str,
        params: dict[str, Any],
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> Any:
        """One guarded request → an httpx ``Response`` (status mapped to tool errors)."""
        import httpx

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
        except ToolSecurityError as exc:
            # The pinned transport refused the connect (rebinding / off-allow-list host).
            raise _ToolDispatchError(ToolErrorCode.INVALID_REQUEST, str(exc)) from exc
        except httpx.TimeoutException as exc:
            # Never echo header/secret values; report the exception type only.
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
        if 300 <= response.status_code < 400:
            # Redirects aren't followed (SSRF guard), so surface it clearly instead of
            # returning the redirect page's HTML as a successful result.
            location = response.headers.get("location", "")
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"upstream returned a redirect ({response.status_code})"
                + (f" to {location}" if location else "")
                + "; point the tool's base_url/path at the final location",
            )
        if response.status_code >= 400:
            raise _ToolDispatchError(
                ToolErrorCode.INVALID_REQUEST,
                f"upstream returned {response.status_code}",
            )
        return response

    async def _http_paginated(
        self,
        client: Any,
        cfg: HttpToolConfig,
        method: str,
        url: str,
        base_params: dict[str, Any],
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> Any:
        """Follow pagination up to ``max_pages``; return ``{"items": [...all records]}``.

        Each next page is re-guarded the same way (the pinned client re-checks the IP),
        so a ``Link``/cursor that points off-host or at an internal address is refused.
        A ``Link`` rel="next" is additionally gated to be SAME-ORIGIN with the original
        request (or on the configured egress allow-list), so the connector's reused auth
        header can't be exfiltrated to a host the operator never authorized. ``page_count``
        counts the pages actually fetched in EVERY mode (not just PAGE).
        """
        page_cfg = cfg.pagination
        max_pages = max(1, page_cfg.max_pages)
        allow_hosts = tuple(cfg.egress_allow_hosts)
        collected: list[Any] = []
        params = dict(base_params)
        next_url = url
        page_num = 1  # 1-based page counter for PAGE-mode query params
        pages_fetched = 0  # the value reported back, counted in every mode

        for _ in range(max_pages):
            if page_cfg.mode is HttpPaginationMode.PAGE:
                params[page_cfg.page_param] = page_num
            response = await self._http_request(
                client, method, next_url, params, body, headers, timeout
            )
            pages_fetched += 1
            try:
                payload = response.json()
            except Exception:
                payload = {"text": response.text}

            page_items = _dig(payload, page_cfg.items_path)
            if isinstance(page_items, list):
                collected.extend(page_items)
            elif page_items is not None:
                collected.append(page_items)

            advanced = False
            if page_cfg.mode is HttpPaginationMode.CURSOR:
                cursor = _dig(payload, page_cfg.cursor_path)
                if cursor:
                    params = {**params, page_cfg.cursor_param: cursor}
                    advanced = True
            elif page_cfg.mode is HttpPaginationMode.PAGE:
                # Stop once a page comes back empty (no more records).
                if isinstance(page_items, list) and page_items:
                    page_num += 1
                    advanced = True
            elif page_cfg.mode is HttpPaginationMode.LINK_HEADER:
                from himmy.toolkit._net import guard_url

                link = _next_link(response.headers.get("link", ""))
                if link:
                    # Default-deny: the Link target must be same-origin with the original
                    # request (or on the egress allow-list) BEFORE we reuse the auth
                    # header on it — a cross-host rel="next" would leak the bearer secret.
                    if not _link_target_allowed(link, next_url, allow_hosts):
                        raise _ToolDispatchError(
                            ToolErrorCode.INVALID_REQUEST,
                            "pagination Link target refused: cross-host rel=\"next\" "
                            "is not same-origin and not on the egress allow-list",
                        )
                    try:
                        guard_url(
                            link,
                            allow_private=cfg.allow_private_hosts,
                            allow_hosts=allow_hosts or None,
                            resolve=False,
                        )
                    except ToolSecurityError as exc:
                        raise _ToolDispatchError(
                            ToolErrorCode.INVALID_REQUEST,
                            f"pagination Link target refused: {exc}",
                        ) from exc
                    next_url, params = link, {}
                    advanced = True
            if not advanced:
                break

        return {"items": collected, "page_count": pages_fetched}

    @staticmethod
    def _build_auth(auth: Any) -> tuple[dict[str, str], dict[str, str]]:
        """Resolve secrets-backed auth into (headers, query-params); both empty if unset.

        The credential is read through the SECRETS LAYER (``get_secret``) — env var,
        vault, cloud secret manager, or file, per deployment config — and never echoed.
        ``BASIC`` pairs ``username`` with the secret (or treats the secret as raw
        ``user:pass`` when no username is set) and base64-encodes here;
        ``PREENCODED_BASIC`` passes an already-encoded credential through verbatim;
        ``API_KEY_QUERY`` returns the secret as a query param instead of a header.
        """
        if auth.mode is HttpAuthMode.NONE or not auth.env_var:
            return {}, {}
        secret = get_secret(auth.env_var)
        if not secret:
            return {}, {}
        if auth.mode is HttpAuthMode.BEARER:
            return {"Authorization": f"Bearer {secret}"}, {}
        if auth.mode is HttpAuthMode.BASIC:
            raw = f"{auth.username}:{secret}" if auth.username else secret
            encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
            return {"Authorization": f"Basic {encoded}"}, {}
        if auth.mode is HttpAuthMode.PREENCODED_BASIC:
            return {"Authorization": f"Basic {secret}"}, {}
        if auth.mode is HttpAuthMode.HEADER:
            header_name = auth.header_name or "Authorization"
            return {header_name: secret}, {}
        if auth.mode is HttpAuthMode.API_KEY_QUERY:
            param = auth.query_param or "api_key"
            return {}, {param: secret}
        return {}, {}

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
        """Bind selected tools as pure-data inference ``BoundTool``s.

        Each returned ``BoundTool`` carries only the tool's schemas and read-only
        flag — execution flows separately through :meth:`tool_executor`, which the
        runtime attaches to the request. Keeping execution out of ``BoundTool`` is
        what decouples the inference layer from the tool layer. The inference kernel
        is imported lazily so this module imports without it.

        P1 self-learning: when a ``reputation_provider`` is injected, the selected tools
        are STABLE-sorted by recent reputation (best first) before binding — ties keep
        insertion order, so the result is deterministic — and a tool below the provider's
        unreliable floor gets a short caution APPENDED to its description (annotate, never
        drop, so the model can still use it if truly needed). With no provider this is
        byte-for-byte the historical behaviour.
        """
        # Imported lazily: the inference kernel may be built in parallel and is
        # not on the core import path for this kernel.
        from himmy.services.inference.models import BoundTool

        selected = (
            self._registry.list()
            if names is None
            else [d for n in names if (d := self._registry.get(n)) is not None]
        )

        provider = self._reputation_provider
        if provider is not None:
            # ``sorted`` is stable: equal scores keep registry/insertion order. Sort by
            # NEGATED score so higher reputation comes first without reversing ties.
            selected = sorted(selected, key=lambda d: -provider.score_for(d.name))

        return [
            BoundTool(
                name=definition.name,
                description=self._annotated_description(definition, provider),
                args_json_schema=definition.args_json_schema,
                output_json_schema=definition.output_json_schema,
                read_only=definition.read_only,
            )
            for definition in selected
        ]

    @staticmethod
    def _annotated_description(
        definition: ToolDefinition, provider: ToolReputationLike | None
    ) -> str:
        """Append a flaky-tool caution to the description when the tool is unreliable."""
        if provider is not None and provider.is_unreliable(definition.name):
            return (
                f"{definition.description} "
                "(note: this tool has failed frequently recently)"
            )
        return definition.description

    def tool_executor(self) -> ToolExecutor:
        """Return the single callback a client manager uses to execute tools by name.

        This is the one explicit seam between inference and tool execution: it routes
        ``(tool_name, args)`` through :meth:`execute` (so every policy hook, timeout,
        retry, and event still applies) and normalizes the result into a
        ``ToolReturnRecord``. The runtime attaches it to ``InferenceRequest`` next to
        ``bound_tools``; unknown names fail closed via :meth:`execute`.
        """
        # Imported lazily: the inference kernel is not on this kernel's import path.
        from himmy.services.inference.models import ToolReturnRecord

        async def _execute(tool_name: str, args: dict[str, Any]) -> ToolReturnRecord:
            res = await self.execute(ToolInvocation(tool_name=tool_name, args=args))
            return ToolReturnRecord(
                tool_call_id=res.tool_call_id,
                tool_name=res.tool_name,
                content=res.result,
                outcome=res.outcome,
                metadata={
                    "error_code": res.error_code.value if res.error_code else None,
                    "error_message": res.error_message,
                    "latency_ms": res.latency_ms,
                },
            )

        return _execute


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
