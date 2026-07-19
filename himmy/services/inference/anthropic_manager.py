"""Inference kernel: a DIRECT Anthropic (Claude) client manager.

Talks to the async ``anthropic`` SDK's Messages API (``messages.create``) with no
pydantic-ai indirection, so a team that just wants Claude pays for one thin adapter
instead of the whole agent framework. ``anthropic`` is an OPTIONAL extra: this module
imports cleanly without it and the SDK is imported lazily inside :meth:`generate`.

It honors the full :class:`InferenceRequest` envelope:

* system messages become the ``system`` parameter; the remaining turns are projected
  into Anthropic ``messages`` (user/assistant/tool-result blocks);
* ``bound_tools`` become Anthropic tool definitions and the model's ``tool_use`` blocks
  are extracted into :class:`ToolCallRecord` (then EXECUTED via ``tool_executor`` like
  the local managers, so the runtime sees real returns);
* TEXT / JSON_OBJECT / STRUCTURED_OUTPUT / AUTO_TOOLS are all supported — JSON and
  structured requests nudge + a single forced ``__structured_output__`` tool so the
  reply is schema-shaped without a provider-native JSON mode;
* ``usage`` fills ``input_tokens``/``output_tokens`` and :mod:`pricing` fills ``cost``;
* :meth:`generate_stream` backs ``InferenceService.run_stream`` (optional).

Failure contract (INF-12): ``generate`` NEVER raises. Every SDK / transport error is
normalized to ``InferenceResponse(status=FAILED, error=InferenceError(...))`` with the
right :class:`InferenceErrorCode` and ``retryable`` flag.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from himmy.core.ids import new_uuid
from himmy.services.inference import pricing
from himmy.services.inference.local import _execute_tool_calls
from himmy.services.inference.models import (
    BoundTool,
    InferenceError,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ModelPrice,
    ResponseFormat,
    ToolCallRecord,
)
from himmy.services.inference.prompt_cache import (
    CacheCapability,
    UsageBreakdown,
    anthropic_system_blocks,
    anthropic_ttl_supported,
    apply_history_cache_breakpoint,
    cache_policy_active,
    cache_savings_usd,
    compute_cached_cost,
    read_anthropic_usage,
    resolve_cache_rates,
    should_cache_prefix,
)
from himmy.services.tools.access import describe_for_model
from himmy.services.tools.schema_normalize import normalize_tool_schema

#: A default Claude model used when the caller doesn't name one (cheap + current).
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-latest"

#: Anthropic requires an explicit ``max_tokens``; this is the framework floor.
_DEFAULT_MAX_TOKENS = 1024

#: Synthetic tool name used to coax a schema-shaped reply out of a model that has no
#: provider-native JSON/structured mode (JSON_OBJECT / STRUCTURED_OUTPUT paths).
_STRUCTURED_TOOL = "__structured_output__"


def _normalize_anthropic_error(exc: BaseException) -> InferenceError:
    """Map an ``anthropic`` SDK / transport exception to a typed InferenceError.

    Keyed on the exception class name so this never imports the SDK on the offline
    path; the mapping mirrors Anthropic's error taxonomy (auth/quota/rate-limit/
    connection/timeout/bad-request). Unknown errors are non-retryable ``UNKNOWN``.
    """
    name = type(exc).__name__
    code, retryable = {
        "AuthenticationError": (InferenceErrorCode.AUTH, False),
        "PermissionDeniedError": (InferenceErrorCode.AUTH, False),
        "RateLimitError": (InferenceErrorCode.RATE_LIMITED, True),
        "APITimeoutError": (InferenceErrorCode.TIMEOUT, True),
        "TimeoutError": (InferenceErrorCode.TIMEOUT, True),
        "APIConnectionError": (InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
        "ConnectionError": (InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
        "InternalServerError": (InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
        "BadRequestError": (InferenceErrorCode.INVALID_REQUEST, False),
        "UnprocessableEntityError": (InferenceErrorCode.INVALID_REQUEST, False),
        "NotFoundError": (InferenceErrorCode.INVALID_REQUEST, False),
        "ConflictError": (InferenceErrorCode.INVALID_REQUEST, False),
    }.get(name, (InferenceErrorCode.UNKNOWN, False))
    # A 429 surfaced as a generic APIStatusError still means rate-limited.
    status = getattr(exc, "status_code", None)
    if status == 429:
        code, retryable = InferenceErrorCode.RATE_LIMITED, True
    elif status in (401, 403):
        code, retryable = InferenceErrorCode.AUTH, False
    elif isinstance(status, int) and status >= 500:
        code, retryable = InferenceErrorCode.PROVIDER_UNAVAILABLE, True
    return InferenceError(code=code, message=str(exc) or name, retryable=retryable)


def _anthropic_tools(
    bound_tools: list[BoundTool], *, provider: str = "anthropic"
) -> list[dict[str, Any]]:
    """Project bound tools into Anthropic's ``tools`` schema (name/description/input).

    Each tool's ``args_json_schema`` is run through the cross-provider
    :func:`normalize_tool_schema` (``$ref`` inlining, nullable-union collapse, …)
    so a schema another provider tolerates does not break Anthropic tool calling.
    Normalization is fail-safe (widen-only), so an already-valid schema is
    unchanged in meaning.
    """
    return [
        {
            "name": tool.name,
            "description": describe_for_model(
                tool.name, tool.description, tool.read_only
            ),
            "input_schema": normalize_tool_schema(
                tool.args_json_schema or {"type": "object"}, provider
            ),
        }
        for tool in bound_tools
    ]


def _tool_use_block(m: InferenceMessage) -> dict[str, Any] | None:
    """Reconstruct the Anthropic assistant ``tool_use`` block a tool result answers.

    Anthropic requires every ``tool_result`` block to reference a ``tool_use_id`` that
    appeared in a PRECEDING assistant ``tool_use`` block, or the request is rejected.
    The runtime's tool-role message carries the originating call's ``tool_call_id``,
    tool name (``name`` / ``metadata['tool_name']``) and arguments
    (``metadata['tool_args']``), so the assistant ``tool_use`` block is reconstructable
    here. Returns ``None`` when there is no ``tool_call_id`` to anchor the pair.
    """
    call_id = m.tool_call_id or m.metadata.get("tool_call_id")
    if not call_id:
        return None
    name = m.name or m.metadata.get("tool_name") or ""
    args = m.metadata.get("tool_args")
    return {
        "type": "tool_use",
        "id": str(call_id),
        "name": name,
        "input": args if isinstance(args, dict) else {},
    }


def _anthropic_messages(request: InferenceRequest) -> list[dict[str, Any]]:
    """Project the request into Anthropic ``messages`` (system handled separately).

    ``tool``-role messages are rebuilt as a ``tool_result`` content block on a user
    turn (Anthropic's required shape); plain user/assistant turns pass through as text.
    Anthropic also requires each ``tool_result`` to reference a ``tool_use`` block in a
    preceding assistant turn, so before each run of tool results this back-fills the
    originating assistant ``tool_use`` blocks (reconstructed from the tool turns'
    metadata) — inserting a synthetic assistant turn when none precedes — which makes a
    multi-turn tool run valid against Anthropic.
    """
    messages: list[dict[str, Any]] = []
    # The assistant turn whose ``tool_use`` blocks the current run of tool results
    # references; reset whenever a non-tool message breaks the run.
    tool_anchor: dict[str, Any] | None = None
    for m in request.messages:
        role = (m.role or "user").lower()
        if role == "system":
            continue
        if role == "tool":
            block = _tool_use_block(m)
            if block is not None:
                if tool_anchor is None:
                    # No assistant turn precedes this tool result: insert a synthetic
                    # assistant turn to carry the originating tool_use block.
                    tool_anchor = {"role": "assistant", "content": []}
                    messages.append(tool_anchor)
                content = tool_anchor["content"]
                if isinstance(content, str):
                    # Lift a plain-string assistant turn into block form so the
                    # tool_use block can ride alongside the original text.
                    content = (
                        [{"type": "text", "text": content}] if content else []
                    )
                    tool_anchor["content"] = content
                content.append(block)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id or "",
                            "content": m.content,
                        }
                    ],
                }
            )
            # A user (tool_result) turn breaks the assistant anchor.
            tool_anchor = None
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.content}
            messages.append(entry)
            tool_anchor = entry
        else:
            messages.append({"role": "user", "content": m.content})
            tool_anchor = None
    return messages


def _system_text(request: InferenceRequest) -> str:
    """Join all system messages into Anthropic's ``system`` parameter string."""
    parts = [m.content for m in request.messages if m.role == "system" and m.content]
    return "\n\n".join(parts)


def _structured_schema(request: InferenceRequest) -> dict[str, Any]:
    """The JSON schema to coax a structured reply toward (STRUCTURED_OUTPUT/JSON)."""
    if request.output_json_schema:
        return request.output_json_schema
    # JSON_OBJECT has no schema: accept any object.
    return {"type": "object", "additionalProperties": True}


class AnthropicClientManager:
    """A :class:`ClientManager` backed directly by the async ``anthropic`` SDK.

    Construction takes an optional ``client`` so tests can inject a fake async client
    (``client.messages.create(...)``) and the whole layer stays offline. With no
    injected client the real ``anthropic.AsyncAnthropic`` is built lazily on first use
    (requiring the ``[anthropic]`` extra + ``ANTHROPIC_API_KEY``).
    """

    #: Anthropic is the explicit-cache provider (``cache_control`` breakpoints). The
    #: runtime resolves this at the call site to decide how to mark the stable prefix.
    cache_capability: CacheCapability = CacheCapability.ANTHROPIC_EXPLICIT

    def __init__(
        self,
        *,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        model_registry: dict[str, str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        provider_name: str = "anthropic",
    ) -> None:
        """Configure the default model, optional key/base_url, and (test) client."""
        self._model = model
        self._registry = dict(model_registry or {})
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._max_tokens = max_tokens
        self.provider_name = provider_name

    def resolve(self, model_key: str) -> str:
        """Map a model key to a Claude model id (registry, else the key/default)."""
        if not model_key or model_key == "default":
            return self._registry.get("default", self._model)
        return self._registry.get(model_key, model_key)

    def _build_client(self) -> Any:
        """Lazily construct the real ``anthropic.AsyncAnthropic`` client (extra-gated)."""
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "anthropic SDK not installed; pip install 'himmy[anthropic]'"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Call ``messages.create`` and map the reply onto an InferenceResponse.

        Never raises: any SDK / transport failure is normalized to a FAILED response.
        """
        model = self.resolve(request.model_key)
        model_path = f"anthropic:{model}"
        started = time.perf_counter()

        params = request.generation_params or {}
        wants_structured = request.response_format in (
            ResponseFormat.STRUCTURED_OUTPUT,
            ResponseFormat.JSON_OBJECT,
        )

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(params.get("max_tokens") or self._max_tokens),
            "messages": _anthropic_messages(request),
        }
        system = _system_text(request)
        if wants_structured:
            # No provider-native JSON mode: force a single synthetic tool whose input
            # schema IS the desired shape, then read the tool input back as the result.
            instruction = (
                f"Call the `{_STRUCTURED_TOOL}` tool with the structured result."
            )
            system = f"{system}\n\n{instruction}" if system else instruction
        if system:
            self._apply_system(request, model, payload, system)
        self._apply_history_cache(request, model, payload)
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            payload["top_p"] = params["top_p"]

        tools = self._payload_tools(request, wants_structured)
        if tools:
            payload["tools"] = tools
            if wants_structured:
                payload["tool_choice"] = {"type": "tool", "name": _STRUCTURED_TOOL}
            else:
                tool_choice = self._tool_choice(request)
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice

        try:
            client = self._build_client()
            message = await client.messages.create(**payload)
        except Exception as exc:  # noqa: BLE001 - normalize ALL provider errors
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=_normalize_anthropic_error(exc),
                provider_name=self.provider_name,
                model_path=model_path,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        return await self._map_message(request, message, model, model_path, started)

    @staticmethod
    def _tool_choice(request: InferenceRequest) -> dict[str, Any] | None:
        """Provider-native ``tool_choice`` for a forced tool, else ``None``.

        ``normalize_forced_tool`` (the service boundary) stamps
        ``metadata['forced_tool']`` with the forced tool's name; we surface that as
        Anthropic's native ``{"type": "tool", "name": ...}`` so forcing no longer
        relies on the prompt nudge alone. Returns ``None`` when no tool is forced so
        the default payload is byte-identical to before.
        """
        forced = request.metadata.get("forced_tool")
        if not isinstance(forced, str) or not forced:
            return None
        if not any(t.name == forced for t in request.bound_tools):
            return None
        return {"type": "tool", "name": forced}

    def _payload_tools(
        self, request: InferenceRequest, wants_structured: bool
    ) -> list[dict[str, Any]]:
        """Build the ``tools`` payload (real bound tools, or the structured shim).

        Anthropic renders the tool array inside the single ``cache_control`` breakpoint
        on the system block, so an active cache policy name-sorts tools (a fresh,
        non-mutating ``sorted``) to keep the cached system+tools prefix byte-stable —
        mirroring the OpenAI adapter. Gated on ``cache_policy_active`` so the no-cache
        path keeps registry order and stays byte-identical to the pinned contract.
        """
        if wants_structured:
            return [
                {
                    "name": _STRUCTURED_TOOL,
                    "description": "Return the structured result for this request.",
                    "input_schema": normalize_tool_schema(
                        _structured_schema(request), self.provider_name
                    ),
                }
            ]
        if request.bound_tools:
            tools = request.bound_tools
            if cache_policy_active(request):
                tools = sorted(tools, key=lambda t: t.name)
            return _anthropic_tools(tools, provider=self.provider_name)
        return []

    def _apply_system(
        self,
        request: InferenceRequest,
        model: str,
        payload: dict[str, Any],
        system: str,
    ) -> None:
        """Set ``payload['system']`` — block-form with a cache breakpoint, or a plain string.

        When the request opts into caching (a non-``None``, ``enabled`` policy whose
        ``system_and_tools``/``system_only`` scope and prefix size clear the per-model
        floor) the system becomes a single block carrying ``cache_control`` — one
        breakpoint caches BOTH tools and system per Anthropic's render order. A ``1h``
        TTL is emitted only behind an SDK-version gate (degrading to 5m on uncertainty),
        and adds the extended-cache beta header. Otherwise the system stays a plain
        string, byte-identical to the no-cache path.
        """
        if not should_cache_prefix(request, model, self.cache_capability):
            payload["system"] = system
            return
        policy = request.cache_policy
        assert policy is not None  # narrowed by should_cache_prefix
        ttl_supported = anthropic_ttl_supported()
        # sec-r3 #5: Anthropic's prompt cache has no out-of-band partition key (unlike
        # OpenAI's ``prompt_cache_key``), so a per-principal ``cache_key`` — which the
        # OpenAI adapter uses to scope its cache — is otherwise a NO-OP here, leaving a
        # shared-key multi-tenant deployment exposed to a cross-tenant cache-read
        # side-channel on a byte-identical prefix. Fold the scope key into the cached
        # prefix as a leading salt block so two principals never share cacheable bytes.
        payload["system"] = anthropic_system_blocks(
            system,
            ttl=policy.ttl,
            ttl_supported=ttl_supported,
            scope_salt=policy.cache_key,
        )
        if policy.ttl == "1h" and ttl_supported:
            payload["extra_headers"] = {
                "anthropic-beta": "extended-cache-ttl-2025-04-11"
            }

    def _apply_history_cache(
        self, request: InferenceRequest, model: str, payload: dict[str, Any]
    ) -> None:
        """Add a SECOND cache breakpoint on the tail of the conversation history.

        The system+tools breakpoint (:meth:`_apply_system`) discounts only the FIXED
        prefix; the growing conversation/tool-result history sits outside it, so a tool
        loop re-pays full price for every prior step (O(N^2) uncached growth). Anthropic
        allows up to FOUR ``cache_control`` breakpoints, so this stamps one ephemeral
        marker on the last content block of the last message — caching the whole
        conversation-so-far so the NEXT turn (one more tool result appended) reads it
        back cheaply.

        Gated on the SAME predicate as the system breakpoint (``should_cache_prefix``),
        which is ``False`` on a cache-busted (compaction) turn because the runtime drops
        the policy that turn — so a rewritten prefix is never marked. A no-op (leaving
        the payload byte-identical) when caching is inactive.
        """
        if not should_cache_prefix(request, model, self.cache_capability):
            return
        policy = request.cache_policy
        assert policy is not None  # narrowed by should_cache_prefix
        apply_history_cache_breakpoint(
            payload["messages"],
            ttl=policy.ttl,
            ttl_supported=anthropic_ttl_supported(),
        )

    async def _map_message(
        self,
        request: InferenceRequest,
        message: Any,
        model: str,
        model_path: str,
        started: float,
    ) -> InferenceResponse:
        """Map an Anthropic ``Message`` onto an :class:`InferenceResponse`."""
        text_parts: list[str] = []
        structured: Any = None
        raw_calls: list[ToolCallRecord] = []
        wants_structured = request.response_format in (
            ResponseFormat.STRUCTURED_OUTPUT,
            ResponseFormat.JSON_OBJECT,
        )

        for block in getattr(message, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    text_parts.append(text)
            elif kind == "tool_use":
                name = getattr(block, "name", "") or ""
                args = getattr(block, "input", {}) or {}
                if not isinstance(args, dict):
                    args = {}
                if wants_structured and name == _STRUCTURED_TOOL:
                    structured = args
                else:
                    raw_calls.append(
                        ToolCallRecord(
                            tool_call_id=getattr(block, "id", None) or new_uuid(),
                            tool_name=name,
                            args=args,
                        )
                    )

        output_text = "\n".join(text_parts) if text_parts else None

        tool_returns: list[Any] = []
        if raw_calls:
            tool_returns = await _execute_tool_calls(
                request.bound_tools, raw_calls, request.tool_executor
            )
            output_text = None  # the reply was a tool call, not a final answer

        if wants_structured and structured is not None:
            output_text = json.dumps(structured)

        breakdown = read_anthropic_usage(getattr(message, "usage", None))
        price = pricing.price_for(model_path)
        if price.input_per_1k == 0.0 and price.output_per_1k == 0.0:
            price = pricing.price_for(request.model_key)
        ttl = self._policy_ttl(request)
        read_rate, write_mult = resolve_cache_rates(
            price,
            capability=self.cache_capability,
            ttl=ttl,
            cache_read_multiplier=price.cache_read_multiplier,
            cache_write_multiplier=price.cache_write_multiplier,
        )
        cost = compute_cached_cost(
            price, breakdown, read_rate=read_rate, write_mult=write_mult
        )

        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=output_text,
            output_structured=structured,
            tool_calls=raw_calls,
            tool_returns=tool_returns,
            # input/output tokens stay FULL totals for back-compat downstream.
            input_tokens=breakdown.total_input,
            output_tokens=breakdown.output,
            cost=cost,
            model_path=model_path,
            provider_name=self.provider_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=_cache_metadata(price, breakdown, read_rate, write_mult),
        )

    @staticmethod
    def _policy_ttl(request: InferenceRequest) -> str:
        """The cache TTL to bill writes at (from the request policy, default 5m)."""
        policy = request.cache_policy
        return policy.ttl if policy is not None else "5m"

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str | InferenceResponse]:
        """Stream text deltas via the Messages stream, then the final response (INF-7).

        Uses ``client.messages.stream(...)`` as an async context manager yielding text
        deltas; the final accumulated message powers a fully-materialized
        :class:`InferenceResponse`. Any failure falls back to a single buffered
        :meth:`generate` so the contract (deltas then a final response) always holds.

        A JSON_OBJECT / STRUCTURED_OUTPUT request relies on the forced
        ``__structured_output__`` tool that ONLY the buffered path applies; a
        forced-schema reply has no meaningful incremental text, so it delegates to
        :meth:`generate` to get the same constraint + populated ``output_structured``
        (otherwise the stream is unconstrained and ``output_structured`` stays ``None``).
        """
        if request.response_format in (
            ResponseFormat.JSON_OBJECT,
            ResponseFormat.STRUCTURED_OUTPUT,
        ):
            yield await self.generate(request)
            return
        model = self.resolve(request.model_key)
        model_path = f"anthropic:{model}"
        started = time.perf_counter()
        params = request.generation_params or {}
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(params.get("max_tokens") or self._max_tokens),
            "messages": _anthropic_messages(request),
        }
        system = _system_text(request)
        if system:
            self._apply_system(request, model, payload, system)
        self._apply_history_cache(request, model, payload)
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]
        if request.bound_tools:
            # Name-sort under an active policy so the cached system+tools prefix stays
            # byte-stable (the stream path bypasses ``_payload_tools``); fresh list, no
            # mutation, and byte-identical registry order when caching is inactive.
            tools = request.bound_tools
            if cache_policy_active(request):
                tools = sorted(tools, key=lambda t: t.name)
            payload["tools"] = _anthropic_tools(
                tools, provider=self.provider_name
            )
            tool_choice = self._tool_choice(request)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        try:
            client = self._build_client()
            async with client.messages.stream(**payload) as stream:
                async for delta in stream.text_stream:
                    if delta:
                        yield str(delta)
                final = await stream.get_final_message()
            yield await self._map_message(request, final, model, model_path, started)
        except Exception:  # noqa: BLE001 - degrade to a buffered generate
            yield await self.generate(request)


def _cache_metadata(
    price: ModelPrice,
    breakdown: UsageBreakdown,
    read_rate: float,
    write_mult: float,
) -> dict[str, Any]:
    """Additive cache observability stamped onto ``InferenceResponse.metadata``.

    Only present when there were cache reads/creations, so the no-cache path stays
    metadata-clean (and byte-identical for downstream consumers that key on absence).
    """
    if breakdown.cache_read == 0 and breakdown.cache_creation == 0:
        return {}
    return {
        "cache_read_tokens": breakdown.cache_read,
        "cache_creation_tokens": breakdown.cache_creation,
        "cache_savings_usd": cache_savings_usd(
            price, breakdown, read_rate=read_rate, write_mult=write_mult
        ),
    }


__all__ = ["AnthropicClientManager", "DEFAULT_ANTHROPIC_MODEL"]
