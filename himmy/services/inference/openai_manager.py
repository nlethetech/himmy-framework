"""Inference kernel: a DIRECT OpenAI (chat-completions) client manager.

Talks to the async ``openai`` SDK's ``chat.completions.create`` with no pydantic-ai
indirection. ``openai`` is an OPTIONAL extra: this module imports cleanly without it and
the SDK is imported lazily inside :meth:`generate`. A ``base_url`` override makes this
work against any OpenAI-compatible endpoint (OpenRouter, vLLM, local servers, …).

It honors the full :class:`InferenceRequest` envelope:

* system / user / assistant / tool messages project straight into chat ``messages``
  (tool-role turns become ``role="tool"`` with their ``tool_call_id``);
* ``bound_tools`` become ``tools=[{type:function, …}]`` and the model's
  ``message.tool_calls`` are extracted into :class:`ToolCallRecord` (then EXECUTED via
  ``tool_executor`` like the local managers, so the runtime sees real returns);
* JSON_OBJECT uses ``response_format={"type":"json_object"}``; STRUCTURED_OUTPUT uses
  ``response_format={"type":"json_schema", …}`` so the reply matches the schema;
* ``usage`` fills ``input_tokens``/``output_tokens`` and :mod:`pricing` fills ``cost``;
* :meth:`generate_stream` backs ``InferenceService.run_stream`` (optional).

Failure contract (INF-12): ``generate`` NEVER raises — every SDK / transport error is
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
    cache_policy_active,
    cache_savings_usd,
    compute_cached_cost,
    is_openai_family_model,
    openrouter_passthrough_backend,
    read_openai_usage,
    resolve_cache_rates,
)
from himmy.services.tools.access import describe_for_model

#: A default OpenAI model used when the caller doesn't name one (cheap + current).
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

#: Schema name handed to ``response_format=json_schema`` (must be a valid identifier).
_STRUCTURED_SCHEMA_NAME = "structured_output"

#: Default per-request timeout (seconds) for the SDK's underlying HTTP client. Bounded so
#: a hung endpoint can't wedge a call; the InferenceService adds its own ceiling on top.
_DEFAULT_TIMEOUT_S = 60.0

#: Default SDK-level retry budget. The openai SDK only retries on its own idempotent,
#: safe-to-retry conditions (429 / connection / 5xx), never on a partially-applied
#: request, so this is safe for side-effecting tool turns.
_DEFAULT_MAX_RETRIES = 2


def _normalize_openai_error(exc: BaseException) -> InferenceError:
    """Map an ``openai`` SDK / transport exception to a typed InferenceError.

    Keyed on the exception class name so this never imports the SDK on the offline
    path; mirrors OpenAI's error taxonomy (auth/quota/rate-limit/connection/timeout/
    bad-request). Unknown errors are non-retryable ``UNKNOWN``.
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
    status = getattr(exc, "status_code", None)
    if status == 429:
        code, retryable = InferenceErrorCode.RATE_LIMITED, True
    elif status in (401, 403):
        code, retryable = InferenceErrorCode.AUTH, False
    elif isinstance(status, int) and status >= 500:
        code, retryable = InferenceErrorCode.PROVIDER_UNAVAILABLE, True
    return InferenceError(code=code, message=str(exc) or name, retryable=retryable)


def _openai_tools(bound_tools: list[BoundTool]) -> list[dict[str, Any]]:
    """Project bound tools into OpenAI's function-tool schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": describe_for_model(
                    tool.name, tool.description, tool.read_only
                ),
                "parameters": tool.args_json_schema or {"type": "object"},
            },
        }
        for tool in bound_tools
    ]


def _tool_call_block(m: InferenceMessage) -> dict[str, Any] | None:
    """Reconstruct the OpenAI assistant ``tool_calls`` entry that a tool turn answers.

    OpenAI strictly requires every ``role="tool"`` message to be preceded by an
    assistant message bearing a ``tool_calls`` array that includes the SAME
    ``tool_call_id`` — otherwise the request is rejected with a 400 (the multi-turn
    tool blocker). The runtime's tool-role message carries the originating call's
    ``tool_call_id``, tool name (``name`` / ``metadata['tool_name']``) and arguments
    (``metadata['tool_args']``), so the assistant call block is fully reconstructable
    here. Returns ``None`` when there is no ``tool_call_id`` to anchor the pair.
    """
    call_id = m.tool_call_id or m.metadata.get("tool_call_id")
    if not call_id:
        return None
    name = m.name or m.metadata.get("tool_name") or ""
    args = m.metadata.get("tool_args")
    try:
        arguments = json.dumps(args if isinstance(args, dict) else {})
    except (TypeError, ValueError):  # pragma: no cover - defensive
        arguments = "{}"
    return {
        "id": str(call_id),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _openai_messages(
    request: InferenceRequest, *, cache_system: bool = False
) -> list[dict[str, Any]]:
    """Project the request into OpenAI chat ``messages`` (role-preserving).

    A ``role="tool"`` turn projects to a ``role="tool"`` message, but OpenAI rejects a
    tool message that is not preceded by an assistant message carrying a matching
    ``tool_calls`` array. So before each run of tool messages this back-fills the
    originating assistant ``tool_calls`` (reconstructed from the tool turns' metadata):
    if the message just before the run is already an assistant message it gains the
    ``tool_calls`` array, otherwise a synthetic assistant message is inserted to anchor
    the pair. This is the symmetric round-trip of the non-streaming tool-call extraction
    and makes a multi-turn tool run (streaming or not) valid against OpenAI/OpenRouter.

    With ``cache_system=True`` the LAST system message's content is rewritten from a
    plain string into a single-block array carrying ``cache_control`` — the shape
    OpenRouter forwards to an Anthropic/Gemini backend for explicit prefix caching. This
    path is taken ONLY for OpenRouter passthrough models with caching enabled; the
    default (``cache_system=False``) emits string content, byte-identical to today and
    pinned by the OpenAI shape contract.
    """
    messages: list[dict[str, Any]] = []
    last_system_idx = -1
    # Tracks the assistant message that the CURRENT contiguous run of tool messages must
    # be anchored to; reset whenever a non-tool message breaks the run.
    tool_anchor: dict[str, Any] | None = None
    for m in request.messages:
        role = (m.role or "user").lower()
        if role == "tool":
            block = _tool_call_block(m)
            if block is not None:
                if tool_anchor is None:
                    # No assistant message precedes this tool turn (e.g. the runtime
                    # appends the tool result before the assistant text): insert a
                    # synthetic assistant message to carry the originating tool_calls.
                    tool_anchor = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    }
                    messages.append(tool_anchor)
                tool_anchor.setdefault("tool_calls", []).append(block)
            messages.append(
                {
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id or "",
                }
            )
        elif role in ("system", "assistant", "user"):
            entry: dict[str, Any] = {"role": role, "content": m.content}
            messages.append(entry)
            # An assistant message can anchor the tool messages that follow it.
            tool_anchor = entry if role == "assistant" else None
            if role == "system":
                last_system_idx = len(messages) - 1
        else:
            messages.append({"role": "user", "content": m.content})
            tool_anchor = None
    if cache_system and last_system_idx >= 0:
        text = messages[last_system_idx]["content"]
        messages[last_system_idx]["content"] = [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return messages


def _structured_schema(request: InferenceRequest) -> dict[str, Any]:
    """The JSON schema for STRUCTURED_OUTPUT (object fallback when none provided)."""
    if request.output_json_schema:
        return request.output_json_schema
    return {"type": "object", "additionalProperties": True}


class OpenAIClientManager:
    """A :class:`ClientManager` backed directly by the async ``openai`` SDK.

    Construction takes an optional ``client`` so tests can inject a fake async client
    (``client.chat.completions.create(...)``) and the whole layer stays offline. With no
    injected client the real ``openai.AsyncOpenAI`` is built lazily on first use
    (requiring the ``[openai]`` extra + ``OPENAI_API_KEY``, unless a ``base_url`` +
    ``api_key`` override targets another OpenAI-compatible endpoint such as OpenRouter,
    Sarvam, Groq, or Together).

    ``default_headers`` are sent on every request — used for OpenRouter's optional
    ``HTTP-Referer`` / ``X-Title`` attribution headers (supported, never required). The
    SDK client is built with a bounded ``timeout`` and a conservative ``max_retries``;
    the SDK only retries its own safe/idempotent conditions, so this never re-applies a
    side-effecting request.
    """

    #: OpenAI-family caches AUTOMATICALLY on a stable prefix (no ``cache_control``).
    #: The cost accounting bills returned ``cached_tokens`` at the family read rate.
    #: (OpenRouter passthrough to an Anthropic/Gemini backend is a C4 refinement.)
    cache_capability: CacheCapability = CacheCapability.OPENAI_AUTOMATIC

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        model_registry: dict[str, str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        client: Any | None = None,
        provider_name: str = "openai",
    ) -> None:
        """Configure the default model, optional key/base_url/headers, and (test) client."""
        self._model = model
        self._registry = dict(model_registry or {})
        self._api_key = api_key
        self._base_url = base_url
        # Drop blank header values so an unset OpenRouter attribution env var never sends
        # an empty header; keep this a copy so callers can't mutate it later.
        self._default_headers = {
            str(k): str(v) for k, v in (default_headers or {}).items() if v
        }
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self.provider_name = provider_name

    def resolve(self, model_key: str) -> str:
        """Map a model key to an OpenAI model id (registry, else the key/default)."""
        if not model_key or model_key == "default":
            return self._registry.get("default", self._model)
        return self._registry.get(model_key, model_key)

    def _build_client(self) -> Any:
        """Lazily construct the real ``openai.AsyncOpenAI`` client (extra-gated)."""
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "openai SDK not installed; pip install 'himmy[openai]'"
            ) from exc
        kwargs: dict[str, Any] = {
            "timeout": self._timeout,
            "max_retries": self._max_retries,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._default_headers:
            kwargs["default_headers"] = dict(self._default_headers)
        self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Call ``chat.completions.create`` and map the reply onto an InferenceResponse.

        Never raises: any SDK / transport failure is normalized to a FAILED response.
        """
        model = self.resolve(request.model_key)
        model_path = f"openai:{model}"
        started = time.perf_counter()
        params = request.generation_params or {}

        cache_system = self._passthrough_caches_system(request, model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": _openai_messages(request, cache_system=cache_system),
        }
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            payload["max_tokens"] = params["max_tokens"]
        if params.get("top_p") is not None:
            payload["top_p"] = params["top_p"]
        if request.bound_tools:
            payload["tools"] = self._tools_payload(request)
        rf = self._response_format(request)
        if rf is not None:
            payload["response_format"] = rf
        self._apply_prompt_cache_key(request, model, payload)
        self._apply_usage_accounting(payload)
        if self._default_headers:
            payload["extra_headers"] = dict(self._default_headers)

        try:
            client = self._build_client()
            completion = await client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - normalize ALL provider errors
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=_normalize_openai_error(exc),
                provider_name=self.provider_name,
                model_path=model_path,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        return await self._map_completion(
            request, completion, model, model_path, started
        )

    @staticmethod
    def _response_format(request: InferenceRequest) -> dict[str, Any] | None:
        """Build the ``response_format`` kwarg for JSON / structured requests."""
        if request.response_format == ResponseFormat.JSON_OBJECT:
            return {"type": "json_object"}
        if request.response_format == ResponseFormat.STRUCTURED_OUTPUT:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": _STRUCTURED_SCHEMA_NAME,
                    "schema": _structured_schema(request),
                },
            }
        return None

    def _passthrough_caches_system(
        self, request: InferenceRequest, model: str
    ) -> bool:
        """True when this is OpenRouter forwarding to an Anthropic/Gemini backend + caching.

        Only ``provider_name == "openrouter"`` with an ``anthropic/...`` or ``google/...``
        underlying model id, and an active (non-``None``, ``enabled``) cache policy,
        warrants rewriting the system message to block form. openai-backed OpenRouter
        models and direct OpenAI/Groq/Together rely on AUTOMATIC prefix caching, so they
        keep the string-content shape the contract test pins.
        """
        if self.provider_name != "openrouter":
            return False
        if not cache_policy_active(request):
            return False
        policy = request.cache_policy
        assert policy is not None  # narrowed by cache_policy_active
        if policy.scope == "none":
            return False
        return openrouter_passthrough_backend(model) is not None

    def _tools_payload(self, request: InferenceRequest) -> list[dict[str, Any]]:
        """Project bound tools, name-sorting them when caching is active for stability.

        OpenAI-family caching is AUTOMATIC on a byte-stable prefix; the tool array is part
        of that prefix, so an active cache policy sorts tools by name for a deterministic
        order (consistent with the response-cache key in ``cache.py``). Sorting is gated
        on the active policy so the default path keeps registry order and stays
        byte-identical to the pinned contract.
        """
        tools = request.bound_tools
        if cache_policy_active(request):
            tools = sorted(tools, key=lambda t: t.name)
        return _openai_tools(tools)

    def _apply_prompt_cache_key(
        self, request: InferenceRequest, model: str, payload: dict[str, Any]
    ) -> None:
        """Add ``prompt_cache_key`` when the policy names one AND the model is OpenAI-family.

        ``prompt_cache_key`` is an OpenAI routing hint; it is omitted for every non-OpenAI
        backend (including OpenRouter's Anthropic/Gemini passthrough) so a non-OpenAI
        endpoint never sees an unsupported param.
        """
        if not cache_policy_active(request):
            return
        policy = request.cache_policy
        assert policy is not None  # narrowed by cache_policy_active
        if policy.cache_key and is_openai_family_model(model):
            payload["prompt_cache_key"] = policy.cache_key

    def _apply_usage_accounting(self, payload: dict[str, Any]) -> None:
        """Opt OpenRouter into detailed usage accounting (``usage: {include: true}``).

        Direct OpenAI/Groq/Together return ``prompt_tokens_details.cached_tokens`` in the
        usage object automatically, so :func:`read_openai_usage` already sees cache reads.
        OpenRouter, however, OMITS ``prompt_tokens_details`` (and upstream cost) UNLESS the
        request opts in with ``usage: {"include": true}``. Without it, the provider still
        serves the cache upstream — the cost is just invisible to our accounting, so
        ``cache_read_tokens`` / ``cache_savings_usd`` would read zero on a real hit. Gated on
        ``provider_name == "openrouter"`` so the direct-OpenAI contract surface is untouched.
        Passed via ``extra_body`` (a non-standard field the SDK forwards verbatim).
        """
        if self.provider_name != "openrouter":
            return
        extra = dict(payload.get("extra_body") or {})
        extra.setdefault("usage", {"include": True})
        payload["extra_body"] = extra

    async def _map_completion(
        self,
        request: InferenceRequest,
        completion: Any,
        model: str,
        model_path: str,
        started: float,
    ) -> InferenceResponse:
        """Map an OpenAI ``ChatCompletion`` onto an :class:`InferenceResponse`."""
        choices = getattr(completion, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        text = getattr(message, "content", None) if message is not None else None

        raw_calls = _parse_openai_tool_calls(message)
        tool_returns: list[Any] = []
        if raw_calls:
            tool_returns = await _execute_tool_calls(
                request.bound_tools, raw_calls, request.tool_executor
            )
            text = None  # the reply was a tool call, not a final answer

        structured: Any = None
        if (
            request.response_format
            in (ResponseFormat.JSON_OBJECT, ResponseFormat.STRUCTURED_OUTPUT)
            and text
        ):
            try:
                structured = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                structured = None

        breakdown = read_openai_usage(getattr(completion, "usage", None))
        price = pricing.price_for(model_path)
        if price.input_per_1k == 0.0 and price.output_per_1k == 0.0:
            price = pricing.price_for(request.model_key)
        read_rate, write_mult = resolve_cache_rates(
            price,
            capability=self.cache_capability,
            cache_read_multiplier=price.cache_read_multiplier,
            cache_write_multiplier=price.cache_write_multiplier,
        )
        cost = compute_cached_cost(
            price, breakdown, read_rate=read_rate, write_mult=write_mult
        )

        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=text,
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

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str | InferenceResponse]:
        """Stream text deltas via ``stream=True``, then the final response (INF-7).

        Accumulates ``choices[0].delta.content`` deltas and rebuilds a final
        :class:`InferenceResponse` from the streamed text + any ``usage`` on the last
        chunk (``stream_options={"include_usage": True}``). ``bound_tools`` are sent on
        the stream request and the model's ``delta.tool_calls`` are reassembled by index
        across chunks, then EXECUTED on stream end exactly like :meth:`generate` — so a
        streamed tool turn is no longer silently dropped. Any failure degrades to a
        single buffered :meth:`generate` so the delta-then-final contract always holds.
        """
        model = self.resolve(request.model_key)
        model_path = f"openai:{model}"
        started = time.perf_counter()
        params = request.generation_params or {}
        cache_system = self._passthrough_caches_system(request, model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": _openai_messages(request, cache_system=cache_system),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            payload["max_tokens"] = params["max_tokens"]
        if request.bound_tools:
            payload["tools"] = self._tools_payload(request)
        self._apply_prompt_cache_key(request, model, payload)
        self._apply_usage_accounting(payload)
        if self._default_headers:
            payload["extra_headers"] = dict(self._default_headers)

        try:
            client = self._build_client()
            stream = await client.chat.completions.create(**payload)
            chunks: list[str] = []
            tool_fragments: dict[int, dict[str, Any]] = {}
            breakdown = UsageBreakdown()
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    piece = getattr(delta, "content", None) if delta else None
                    if piece:
                        chunks.append(str(piece))
                        yield str(piece)
                    _accumulate_tool_deltas(delta, tool_fragments)
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    breakdown = read_openai_usage(usage)
        except Exception:  # noqa: BLE001 - degrade to a buffered generate
            yield await self.generate(request)
            return

        raw_calls = _tool_calls_from_fragments(tool_fragments)
        tool_returns: list[Any] = []
        if raw_calls:
            tool_returns = await _execute_tool_calls(
                request.bound_tools, raw_calls, request.tool_executor
            )

        text: str | None = "".join(chunks)
        if raw_calls:
            text = None  # the reply was a tool call, not a final answer
        price = pricing.price_for(model_path)
        if price.input_per_1k == 0.0 and price.output_per_1k == 0.0:
            price = pricing.price_for(request.model_key)
        read_rate, write_mult = resolve_cache_rates(
            price,
            capability=self.cache_capability,
            cache_read_multiplier=price.cache_read_multiplier,
            cache_write_multiplier=price.cache_write_multiplier,
        )
        yield InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=text,
            tool_calls=raw_calls,
            tool_returns=tool_returns,
            input_tokens=breakdown.total_input,
            output_tokens=breakdown.output,
            cost=compute_cached_cost(
                price, breakdown, read_rate=read_rate, write_mult=write_mult
            ),
            model_path=model_path,
            provider_name=self.provider_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=_cache_metadata(price, breakdown, read_rate, write_mult),
        )


def _accumulate_tool_deltas(
    delta: Any, fragments: dict[int, dict[str, Any]]
) -> None:
    """Fold a streamed ``delta.tool_calls`` slice into per-index fragment buffers.

    OpenAI streams a tool call across many chunks: the first delta for an ``index``
    carries the ``id`` (and the function ``name``), and later deltas append more of the
    JSON ``arguments`` string. This reassembles them by ``index`` so the completed call
    can be parsed like the non-streaming ``message.tool_calls`` shape.
    """
    if delta is None:
        return
    for tc in getattr(delta, "tool_calls", None) or []:
        idx = getattr(tc, "index", None)
        if idx is None:
            continue
        slot = fragments.setdefault(idx, {"id": None, "name": "", "arguments": ""})
        tc_id = getattr(tc, "id", None)
        if tc_id:
            slot["id"] = tc_id
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            if name:
                slot["name"] = name
            args = getattr(fn, "arguments", None)
            if args:
                slot["arguments"] += str(args)


def _tool_calls_from_fragments(
    fragments: dict[int, dict[str, Any]],
) -> list[ToolCallRecord]:
    """Parse reassembled streamed tool-call fragments into typed :class:`ToolCallRecord`.

    Ordered by stream ``index``; the accumulated ``arguments`` JSON string is decoded
    (an empty / malformed buffer degrades to ``{}``), mirroring the non-streaming
    :func:`_parse_openai_tool_calls` shape.
    """
    calls: list[ToolCallRecord] = []
    for idx in sorted(fragments):
        slot = fragments[idx]
        name = slot.get("name") or ""
        if not name:
            continue
        raw_args = slot.get("arguments") or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
        calls.append(
            ToolCallRecord(
                tool_call_id=slot.get("id") or new_uuid(),
                tool_name=name,
                args=args if isinstance(args, dict) else {},
            )
        )
    return calls


def _parse_openai_tool_calls(message: Any) -> list[ToolCallRecord]:
    """Parse ``message.tool_calls`` into typed :class:`ToolCallRecord`."""
    if message is None:
        return []
    raw = getattr(message, "tool_calls", None) or []
    calls: list[ToolCallRecord] = []
    for item in raw:
        fn = getattr(item, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", "") or ""
        args = getattr(fn, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(
            ToolCallRecord(
                tool_call_id=getattr(item, "id", None) or new_uuid(),
                tool_name=name,
                args=args if isinstance(args, dict) else {},
            )
        )
    return calls


def _cache_metadata(
    price: ModelPrice,
    breakdown: UsageBreakdown,
    read_rate: float,
    write_mult: float,
) -> dict[str, Any]:
    """Additive cache observability stamped onto ``InferenceResponse.metadata``.

    Only present when the provider returned cached tokens, so the no-cache path stays
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


__all__ = ["OpenAIClientManager", "DEFAULT_OPENAI_MODEL"]
