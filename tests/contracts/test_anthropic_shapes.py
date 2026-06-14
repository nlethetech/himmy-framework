"""Anthropic contract, layer 1 (always runs): SDK shape parity without the SDK.

The fixture classes below hand-build objects mirroring the EXACT attribute shapes
the ``anthropic`` SDK returns — verified against **anthropic 0.106.0** (``Message``
with ``content`` blocks of ``type="text"|"tool_use"``, ``usage`` with the cache
token split, the ``messages.stream`` context-manager seam, and the error taxonomy
carrying ``status_code``). They are fed through :class:`AnthropicClientManager`
via an injected fake client, pinning the adapter's side of the provider contract:

* request projection — system join, tool-role → ``tool_result`` user block,
  generation params, tool definitions, forced structured ``tool_choice``;
* response normalization — text-block join, ``tool_use`` extraction + execution,
  usage (cache reads folded into input) and cost;
* streaming — text deltas then a final :class:`InferenceResponse`, with mid-stream
  failure degrading to a buffered ``generate``;
* error normalization — the full class-name taxonomy plus the ``status_code``
  fallbacks (429 → RATE_LIMITED, 529 overloaded → PROVIDER_UNAVAILABLE).

Layer 2 (``test_anthropic_sdk.py``) replays these scenarios with REAL SDK types,
so an SDK bump that changes shapes fails loudly there while this file keeps
guarding the adapter logic everywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference import pricing
from himmy.services.inference.anthropic_manager import AnthropicClientManager
from himmy.services.inference.models import (
    BoundTool,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ResponseFormat,
    ToolReturnRecord,
)
from tests.conftest import run_async

#: A model present in the bundled price table so cost mapping is exercised.
PRICED_MODEL = "claude-3-5-haiku-latest"


# ------------------------------------------------------------ fixture SDK shapes
class _TextBlock:
    """Mirrors ``anthropic.types.TextBlock`` (``type="text"``, ``text``)."""

    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _ToolUseBlock:
    """Mirrors ``anthropic.types.ToolUseBlock`` (``type/id/name/input``)."""

    type = "tool_use"

    def __init__(self, *, id: str | None, name: str, input: Any) -> None:  # noqa: A002
        self.id = id
        self.name = name
        self.input = input


class _Usage:
    """Mirrors ``anthropic.types.Usage`` including the prompt-cache token split."""

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _Message:
    """Mirrors ``anthropic.types.Message`` (``content``/``usage``/``stop_reason``)."""

    def __init__(
        self, content: list[Any], usage: _Usage, *, stop_reason: str = "end_turn"
    ) -> None:
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason
        self.role = "assistant"
        self.type = "message"


class _Messages:
    """Fake ``client.messages`` namespace capturing ``create`` kwargs."""

    def __init__(self, message: _Message) -> None:
        self._message = message
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _Message:
        self.seen.update(kwargs)
        return self._message


class _Client:
    """Fake async Anthropic client (``client.messages.create``)."""

    def __init__(self, message: _Message) -> None:
        self.messages = _Messages(message)


def _client(content: list[Any], usage: _Usage | None = None) -> _Client:
    return _Client(_Message(content, usage or _Usage(11, 7)))


def _req(**kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": PRICED_MODEL,
        "messages": [InferenceMessage(role="user", content="hello")],
    }
    base.update(kw)
    return InferenceRequest(**base)


# --------------------------------------------------------- request projection
def test_request_projection_full_envelope() -> None:
    """The full envelope projects to the exact ``messages.create`` kwarg shapes."""
    client = _client([_TextBlock("ok")])
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)
    req = _req(
        messages=[
            InferenceMessage(role="system", content="be brief"),
            InferenceMessage(role="system", content="answer in english"),
            InferenceMessage(role="user", content="what are the rates?"),
            InferenceMessage(role="assistant", content="let me check"),
            InferenceMessage(role="tool", content='{"rate": 5}', tool_call_id="tc_1"),
        ],
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[
            BoundTool(
                name="lookup",
                description="look something up",
                args_json_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
                read_only=True,
            )
        ],
        generation_params={"temperature": 0.2, "top_p": 0.9, "max_tokens": 77},
    )
    run_async(mgr.generate(req))
    seen = client.messages.seen

    # Exact kwarg surface: nothing extra leaks into the provider call.
    assert set(seen) == {
        "model",
        "max_tokens",
        "messages",
        "system",
        "temperature",
        "top_p",
        "tools",
    }
    assert seen["model"] == PRICED_MODEL
    assert seen["max_tokens"] == 77
    assert seen["temperature"] == 0.2
    assert seen["top_p"] == 0.9
    # System turns are split out of messages and joined with a blank line.
    assert seen["system"] == "be brief\n\nanswer in english"
    # Tool-role turns become a tool_result content block on a USER turn.
    assert seen["messages"] == [
        {"role": "user", "content": "what are the rates?"},
        {"role": "assistant", "content": "let me check"},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tc_1",
                    "content": '{"rate": 5}',
                }
            ],
        },
    ]
    # Bound tools project to Anthropic's name/description/input_schema shape.
    (tool,) = seen["tools"]
    assert set(tool) == {"name", "description", "input_schema"}
    assert tool["name"] == "lookup"
    assert tool["input_schema"]["properties"]["q"] == {"type": "string"}


def test_structured_request_forces_synthetic_tool_choice() -> None:
    """STRUCTURED_OUTPUT forces ``tool_choice`` onto the synthetic schema tool."""
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    client = _client(
        [
            _ToolUseBlock(
                id="t1", name="__structured_output__", input={"city": "Lumbini"}
            )
        ]
    )
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(
        mgr.generate(
            _req(
                response_format=ResponseFormat.STRUCTURED_OUTPUT,
                output_json_schema=schema,
            )
        )
    )
    seen = client.messages.seen
    assert seen["tool_choice"] == {"type": "tool", "name": "__structured_output__"}
    assert seen["tools"][0]["input_schema"] == schema
    # The synthetic tool's input IS the structured result, never a tool call.
    assert resp.output_structured == {"city": "Lumbini"}
    assert resp.tool_calls == []


# ------------------------------------------------------ response normalization
def test_text_blocks_join_and_cache_usage_bills_reads_at_discount() -> None:
    """Text blocks join; input_tokens stays the FULL total but cache tokens bill cheap.

    The cross-provider cost-accounting fix (C2): ``input_tokens`` remains the full
    total (uncached + cache_read + cache_creation) for back-compat, but ``cost`` now
    bills cache READS at Anthropic's 0.1x read rate and cache WRITES at the 1.25x (5m)
    premium — NOT the previous full 1.0x on every folded token — and the discount is
    surfaced on ``metadata``. (Previously this asserted full-rate cost on all 42
    tokens; that was the ~10x mis-billing this item fixes.)
    """
    client = _client(
        [_TextBlock("first"), _TextBlock("second")],
        _Usage(
            10,
            5,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=2,
        ),
    )
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(mgr.generate(_req()))

    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "first\nsecond"
    # FULL total preserved for downstream consumers: 10 + 30 read + 2 creation.
    assert resp.input_tokens == 42
    assert resp.output_tokens == 5

    price = pricing.price_for(f"anthropic:{PRICED_MODEL}")
    # Uncached input (10) + output (5) at full rate, then 30 reads at 0.1x and
    # 2 writes at the 5m 1.25x premium of the input rate.
    expected = price.cost(input_tokens=10, output_tokens=5) + (
        price.input_per_1k / 1000.0
    ) * (30 * 0.1 + 2 * 1.25)
    assert resp.cost == pytest.approx(expected)
    assert resp.cost > 0.0
    # The discount must be strictly cheaper than the old full-rate-on-everything math.
    full_rate = price.cost(input_tokens=42, output_tokens=5)
    assert resp.cost < full_rate
    # Additive metadata carries the cache breakdown + dollar savings.
    assert resp.metadata["cache_read_tokens"] == 30
    assert resp.metadata["cache_creation_tokens"] == 2
    expected_savings = (price.input_per_1k / 1000.0) * (
        (30 + 2) - (30 * 0.1 + 2 * 1.25)
    )
    assert resp.metadata["cache_savings_usd"] == pytest.approx(expected_savings)


def test_tool_use_blocks_normalize_ids_args_and_execute() -> None:
    """tool_use blocks map to ToolCallRecords (id fallback, args coercion) and run."""
    executed: list[tuple[str, dict[str, Any]]] = []

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        executed.append((name, args))
        return ToolReturnRecord(tool_call_id="x", tool_name=name, content="ok")

    client = _client(
        [
            _TextBlock("calling tools"),
            _ToolUseBlock(id="toolu_1", name="lookup", input={"q": "rates"}),
            # Missing id and a non-dict input: both must be repaired, not crash.
            _ToolUseBlock(id=None, name="fetch", input="not-a-dict"),
        ],
        _Usage(9, 4),
    )
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(
        mgr.generate(
            _req(
                response_format=ResponseFormat.AUTO_TOOLS,
                bound_tools=[
                    BoundTool(name="lookup", description="d"),
                    BoundTool(name="fetch", description="d"),
                ],
                tool_executor=executor,
            )
        )
    )

    assert resp.status == InferenceStatus.SUCCESS
    assert [c.tool_name for c in resp.tool_calls] == ["lookup", "fetch"]
    assert resp.tool_calls[0].tool_call_id == "toolu_1"
    assert resp.tool_calls[0].args == {"q": "rates"}
    # A missing block id is replaced with a generated one (never empty).
    assert resp.tool_calls[1].tool_call_id
    # A non-dict input is coerced to an empty args dict.
    assert resp.tool_calls[1].args == {}
    # Calls were executed and the returns paired by the call ids.
    assert executed == [("lookup", {"q": "rates"}), ("fetch", {})]
    assert [r.tool_call_id for r in resp.tool_returns] == [
        c.tool_call_id for c in resp.tool_calls
    ]
    # A tool-calling turn is not a final answer even when text blocks are present.
    assert resp.output_text is None


# ----------------------------------------------------------------- streaming
class _Stream:
    """Mirrors the ``messages.stream`` context manager (text_stream + final)."""

    def __init__(self, deltas: list[str], final: _Message) -> None:
        self._deltas = deltas
        self._final = final

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    @property
    def text_stream(self) -> Any:
        async def _gen() -> Any:
            for piece in self._deltas:
                yield piece

        return _gen()

    async def get_final_message(self) -> _Message:
        return self._final


class _StreamClient:
    """Fake client exposing ``messages.stream`` (and a failing ``create``)."""

    class _M:
        def __init__(self, deltas: list[str], final: _Message) -> None:
            self._deltas = deltas
            self._final = final
            self.seen: dict[str, Any] = {}

        def stream(self, **kwargs: Any) -> _Stream:
            self.seen.update(kwargs)
            return _Stream(self._deltas, self._final)

    def __init__(self, deltas: list[str], final: _Message) -> None:
        self.messages = _StreamClient._M(deltas, final)


def test_stream_event_sequence_yields_deltas_then_final() -> None:
    """The stream contract: every text delta in order, then ONE final response."""
    final = _Message([_TextBlock("Hello Nepal")], _Usage(6, 3))
    client = _StreamClient(["Hel", "lo ", "Nepal"], final)
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)

    async def _collect() -> list[Any]:
        return [piece async for piece in mgr.generate_stream(_req())]

    pieces = run_async(_collect())
    assert pieces[:-1] == ["Hel", "lo ", "Nepal"]
    last = pieces[-1]
    assert isinstance(last, InferenceResponse)
    assert last.status == InferenceStatus.SUCCESS
    assert last.output_text == "Hello Nepal"
    assert last.input_tokens == 6 and last.output_tokens == 3


def test_stream_failure_degrades_to_buffered_generate() -> None:
    """A mid-stream failure degrades to ONE buffered generate (contract holds)."""

    class _BrokenStreamClient:
        class _M:
            def stream(self, **kwargs: Any) -> Any:
                raise RuntimeError("stream transport down")

            async def create(self, **kwargs: Any) -> _Message:
                return _Message([_TextBlock("buffered fallback")], _Usage(2, 2))

        def __init__(self) -> None:
            self.messages = _BrokenStreamClient._M()

    mgr = AnthropicClientManager(model=PRICED_MODEL, client=_BrokenStreamClient())

    async def _collect() -> list[Any]:
        return [piece async for piece in mgr.generate_stream(_req())]

    pieces = run_async(_collect())
    assert len(pieces) == 1
    assert pieces[0].status == InferenceStatus.SUCCESS
    assert pieces[0].output_text == "buffered fallback"


# ---------------------------------------------------------- error normalization
#: (SDK exception class name, status_code attr, expected code, expected retryable).
#: Mirrors anthropic 0.106.0's error taxonomy. ``OverloadedError`` (HTTP 529) is
#: intentionally ABSENT from the adapter's name map — the ``status >= 500``
#: fallback is the load-bearing path for it, which this table pins.
_ERROR_CONTRACT: list[tuple[str, int | None, InferenceErrorCode, bool]] = [
    ("RateLimitError", 429, InferenceErrorCode.RATE_LIMITED, True),
    ("AuthenticationError", 401, InferenceErrorCode.AUTH, False),
    ("PermissionDeniedError", 403, InferenceErrorCode.AUTH, False),
    ("BadRequestError", 400, InferenceErrorCode.INVALID_REQUEST, False),
    ("UnprocessableEntityError", 422, InferenceErrorCode.INVALID_REQUEST, False),
    ("NotFoundError", 404, InferenceErrorCode.INVALID_REQUEST, False),
    ("InternalServerError", 500, InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
    ("OverloadedError", 529, InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
    ("APIStatusError", 429, InferenceErrorCode.RATE_LIMITED, True),
    ("APITimeoutError", None, InferenceErrorCode.TIMEOUT, True),
    ("APIConnectionError", None, InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
]


@pytest.mark.parametrize(
    ("name", "status_code", "expected_code", "expected_retryable"),
    _ERROR_CONTRACT,
    ids=[row[0] + (f"_{row[1]}" if row[1] else "") for row in _ERROR_CONTRACT],
)
def test_sdk_error_shapes_normalize_to_failed(
    name: str,
    status_code: int | None,
    expected_code: InferenceErrorCode,
    expected_retryable: bool,
) -> None:
    """Every SDK error shape becomes FAILED with the right code; generate never raises."""
    attrs: dict[str, Any] = {"status_code": status_code} if status_code else {}
    exc_cls = type(name, (Exception,), attrs)

    class _BoomClient:
        class _M:
            async def create(self, **kwargs: Any) -> Any:
                raise exc_cls("provider said no")

        def __init__(self) -> None:
            self.messages = _BoomClient._M()

    mgr = AnthropicClientManager(model=PRICED_MODEL, client=_BoomClient())
    resp = run_async(mgr.generate(_req()))

    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == expected_code
    assert resp.error.retryable is expected_retryable
    assert resp.model_path == f"anthropic:{PRICED_MODEL}"
