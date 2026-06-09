"""Tests for the DIRECT Anthropic client manager (all offline, fake SDK client).

A fake async client (``client.messages.create`` / ``client.messages.stream``) is
injected so nothing touches the network or needs a key. Covers text / json / structured
/ tool / stream paths, usage+cost extraction, and SDK-error → FAILED normalization.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference import pricing
from himmy.services.inference.anthropic_manager import (
    AnthropicClientManager,
    _normalize_anthropic_error,
)
from himmy.services.inference.models import (
    BoundTool,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ResponseFormat,
    ToolReturnRecord,
)
from tests.conftest import run_async

PRICED_MODEL = "claude-3-5-haiku-latest"


# ---------------------------------------------------------------- fake SDK types
class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int, **kw: Any) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.__dict__.update(kw)


class _Message:
    def __init__(self, content: list[_Block], usage: _Usage) -> None:
        self.content = content
        self.usage = usage


class _FakeMessages:
    def __init__(self, message: _Message) -> None:
        self._message = message
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _Message:
        self.seen.update(kwargs)
        return self._message


class _FakeClient:
    def __init__(self, message: _Message) -> None:
        self.messages = _FakeMessages(message)


def _client(
    content: list[_Block], *, input_tokens: int = 11, output_tokens: int = 7
) -> _FakeClient:
    return _FakeClient(_Message(content, _Usage(input_tokens, output_tokens)))


def _req(**kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": PRICED_MODEL,
        "messages": [
            InferenceMessage(role="system", content="be brief"),
            InferenceMessage(role="user", content="hello"),
        ],
    }
    base.update(kw)
    return InferenceRequest(**base)


# ---------------------------------------------------------------------- text
def test_text_reply_maps_to_success_with_usage_and_cost() -> None:
    """A text block maps to SUCCESS with token counts and a non-zero priced cost."""
    client = _client(
        [_Block(type="text", text="नमस्ते from claude")],
        input_tokens=120,
        output_tokens=12,
    )
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(
        mgr.generate(_req(generation_params={"temperature": 0.1, "max_tokens": 50}))
    )

    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "नमस्ते from claude"
    assert resp.input_tokens == 120 and resp.output_tokens == 12
    assert resp.model_path == "anthropic:claude-3-5-haiku-latest"
    assert resp.provider_name == "anthropic"
    # cost matches the bundled price table for this priced model.
    expected = pricing.price_for("anthropic:claude-3-5-haiku-latest").cost(
        input_tokens=120, output_tokens=12
    )
    assert resp.cost == pytest.approx(expected)
    assert resp.cost > 0.0
    # system is split out; the user turn is projected into messages; max_tokens honored.
    seen = client.messages.seen
    assert seen["system"] == "be brief"
    assert seen["max_tokens"] == 50
    assert seen["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["temperature"] == 0.1


def test_max_tokens_defaults_when_unspecified() -> None:
    """Anthropic requires max_tokens; the manager supplies its floor by default."""
    client = _client([_Block(type="text", text="hi")])
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client, max_tokens=256)
    run_async(mgr.generate(_req()))
    assert client.messages.seen["max_tokens"] == 256


# --------------------------------------------------------------------- json
def test_json_object_uses_structured_tool_and_returns_object() -> None:
    """JSON_OBJECT forces the synthetic tool; its input becomes the structured result."""
    client = _client(
        [
            _Block(
                type="tool_use",
                id="t1",
                name="__structured_output__",
                input={"answer": 42},
            )
        ]
    )
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(mgr.generate(_req(response_format=ResponseFormat.JSON_OBJECT)))

    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_structured == {"answer": 42}
    assert resp.output_text == '{"answer": 42}'
    # the synthetic tool is NOT surfaced as a real tool call.
    assert resp.tool_calls == []
    seen = client.messages.seen
    assert seen["tool_choice"] == {"type": "tool", "name": "__structured_output__"}
    assert seen["tools"][0]["name"] == "__structured_output__"


# --------------------------------------------------------------- structured
def test_structured_output_uses_request_schema() -> None:
    """STRUCTURED_OUTPUT hands the request schema to the synthetic tool's input_schema."""
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    client = _client(
        [
            _Block(
                type="tool_use",
                id="t1",
                name="__structured_output__",
                input={"city": "Pokhara"},
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
    assert resp.output_structured == {"city": "Pokhara"}
    assert client.messages.seen["tools"][0]["input_schema"] == schema


# ------------------------------------------------------------------- tools
def test_tool_use_block_is_extracted_and_executed() -> None:
    """A tool_use block becomes a ToolCallRecord and is executed via the executor."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        calls.append((name, args))
        return ToolReturnRecord(tool_call_id="x", tool_name=name, content={"ran": True})

    client = _client(
        [_Block(type="tool_use", id="call_1", name="lookup", input={"q": "rates"})]
    )
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=client)
    req = _req(
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[
            BoundTool(name="lookup", description="look something up", read_only=True)
        ],
        tool_executor=executor,
    )
    resp = run_async(mgr.generate(req))

    assert resp.status == InferenceStatus.SUCCESS
    assert [c.tool_name for c in resp.tool_calls] == ["lookup"]
    assert resp.tool_calls[0].args == {"q": "rates"}
    assert resp.tool_calls[0].tool_call_id == "call_1"
    assert len(resp.tool_returns) == 1
    assert resp.tool_returns[0].content == {"ran": True}
    # a tool call is not a final answer
    assert resp.output_text is None
    assert calls == [("lookup", {"q": "rates"})]
    # the bound tool is advertised in the payload
    assert client.messages.seen["tools"][0]["name"] == "lookup"


# ------------------------------------------------------------------ stream
def test_generate_stream_yields_deltas_then_response() -> None:
    """generate_stream yields text deltas, then a final InferenceResponse."""

    class _Stream:
        def __init__(self, msg: _Message) -> None:
            self._msg = msg

        async def __aenter__(self) -> _Stream:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        @property
        def text_stream(self) -> Any:
            async def _gen() -> Any:
                for piece in ["Hel", "lo!"]:
                    yield piece

            return _gen()

        async def get_final_message(self) -> _Message:
            return self._msg

    class _StreamMessages:
        def __init__(self, msg: _Message) -> None:
            self._msg = msg

        def stream(self, **kwargs: Any) -> _Stream:
            return _Stream(self._msg)

    class _StreamClient:
        def __init__(self, msg: _Message) -> None:
            self.messages = _StreamMessages(msg)

    msg = _Message([_Block(type="text", text="Hello!")], _Usage(3, 2))
    mgr = AnthropicClientManager(model=PRICED_MODEL, client=_StreamClient(msg))

    async def _collect() -> tuple[list[str], Any]:
        deltas: list[str] = []
        final: Any = None
        async for piece in mgr.generate_stream(_req()):
            if isinstance(piece, str):
                deltas.append(piece)
            else:
                final = piece
        return deltas, final

    deltas, final = run_async(_collect())
    assert deltas == ["Hel", "lo!"]
    assert final is not None
    assert final.status == InferenceStatus.SUCCESS
    assert final.output_text == "Hello!"


# ------------------------------------------------------- error normalization
def test_sdk_error_is_normalized_to_failed_never_raises() -> None:
    """An exception from messages.create becomes a FAILED response (never raises)."""

    class _BoomClient:
        class _M:
            async def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("kaboom")

        def __init__(self) -> None:
            self.messages = _BoomClient._M()

    mgr = AnthropicClientManager(model=PRICED_MODEL, client=_BoomClient())
    resp = run_async(mgr.generate(_req()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.UNKNOWN
    assert resp.model_path == "anthropic:claude-3-5-haiku-latest"


def test_rate_limit_error_is_retryable() -> None:
    """A RateLimitError-named exception maps to a retryable RATE_LIMITED code."""

    class RateLimitError(Exception):
        pass

    err = _normalize_anthropic_error(RateLimitError("slow down"))
    assert err.code == InferenceErrorCode.RATE_LIMITED
    assert err.retryable is True


def test_auth_error_is_not_retryable() -> None:
    """An AuthenticationError-named exception maps to a non-retryable AUTH code."""

    class AuthenticationError(Exception):
        pass

    err = _normalize_anthropic_error(AuthenticationError("bad key"))
    assert err.code == InferenceErrorCode.AUTH
    assert err.retryable is False


def test_status_code_429_overrides_to_rate_limited() -> None:
    """A generic APIStatusError carrying status 429 still maps to RATE_LIMITED."""

    class APIStatusError(Exception):
        status_code = 429

    err = _normalize_anthropic_error(APIStatusError("too many"))
    assert err.code == InferenceErrorCode.RATE_LIMITED
    assert err.retryable is True
