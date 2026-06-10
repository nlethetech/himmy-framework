"""OpenAI contract, layer 1 (always runs): SDK shape parity without the SDK.

The fixture classes below hand-build objects mirroring the EXACT attribute shapes
the ``openai`` SDK returns — verified against **openai 2.33.0** (``ChatCompletion``
with ``choices[0].message`` carrying ``content``/``tool_calls`` whose
``function.arguments`` is a JSON STRING, ``usage`` with prompt/completion tokens,
``ChatCompletionChunk`` streaming with ``choices[0].delta.content`` and a final
usage-bearing chunk, and the error taxonomy carrying ``status_code``). They are
fed through :class:`OpenAIClientManager` via an injected fake client, pinning:

* request projection — role-preserving messages (tool turns keep
  ``tool_call_id``), function-tool definitions, ``response_format`` for
  JSON_OBJECT / STRUCTURED_OUTPUT, generation params;
* response normalization — text, tool_calls (JSON-string args parsed, invalid
  args and missing ids repaired) + execution, usage and cost, JSON text parsed
  into ``output_structured``;
* streaming — delta-then-final with ``stream_options={"include_usage": True}``,
  mid-stream failure degrading to a buffered ``generate``;
* error normalization — the full class-name taxonomy plus ``status_code``
  fallbacks (429 → RATE_LIMITED, 5xx → PROVIDER_UNAVAILABLE).

Layer 2 (``test_openai_sdk.py``) replays these scenarios with REAL SDK types, so
an SDK bump that changes shapes fails loudly there while this file keeps guarding
the adapter logic everywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference import pricing
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
from himmy.services.inference.openai_manager import OpenAIClientManager
from tests.conftest import run_async

#: A model present in the bundled price table so cost mapping is exercised.
PRICED_MODEL = "gpt-4o-mini"


# ------------------------------------------------------------ fixture SDK shapes
class _Function:
    """Mirrors the tool call's ``function`` (``name`` + JSON-string ``arguments``)."""

    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """Mirrors ``ChatCompletionMessageFunctionToolCall`` (``id/type/function``)."""

    type = "function"

    def __init__(self, *, id: str | None, function: _Function | None) -> None:  # noqa: A002
        self.id = id
        self.function = function


class _ChatMessage:
    """Mirrors ``ChatCompletionMessage`` (``role/content/tool_calls``)."""

    role = "assistant"

    def __init__(
        self, content: str | None, tool_calls: list[_ToolCall] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    """Mirrors ``chat_completion.Choice`` (``index/finish_reason/message``)."""

    def __init__(self, message: _ChatMessage, finish_reason: str = "stop") -> None:
        self.index = 0
        self.finish_reason = finish_reason
        self.message = message


class _Usage:
    """Mirrors ``CompletionUsage`` (``prompt/completion/total`` tokens)."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Completion:
    """Mirrors ``ChatCompletion`` (``choices`` + ``usage``)."""

    def __init__(self, choices: list[_Choice], usage: _Usage | None) -> None:
        self.choices = choices
        self.usage = usage


class _Completions:
    """Fake ``client.chat.completions`` namespace capturing ``create`` kwargs."""

    def __init__(self, completion: Any) -> None:
        self._completion = completion
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.seen.update(kwargs)
        return self._completion


class _Chat:
    def __init__(self, completion: Any) -> None:
        self.completions = _Completions(completion)


class _Client:
    """Fake async OpenAI client (``client.chat.completions.create``)."""

    def __init__(self, completion: Any) -> None:
        self.chat = _Chat(completion)


def _client(
    message: _ChatMessage,
    *,
    usage: _Usage | None = None,
    finish_reason: str = "stop",
) -> _Client:
    return _Client(
        _Completion([_Choice(message, finish_reason)], usage or _Usage(11, 7))
    )


def _req(**kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": PRICED_MODEL,
        "messages": [InferenceMessage(role="user", content="hello")],
    }
    base.update(kw)
    return InferenceRequest(**base)


# --------------------------------------------------------- request projection
def test_request_projection_full_envelope() -> None:
    """The full envelope projects to the exact ``chat.completions.create`` kwargs."""
    client = _client(_ChatMessage("ok"))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    req = _req(
        messages=[
            InferenceMessage(role="system", content="be brief"),
            InferenceMessage(role="user", content="what are the rates?"),
            InferenceMessage(role="assistant", content="let me check"),
            InferenceMessage(role="tool", content='{"rate": 5}', tool_call_id="call_9"),
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
    seen = client.chat.completions.seen

    # Exact kwarg surface: nothing extra leaks into the provider call.
    assert set(seen) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "tools",
    }
    assert seen["model"] == PRICED_MODEL
    assert seen["temperature"] == 0.2
    assert seen["top_p"] == 0.9
    assert seen["max_tokens"] == 77
    # Roles are preserved; tool turns keep role="tool" + tool_call_id.
    assert seen["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "what are the rates?"},
        {"role": "assistant", "content": "let me check"},
        {"role": "tool", "content": '{"rate": 5}', "tool_call_id": "call_9"},
    ]
    # Bound tools project to OpenAI's function-tool shape.
    (tool,) = seen["tools"]
    assert tool["type"] == "function"
    assert set(tool["function"]) == {"name", "description", "parameters"}
    assert tool["function"]["name"] == "lookup"
    assert tool["function"]["parameters"]["properties"]["q"] == {"type": "string"}


def test_json_object_and_structured_response_format_kwargs() -> None:
    """JSON_OBJECT / STRUCTURED_OUTPUT map to the two ``response_format`` shapes."""
    client = _client(_ChatMessage('{"ok": true}'))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    run_async(mgr.generate(_req(response_format=ResponseFormat.JSON_OBJECT)))
    assert client.chat.completions.seen["response_format"] == {"type": "json_object"}

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    client = _client(_ChatMessage('{"city": "Pokhara"}'))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(
        mgr.generate(
            _req(
                response_format=ResponseFormat.STRUCTURED_OUTPUT,
                output_json_schema=schema,
            )
        )
    )
    assert client.chat.completions.seen["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "structured_output", "schema": schema},
    }
    assert resp.output_structured == {"city": "Pokhara"}


# ------------------------------------------------------ response normalization
def test_text_reply_maps_usage_and_cost() -> None:
    """``choices[0].message.content`` + usage map to text/tokens/priced cost."""
    client = _client(_ChatMessage("नमस्ते from gpt"), usage=_Usage(120, 12))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(mgr.generate(_req()))

    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "नमस्ते from gpt"
    assert resp.input_tokens == 120 and resp.output_tokens == 12
    assert resp.model_path == f"openai:{PRICED_MODEL}"
    assert resp.provider_name == "openai"
    expected = pricing.price_for(f"openai:{PRICED_MODEL}").cost(
        input_tokens=120, output_tokens=12
    )
    assert resp.cost == pytest.approx(expected)
    assert resp.cost > 0.0


def test_tool_calls_parse_json_string_args_and_execute() -> None:
    """tool_calls map to ToolCallRecords (JSON-string args, repairs) and execute."""
    executed: list[tuple[str, dict[str, Any]]] = []

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        executed.append((name, args))
        return ToolReturnRecord(tool_call_id="x", tool_name=name, content="ok")

    message = _ChatMessage(
        None,
        tool_calls=[
            _ToolCall(id="call_1", function=_Function("lookup", '{"q": "rates"}')),
            # Missing id + INVALID JSON arguments: both repaired, not crashed on.
            _ToolCall(id=None, function=_Function("fetch", "{not json")),
            # A tool call with no function payload is skipped entirely.
            _ToolCall(id="call_3", function=None),
        ],
    )
    client = _client(message, finish_reason="tool_calls")
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
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
    assert resp.tool_calls[0].tool_call_id == "call_1"
    assert resp.tool_calls[0].args == {"q": "rates"}
    assert resp.tool_calls[1].tool_call_id  # generated, never empty
    assert resp.tool_calls[1].args == {}  # invalid JSON arguments → {}
    assert executed == [("lookup", {"q": "rates"}), ("fetch", {})]
    assert [r.tool_call_id for r in resp.tool_returns] == [
        c.tool_call_id for c in resp.tool_calls
    ]
    # A tool-calling turn is not a final answer.
    assert resp.output_text is None


def test_non_json_text_under_json_format_yields_no_structured() -> None:
    """A non-JSON reply under JSON_OBJECT keeps the text but no structured value."""
    client = _client(_ChatMessage("sorry, plain prose"))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(mgr.generate(_req(response_format=ResponseFormat.JSON_OBJECT)))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "sorry, plain prose"
    assert resp.output_structured is None


# ----------------------------------------------------------------- streaming
class _ChunkDelta:
    """Mirrors ``ChoiceDelta`` (``content``)."""

    def __init__(self, content: str | None) -> None:
        self.content = content


class _ChunkChoice:
    """Mirrors ``chat_completion_chunk.Choice`` (``delta``)."""

    def __init__(self, delta: _ChunkDelta | None) -> None:
        self.index = 0
        self.delta = delta
        self.finish_reason = None


class _Chunk:
    """Mirrors ``ChatCompletionChunk`` (``choices`` + optional final ``usage``)."""

    def __init__(self, choices: list[_ChunkChoice], usage: Any = None) -> None:
        self.choices = choices
        self.usage = usage


class _ChunkUsage:
    """Mirrors the usage object on the final ``include_usage`` chunk."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


async def _aiter(chunks: list[_Chunk]) -> Any:
    for chunk in chunks:
        yield chunk


def test_stream_chunk_sequence_yields_deltas_then_final_with_usage() -> None:
    """Delta chunks stream in order; the empty-choices usage chunk fills tokens."""
    chunks = [
        _Chunk([_ChunkChoice(_ChunkDelta("Hel"))]),
        _Chunk([_ChunkChoice(_ChunkDelta("lo "))]),
        _Chunk([_ChunkChoice(_ChunkDelta(None))]),  # role/finish chunks carry no text
        _Chunk([_ChunkChoice(_ChunkDelta("Nepal"))]),
        _Chunk([], usage=_ChunkUsage(9, 4)),  # final include_usage chunk
    ]

    class _StreamCompletions:
        def __init__(self) -> None:
            self.seen: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.seen.update(kwargs)
            return _aiter(chunks)

    class _StreamClient:
        def __init__(self) -> None:
            self.chat = type("c", (), {"completions": _StreamCompletions()})()

    client = _StreamClient()
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)

    async def _collect() -> list[Any]:
        return [piece async for piece in mgr.generate_stream(_req())]

    pieces = run_async(_collect())
    assert pieces[:-1] == ["Hel", "lo ", "Nepal"]
    last = pieces[-1]
    assert isinstance(last, InferenceResponse)
    assert last.status == InferenceStatus.SUCCESS
    assert last.output_text == "Hello Nepal"
    assert last.input_tokens == 9 and last.output_tokens == 4
    # The stream request opts into usage chunks.
    seen = client.chat.completions.seen
    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


def test_stream_failure_degrades_to_buffered_generate() -> None:
    """A failing stream degrades to ONE buffered generate (contract holds)."""

    class _FlakyCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            if kwargs.get("stream"):
                raise RuntimeError("stream transport down")
            return _Completion(
                [_Choice(_ChatMessage("buffered fallback"))], _Usage(2, 2)
            )

    class _FlakyClient:
        def __init__(self) -> None:
            self.chat = type("c", (), {"completions": _FlakyCompletions()})()

    mgr = OpenAIClientManager(model=PRICED_MODEL, client=_FlakyClient())

    async def _collect() -> list[Any]:
        return [piece async for piece in mgr.generate_stream(_req())]

    pieces = run_async(_collect())
    assert len(pieces) == 1
    assert pieces[0].status == InferenceStatus.SUCCESS
    assert pieces[0].output_text == "buffered fallback"


# ---------------------------------------------------------- error normalization
#: (SDK exception class name, status_code attr, expected code, expected retryable).
#: Mirrors openai 2.33.0's error taxonomy; a 5xx APIStatusError relies on the
#: ``status >= 500`` fallback, which this table pins.
_ERROR_CONTRACT: list[tuple[str, int | None, InferenceErrorCode, bool]] = [
    ("RateLimitError", 429, InferenceErrorCode.RATE_LIMITED, True),
    ("AuthenticationError", 401, InferenceErrorCode.AUTH, False),
    ("PermissionDeniedError", 403, InferenceErrorCode.AUTH, False),
    ("BadRequestError", 400, InferenceErrorCode.INVALID_REQUEST, False),
    ("UnprocessableEntityError", 422, InferenceErrorCode.INVALID_REQUEST, False),
    ("NotFoundError", 404, InferenceErrorCode.INVALID_REQUEST, False),
    ("InternalServerError", 500, InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
    ("APIStatusError", 503, InferenceErrorCode.PROVIDER_UNAVAILABLE, True),
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

    class _BoomCompletions:
        async def create(self, **kwargs: Any) -> Any:
            raise exc_cls("provider said no")

    class _BoomClient:
        def __init__(self) -> None:
            self.chat = type("c", (), {"completions": _BoomCompletions()})()

    mgr = OpenAIClientManager(model=PRICED_MODEL, client=_BoomClient())
    resp = run_async(mgr.generate(_req()))

    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == expected_code
    assert resp.error.retryable is expected_retryable
    assert resp.model_path == f"openai:{PRICED_MODEL}"
