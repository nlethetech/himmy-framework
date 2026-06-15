"""OpenAI contract, layer 2 (SDK required): real SDK types through the adapter.

Skipped wholesale when the ``openai`` extra is absent (the always-run shape layer
is ``test_openai_shapes.py``). With the SDK installed these tests:

* pin the SDK major version the contract was verified against, so a major bump
  fails loudly and forces a re-verification of the fixture shapes;
* validate the adapter's ``chat.completions.create`` kwargs against the SDK's
  OWN request TypedDicts (``CompletionCreateParamsNonStreaming`` /
  ``ChatCompletionFunctionToolParam`` / ``FunctionDefinition`` /
  ``ChatCompletionToolMessageParam``), so a renamed or removed request field
  fails here before it fails in production;
* feed REAL ``ChatCompletion`` / ``ChatCompletionMessage`` / tool-call /
  ``CompletionUsage`` / ``ChatCompletionChunk`` instances through the
  normalization paths (text, tool_calls + execution, usage, streaming);
* raise REAL SDK exceptions (constructed with real ``httpx`` responses, incl.
  429 rate-limit) inside ``generate`` and assert the normalized FAILED contract.

The network is never touched: a fake transport (injected ``client``) carries the
real SDK objects.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference.models import (
    BoundTool,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ResponseFormat,
    ToolReturnRecord,
)
from himmy.services.inference.openai_manager import OpenAIClientManager
from tests.conftest import run_async

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

#: The SDK version this contract file was authored and verified against.
#: On a MAJOR bump: re-verify every fixture in tests/contracts/ against the new
#: request/response shapes, then update this pin.
_VERIFIED_VERSION = "2.33.0"
_VERIFIED_MAJOR = 2

MODEL = "gpt-4o-mini"


def _typed_dict_keys(td: Any) -> set[str]:
    """All declared keys of a TypedDict (required + optional, incl. inherited)."""
    return set(td.__required_keys__) | set(td.__optional_keys__)


def _real_completion(
    *,
    content: str | None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> Any:
    """Build a REAL ``ChatCompletion`` (validated by the SDK's pydantic models)."""
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    return ChatCompletion(
        id="chatcmpl-contract",
        created=1,
        model=MODEL,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,  # type: ignore[arg-type]
                message=ChatCompletionMessage(
                    role="assistant", content=content, tool_calls=tool_calls
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class _Transport:
    """Fake transport returning (or raising) real SDK objects from ``create``."""

    class _Completions:
        def __init__(self, result: Any) -> None:
            self._result = result
            self.seen: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.seen.update(kwargs)
            if isinstance(self._result, BaseException):
                raise self._result
            return self._result

    class _Chat:
        def __init__(self, result: Any) -> None:
            self.completions = _Transport._Completions(result)

    def __init__(self, result: Any) -> None:
        self.chat = _Transport._Chat(result)


def _req(**kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": MODEL,
        "messages": [InferenceMessage(role="user", content="hello")],
    }
    base.update(kw)
    return InferenceRequest(**base)


# ------------------------------------------------------------------ version pin
def test_contract_verified_against_installed_sdk_major() -> None:
    """The installed SDK major must match the version this contract was built on."""
    major = int(openai.__version__.split(".")[0])
    assert major == _VERIFIED_MAJOR, (
        f"openai SDK major changed: contract verified against {_VERIFIED_VERSION}, "
        f"installed {openai.__version__}. Re-verify the request/response fixture "
        f"shapes in tests/contracts/ and update the pin."
    )


# --------------------------------------------------------- request -> SDK kwargs
def test_generate_kwargs_validate_against_sdk_request_typeddicts() -> None:
    """Every kwarg/shape the adapter sends exists in the SDK's request schema."""
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionFunctionToolParam,
        ChatCompletionToolMessageParam,
        completion_create_params,
    )
    from openai.types.shared_params import FunctionDefinition

    transport = _Transport(_real_completion(content="ok"))
    mgr = OpenAIClientManager(model=MODEL, client=transport)
    run_async(
        mgr.generate(
            _req(
                messages=[
                    InferenceMessage(role="system", content="be brief"),
                    InferenceMessage(role="user", content="rates?"),
                    InferenceMessage(role="assistant", content="checking"),
                    InferenceMessage(
                        role="tool",
                        content="5%",
                        tool_call_id="call_1",
                        name="lookup",
                        metadata={"tool_name": "lookup", "tool_args": {"q": "rates"}},
                    ),
                ],
                response_format=ResponseFormat.AUTO_TOOLS,
                bound_tools=[
                    BoundTool(
                        name="lookup",
                        description="look something up",
                        args_json_schema={"type": "object"},
                    )
                ],
                generation_params={"temperature": 0.1, "top_p": 0.9, "max_tokens": 64},
            )
        )
    )
    seen = transport.chat.completions.seen

    create_keys = _typed_dict_keys(
        completion_create_params.CompletionCreateParamsNonStreaming
    )
    assert set(seen) <= create_keys, f"unknown create kwargs: {set(seen) - create_keys}"
    # The tool-role message must satisfy the SDK's tool-message param shape.
    tool_msg_keys = _typed_dict_keys(ChatCompletionToolMessageParam)
    (tool_msg,) = [m for m in seen["messages"] if m["role"] == "tool"]
    assert set(tool_msg) <= tool_msg_keys
    assert tool_msg["tool_call_id"] == "call_1"
    # The preceding assistant message carries the reconstructed tool_calls array;
    # both the message and its tool_calls entries must satisfy the SDK param shapes.
    assistant_keys = _typed_dict_keys(ChatCompletionAssistantMessageParam)
    (assistant_msg,) = [
        m for m in seen["messages"] if m["role"] == "assistant" and "tool_calls" in m
    ]
    assert set(assistant_msg) <= assistant_keys
    (call,) = assistant_msg["tool_calls"]
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup"
    assert call["function"]["arguments"] == '{"q": "rates"}'
    # Tool definitions must satisfy the SDK's function-tool param shape.
    tool_keys = _typed_dict_keys(ChatCompletionFunctionToolParam)
    fn_keys = _typed_dict_keys(FunctionDefinition)
    for tool in seen["tools"]:
        assert set(tool) <= tool_keys, f"unknown tool keys: {set(tool) - tool_keys}"
        assert set(tool["function"]) <= fn_keys, (
            f"unknown function keys: {set(tool['function']) - fn_keys}"
        )


def test_stream_kwargs_validate_against_sdk_request_typeddicts() -> None:
    """The streaming call's kwargs (stream/stream_options) exist in the SDK schema."""
    from openai.types.chat import completion_create_params

    chunks_done: list[Any] = []

    class _StreamTransport:
        class _Completions:
            def __init__(self) -> None:
                self.seen: dict[str, Any] = {}

            async def create(self, **kwargs: Any) -> Any:
                self.seen.update(kwargs)

                async def _gen() -> Any:
                    for chunk in chunks_done:
                        yield chunk

                return _gen()

        def __init__(self) -> None:
            self.chat = type(
                "c", (), {"completions": _StreamTransport._Completions()}
            )()

    transport = _StreamTransport()
    mgr = OpenAIClientManager(model=MODEL, client=transport)

    async def _drain() -> None:
        async for _ in mgr.generate_stream(_req()):
            pass

    run_async(_drain())
    seen = transport.chat.completions.seen
    create_keys = _typed_dict_keys(
        completion_create_params.CompletionCreateParamsStreaming
    )
    assert set(seen) <= create_keys, f"unknown create kwargs: {set(seen) - create_keys}"
    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


def _real_tool_chunk(
    *,
    index: int,
    id: str | None = None,  # noqa: A002 - mirrors the SDK attribute name
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    """Build a REAL ``ChatCompletionChunk`` carrying one streamed tool-call fragment."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    return ChatCompletionChunk(
        id="chatcmpl-stream",
        created=1,
        model=MODEL,
        object="chat.completion.chunk",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=index,
                            id=id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name=name, arguments=arguments
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def _real_usage_chunk(*, prompt_tokens: int, completion_tokens: int) -> Any:
    """Build a REAL final ``ChatCompletionChunk`` carrying only ``usage``."""
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletionChunk

    return ChatCompletionChunk(
        id="chatcmpl-stream",
        created=1,
        model=MODEL,
        object="chat.completion.chunk",
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def test_real_stream_tool_call_deltas_reassemble_and_execute() -> None:
    """REAL streamed ``delta.tool_calls`` fragments reassemble + execute on stream end.

    The SDK streams a tool call across chunks (id+name first, then the ``arguments``
    JSON in pieces). Through REAL ``ChatCompletionChunk`` / ``ChoiceDeltaToolCall``
    types, the adapter must send the bound ``tools``, reassemble the fragments by
    ``index`` and execute the completed call — the streaming-tools blocker fix.
    """
    executed: list[tuple[str, dict[str, Any]]] = []

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        executed.append((name, args))
        return ToolReturnRecord(
            tool_call_id="x", tool_name=name, content="18C and clear"
        )

    chunks = [
        _real_tool_chunk(index=0, id="call_w", name="get_weather"),
        _real_tool_chunk(index=0, arguments='{"city"'),
        _real_tool_chunk(index=0, arguments=': "Paris"}'),
        _real_usage_chunk(prompt_tokens=20, completion_tokens=6),
    ]

    class _StreamTransport:
        class _Completions:
            def __init__(self) -> None:
                self.seen: dict[str, Any] = {}

            async def create(self, **kwargs: Any) -> Any:
                self.seen.update(kwargs)

                async def _gen() -> Any:
                    for chunk in chunks:
                        yield chunk

                return _gen()

        def __init__(self) -> None:
            self.chat = type(
                "c", (), {"completions": _StreamTransport._Completions()}
            )()

    transport = _StreamTransport()
    mgr = OpenAIClientManager(model=MODEL, client=transport)
    req = _req(
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[BoundTool(name="get_weather", description="weather")],
        tool_executor=executor,
    )

    async def _collect() -> list[Any]:
        return [piece async for piece in mgr.generate_stream(req)]

    pieces = run_async(_collect())

    # Bound tools ride on the stream request (the blocker).
    seen = transport.chat.completions.seen
    assert [t["function"]["name"] for t in seen["tools"]] == ["get_weather"]
    # Only the final response is yielded (a tool turn has no text deltas).
    (final,) = pieces
    assert final.status == InferenceStatus.SUCCESS
    assert [c.tool_name for c in final.tool_calls] == ["get_weather"]
    assert final.tool_calls[0].tool_call_id == "call_w"
    assert final.tool_calls[0].args == {"city": "Paris"}
    assert executed == [("get_weather", {"city": "Paris"})]
    assert final.tool_returns[0].content == "18C and clear"
    assert final.output_text is None
    assert final.input_tokens == 20 and final.output_tokens == 6


# --------------------------------------------------- SDK response -> normalized
def test_real_completion_text_normalizes_with_usage() -> None:
    """A real ChatCompletion maps to text + token usage + priced cost."""
    completion = _real_completion(
        content="hello from gpt", prompt_tokens=120, completion_tokens=12
    )
    mgr = OpenAIClientManager(model=MODEL, client=_Transport(completion))
    resp = run_async(mgr.generate(_req()))

    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "hello from gpt"
    assert resp.input_tokens == 120 and resp.output_tokens == 12
    assert resp.cost > 0.0
    assert resp.model_path == f"openai:{MODEL}"


def test_real_tool_calls_map_and_execute() -> None:
    """Real tool_calls (JSON-string arguments) map to records and execute."""
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )

    executed: list[tuple[str, dict[str, Any]]] = []

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        executed.append((name, args))
        return ToolReturnRecord(tool_call_id="x", tool_name=name, content={"ok": True})

    completion = _real_completion(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_01",
                type="function",
                function=Function(name="lookup", arguments='{"q": "rates"}'),
            )
        ],
    )
    mgr = OpenAIClientManager(model=MODEL, client=_Transport(completion))
    resp = run_async(
        mgr.generate(
            _req(
                response_format=ResponseFormat.AUTO_TOOLS,
                bound_tools=[BoundTool(name="lookup", description="d")],
                tool_executor=executor,
            )
        )
    )

    assert resp.status == InferenceStatus.SUCCESS
    assert [c.tool_name for c in resp.tool_calls] == ["lookup"]
    assert resp.tool_calls[0].tool_call_id == "call_01"
    # The SDK ships arguments as a JSON STRING; the adapter must parse it.
    assert resp.tool_calls[0].args == {"q": "rates"}
    assert executed == [("lookup", {"q": "rates"})]
    assert resp.tool_returns[0].tool_call_id == "call_01"
    # A tool_calls finish is not a final answer.
    assert resp.output_text is None


def test_real_stream_chunks_yield_deltas_then_final_with_usage() -> None:
    """Real ChatCompletionChunks stream deltas; the usage chunk fills tokens."""
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    def _chunk(content: str | None, usage: Any = None) -> Any:
        return ChatCompletionChunk(
            id="chatcmpl-contract",
            created=1,
            model=MODEL,
            object="chat.completion.chunk",
            choices=(
                []
                if content is None
                else [
                    ChunkChoice(
                        index=0, delta=ChoiceDelta(content=content), finish_reason=None
                    )
                ]
            ),
            usage=usage,
        )

    chunks = [
        _chunk("Hel"),
        _chunk("lo "),
        _chunk("Nepal"),
        _chunk(
            None,
            usage=CompletionUsage(
                prompt_tokens=9, completion_tokens=4, total_tokens=13
            ),
        ),
    ]

    class _StreamTransport:
        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                async def _gen() -> Any:
                    for chunk in chunks:
                        yield chunk

                return _gen()

        def __init__(self) -> None:
            self.chat = type(
                "c", (), {"completions": _StreamTransport._Completions()}
            )()

    mgr = OpenAIClientManager(model=MODEL, client=_StreamTransport())

    async def _collect() -> list[Any]:
        return [piece async for piece in mgr.generate_stream(_req())]

    pieces = run_async(_collect())
    assert pieces[:-1] == ["Hel", "lo ", "Nepal"]
    last = pieces[-1]
    assert last.status == InferenceStatus.SUCCESS
    assert last.output_text == "Hello Nepal"
    assert last.input_tokens == 9 and last.output_tokens == 4


# ----------------------------------------------------------- error normalization
def _status_error(exc_cls: Any, status: int, message: str) -> BaseException:
    """Construct a real SDK APIStatusError subclass with a real httpx response."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    exc: BaseException = exc_cls(
        message, response=httpx.Response(status, request=request), body=None
    )
    return exc


def test_real_sdk_errors_normalize_to_failed() -> None:
    """Real SDK exceptions (incl. 429) map to FAILED responses, never raise."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    cases: list[tuple[BaseException, InferenceErrorCode, bool]] = [
        (
            _status_error(openai.RateLimitError, 429, "rate limited"),
            InferenceErrorCode.RATE_LIMITED,
            True,
        ),
        (
            _status_error(openai.AuthenticationError, 401, "bad key"),
            InferenceErrorCode.AUTH,
            False,
        ),
        (
            _status_error(openai.BadRequestError, 400, "bad request"),
            InferenceErrorCode.INVALID_REQUEST,
            False,
        ),
        (
            _status_error(openai.InternalServerError, 500, "boom"),
            InferenceErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
        (
            openai.APITimeoutError(request=request),
            InferenceErrorCode.TIMEOUT,
            True,
        ),
        (
            openai.APIConnectionError(request=request),
            InferenceErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
    ]
    for exc, expected_code, expected_retryable in cases:
        mgr = OpenAIClientManager(model=MODEL, client=_Transport(exc))
        resp = run_async(mgr.generate(_req()))
        label = type(exc).__name__
        assert resp.status == InferenceStatus.FAILED, label
        assert resp.error is not None, label
        assert resp.error.code == expected_code, label
        assert resp.error.retryable is expected_retryable, label
