"""Anthropic contract, layer 2 (SDK required): real SDK types through the adapter.

Skipped wholesale when the ``anthropic`` extra is absent (the always-run shape
layer is ``test_anthropic_shapes.py``). With the SDK installed these tests:

* pin the SDK major version the contract was verified against, so a major bump
  fails loudly and forces a re-verification of the fixture shapes;
* validate the adapter's ``messages.create`` kwargs against the SDK's OWN request
  TypedDicts (``MessageCreateParamsNonStreaming`` / ``MessageParam`` /
  ``ToolParam`` / ``ToolResultBlockParam`` / ``ToolChoiceToolParam``), so a
  renamed or removed request field fails here before it fails in production;
* feed REAL ``Message`` / ``TextBlock`` / ``ToolUseBlock`` / ``Usage`` instances
  through the normalization path (text, tool_use + execution, cache-token
  folding, ``stop_reason``-driven semantics);
* raise REAL SDK exceptions (constructed with real ``httpx`` responses, incl.
  429 rate-limit and 529 overloaded) inside ``generate`` and assert the
  normalized FAILED contract.

The network is never touched: a fake transport (injected ``client``) carries the
real SDK objects.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference.anthropic_manager import AnthropicClientManager
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

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

#: The SDK version this contract file was authored and verified against.
#: On a MAJOR bump: re-verify every fixture in tests/contracts/ against the new
#: request/response shapes, then update this pin.
_VERIFIED_VERSION = "0.106.0"
_VERIFIED_MAJOR = 0

MODEL = "claude-3-5-haiku-latest"


def _typed_dict_keys(td: Any) -> set[str]:
    """All declared keys of a TypedDict (required + optional, incl. inherited)."""
    return set(td.__required_keys__) | set(td.__optional_keys__)


def _real_message(
    content: list[Any],
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int | None = None,
    cache_creation: int | None = None,
) -> Any:
    """Build a REAL ``anthropic.types.Message`` (validated by the SDK's models)."""
    from anthropic.types import Message, Usage

    return Message(
        id="msg_contract",
        model=MODEL,
        role="assistant",
        type="message",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        content=content,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        ),
    )


class _Transport:
    """Fake transport returning (or raising) real SDK objects from ``create``."""

    class _M:
        def __init__(self, result: Any) -> None:
            self._result = result
            self.seen: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.seen.update(kwargs)
            if isinstance(self._result, BaseException):
                raise self._result
            return self._result

    def __init__(self, result: Any) -> None:
        self.messages = _Transport._M(result)


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
    major = int(anthropic.__version__.split(".")[0])
    assert major == _VERIFIED_MAJOR, (
        f"anthropic SDK major changed: contract verified against "
        f"{_VERIFIED_VERSION}, installed {anthropic.__version__}. Re-verify the "
        f"request/response fixture shapes in tests/contracts/ and update the pin."
    )


# --------------------------------------------------------- request -> SDK kwargs
def test_generate_kwargs_validate_against_sdk_request_typeddicts() -> None:
    """Every kwarg/shape the adapter sends exists in the SDK's request schema."""
    from anthropic.types import (
        MessageParam,
        ToolParam,
        ToolResultBlockParam,
        message_create_params,
    )

    transport = _Transport(_real_message([]))
    mgr = AnthropicClientManager(model=MODEL, client=transport)
    run_async(
        mgr.generate(
            _req(
                messages=[
                    InferenceMessage(role="system", content="be brief"),
                    InferenceMessage(role="user", content="rates?"),
                    InferenceMessage(role="assistant", content="checking"),
                    InferenceMessage(role="tool", content="5%", tool_call_id="tc_1"),
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
    seen = transport.messages.seen

    create_keys = _typed_dict_keys(
        message_create_params.MessageCreateParamsNonStreaming
    )
    assert set(seen) <= create_keys, f"unknown create kwargs: {set(seen) - create_keys}"
    msg_keys = _typed_dict_keys(MessageParam)
    for msg in seen["messages"]:
        assert set(msg) <= msg_keys, f"unknown message keys: {set(msg) - msg_keys}"
    tool_keys = _typed_dict_keys(ToolParam)
    for tool in seen["tools"]:
        assert set(tool) <= tool_keys, f"unknown tool keys: {set(tool) - tool_keys}"
    # The tool-role turn was projected into a tool_result block: validate it too.
    block_keys = _typed_dict_keys(ToolResultBlockParam)
    (tool_result_turn,) = [
        m for m in seen["messages"] if isinstance(m["content"], list)
    ]
    for block in tool_result_turn["content"]:
        assert block["type"] == "tool_result"
        assert set(block) <= block_keys, (
            f"unknown block keys: {set(block) - block_keys}"
        )


def test_structured_tool_choice_validates_against_sdk_typeddict() -> None:
    """The forced structured ``tool_choice`` matches the SDK's ToolChoiceToolParam."""
    from anthropic.types import TextBlock, ToolChoiceToolParam

    transport = _Transport(_real_message([TextBlock(type="text", text="{}")]))
    mgr = AnthropicClientManager(model=MODEL, client=transport)
    run_async(mgr.generate(_req(response_format=ResponseFormat.JSON_OBJECT)))
    seen = transport.messages.seen
    choice_keys = _typed_dict_keys(ToolChoiceToolParam)
    assert set(seen["tool_choice"]) <= choice_keys
    assert seen["tool_choice"]["type"] == "tool"


# --------------------------------------------------- SDK response -> normalized
def test_real_text_message_normalizes_with_cache_usage_total() -> None:
    """A real Message maps to text + a FULL input total (cache reads/writes included)."""
    from anthropic.types import TextBlock

    message = _real_message(
        [TextBlock(type="text", text="first"), TextBlock(type="text", text="second")],
        input_tokens=10,
        output_tokens=5,
        cache_read=30,
        cache_creation=2,
    )
    mgr = AnthropicClientManager(model=MODEL, client=_Transport(message))
    resp = run_async(mgr.generate(_req()))

    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "first\nsecond"
    # input_tokens stays the FULL total; cost bills reads/writes at their cache rates.
    assert resp.input_tokens == 42  # 10 uncached + 30 cache-read + 2 cache-creation
    assert resp.output_tokens == 5
    assert resp.cost > 0.0
    assert resp.metadata["cache_read_tokens"] == 30
    assert resp.metadata["cache_creation_tokens"] == 2
    assert resp.model_path == f"anthropic:{MODEL}"


def test_real_tool_use_message_maps_and_executes() -> None:
    """A real tool_use stop maps blocks to ToolCallRecords and executes them."""
    from anthropic.types import TextBlock, ToolUseBlock

    executed: list[tuple[str, dict[str, Any]]] = []

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        executed.append((name, args))
        return ToolReturnRecord(tool_call_id="x", tool_name=name, content={"ok": True})

    message = _real_message(
        [
            TextBlock(type="text", text="let me look that up"),
            ToolUseBlock(
                type="tool_use", id="toolu_01", name="lookup", input={"q": "rates"}
            ),
        ],
        stop_reason="tool_use",
    )
    mgr = AnthropicClientManager(model=MODEL, client=_Transport(message))
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
    assert resp.tool_calls[0].tool_call_id == "toolu_01"
    assert resp.tool_calls[0].args == {"q": "rates"}
    assert executed == [("lookup", {"q": "rates"})]
    assert resp.tool_returns[0].tool_call_id == "toolu_01"
    # A tool_use stop is not a final answer.
    assert resp.output_text is None


# ----------------------------------------------------------- error normalization
def _status_error(exc_cls: Any, status: int, message: str) -> BaseException:
    """Construct a real SDK APIStatusError subclass with a real httpx response."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc: BaseException = exc_cls(
        message, response=httpx.Response(status, request=request), body=None
    )
    return exc


def test_real_sdk_errors_normalize_to_failed() -> None:
    """Real SDK exceptions (incl. 429 + 529 overloaded) map to FAILED, never raise."""
    # OverloadedError is not re-exported at the package top level in 0.106.0; it
    # is the REAL class the SDK raises on HTTP 529, so reach into _exceptions.
    from anthropic._exceptions import OverloadedError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    cases: list[tuple[BaseException, InferenceErrorCode, bool]] = [
        (
            _status_error(anthropic.RateLimitError, 429, "rate limited"),
            InferenceErrorCode.RATE_LIMITED,
            True,
        ),
        (
            _status_error(anthropic.AuthenticationError, 401, "bad key"),
            InferenceErrorCode.AUTH,
            False,
        ),
        (
            _status_error(anthropic.BadRequestError, 400, "bad request"),
            InferenceErrorCode.INVALID_REQUEST,
            False,
        ),
        (
            _status_error(anthropic.InternalServerError, 500, "boom"),
            InferenceErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
        # HTTP 529 "overloaded_error": no name-map entry; the status>=500
        # fallback must classify it retryable PROVIDER_UNAVAILABLE.
        (
            _status_error(OverloadedError, 529, "overloaded"),
            InferenceErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
        (
            anthropic.APITimeoutError(request=request),
            InferenceErrorCode.TIMEOUT,
            True,
        ),
        (
            anthropic.APIConnectionError(request=request),
            InferenceErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
    ]
    for exc, expected_code, expected_retryable in cases:
        mgr = AnthropicClientManager(model=MODEL, client=_Transport(exc))
        resp = run_async(mgr.generate(_req()))
        label = type(exc).__name__
        assert resp.status == InferenceStatus.FAILED, label
        assert resp.error is not None, label
        assert resp.error.code == expected_code, label
        assert resp.error.retryable is expected_retryable, label


# ------------------------------------------------------------------- streaming
def test_stream_final_uses_real_sdk_message() -> None:
    """The stream seam normalizes a REAL final Message after the text deltas."""
    from anthropic.types import TextBlock

    final = _real_message(
        [TextBlock(type="text", text="Hello Nepal")], input_tokens=6, output_tokens=3
    )

    class _Stream:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        @property
        def text_stream(self) -> Any:
            async def _gen() -> Any:
                for piece in ["Hel", "lo ", "Nepal"]:
                    yield piece

            return _gen()

        async def get_final_message(self) -> Any:
            return final

    class _StreamTransport:
        class _M:
            def stream(self, **kwargs: Any) -> Any:
                return _Stream()

        def __init__(self) -> None:
            self.messages = _StreamTransport._M()

    mgr = AnthropicClientManager(model=MODEL, client=_StreamTransport())

    async def _collect() -> list[Any]:
        return [piece async for piece in mgr.generate_stream(_req())]

    pieces = run_async(_collect())
    assert pieces[:-1] == ["Hel", "lo ", "Nepal"]
    last = pieces[-1]
    assert last.status == InferenceStatus.SUCCESS
    assert last.output_text == "Hello Nepal"
    assert last.input_tokens == 6 and last.output_tokens == 3
