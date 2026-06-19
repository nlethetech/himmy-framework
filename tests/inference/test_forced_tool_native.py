"""Native ``tool_choice`` for forced tools + the streaming-path fix (QUICK WIN 2).

Forcing a tool used to be a prompt-nudge only (``normalize_forced_tool`` appends a
system message) with NO provider-native ``tool_choice`` on OpenAI, and ``run_stream``
never even called ``normalize_forced_tool`` — so a streamed ``ResponseFormat.TOOL``
silently lost its forcing. These tests pin:

* OpenAI sends ``{"type": "function", "function": {"name": ...}}`` for a forced tool;
* Anthropic sends ``{"type": "tool", "name": ...}`` for a forced tool;
* the system-prompt nudge is RETAINED (for local backends with no native choice);
* ``run_stream`` normalizes the forced tool and drives the forced call.
"""

from __future__ import annotations

from typing import Any

from himmy.services.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceService,
    StubClientManager,
)
from himmy.services.inference.anthropic_manager import AnthropicClientManager
from himmy.services.inference.client_manager import normalize_forced_tool
from himmy.services.inference.models import (
    BoundTool,
    InferenceStatus,
    ResponseFormat,
)
from himmy.services.inference.openai_manager import OpenAIClientManager
from tests.conftest import run_async


def _tool(name: str) -> BoundTool:
    return BoundTool(
        name=name,
        description=f"the {name} tool",
        args_json_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )


def _forced_req(model_key: str, tools: list[BoundTool], **kw: Any) -> InferenceRequest:
    """A normalized forced-tool request (as the service would hand the manager)."""
    req = InferenceRequest(
        model_key=model_key,
        messages=[InferenceMessage(role="user", content="do it")],
        response_format=ResponseFormat.TOOL,
        bound_tools=tools,
        **kw,
    )
    return normalize_forced_tool(req)


# ----------------------------------------------------------------- OpenAI native
class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _FakeCompletions:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.seen.update(kwargs)
        return self._result


class _FakeChat:
    def __init__(self, result: Any) -> None:
        self.completions = _FakeCompletions(result)


class _FakeClient:
    def __init__(self, result: Any) -> None:
        self.chat = _FakeChat(result)


def _completion(text: str = "ok") -> _Obj:
    message = _Obj(content=text, tool_calls=None)
    usage = _Obj(prompt_tokens=5, completion_tokens=2)
    return _Obj(choices=[_Obj(message=message)], usage=usage)


def test_openai_serializes_native_tool_choice_for_forced_tool() -> None:
    client = _FakeClient(_completion())
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    req = _forced_req("gpt-4o-mini", [_tool("a"), _tool("b")], tool_names_override=["b"])
    run_async(mgr.generate(req))
    assert client.chat.completions.seen["tool_choice"] == {
        "type": "function",
        "function": {"name": "b"},
    }
    # The nudge is RETAINED so local backends still get the instruction.
    assert any("`b`" in m.content for m in req.messages if m.role == "system")


def test_openai_no_tool_choice_without_forcing() -> None:
    client = _FakeClient(_completion())
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    req = InferenceRequest(
        model_key="gpt-4o-mini",
        messages=[InferenceMessage(role="user", content="hi")],
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[_tool("a")],
    )
    run_async(mgr.generate(req))
    assert "tool_choice" not in client.chat.completions.seen


# ----------------------------------------------------------------- Anthropic native
class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Usage:
    def __init__(self) -> None:
        self.input_tokens = 5
        self.output_tokens = 2


class _Message:
    def __init__(self, content: list[_Block]) -> None:
        self.content = content
        self.usage = _Usage()


class _FakeMessages:
    def __init__(self, message: _Message) -> None:
        self._message = message
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _Message:
        self.seen.update(kwargs)
        return self._message


class _FakeAnthropicClient:
    def __init__(self, message: _Message) -> None:
        self.messages = _FakeMessages(message)


def test_anthropic_serializes_native_tool_choice_for_forced_tool() -> None:
    client = _FakeAnthropicClient(_Message([_Block(type="text", text="ok")]))
    mgr = AnthropicClientManager(model="claude-3-5-haiku-latest", client=client)
    req = _forced_req(
        "claude-3-5-haiku-latest", [_tool("a"), _tool("b")], tool_names_override=["a"]
    )
    run_async(mgr.generate(req))
    assert client.messages.seen["tool_choice"] == {"type": "tool", "name": "a"}
    assert any("`a`" in m.content for m in req.messages if m.role == "system")


def test_anthropic_no_tool_choice_without_forcing() -> None:
    client = _FakeAnthropicClient(_Message([_Block(type="text", text="ok")]))
    mgr = AnthropicClientManager(model="claude-3-5-haiku-latest", client=client)
    req = InferenceRequest(
        model_key="claude-3-5-haiku-latest",
        messages=[InferenceMessage(role="user", content="hi")],
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[_tool("a")],
    )
    run_async(mgr.generate(req))
    assert "tool_choice" not in client.messages.seen


# ----------------------------------------------------------------- nudge for local
def test_nudge_retained_for_local_backends() -> None:
    """The normalized request still carries the explicit system nudge."""
    req = _forced_req("default", [_tool("a")])
    assert req.metadata["forced_tool"] == "a"
    assert any("`a`" in m.content for m in req.messages if m.role == "system")


# ----------------------------------------------------------------- run_stream fix
def test_run_stream_forces_the_tool_via_fallback() -> None:
    """A streamed ResponseFormat.TOOL now forces the call (was silently dropped).

    The stub has no ``generate_stream`` so ``run_stream`` takes the offline fallback,
    which must STILL normalize the forced tool and surface exactly one forced call.
    """
    svc = InferenceService(StubClientManager(), max_retries=0)
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="do it")],
        response_format=ResponseFormat.TOOL,
        bound_tools=[_tool("a"), _tool("b")],
        tool_names_override=["b"],
    )

    async def _collect() -> Any:
        final = None
        async for delta in svc.run_stream(req):
            if delta.done:
                final = delta.response
        return final

    final = run_async(_collect())
    assert final is not None
    assert final.status == InferenceStatus.SUCCESS
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].tool_name == "b"


def test_run_stream_forced_tool_with_no_bound_tool_fails_cleanly() -> None:
    """A streamed forced-tool request with no bound tool yields a FAILED terminal."""
    svc = InferenceService(StubClientManager(), max_retries=0)
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="do it")],
        response_format=ResponseFormat.TOOL,
        bound_tools=[],
    )

    async def _collect() -> Any:
        final = None
        async for delta in svc.run_stream(req):
            if delta.done:
                final = delta.response
        return final

    final = run_async(_collect())
    assert final is not None
    assert final.status == InferenceStatus.FAILED
    assert final.error is not None
