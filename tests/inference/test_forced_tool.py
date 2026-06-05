"""ResponseFormat.TOOL: force the model to call one named tool (was NotImplementedError)."""

from __future__ import annotations

import pytest

from himmy.core.errors import HimmyError
from himmy.services.inference.client_manager import (
    StubClientManager,
    normalize_forced_tool,
)
from himmy.services.inference.models import (
    BoundTool,
    InferenceMessage,
    InferenceRequest,
    ResponseFormat,
    ToolReturnRecord,
)
from tests.conftest import run_async


async def _handler(args: dict) -> ToolReturnRecord:
    return ToolReturnRecord(tool_call_id="", tool_name="lookup", content={"ok": args})


def _tool(name: str) -> BoundTool:
    return BoundTool(
        name=name,
        description=f"the {name} tool",
        args_json_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        handler=_handler,
    )


def _req(fmt: ResponseFormat, tools: list[BoundTool], **kw) -> InferenceRequest:
    return InferenceRequest(
        messages=[InferenceMessage(role="user", content="do the thing")],
        response_format=fmt,
        bound_tools=tools,
        **kw,
    )


def test_normalize_rewrites_to_single_tool_auto_tools() -> None:
    out = normalize_forced_tool(_req(ResponseFormat.TOOL, [_tool("a"), _tool("b")]))
    assert out.response_format == ResponseFormat.AUTO_TOOLS
    assert [t.name for t in out.bound_tools] == ["a"]  # first tool forced
    assert out.metadata["forced_tool"] == "a"
    assert any("`a`" in m.content for m in out.messages if m.role == "system")


def test_normalize_honors_tool_names_override() -> None:
    out = normalize_forced_tool(
        _req(ResponseFormat.TOOL, [_tool("a"), _tool("b")], tool_names_override=["b"])
    )
    assert [t.name for t in out.bound_tools] == ["b"]


def test_normalize_requires_a_tool() -> None:
    with pytest.raises(HimmyError):
        normalize_forced_tool(_req(ResponseFormat.TOOL, []))


def test_normalize_passthrough_for_non_tool_formats() -> None:
    req = _req(ResponseFormat.TEXT, [_tool("a")])
    assert normalize_forced_tool(req) is req


def test_stub_forces_the_tool_call() -> None:
    # The exact thing that used to raise NotImplementedError now produces the call.
    resp = run_async(
        StubClientManager().generate(
            _req(ResponseFormat.TOOL, [_tool("a"), _tool("b")])
        )
    )
    assert resp.status.value == "SUCCESS"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "a"
