"""Tests for real tool-calling on the local providers (Ollama native + Claude CLI ReAct)."""

from __future__ import annotations

from himmy.services.inference.local import ClaudeCliClientManager, OllamaClientManager
from himmy.services.inference.models import (
    BoundTool,
    InferenceMessage,
    InferenceRequest,
    ToolReturnRecord,
)
from tests.conftest import executor_from, run_async


async def _search_handler(args: dict) -> ToolReturnRecord:
    return ToolReturnRecord(
        tool_call_id="", tool_name="web_search", content={"hits": [args.get("query")]}
    )


_TOOL = BoundTool(
    name="web_search",
    description="search the web",
    args_json_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

_EXECUTOR = executor_from({"web_search": _search_handler})


def _req(tools: bool = True) -> InferenceRequest:
    return InferenceRequest(
        messages=[InferenceMessage(role="user", content="find permaculture")],
        bound_tools=[_TOOL] if tools else [],
        tool_executor=_EXECUTOR if tools else None,
    )


# --------------------------------------------------------------------- Ollama


def test_ollama_injects_tools_and_parses_tool_calls() -> None:
    seen: dict = {}

    def transport(path: str, payload: dict) -> dict:
        seen.update(payload)
        return {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": {"query": "pc"}}}
                ],
            }
        }

    mgr = OllamaClientManager(transport=transport)
    resp = run_async(mgr.generate(_req()))
    assert "tools" in seen  # tools injected into the payload
    assert seen["tools"][0]["function"]["name"] == "web_search"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "web_search"
    assert resp.tool_calls[0].args == {"query": "pc"}
    # The manager also EXECUTES the bound handler and pairs the return (so the
    # model sees the result next turn instead of looping).
    assert len(resp.tool_returns) == 1
    assert resp.tool_returns[0].tool_call_id == resp.tool_calls[0].tool_call_id
    assert resp.tool_returns[0].content == {"hits": ["pc"]}


def test_ollama_parses_stringified_arguments() -> None:
    def transport(path: str, payload: dict) -> dict:
        return {
            "message": {
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": '{"query": "x"}'}}
                ]
            }
        }

    resp = run_async(OllamaClientManager(transport=transport).generate(_req()))
    assert resp.tool_calls[0].args == {"query": "x"}


def test_ollama_structured_output() -> None:
    """Ollama gets the schema via `format` and the JSON reply parses to output_structured."""
    from himmy.services.inference.models import ResponseFormat

    seen: dict = {}

    def transport(path: str, payload: dict) -> dict:
        seen.update(payload)
        return {"message": {"content": '{"steps": ["a", "b"]}'}}

    schema = {
        "type": "object",
        "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
        "required": ["steps"],
    }
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="plan it")],
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema=schema,
    )
    resp = run_async(OllamaClientManager(transport=transport).generate(req))
    assert seen["format"] == schema  # schema sent to Ollama
    assert resp.output_structured == {"steps": ["a", "b"]}


def test_ollama_no_tools_stays_text() -> None:
    def transport(path: str, payload: dict) -> dict:
        assert "tools" not in payload  # none bound → no tools key
        return {"message": {"content": "plain answer"}}

    resp = run_async(
        OllamaClientManager(transport=transport).generate(_req(tools=False))
    )
    assert resp.output_text == "plain answer"
    assert resp.tool_calls == []


# ------------------------------------------------------------------ Claude CLI


def test_claude_cli_emits_manifest_and_parses_tool_call() -> None:
    seen: dict = {}

    def runner(argv: list[str], stdin: str) -> str:
        seen["stdin"] = stdin
        return 'TOOL_CALL web_search {"query": "permaculture"}'

    resp = run_async(ClaudeCliClientManager(runner=runner).generate(_req()))
    assert "TOOL_CALL" in seen["stdin"]  # ReAct manifest appended
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "web_search"
    assert resp.tool_calls[0].args == {"query": "permaculture"}
    assert resp.output_text == ""  # a tool call suppresses the text
    assert resp.tool_returns[0].content == {"hits": ["permaculture"]}  # executed


def test_claude_cli_plain_answer_stays_text() -> None:
    resp = run_async(
        ClaudeCliClientManager(runner=lambda a, s: "Here is the answer.").generate(
            _req()
        )
    )
    assert resp.tool_calls == []
    assert resp.output_text == "Here is the answer."


def test_claude_cli_no_tools_no_manifest() -> None:
    def runner(argv: list[str], stdin: str) -> str:
        assert "TOOL_CALL" not in stdin
        return "answer"

    resp = run_async(ClaudeCliClientManager(runner=runner).generate(_req(tools=False)))
    assert resp.output_text == "answer"
