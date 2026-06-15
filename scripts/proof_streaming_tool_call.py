"""REAL end-to-end proof for the streaming tool-call blocker fix.

Drives himmy's STREAMING agent loop against the LIVE OpenRouter API with one
deterministic tool bound, then asserts the tool was actually invoked OVER the
streaming path, the run completed, and the final answer reflects the tool's
returned value. A non-streaming run is included as a control.

Run (env must be exported in the SAME shell first):
    set -a; source .env; set +a
    .venv/bin/python scripts/proof_streaming_tool_call.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Make the repo importable without a prior install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.cli.provider import build_inference_for
from himmy.services.inference.models import LLMConfig, ResponseFormat
from himmy.services.inference.service import StreamDelta
from himmy.services.tools.registry import ToolRegistry, register_local_tool

MODEL = "openai/gpt-4o-mini"  # confirmed to emit fragmented tool_call deltas over SSE

# Module-level flag flipped by the tool handler itself: the ONLY way this becomes
# True is if himmy actually executed the bound Python callable. So this is a hard,
# independent witness that the tool ran (not just that the model emitted a call).
_HANDLER_FIRED: dict[str, int] = {"count": 0}
_WEATHER_ANSWER = "18C and clear"


def _get_weather(args: dict) -> str:
    """Deterministic weather tool. himmy calls local handlers as handler(args_dict)."""
    _HANDLER_FIRED["count"] += 1
    city = (args or {}).get("city", "unknown")
    print(f"   [tool handler] get_weather(city={city!r}) -> {_WEATHER_ANSWER!r}")
    return _WEATHER_ANSWER


def _build_runtime_and_registry():
    """Build the OpenRouter runtime EXACTLY the way himmy's CLI does, with one tool."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="get_weather",
        handler=_get_weather,
        description="Get the current weather for a city. Returns a short text report.",
        args_json_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"}
            },
            "required": ["city"],
        },
        read_only=True,
    )
    # CLI provider path: provider 'openrouter' + model -> OpenAIClientManager(provider_name='openrouter')
    inference = build_inference_for("openrouter", MODEL)
    mgr = inference._client_manager
    print(
        f"   manager={type(mgr).__name__} provider_name="
        f"{getattr(mgr, 'provider_name', '?')!r} model={MODEL!r}"
    )
    runtime, _inf, _tools = build_runtime(inference=inference, tool_registry=registry)
    return runtime


def _persona_task():
    persona = Persona(
        name="weather-helper",
        description="You report the weather by calling the get_weather tool, then state the answer.",
        instructions=["Use the get_weather tool for any weather question."],
    )
    task = Task(
        title="weather",
        prompt="What's the weather in Paris? Use the get_weather tool.",
        context={"tool_names": ["get_weather"]},
    )
    return persona, task


async def run_streaming(runtime) -> dict:
    """Drive the STREAMING agent loop, consume to completion, collect observations."""
    persona, task = _persona_task()
    _HANDLER_FIRED["count"] = 0  # reset witness for this path

    text_deltas: list[str] = []
    tool_call_events: list[dict] = []
    tool_result_events: list[dict] = []
    final_result = None
    saw_done = False

    gen = runtime.stream_agent_loop(
        persona,
        task,
        max_turns=4,
        llm_config=LLMConfig(response_format=ResponseFormat.AUTO_TOOLS),
    )
    async for delta in gen:
        assert isinstance(delta, StreamDelta)
        if delta.event_type is None and not delta.done and delta.delta:
            text_deltas.append(delta.delta)
        elif delta.event_type == "tool_call":
            tool_call_events.append(delta.event_payload or {})
        elif delta.event_type == "tool_result":
            tool_result_events.append(delta.event_payload or {})
        if delta.done:
            saw_done = True
            final_result = (delta.event_payload or {}).get("result")

    streamed_text = "".join(text_deltas)
    # Gather the typed tool exchanges from the materialized AgentLoopResult too.
    loop_tool_calls = []
    loop_tool_returns = []
    final_answer = ""
    stopped_reason = None
    turn_count = None
    turn_errors = []
    if final_result is not None:
        stopped_reason = final_result.stopped_reason
        turn_count = final_result.turn_count
        final_answer = (final_result.final.output_text or "").strip()
        for i, turn in enumerate(final_result.turns):
            turn_errors.append((i, turn.status, turn.error_code, turn.error))
            for c in turn.tool_calls:
                loop_tool_calls.append((c.tool_name, c.args))
            for r in turn.tool_returns:
                loop_tool_returns.append((r.tool_name, r.content))

    # The final answer can be either the streamed text or the materialized final.
    combined_answer = (final_answer or streamed_text).strip()

    return {
        "handler_fired": _HANDLER_FIRED["count"],
        "tool_call_events": tool_call_events,
        "tool_result_events": tool_result_events,
        "loop_tool_calls": loop_tool_calls,
        "loop_tool_returns": loop_tool_returns,
        "streamed_text": streamed_text,
        "final_answer": combined_answer,
        "stopped_reason": stopped_reason,
        "turn_count": turn_count,
        "saw_done": saw_done,
        "has_result": final_result is not None,
        "turn_errors": turn_errors,
    }


async def run_nonstreaming(runtime) -> dict:
    """Control: the NON-streaming loop with the same tool must also work."""
    persona, task = _persona_task()
    _HANDLER_FIRED["count"] = 0  # reset witness for this path

    loop = await runtime.run_agent_loop(
        persona,
        task,
        max_turns=4,
        llm_config=LLMConfig(response_format=ResponseFormat.AUTO_TOOLS),
    )
    tool_calls = []
    tool_returns = []
    for turn in loop.turns:
        for c in turn.tool_calls:
            tool_calls.append((c.tool_name, c.args))
        for r in turn.tool_returns:
            tool_returns.append((r.tool_name, r.content))
    return {
        "handler_fired": _HANDLER_FIRED["count"],
        "tool_calls": tool_calls,
        "tool_returns": tool_returns,
        "final_answer": (loop.final.output_text or "").strip(),
        "stopped_reason": loop.stopped_reason,
        "turn_count": loop.turn_count,
    }


def _check_streaming(s: dict) -> tuple[bool, list[str]]:
    """Assert the three required conditions for the STREAMING path."""
    fails: list[str] = []
    # (a) the tool was ACTUALLY invoked over the streaming path.
    tool_invoked = (
        s["handler_fired"] > 0
        and any(name == "get_weather" for name, _ in s["loop_tool_calls"])
    )
    if not tool_invoked:
        fails.append(
            "tool was NOT invoked over the streaming path "
            f"(handler_fired={s['handler_fired']}, tool_calls={s['loop_tool_calls']})"
        )
    # extra signal: streaming surfaced a tool_call structured event
    if not s["tool_call_events"]:
        fails.append("no 'tool_call' structured stream event was emitted")
    # (b) the run completed.
    if not s["saw_done"] or not s["has_result"]:
        fails.append("stream did not complete with a terminal done+result")
    if s["stopped_reason"] != "final":
        fails.append(f"stopped_reason was {s['stopped_reason']!r}, expected 'final'")
    # (c) the final answer reflects the tool's returned value.
    ans = s["final_answer"].lower()
    reflects = ("18" in ans) and ("clear" in ans)
    if not reflects:
        fails.append(
            "final streamed answer does not reflect the tool result "
            f"('18C and clear'): {s['final_answer']!r}"
        )
    # the tool's return must have flowed back into the loop as a tool_return
    if not any(
        name == "get_weather" and _WEATHER_ANSWER in str(content)
        for name, content in s["loop_tool_returns"]
    ):
        fails.append(
            f"tool return value never reassembled into the loop: {s['loop_tool_returns']}"
        )
    return (not fails), fails


def _check_nonstreaming(n: dict) -> tuple[bool, list[str]]:
    fails: list[str] = []
    if n["handler_fired"] <= 0 or not any(
        name == "get_weather" for name, _ in n["tool_calls"]
    ):
        fails.append(
            f"non-streaming control: tool not invoked (handler={n['handler_fired']}, "
            f"calls={n['tool_calls']})"
        )
    ans = n["final_answer"].lower()
    if not (("18" in ans) and ("clear" in ans)):
        fails.append(
            f"non-streaming control: answer does not reflect tool result: {n['final_answer']!r}"
        )
    return (not fails), fails


async def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("FAIL: OPENROUTER_API_KEY not set in this shell.")
        return 1

    runtime = _build_runtime_and_registry()

    print("\n=== STREAMING agent loop (the FIXED path) ===")
    s = await run_streaming(runtime)
    print(f"   handler_fired      : {s['handler_fired']}")
    print(f"   stream tool_call ev: {s['tool_call_events']}")
    print(f"   loop tool_calls    : {s['loop_tool_calls']}")
    print(f"   loop tool_returns  : {s['loop_tool_returns']}")
    print(f"   stopped_reason     : {s['stopped_reason']}   turns: {s['turn_count']}")
    print(f"   turn_errors        : {s['turn_errors']}")
    print(f"   streamed_text      : {s['streamed_text']!r}")
    print(f"   final_answer       : {s['final_answer']!r}")
    stream_ok, stream_fails = _check_streaming(s)

    print("\n=== NON-STREAMING control loop ===")
    n = await run_nonstreaming(runtime)
    print(f"   handler_fired      : {n['handler_fired']}")
    print(f"   tool_calls         : {n['tool_calls']}")
    print(f"   tool_returns       : {n['tool_returns']}")
    print(f"   stopped_reason     : {n['stopped_reason']}   turns: {n['turn_count']}")
    print(f"   final_answer       : {n['final_answer']!r}")
    nonstream_ok, nonstream_fails = _check_nonstreaming(n)

    print("\n=== VERDICT ===")
    for f in stream_fails + nonstream_fails:
        print(f"   - {f}")
    overall = stream_ok and nonstream_ok
    print(f"   STREAMING_TOOL_CALLED : {s['handler_fired'] > 0}")
    print(f"   STREAMING_COMPLETED   : {s['saw_done'] and s['has_result']}")
    print(f"   TOOL_RESULT_REFLECTED : {('18' in s['final_answer'].lower()) and ('clear' in s['final_answer'].lower())}")
    print(f"   NONSTREAMING_CONTROL  : {nonstream_ok}")
    print("\n" + ("PASS: streaming tool-call blocker is FIXED." if overall else "FAIL: streaming tool-call blocker is NOT fully fixed."))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
