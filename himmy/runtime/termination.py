"""Agent-loop termination helpers: a `final_answer` tool + no-progress detection.

Reactive agent loops stop only when the model emits no tool calls. Two failure modes
remain: a model that keeps calling tools never returns "done", and a model that repeats
the same call spins until ``max_turns``. These helpers add an explicit terminal signal
(`final_answer`) and a no-progress guard, used by both the single-agent loop and the
multi-agent orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.runtime.single_agent import RunResult

#: The name of the synthetic terminal tool an agent calls to end its loop.
FINAL_ANSWER_TOOL = "final_answer"

_FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The agent's final answer."}
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def register_final_answer_tool(
    registry: ToolRegistry, *, name: str = FINAL_ANSWER_TOOL
) -> None:
    """Register the ``final_answer`` tool; calling it ends the agent's loop."""

    def _handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"answer": str(args.get("answer", ""))}

    register_local_tool(
        registry,
        name=name,
        handler=_handler,
        description="Provide your final answer and end the task.",
        args_json_schema=_FINAL_ANSWER_SCHEMA,
        metadata={"synthetic": "termination"},
    )


def final_answer_text(
    result: RunResult, *, name: str = FINAL_ANSWER_TOOL
) -> str | None:
    """Return the answer if ``result`` called ``final_answer``, else ``None``."""
    for call in result.tool_calls:
        if call.tool_name == name:
            return str(call.args.get("answer", ""))
    return None


def calls_signature(result: RunResult) -> tuple[tuple[str, str], ...]:
    """A hashable signature of a turn's tool calls (name + sorted args) for dedup."""
    import json

    return tuple(
        (c.tool_name, json.dumps(c.args, sort_keys=True, default=str))
        for c in result.tool_calls
    )


def is_no_progress(turns: list[RunResult]) -> bool:
    """True when the last two turns made the identical (non-empty) tool calls."""
    if len(turns) < 2:
        return False
    last = calls_signature(turns[-1])
    return bool(last) and last == calls_signature(turns[-2])


__all__ = [
    "FINAL_ANSWER_TOOL",
    "register_final_answer_tool",
    "final_answer_text",
    "calls_signature",
    "is_no_progress",
]
