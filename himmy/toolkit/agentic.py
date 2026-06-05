"""Agentic pack: the tools that make an agent *act like one*, not just answer.

Three capabilities that turn a reactive tool-caller into an autonomous worker:

* ``ask_human`` — pause mid-run and ask the operator a question, then continue with
  their answer (human-in-the-loop, agent-initiated). On a TTY it prompts; when there is
  no interactive human (pipe / CI) it returns ``answered: false`` so the model can adapt.
* ``scratchpad_set`` / ``scratchpad_get`` — a keyed working-memory notepad the agent
  manages itself within a run (distinct from the durable semantic ``memory`` pack).
* ``todo_write`` / ``todo_read`` — a self-managed task list (write the whole list with
  per-item status each time; read it back) so the agent can plan and track its own work.

State lives for the process (one registry → one shared scratchpad/todo), so it persists
across turns of a ``himmy chat`` session and resets between separate ``himmy run`` calls —
which is the right lifetime for *working* memory.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig

#: An injectable responder for ``ask_human`` (sync or async). When set, it overrides the
#: default terminal prompt — used to embed himmy in another UI, or in tests.
HumanResponder = Callable[[str], str | Awaitable[str]]
_HUMAN_RESPONDER: HumanResponder | None = None


def set_human_responder(responder: HumanResponder | None) -> None:
    """Install (or clear) the callback ``ask_human`` uses to reach a human."""
    global _HUMAN_RESPONDER
    _HUMAN_RESPONDER = responder


async def _default_ask(question: str) -> dict[str, Any]:
    """Prompt on the terminal; report ``answered: false`` when non-interactive."""
    if not sys.stdin.isatty():
        return {
            "question": question,
            "answer": "",
            "answered": False,
            "reason": "no interactive human available",
        }
    print(f"\n[agent asks] {question}", file=sys.stderr, flush=True)
    answer = (await asyncio.to_thread(input, "your answer> ")).strip()
    return {"question": question, "answer": answer, "answered": True}


_ASK_HUMAN_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to put to the human operator.",
        }
    },
    "required": ["question"],
    "additionalProperties": False,
}

_SCRATCH_SET_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {"type": "string", "description": "Name of the note."},
        "value": {"type": "string", "description": "The note's contents."},
    },
    "required": ["key", "value"],
    "additionalProperties": False,
}

_SCRATCH_GET_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "description": "Note to read; omit to read every note.",
        }
    },
    "additionalProperties": False,
}

# A FLAT array-of-strings schema: small local models (and Ollama's tool grammar)
# reliably emit this, but choke on nested array-of-objects and return nothing. Status
# is tracked via the separate todo_complete tool rather than an inline object field.
_TODO_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The task descriptions, in order.",
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

_TODO_COMPLETE_SCHEMA = {
    "type": "object",
    "properties": {
        "item": {
            "type": "string",
            "description": "The task to mark done (matched by content prefix).",
        }
    },
    "required": ["item"],
    "additionalProperties": False,
}

_TODO_READ_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def register_agentic_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``ask_human``, the scratchpad tools, and the todo-list tools."""
    scratchpad: dict[str, str] = {}
    todo: list[dict[str, Any]] = []

    async def ask_human(args: dict[str, Any]) -> dict[str, Any]:
        question = str(args["question"])
        responder = _HUMAN_RESPONDER
        if responder is None:
            return await _default_ask(question)
        result = responder(question)
        answer = await result if inspect.isawaitable(result) else result
        return {"question": question, "answer": str(answer), "answered": True}

    def scratchpad_set(args: dict[str, Any]) -> dict[str, Any]:
        key = str(args["key"])
        scratchpad[key] = str(args["value"])
        return {"key": key, "ok": True, "keys": sorted(scratchpad)}

    def scratchpad_get(args: dict[str, Any]) -> dict[str, Any]:
        key = args.get("key")
        if key is None:
            return {"notes": dict(scratchpad)}
        return {"key": str(key), "value": scratchpad.get(str(key))}

    def _todo_result() -> dict[str, Any]:
        done = sum(1 for t in todo if t["status"] == "completed")
        return {"count": len(todo), "completed": done, "items": list(todo)}

    def todo_write(args: dict[str, Any]) -> dict[str, Any]:
        items = args.get("items") or []
        todo.clear()
        for item in items:
            content = str(item).strip()
            if content:
                todo.append({"content": content, "status": "pending"})
        return _todo_result()

    def todo_complete(args: dict[str, Any]) -> dict[str, Any]:
        target = str(args["item"]).strip().lower()
        matched = False
        for entry in todo:
            content = entry["content"].lower()
            if content == target or content.startswith(target) or target in content:
                entry["status"] = "completed"
                matched = True
                break
        return {"matched": matched, **_todo_result()}

    def todo_read(_args: dict[str, Any]) -> dict[str, Any]:
        return _todo_result()

    register_local_tool(
        registry,
        name="ask_human",
        handler=ask_human,
        description=(
            "Ask the human operator a question and wait for their answer. Use when you "
            "are blocked on a decision only the user can make."
        ),
        args_json_schema=_ASK_HUMAN_SCHEMA,
        metadata={"pack": "agentic"},
    )
    register_local_tool(
        registry,
        name="scratchpad_set",
        handler=scratchpad_set,
        description="Save a named note to your working scratchpad (overwrites the key).",
        args_json_schema=_SCRATCH_SET_SCHEMA,
        metadata={"pack": "agentic"},
    )
    register_local_tool(
        registry,
        name="scratchpad_get",
        handler=scratchpad_get,
        description="Read a note from your scratchpad (omit `key` to read all of them).",
        args_json_schema=_SCRATCH_GET_SCHEMA,
        metadata={"pack": "agentic"},
    )
    register_local_tool(
        registry,
        name="todo_write",
        handler=todo_write,
        description=(
            "Replace your task list with `items`, a flat list of task descriptions "
            "(strings). Use it to plan multi-step work before you start."
        ),
        args_json_schema=_TODO_WRITE_SCHEMA,
        metadata={"pack": "agentic"},
    )
    register_local_tool(
        registry,
        name="todo_complete",
        handler=todo_complete,
        description="Mark a task on your list done (matched by its content).",
        args_json_schema=_TODO_COMPLETE_SCHEMA,
        metadata={"pack": "agentic"},
    )
    register_local_tool(
        registry,
        name="todo_read",
        handler=todo_read,
        description="Read your current task list and how many items are completed.",
        args_json_schema=_TODO_READ_SCHEMA,
        metadata={"pack": "agentic"},
    )


AGENTIC_TOOL_NAMES = (
    "ask_human",
    "scratchpad_set",
    "scratchpad_get",
    "todo_write",
    "todo_complete",
    "todo_read",
)

__all__ = ["register_agentic_pack", "set_human_responder", "AGENTIC_TOOL_NAMES"]
