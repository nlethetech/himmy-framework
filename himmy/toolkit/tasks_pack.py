"""Tasks pack: ``list_tasks`` + ``add_task`` + ``complete_task``.

Lets an agent manage the same durable task list the Studio Tasks screen shows
(``.himmy/tasks.db`` / ``HIMMY_TASKS_PATH``) — a shared to-do board between you and
your agents. (Distinct from the per-run ``todo_*`` scratch tools in the agentic pack.)
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig

_ADD_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The task to add."},
        "due": {"type": "string", "description": "Optional due date (YYYY-MM-DD)."},
        "priority": {
            "type": "integer",
            "description": "Optional priority 0–3 (0 none, 1 low, 2 medium, 3 high).",
            "minimum": 0,
            "maximum": 3,
        },
    },
    "required": ["title"],
    "additionalProperties": False,
}

_COMPLETE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string", "description": "Title to mark done."}},
    "required": ["title"],
    "additionalProperties": False,
}


def register_tasks_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``list_tasks`` / ``add_task`` / ``complete_task`` over the tasks store."""
    from himmy.api.studio_tasks import get_tasks_store

    def list_tasks(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "tasks": [
                {"title": t.title, "done": t.done, "due": t.due, "priority": t.priority}
                for t in get_tasks_store().list()
            ]
        }

    def add_task(args: dict[str, Any]) -> dict[str, Any]:
        # ``priority`` is optional + best-effort: a non-int value falls back to 0 so a model
        # that emits "high" instead of 3 can't break the add.
        try:
            priority = int(args.get("priority") or 0)
        except (TypeError, ValueError):
            priority = 0
        t = get_tasks_store().add(
            str(args["title"]), due=args.get("due"), priority=priority
        )
        return {"added": True, "title": t.title, "due": t.due, "priority": t.priority}

    def complete_task(args: dict[str, Any]) -> dict[str, Any]:
        return {"completed": get_tasks_store().complete_by_title(str(args["title"]))}

    register_local_tool(
        registry,
        name="list_tasks",
        read_only=True,
        handler=list_tasks,
        description="List the shared task board (open + done).",
        args_json_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        metadata={"pack": "tasks"},
    )
    register_local_tool(
        registry,
        name="add_task",
        read_only=False,
        handler=add_task,
        description="Add a task to the shared board.",
        args_json_schema=_ADD_SCHEMA,
        metadata={"pack": "tasks"},
    )
    register_local_tool(
        registry,
        name="complete_task",
        read_only=False,
        handler=complete_task,
        description="Mark a task done by its title.",
        args_json_schema=_COMPLETE_SCHEMA,
        metadata={"pack": "tasks"},
    )


__all__ = ["register_tasks_pack"]
