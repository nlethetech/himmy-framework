"""Memory pack: ``remember`` + ``recall`` — the agent's durable long-term memory.

Wraps :class:`~himmy.services.memory.service.MemoryService` so an agent can save facts
and recall them semantically across runs. With ``HIMMY_MEMORY_PATH`` set, memories
persist to a SQLite file (durable); otherwise they live for the process. The subject
(whose memory this is) comes from ``HIMMY_MEMORY_SUBJECT`` (default ``"default"``).
"""

from __future__ import annotations

from typing import Any

from himmy.services.memory.service import MemoryService
from himmy.services.memory.store import InMemoryMemoryStore, SqliteMemoryStore
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig

_REMEMBER_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "A fact or note to remember."},
        "kind": {"type": "string", "default": "semantic"},
    },
    "required": ["text"],
    "additionalProperties": False,
}

_RECALL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to recall."},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def register_memory_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``remember`` and ``recall`` over a (durable) memory service."""
    store = (
        SqliteMemoryStore(config.memory_path)
        if config.memory_path
        else InMemoryMemoryStore()
    )
    memory = MemoryService(store)
    subject = config.memory_subject

    def remember(args: dict[str, Any]) -> dict[str, Any]:
        record = memory.remember(
            str(args["text"]),
            subject_id=subject,
            kind=str(args.get("kind", "semantic")),
        )
        return {"memory_id": record.memory_id, "subject": subject}

    async def recall(args: dict[str, Any]) -> dict[str, Any]:
        hits = await memory.recall(
            str(args["query"]),
            subject_id=subject,
            top_k=int(args.get("top_k", 5)),
        )
        return {
            "query": args["query"],
            "results": [
                {"text": h.record.text, "similarity": h.similarity} for h in hits
            ],
        }

    register_local_tool(
        registry,
        name="remember",
        handler=remember,
        description="Save a fact to the agent's long-term memory.",
        args_json_schema=_REMEMBER_SCHEMA,
        metadata={"pack": "memory"},
    )
    register_local_tool(
        registry,
        name="recall",
        handler=recall,
        description="Recall relevant facts from the agent's long-term memory.",
        args_json_schema=_RECALL_SCHEMA,
        metadata={"pack": "memory"},
    )


__all__ = ["register_memory_pack"]
