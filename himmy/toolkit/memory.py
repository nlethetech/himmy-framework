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
        "similarity_threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Optional minimum similarity; below it a memory is not recalled. "
                "Omit to always return the single most relevant memory."
            ),
        },
        "as_of": {
            "type": "string",
            "description": (
                "Optional ISO timestamp: recall facts that were TRUE at that instant "
                "(bi-temporal point-in-time recall)."
            ),
        },
        "active_only": {
            "type": "boolean",
            "default": False,
            "description": "Exclude facts that have been invalidated/superseded.",
        },
        "tier": {
            "type": "string",
            "enum": ["core", "recall", "archival"],
            "description": "Optional tier to scope the recall to.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "A new fact to reconcile."},
    },
    "required": ["text"],
    "additionalProperties": False,
}


def effective_memory_path(
    config: ToolkitConfig, server: bool | None = None
) -> str | None:
    """The memory store path agents should actually use.

    Explicit config (``HIMMY_MEMORY_PATH`` / project toolkit config) always wins.
    Otherwise, inside a SERVER entrypoint (Studio/BFF) memories default to the
    SAME durable store the Memory API reads — ``.himmy/memory.db`` under the
    project root — so an agent's ``remember`` survives the run and shows up in
    the GUI. On the one-shot CLI path the default stays in-memory (zero setup),
    unchanged. (Found live: agents remembered into an ephemeral store while the
    Brain/Memory screens read the durable one — memories silently vanished.)
    """
    if config.memory_path:
        return config.memory_path
    from himmy.services.storage.factory import in_server_context

    if not (server if server is not None else in_server_context()):
        return None
    from pathlib import Path

    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "memory.db")


def register_memory_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``remember`` + ``recall`` (and ``consolidate`` when enabled).

    With ``memory_consolidate`` set, the pack also exposes a ``consolidate`` tool that
    reconciles a new fact against existing ones (ADD/UPDATE/DELETE/NOOP) rather than
    blindly appending — the consolidation decision is itself audited on the spine when
    a registry/event sink is wired.
    """
    path = effective_memory_path(config, server=config.server_context or None)
    store = SqliteMemoryStore(path) if path else InMemoryMemoryStore()
    embedder, _dim = config.build_embedder_and_dim()
    memory = MemoryService(
        store, embedder=embedder, min_similarity=config.memory_min_similarity
    )
    subject = config.memory_subject

    def remember(args: dict[str, Any]) -> dict[str, Any]:
        record = memory.remember(
            str(args["text"]),
            subject_id=subject,
            kind=str(args.get("kind", "semantic")),
        )
        return {"memory_id": record.memory_id, "subject": subject}

    async def recall(args: dict[str, Any]) -> dict[str, Any]:
        threshold = args.get("similarity_threshold")
        tier = args.get("tier")
        as_of = args.get("as_of")
        hits = await memory.recall(
            str(args["query"]),
            subject_id=subject,
            top_k=int(args.get("top_k", 5)),
            similarity_threshold=(float(threshold) if threshold is not None else None),
            as_of=str(as_of) if as_of else None,
            tier=str(tier) if tier else None,
            active_only=bool(args.get("active_only", False)),
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
        read_only=False,
        handler=remember,
        description="Save a fact to the agent's long-term memory.",
        args_json_schema=_REMEMBER_SCHEMA,
        metadata={"pack": "memory"},
    )
    register_local_tool(
        registry,
        name="recall",
        read_only=True,
        handler=recall,
        description="Recall relevant facts from the agent's long-term memory.",
        args_json_schema=_RECALL_SCHEMA,
        metadata={"pack": "memory"},
    )

    if config.memory_consolidate:
        from himmy.services.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(memory)

        async def consolidate(args: dict[str, Any]) -> dict[str, Any]:
            result = await consolidator.consolidate(
                str(args["text"]), subject_id=subject
            )
            return {
                "action": result.action,
                "reason": result.reason,
                "memory_id": result.record.memory_id if result.record else None,
                "target_id": result.target_id,
            }

        register_local_tool(
            registry,
            name="consolidate",
            read_only=False,
            handler=consolidate,
            description=(
                "Reconcile a new fact against existing memory "
                "(add / update / delete / noop)."
            ),
            args_json_schema=_CONSOLIDATE_SCHEMA,
            metadata={"pack": "memory"},
        )


__all__ = ["register_memory_pack"]
