"""Knowledge pack: ``kb_ingest`` + ``kb_search`` — an agent's own RAG memory.

These wrap the framework's existing, fully-tested
:class:`~himmy.services.knowledge.service.KnowledgeBase`: an agent can ingest text or
a file and later semantically search it. The pack builds one in-process knowledge base
(offline :class:`DeterministicEmbedder`, in-memory storage) bound to a single ``kb_id``,
so ingest-then-search works within a run with no setup. Persistence across processes is
a pgvector-backed `KnowledgeBase` the caller can wire instead — out of scope here.
"""

from __future__ import annotations

from typing import Any

from himmy.services.knowledge import (
    DocumentInput,
    KnowledgeBase,
)
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.files import _safe_path

_DEFAULT_KB_ID = "default"

_INGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Raw text to ingest."},
        "path": {"type": "string", "description": "A file under the sandbox root."},
        "title": {"type": "string"},
        "source_uri": {"type": "string"},
    },
    "additionalProperties": False,
}

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def _build_durable_kb(
    dsn: str, embedder: Any, vector_dim: int, scope: tuple[str, str]
) -> tuple[KnowledgeBase, str]:
    """Build a Postgres+pgvector-backed KB, persisting across processes by name.

    ``scope`` is the ``(workspace_id, client_id)`` the KB is resolved/created under. On a
    SHARED durable backend a fixed ``("local", "local")`` scope would hand EVERY tenant the
    same KB (t1's chunks returned by t2's ``kb_search``); the caller passes the run's
    tenant-namespaced scope so each tenant gets its own isolated KB. ``("local", "local")``
    (the offline default) preserves the historical single shared KB byte-for-byte.
    """
    try:
        from himmy.services.storage.postgres import PostgresStorageService
    except Exception as exc:  # pragma: no cover - optional extra missing
        raise ValueError(
            "kb_dsn (durable knowledge) needs the 'postgres' extra: "
            "pip install 'himmy[postgres]'"
        ) from exc

    workspace_id, client_id = scope
    storage = await PostgresStorageService.connect(dsn)
    await storage.create_knowledge_schema(vector_dim=vector_dim)
    kb = KnowledgeBase(
        storage=storage,
        embedder=embedder,
        backend=storage.knowledge_backend(),
    )
    existing = await kb.resolve_kb(
        workspace_id=workspace_id, client_id=client_id, name=_DEFAULT_KB_ID
    )
    if existing is not None:
        return kb, existing.kb_id
    record = await kb.create_kb(
        workspace_id=workspace_id,
        client_id=client_id,
        name=_DEFAULT_KB_ID,
        vector_dim=vector_dim,
    )
    return kb, record.kb_id


def register_knowledge_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``kb_ingest`` and ``kb_search`` over a shared KB.

    By default the KB is in-process (per run). When ``config.kb_dsn`` is set, the KB is
    backed by Postgres + pgvector so ``kb_ingest``/``kb_search`` persist across processes
    (resolved by a fixed KB name); that path needs the ``postgres`` extra.
    """
    state: dict[str, Any] = {}
    embedder, vector_dim = config.build_embedder_and_dim()
    # P1 tenancy: the KB scope keys are namespaced by the run's ``tenant_scope`` when the
    # server threads it in, so a SHARED durable (pgvector) backend never hands one tenant's
    # ingested chunks to another's ``kb_search``. ``None`` → ("local", "local"), the
    # historical single shared KB — the offline/in-process default is byte-for-byte unchanged.
    kb_workspace_id, kb_client_id = config.scoped_kb_keys()

    async def _ensure_kb() -> tuple[KnowledgeBase, str]:
        """Build the KB (in-process or durable pgvector) on first use; cache it."""
        if "kb" not in state:
            if config.kb_dsn:
                kb, kb_id = await _build_durable_kb(
                    config.kb_dsn, embedder, vector_dim, (kb_workspace_id, kb_client_id)
                )
            else:
                kb = KnowledgeBase(storage=StorageService(), embedder=embedder)
                record = await kb.create_kb(
                    workspace_id=kb_workspace_id,
                    client_id=kb_client_id,
                    name=_DEFAULT_KB_ID,
                    vector_dim=vector_dim,
                )
                kb_id = record.kb_id
            state["kb"], state["kb_id"] = kb, kb_id
        return state["kb"], state["kb_id"]

    async def kb_ingest(args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("text")
        path = args.get("path")
        if not text and not path:
            raise ValueError("kb_ingest requires either 'text' or 'path'")
        if path:
            target = _safe_path(config.fs_root, str(path))
            doc = DocumentInput(
                file=str(target),
                title=args.get("title"),
                source_uri=args.get("source_uri") or str(path),
            )
        else:
            doc = DocumentInput(
                text=str(text),
                title=args.get("title"),
                source_uri=args.get("source_uri"),
            )
        kb, kb_id = await _ensure_kb()
        docs = await kb.ingest_documents(kb_id, [doc])
        return {"ingested": len(docs), "document_ids": [d.document_id for d in docs]}

    async def kb_search(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        top_k = max(1, min(int(args.get("top_k", 5)), 20))
        kb, kb_id = await _ensure_kb()
        chunks = await kb.search(kb_id, query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "text": c.text,
                    "similarity": c.similarity,
                    "source_uri": c.source_uri,
                }
                for c in chunks
            ],
        }

    register_local_tool(
        registry,
        name="kb_ingest",
        read_only=False,
        handler=kb_ingest,
        description="Ingest text or a file into the agent's knowledge base.",
        args_json_schema=_INGEST_SCHEMA,
        metadata={"pack": "knowledge"},
    )
    register_local_tool(
        registry,
        name="kb_search",
        read_only=True,
        handler=kb_search,
        description="Semantically search the agent's knowledge base.",
        args_json_schema=_SEARCH_SCHEMA,
        metadata={"pack": "knowledge"},
    )


__all__ = ["register_knowledge_pack"]
