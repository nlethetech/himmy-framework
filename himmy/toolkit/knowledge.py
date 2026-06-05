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
    DeterministicEmbedder,
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


def register_knowledge_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``kb_ingest`` and ``kb_search`` over a shared in-process KB."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    state: dict[str, str] = {}

    async def _ensure_kb_id() -> str:
        """Create the backing KB on first use and cache its id (dim matches embedder)."""
        if "kb_id" not in state:
            record = await kb.create_kb(
                workspace_id="local", client_id="local", name=_DEFAULT_KB_ID
            )
            state["kb_id"] = record.kb_id
        return state["kb_id"]

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
        docs = await kb.ingest_documents(await _ensure_kb_id(), [doc])
        return {"ingested": len(docs), "document_ids": [d.document_id for d in docs]}

    async def kb_search(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        top_k = max(1, min(int(args.get("top_k", 5)), 20))
        chunks = await kb.search(await _ensure_kb_id(), query, top_k=top_k)
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
        handler=kb_ingest,
        description="Ingest text or a file into the agent's knowledge base.",
        args_json_schema=_INGEST_SCHEMA,
        metadata={"pack": "knowledge"},
    )
    register_local_tool(
        registry,
        name="kb_search",
        handler=kb_search,
        description="Semantically search the agent's knowledge base.",
        args_json_schema=_SEARCH_SCHEMA,
        metadata={"pack": "knowledge"},
    )


__all__ = ["register_knowledge_pack"]
