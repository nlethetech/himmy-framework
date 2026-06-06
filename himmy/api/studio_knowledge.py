"""Studio Knowledge: create a knowledge base, ingest text, and test retrieval.

Wraps :class:`~himmy.services.knowledge.service.KnowledgeBase` as a process-wide
singleton scoped to a single local user (fixed workspace/client). The default store
is in-memory (persists for the session); set ``HIMMY_KB_DSN`` to back it with
Postgres+pgvector for durability. The same ``kb_search`` an agent uses reads it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_WORKSPACE = "studio"
_CLIENT = "local"

_KB: Any = None
_NAMES: dict[str, str] = {}  # name → kb_id (this session)
_DOCS: dict[str, int] = {}  # kb_id → ingested doc count


class KbInfo(BaseModel):
    kb_id: str
    name: str
    documents: int = 0


class IngestResult(BaseModel):
    document_id: str
    title: str | None = None


class SearchHit(BaseModel):
    text: str
    similarity: float
    source_uri: str | None = None


def _service() -> Any:
    global _KB
    if _KB is None:
        from himmy.runtime.builder import build_storage
        from himmy.services.knowledge.service import KnowledgeBase
        from himmy.toolkit.config import ToolkitConfig

        embedder, _dim = ToolkitConfig.from_env().build_embedder_and_dim()
        _KB = KnowledgeBase(storage=build_storage(), embedder=embedder)
    return _KB


def reset_kb_service() -> None:
    global _KB, _NAMES, _DOCS
    _KB = None
    _NAMES = {}
    _DOCS = {}


def list_kbs() -> list[KbInfo]:
    return [
        KbInfo(kb_id=kb_id, name=name, documents=_DOCS.get(kb_id, 0))
        for name, kb_id in _NAMES.items()
    ]


async def create_kb(name: str) -> KbInfo:
    if name in _NAMES:
        return KbInfo(
            kb_id=_NAMES[name], name=name, documents=_DOCS.get(_NAMES[name], 0)
        )
    rec = await _service().create_kb(
        workspace_id=_WORKSPACE, client_id=_CLIENT, name=name
    )
    _NAMES[name] = rec.kb_id
    _DOCS[rec.kb_id] = 0
    return KbInfo(kb_id=rec.kb_id, name=name, documents=0)


async def ingest_text(
    kb_id: str, text: str, *, title: str | None = None
) -> IngestResult:
    doc = await _service().ingest_text(kb_id, text, title=title)
    _DOCS[kb_id] = _DOCS.get(kb_id, 0) + 1
    return IngestResult(document_id=doc.document_id, title=doc.title)


async def search(kb_id: str, query: str, *, top_k: int = 5) -> list[SearchHit]:
    chunks = await _service().search(kb_id, query, top_k=top_k)
    return [
        SearchHit(
            text=c.text or c.context_window or "",
            similarity=c.similarity,
            source_uri=c.source_uri,
        )
        for c in chunks
    ]


async def delete_kb(kb_id: str) -> bool:
    await _service().delete_kb(kb_id, workspace_id=_WORKSPACE, client_id=_CLIENT)
    for name, kid in list(_NAMES.items()):
        if kid == kb_id:
            del _NAMES[name]
    _DOCS.pop(kb_id, None)
    return True


__all__ = [
    "KbInfo",
    "IngestResult",
    "SearchHit",
    "reset_kb_service",
    "list_kbs",
    "create_kb",
    "ingest_text",
    "search",
    "delete_kb",
]
