"""Studio Memory: browse, add, and recall an agent's long-term memory.

Wraps :class:`~himmy.services.memory.service.MemoryService` over a durable
:class:`SqliteMemoryStore` at ``.himmy/memory.db`` (the same store an agent uses when
``HIMMY_MEMORY_PATH`` points here), so what you add in the GUI is what the ``memory``
tool pack recalls. The store + embedder are a process-wide singleton (cwd-keyed).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from himmy.services.memory.service import MemoryService
from himmy.services.memory.store import SqliteMemoryStore


class MemoryItem(BaseModel):
    memory_id: str
    subject_id: str
    kind: str
    text: str
    created_at: str


class MemoryHitItem(BaseModel):
    memory_id: str
    text: str
    similarity: float


_SERVICE: MemoryService | None = None
_STORE: SqliteMemoryStore | None = None
_PATH: str | None = None


def _db_path() -> str:
    import os

    env = os.environ.get("HIMMY_MEMORY_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "memory.db")


def get_memory_service() -> MemoryService:
    """Process-wide memory service over the durable store (cwd/env-keyed)."""
    global _SERVICE, _STORE, _PATH
    path = _db_path()
    if _SERVICE is None or _PATH != path:
        from himmy.api.studio_entities import get_entity_registry
        from himmy.toolkit.config import ToolkitConfig

        if _STORE is not None:
            _STORE.close()
        _STORE = SqliteMemoryStore(path)
        embedder, _dim = ToolkitConfig.from_env().build_embedder_and_dim()
        # P0-C: project every memory mutation onto the durable Studio entity spine
        # (memory_fact records + MEMORY_* audit events) — the dormant audit path,
        # now on. The registry is the same durable spine run feedback writes to.
        # SqliteEntityRegistry satisfies EntityRegistryProtocol (no cast needed).
        _SERVICE = MemoryService(
            _STORE,
            embedder=embedder,
            registry=get_entity_registry(),
        )
        _PATH = path
    return _SERVICE


def reset_memory_service() -> None:
    global _SERVICE, _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _SERVICE = None
    _STORE = None
    _PATH = None


def _store() -> SqliteMemoryStore:
    get_memory_service()
    assert _STORE is not None
    return _STORE


def list_subjects() -> list[str]:
    """Distinct subject ids that have memories (newest-active first)."""
    seen: list[str] = []
    for r in _store().list():
        if r.subject_id not in seen:
            seen.append(r.subject_id)
    return sorted(seen) or ["default"]


def list_memories(subject_id: str) -> list[MemoryItem]:
    """All memories for a subject, newest first."""
    records = _store().list(subject_id)
    records.sort(key=lambda r: r.created_at, reverse=True)
    return [
        MemoryItem(
            memory_id=r.memory_id,
            subject_id=r.subject_id,
            kind=r.kind,
            text=r.text,
            created_at=r.created_at,
        )
        for r in records
    ]


def add_memory(
    text: str, *, subject_id: str = "default", kind: str = "semantic"
) -> MemoryItem:
    rec = get_memory_service().remember(text, subject_id=subject_id, kind=kind)
    return MemoryItem(
        memory_id=rec.memory_id,
        subject_id=rec.subject_id,
        kind=rec.kind,
        text=rec.text,
        created_at=rec.created_at,
    )


def forget(memory_id: str) -> bool:
    return get_memory_service().forget(memory_id)


async def recall(
    query: str,
    *,
    subject_id: str = "default",
    top_k: int = 5,
    similarity_threshold: float | None = None,
    as_of: str | None = None,
    active_only: bool = False,
) -> list[MemoryHitItem]:
    hits = await get_memory_service().recall(
        query,
        subject_id=subject_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        as_of=as_of,
        active_only=active_only,
    )
    return [
        MemoryHitItem(
            memory_id=h.record.memory_id, text=h.record.text, similarity=h.similarity
        )
        for h in hits
    ]


class MemoryTimelineItem(BaseModel):
    """One point in a fact's bi-temporal life (for the GUI timeline)."""

    memory_id: str
    text: str
    valid_from: str
    valid_to: str | None
    superseded_by: str | None
    tier: str
    source: str


def list_memory_history(
    stable_key: str, *, subject_id: str = "default"
) -> list[MemoryTimelineItem]:
    """All versions of a fact (by ``stable_key``) ordered oldest-first.

    Surfaces the bi-temporal evolution of a single fact — original value, each
    supersession, and the current truth — so a GUI can render "how this changed
    over time". Falls back to a single record when a fact was remembered without an
    explicit ``stable_key`` (its ``memory_id`` is its own key).
    """
    records = [
        r
        for r in _store().list(subject_id)
        if (r.stable_key or r.memory_id) == stable_key
    ]
    records.sort(key=lambda r: r.valid_from)
    return [
        MemoryTimelineItem(
            memory_id=r.memory_id,
            text=r.text,
            valid_from=r.valid_from,
            valid_to=r.valid_to,
            superseded_by=r.superseded_by,
            tier=r.tier,
            source=r.source,
        )
        for r in records
    ]


__all__ = [
    "MemoryItem",
    "MemoryHitItem",
    "MemoryTimelineItem",
    "get_memory_service",
    "reset_memory_service",
    "list_subjects",
    "list_memories",
    "add_memory",
    "forget",
    "recall",
    "list_memory_history",
]
