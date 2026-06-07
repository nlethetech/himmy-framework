"""Memory service: remember facts and recall them semantically.

``remember`` persists a :class:`MemoryRecord` to the (possibly durable) store;
``recall`` ranks a subject's memories against a query by cosine similarity of their
embeddings. Recall reads from the store every time, so a :class:`SqliteMemoryStore`
makes memory survive restarts; embeddings are cached per memory id within the process.
The default embedder is the offline, deterministic one, so this works with no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from himmy.services.knowledge.embedder import DeterministicEmbedder, EmbedderProtocol
from himmy.services.memory.store import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryStore,
)


@dataclass
class MemoryHit:
    """A recalled memory paired with its similarity to the query."""

    record: MemoryRecord
    similarity: float


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryService:
    """Long-term memory: persist memories and recall them by semantic similarity."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        """Wire a (durable or in-memory) store and an embedder for recall."""
        self._store = store or InMemoryMemoryStore()
        self._embedder = embedder or DeterministicEmbedder()
        self._vectors: dict[str, list[float]] = {}

    def remember(
        self,
        text: str,
        *,
        subject_id: str = "default",
        kind: str = "semantic",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Persist ``text`` as a memory for ``subject_id`` and return the record."""
        record = MemoryRecord(
            subject_id=subject_id,
            kind=kind,
            text=text,
            metadata=dict(metadata or {}),
        )
        return self._store.save(record)

    async def recall(
        self, query: str, *, subject_id: str = "default", top_k: int = 5
    ) -> list[MemoryHit]:
        """Return the ``top_k`` memories most similar to ``query`` for ``subject_id``."""
        records = self._store.list(subject_id)
        if not records:
            return []
        missing = [r for r in records if r.memory_id not in self._vectors]
        if missing:
            vecs = await self._embedder.embed_documents([r.text for r in missing])
            for record, vec in zip(missing, vecs, strict=False):
                self._vectors[record.memory_id] = vec
        query_vec = await self._embedder.embed_query(query)
        scored = [
            MemoryHit(
                record=r, similarity=_cosine(query_vec, self._vectors[r.memory_id])
            )
            for r in records
        ]
        scored.sort(key=lambda h: h.similarity, reverse=True)
        return scored[: max(1, top_k)]

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by id (and drop its cached vector)."""
        self._vectors.pop(memory_id, None)
        return self._store.delete(memory_id)


__all__ = ["MemoryService", "MemoryHit"]
