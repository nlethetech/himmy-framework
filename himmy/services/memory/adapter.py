"""Memory context adapter: auto-inject recalled memories into a prompt.

Registered on a :class:`~himmy.services.context.service.ContextService`, this adapter
turns "remembered" facts into grounding context: when the runtime builds a snapshot for
a subject, it recalls the most relevant memories and returns them as a rendered
:class:`ContextField`, so the agent sees its long-term memory without any tool call.
"""

from __future__ import annotations

from typing import Any

from himmy.services.context.adapters import ContextAdapter
from himmy.services.context.models import ContextField
from himmy.services.memory.service import ALWAYS_INCLUDE, MemoryService


class MemoryContextAdapter(ContextAdapter):
    """A :class:`ContextAdapter` that recalls a subject's memories for a key."""

    name = "memory"

    def __init__(
        self,
        memory: MemoryService,
        *,
        top_k: int = 5,
        subject_id: str | None = None,
        similarity_threshold: float | None = None,
        tiers: tuple[str, ...] | None = None,
        active_only: bool = False,
        max_hops: int | None = None,
    ) -> None:
        """Wrap a :class:`MemoryService`; ``top_k`` caps how many memories inject.

        ``subject_id`` pins the recall subject (overriding the run's scope) so the
        adapter reads the same subject that facts were remembered under.
        ``similarity_threshold`` (optional) raises/lowers the recall floor that drops
        weakly-related memories so an off-topic turn injects no memory context instead
        of a best-of-the-irrelevant one; ``None`` (the default) uses the service's safe
        ``> 0.0`` noise floor. ``tiers`` (optional) restricts injection to the given
        Letta tiers — e.g.
        ``("core", "recall")`` injects always-on core facts plus thresholded recall
        facts but leaves ``archival`` for an explicit ``recall`` tool call; ``None``
        (the default) injects from every tier. ``active_only`` drops invalidated facts.

        ``max_hops`` (default ``None`` = off) opts into *relational* injection: after
        the semantic hits, each hit's graph neighbourhood is walked up to ``max_hops``
        hops (honouring ``active_only`` and the recall subject) and the related
        memories are injected as a separate, deduped ``related_memories`` section. With
        ``None`` the rendered output is byte-for-byte the historical behaviour.
        """
        self._memory = memory
        self._top_k = top_k
        self._subject_id = subject_id
        self._similarity_threshold = similarity_threshold
        self._tiers = tiers
        self._active_only = active_only
        self._max_hops = max_hops

    async def fetch(self, key: str, scope: dict[str, Any]) -> ContextField | None:
        """Recall memories for the scope's subject and render them as a field.

        When ``tiers`` is set, ``core`` facts are injected unconditionally (an
        always-in-context working set) and the remaining requested tiers go through
        the thresholded semantic recall — so an off-topic turn still sees core facts
        but no irrelevant recall noise.
        """
        subject_id = (
            self._subject_id
            or scope.get("subject_id")
            or scope.get("client_id")
            or "default"
        )
        query = str(
            scope.get("query") or scope.get("spec_metadata", {}).get("query") or ""
        )
        if not query:
            return None
        hits = await self._gather(subject_id, query)
        if not hits:
            return None
        rendered = "\n".join(f"- {h.record.text}" for h in hits)
        value: dict[str, Any] = {
            "rendered_text": rendered,
            "memories": [
                {
                    "text": h.record.text,
                    "similarity": h.similarity,
                    "tier": h.record.tier,
                }
                for h in hits
            ],
        }
        if self._max_hops is not None:
            related = await self._gather_related(subject_id, hits)
            if related:
                value["related_memories"] = [
                    {"text": r.text, "tier": r.tier} for r in related
                ]
                related_text = "\n".join(f"- {r.text}" for r in related)
                value["rendered_text"] = (
                    f"{rendered}\nRelated memories:\n{related_text}"
                )
        return ContextField(
            key=key,
            value=value,
            source="memory",
            confidence=hits[0].similarity,
        )

    async def _gather_related(self, subject_id: str, hits: list[Any]) -> list[Any]:
        """Walk each hit's graph neighbourhood and return deduped related memories.

        Deduped against the semantic hits and each other; the seed hits themselves are
        excluded (they already render in the main section). Returns a list of
        :class:`~himmy.services.memory.store.MemoryRecord`. Only called when
        ``max_hops`` is set, so the ``None`` path injects no related section at all.
        """
        max_hops = self._max_hops
        if max_hops is None:  # pragma: no cover - guarded by caller
            return []
        seen = {h.record.memory_id for h in hits}
        related: list[Any] = []
        for hit in hits:
            graph = await self._memory.traverse_graph(
                hit.record.memory_id,
                max_depth=max_hops,
                active_only=self._active_only,
                subject_id=subject_id,
            )
            for memory_id, record in graph.nodes.items():
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                related.append(record)
        return related

    async def _gather(self, subject_id: str, query: str) -> list[Any]:
        """Collect the hits to inject, honoring tier semantics (core always-on)."""
        if self._tiers is None:
            return await self._memory.recall(
                query,
                subject_id=subject_id,
                top_k=self._top_k,
                similarity_threshold=self._similarity_threshold,
                active_only=self._active_only,
            )
        collected: list[Any] = []
        seen: set[str] = set()
        for tier in self._tiers:
            # ``core`` is the always-in-context working set: inject it regardless of
            # query similarity (the explicit always-include path). Other tiers go
            # through the floor (``None`` is the service's safe ``> 0.0`` noise floor).
            threshold = ALWAYS_INCLUDE if tier == "core" else self._similarity_threshold
            tier_hits = await self._memory.recall(
                query,
                subject_id=subject_id,
                top_k=self._top_k,
                similarity_threshold=threshold,
                tier=tier,
                active_only=self._active_only,
            )
            for hit in tier_hits:
                if hit.record.memory_id not in seen:
                    seen.add(hit.record.memory_id)
                    collected.append(hit)
        return collected


__all__ = ["MemoryContextAdapter"]
