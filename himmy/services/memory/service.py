"""Memory service: remember facts and recall them semantically and temporally.

``remember`` persists a :class:`MemoryRecord` to the (possibly durable) store;
``recall`` ranks a subject's memories against a query by cosine similarity of their
embeddings. Recall reads from the store every time, so a :class:`SqliteMemoryStore`
makes memory survive restarts; embeddings are cached per memory id within the process.
The default embedder is the offline, deterministic one, so this works with no network.

Beyond ranking, recall is *bi-temporal*: ``as_of`` answers "what was true at this
instant", ``active_only`` drops invalidated facts, and ``tier`` scopes to one Letta
tier. When a registry / event sink is wired (both optional, default off), every write
is also projected onto the EntityRecord spine as a ``memory_fact`` and audited with a
``MEMORY_REMEMBERED`` event — so memory lives ON the spine, not beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from himmy.core.events import EventType, RunEvent
from himmy.services.knowledge.embedder import DeterministicEmbedder, EmbedderProtocol
from himmy.services.memory.graph import (
    DEFAULT_TRAVERSAL_DEPTH,
    traverse_memory_graph,
)
from himmy.services.memory.store import (
    MEMORY_TIERS,
    RELATES_TO,
    InMemoryMemoryStore,
    MemoryGraph,
    MemoryLink,
    MemoryRecord,
    MemoryStore,
)
from himmy.services.memory.temporal import filter_as_of, is_valid_at

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.core.events import EventSink
    from himmy.entities.protocol import EntityRegistryProtocol


class _AlwaysInclude:
    """Sentinel ``similarity_threshold`` requesting an unconditional recall.

    Distinct from ``None`` (the general default, which now applies a ``> 0.0`` noise
    floor): passing :data:`ALWAYS_INCLUDE` returns the top hits regardless of similarity,
    preserving the ``core`` tier's "always-in-context working set" contract.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "ALWAYS_INCLUDE"


ALWAYS_INCLUDE = _AlwaysInclude()
"""Sentinel for :meth:`MemoryService.recall`'s ``similarity_threshold``: always return."""


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
    """Long-term memory: persist memories and recall them by semantic similarity.

    A registry and/or event sink may be injected (both default ``None``) to project
    each write onto the EntityRecord spine and audit it through the run-event stream.
    When neither is wired the service behaves exactly as before — a pure save/recall
    over the store, with no spine side effects.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        embedder: EmbedderProtocol | None = None,
        min_similarity: float | None = None,
        registry: EntityRegistryProtocol | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        """Wire a (durable or in-memory) store and an embedder for recall.

        ``min_similarity`` is a default recall floor applied when a call does not pass
        its own ``similarity_threshold``; ``None`` (the default) applies the safe
        ``> 0.0`` noise floor (an orthogonal query recalls nothing) — mirroring the
        sibling knowledge service. Pass a float to raise/lower that floor. The
        ``core``-tier always-in-context path uses :data:`ALWAYS_INCLUDE` to bypass it.
        ``registry``/``event_sink`` opt the service into spine projection + audit
        (both default off).
        """
        self._store = store or InMemoryMemoryStore()
        self._embedder = embedder or DeterministicEmbedder()
        self._vectors: dict[str, list[float]] = {}
        self._min_similarity = min_similarity
        self._registry = registry
        self._event_sink = event_sink

    @property
    def store(self) -> MemoryStore:
        """The underlying (possibly durable) memory store."""
        return self._store

    @property
    def embedder(self) -> EmbedderProtocol:
        """The embedder used for recall ranking."""
        return self._embedder

    def remember(
        self,
        text: str,
        *,
        subject_id: str = "default",
        kind: str = "semantic",
        metadata: dict[str, Any] | None = None,
        tier: str = "recall",
        source: str = "user",
        confidence: float = 1.0,
        stable_key: str | None = None,
        project_version: int = 1,
    ) -> MemoryRecord:
        """Persist ``text`` as a memory for ``subject_id`` and return the record.

        With a registry/event sink wired, the saved fact is also projected onto the
        spine as a ``memory_fact`` record and a ``MEMORY_REMEMBERED`` event is emitted;
        without them this is a pure store write (unchanged from the historical API).
        ``project_version`` is the spine version a consolidator uses when a new fact
        continues an existing fact's version chain (a shared ``stable_key``); it
        defaults to ``1`` for a fresh fact.
        """
        record = MemoryRecord(
            subject_id=subject_id,
            kind=kind,
            text=text,
            metadata=dict(metadata or {}),
            tier=tier if tier in MEMORY_TIERS else "recall",
            source=source,
            confidence=confidence,
            stable_key=stable_key,
        )
        saved = self._store.save(record)
        self._project(saved, version=project_version)
        self._emit(
            EventType.MEMORY_REMEMBERED,
            {
                "memory_id": saved.memory_id,
                "subject_id": saved.subject_id,
                "tier": saved.tier,
                "source": saved.source,
            },
        )
        return saved

    async def recall(
        self,
        query: str,
        *,
        subject_id: str = "default",
        top_k: int = 5,
        similarity_threshold: float | _AlwaysInclude | None = None,
        as_of: str | None = None,
        tier: str | None = None,
        active_only: bool = False,
        max_hops: int | None = None,
    ) -> list[MemoryHit]:
        """Return the ``top_k`` memories most similar to ``query`` for ``subject_id``.

        ``similarity_threshold`` closes the "best-of-the-irrelevant" footgun, mirroring
        the sibling knowledge service's noise floor:

        - ``None`` (and no service-level ``min_similarity``) applies the safe ``> 0.0``
          noise floor: a hit with zero overlap is dropped, so an orthogonal query
          correctly recalls nothing instead of surfacing a similarity-``0.0`` hit.
        - An explicit float (or the service default ``min_similarity``) keeps every hit
          with ``similarity >= threshold`` — so a caller who deliberately wants ``0.0``
          (or any cutoff) gets exactly that.
        - :data:`ALWAYS_INCLUDE` bypasses the floor entirely and returns the top hits
          regardless of similarity — the ``core``-tier always-in-context working set.

        ``as_of`` (ISO timestamp) restricts to facts whose ``[valid_from, valid_to)``
        window contains that instant — bi-temporal point-in-time recall. ``active_only``
        drops invalidated facts; ``tier`` scopes to one Letta tier. The result is
        capped at ``top_k`` and ordered most-similar-first.

        ``max_hops`` (default ``None`` = off, exactly the historical behaviour) opts
        into *relational* recall: after ranking, each hit's graph neighbourhood is
        walked up to ``max_hops`` hops (honouring the same ``as_of``/``active_only``/
        ``tier``/``subject_id`` scoping) and the related memories are appended as extra
        hits (similarity ``0.0``, deduped against the semantic hits and each other).
        The semantic ``top_k`` hits always lead; related memories follow.
        """
        records = self._store.list(subject_id, active_only=active_only, tier=tier)
        if as_of is not None:
            records = filter_as_of(records, as_of)
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
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._min_similarity
        )
        if isinstance(threshold, _AlwaysInclude):
            # Always-in-context working set (e.g. the core tier): no floor at all.
            hits = scored[: max(1, top_k)]
        elif threshold is None:
            # General default: the safe ``> 0.0`` noise floor — an orthogonal query
            # recalls nothing instead of a best-of-the-irrelevant similarity-0.0 hit.
            hits = [h for h in scored if h.similarity > 0.0][:top_k]
        else:
            hits = [h for h in scored if h.similarity >= threshold][:top_k]
        if max_hops is not None and max_hops > 0 and hits:
            hits = self._expand_with_related(
                hits,
                max_hops=max_hops,
                as_of=as_of,
                active_only=active_only,
                tier=tier,
                subject_id=subject_id,
            )
        self._emit(
            EventType.MEMORY_RECALLED,
            {
                "subject_id": subject_id,
                # Privacy: the raw query is a user prompt — record its size, not its
                # text, so the audit spine never persists PII (P0-C). The returned
                # count + top similarity remain as the useful, non-sensitive signal.
                "query_chars": len(query),
                "returned": len(hits),
                "top_similarity": hits[0].similarity if hits else 0.0,
            },
        )
        return hits

    def _expand_with_related(
        self,
        hits: list[MemoryHit],
        *,
        max_hops: int,
        as_of: str | None,
        active_only: bool,
        tier: str | None,
        subject_id: str,
    ) -> list[MemoryHit]:
        """Append each hit's related memories (graph-reachable) as similarity-0.0 hits.

        Deduped against the semantic hits and each other; the semantic hits always
        lead and keep their order. Used only when ``recall(max_hops=...)`` is set.
        """
        seen = {h.record.memory_id for h in hits}
        expanded = list(hits)
        for hit in hits:
            graph = self._traverse(
                hit.record.memory_id,
                max_depth=max_hops,
                as_of=as_of,
                active_only=active_only,
                tier=tier,
                subject_id=subject_id,
            )
            for memory_id, record in graph.nodes.items():
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                expanded.append(MemoryHit(record=record, similarity=0.0))
        return expanded

    # -- relational / graph memory ---------------------------------------------

    def link(
        self,
        from_memory_id: str,
        to_memory_id: str,
        *,
        relation: str = RELATES_TO,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryLink:
        """Draw a typed directed link from one memory to another and persist it.

        ``relation`` is an open string; the canonical vocabulary lives in
        :data:`himmy.services.memory.store.MEMORY_RELATIONS` (``relates_to``,
        ``caused_by``, ``contradicts``, ``about_entity``, ``supersedes``, ``part_of``,
        ``depends_on``). The link is stored alongside the memories so a later
        :meth:`traverse_graph` (or ``recall(max_hops=...)``) can walk it. Persisting a
        link is content-neutral, so unlike :meth:`remember` it draws no spine record
        and emits no event (graph edges are read-side only).
        """
        link = MemoryLink(
            from_memory_id=from_memory_id,
            to_memory_id=to_memory_id,
            relation=relation,
            metadata=dict(metadata or {}),
        )
        return self._store.save_link(link)

    async def traverse_graph(
        self,
        seed_memory_id: str,
        *,
        max_depth: int = DEFAULT_TRAVERSAL_DEPTH,
        as_of: str | None = None,
        active_only: bool = False,
        tier: str | None = None,
        subject_id: str | None = None,
    ) -> MemoryGraph:
        """Walk the typed memory graph from ``seed_memory_id`` and return the subgraph.

        Breadth-first up to ``max_depth`` hops over the persisted :class:`MemoryLink`s
        (both directions), honouring the same scoping recall uses: ``as_of`` skips
        endpoints whose ``[valid_from, valid_to)`` window does not contain that instant
        (bi-temporal point-in-time), ``active_only`` drops invalidated endpoints,
        ``tier`` scopes to one Letta tier, and ``subject_id`` (optional) confines the
        walk to one subject — an endpoint failing any filter is pruned (not surfaced
        and not expanded). The returned :class:`MemoryGraph` carries the reached
        records and traversed links; ``truncated`` is True when the depth budget cut
        off an accepted, reachable, unexpanded neighbour.

        ``async`` for API symmetry with :meth:`recall` (it does no awaiting itself).
        """
        return self._traverse(
            seed_memory_id,
            max_depth=max_depth,
            as_of=as_of,
            active_only=active_only,
            tier=tier,
            subject_id=subject_id,
        )

    def _traverse(
        self,
        seed_memory_id: str,
        *,
        max_depth: int,
        as_of: str | None,
        active_only: bool,
        tier: str | None,
        subject_id: str | None,
    ) -> MemoryGraph:
        """Shared sync traversal core for ``traverse_graph`` and ``recall`` expansion.

        Builds the neighbour + endpoint-acceptance callables (the I/O policy) and
        delegates the pure walk to :func:`traverse_memory_graph`.
        """
        store = self._store

        def neighbors(memory_id: str) -> list[MemoryLink]:
            # Combine both directions; the pure walker dedups edges by ``link_id``.
            return [*store.links_from(memory_id), *store.links_to(memory_id)]

        def accept(record: MemoryRecord) -> bool:
            if subject_id is not None and record.subject_id != subject_id:
                return False
            if active_only and record.valid_to is not None:
                return False
            if tier is not None and record.tier != tier:
                return False
            if as_of is not None and not is_valid_at(record, as_of):
                return False
            return True

        return traverse_memory_graph(
            seed_memory_id,
            neighbors=neighbors,
            get=store.get,
            accept=accept,
            max_depth=max_depth,
        )

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by id (and drop its cached vector)."""
        self._vectors.pop(memory_id, None)
        return self._store.delete(memory_id)

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Return one memory by id, or ``None``."""
        return self._store.get(memory_id)

    def invalidate(
        self, memory_id: str, *, valid_to: str, superseded_by: str | None = None
    ) -> MemoryRecord | None:
        """Close a fact's validity window (Graphiti invalidate-not-delete).

        Stamps ``valid_to`` (and optionally ``superseded_by``) on the stored record so
        the fact disappears from ``active_only``/``as_of>=valid_to`` recall while
        remaining queryable as of an earlier instant. The spine record is NOT mutated:
        ``valid_to`` lives in record metadata (not the content-addressed payload), so a
        validity transition is expressed by the typed ``invalidated_by``/``supersedes``
        EntityLinks the consolidator draws — never by editing a registered payload
        (which would trip the content-address-violation guard). Returns the updated
        record, or ``None`` when the id is unknown.
        """
        record = self._store.get(memory_id)
        if record is None:
            return None
        updated = record.model_copy(
            update={"valid_to": valid_to, "superseded_by": superseded_by}
        )
        return self._store.save(updated)

    def promote(self, memory_id: str, tier: str) -> MemoryRecord | None:
        """Move a fact to a different Letta tier (core/recall/archival), audited.

        Tier changes are content-neutral, so this re-projects the fact as a new spine
        version and emits a ``MEMORY_CONSOLIDATED`` event recording the move — the
        same audited-self-edit discipline the consolidator uses.
        """
        if tier not in MEMORY_TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {MEMORY_TIERS}")
        record = self._store.get(memory_id)
        if record is None:
            return None
        if record.tier == tier:
            return record
        old_tier = record.tier
        updated = record.model_copy(update={"tier": tier})
        saved = self._store.save(updated)
        self._project(saved, version=2)
        self._emit(
            EventType.MEMORY_CONSOLIDATED,
            {
                "action": "PROMOTE",
                "memory_id": saved.memory_id,
                "from_tier": old_tier,
                "to_tier": tier,
            },
        )
        return saved

    # -- spine projection + audit (no-ops when registry/sink are absent) -------

    def _project(self, record: MemoryRecord, *, version: int) -> None:
        """Register the fact's ``memory_fact`` record on the spine (best-effort)."""
        if self._registry is None:
            return
        try:
            self._registry.register(record.to_record(version=version))
        except Exception:  # pragma: no cover - defensive, a sink must not break a write
            pass

    def _emit(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Project an audit RunEvent to the registry + best-effort to the sink.

        Mirrors the runtime ``_emit`` fan-out: the event lands as a ``run_event``
        record on the spine (synchronously) and is offered to an async ``EventSink``
        when one is wired and an event loop is running. Each sink is isolated so a
        memory write is never broken by an audit failure.
        """
        if self._registry is None and self._event_sink is None:
            return
        event = RunEvent(event_type=event_type, payload=payload)
        if self._registry is not None:
            try:
                self._registry.register(event.to_record())
            except Exception:  # pragma: no cover - defensive
                pass
        if self._event_sink is not None:
            self._dispatch_to_sink(event)

    def _dispatch_to_sink(self, event: RunEvent) -> None:
        """Hand an event to the async sink, scheduling on the running loop if any."""
        import asyncio

        sink = self._event_sink
        if sink is None:  # pragma: no cover - guarded by caller
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(sink.append_event(event))
        else:
            try:
                asyncio.run(sink.append_event(event))
            except Exception:  # pragma: no cover - defensive
                pass


__all__ = [
    "ALWAYS_INCLUDE",
    "MemoryGraph",
    "MemoryHit",
    "MemoryLink",
    "MemoryService",
]
