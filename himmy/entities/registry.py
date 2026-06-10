"""Entities kernel: the in-memory entity registry (lineage backbone)."""

from __future__ import annotations

import threading
from collections.abc import Collection
from typing import Any

from himmy.core.errors import HimmyError
from himmy.entities.lineage import (
    DEFAULT_TRACE_DEPTH,
    LineageDirection,
    LineageGraph,
)
from himmy.entities.records import (
    EntityLink,
    EntityQuery,
    EntityRecord,
    metadata_contains,
    record_id_for,
)

#: Record kinds that form the audit spine (security/audit history). Records of
#: these kinds get the write-once guard in :class:`EntityRegistry`: the registry
#: stores and returns DEEP COPIES so no caller-held reference aliases the stored
#: ``payload``/``metadata`` dicts.
AUDIT_RECORD_KINDS: frozenset[str] = frozenset(
    {"security_event", "consent", "erasure_tombstone", "privacy_audit_report"}
)


class EntityRegistry:
    """An in-memory, versioned registry of entity records and their links.

    Records are immutable and keyed by ``record_id``; versions of the same
    artefact share a ``stable_id``. Registration is idempotent on ``record_id``.

    **Write-once guard for audit kinds.** :class:`EntityRecord` is a frozen model,
    but its ``payload``/``metadata`` dicts are mutable, so a stored record shared
    by reference could have its history rewritten through that alias. For the
    audit kinds (``audit_kinds``, default :data:`AUDIT_RECORD_KINDS`) the registry
    therefore stores a deep copy on write and hands out deep copies on read —
    mutating any record a caller holds can never alter the stored history, and
    re-registering an existing audit record with different content raises (as for
    every kind). NOTE the weaker offline guarantee: this is an in-process,
    cooperative defence only — code that reaches into the registry's private state
    (or simply swaps the registry object) is not stopped. For evidence that holds
    up after the fact, pair it with the signed exports in
    :mod:`himmy.entities.integrity` (Merkle bundle, or the opt-in hash chain for
    ordered trails) or a durable backend.

    **Thread safety.** All reads and writes are serialised behind one reentrant
    process-level lock, mirroring the SQLite backend's write lock. Without it the
    read-latest -> insert pair in :meth:`new_version` races: two threads both read
    version N and one update is silently lost (or surfaces as a spurious
    content-address violation), corrupting the version chain of the audit spine.
    """

    def __init__(self, *, audit_kinds: Collection[str] | None = None) -> None:
        self._records: dict[str, EntityRecord] = {}
        self._by_stable_id: dict[str, list[EntityRecord]] = {}
        self._links: list[EntityLink] = []
        self._audit_kinds: frozenset[str] = (
            AUDIT_RECORD_KINDS if audit_kinds is None else frozenset(audit_kinds)
        )
        # Reentrant: new_version() -> register() and trace() -> neighbors() -> get()
        # all re-acquire the same lock on the same thread.
        self._lock = threading.RLock()

    def _shield(self, record: EntityRecord) -> EntityRecord:
        """Return a deep copy for audit-kind records (write-once guard); else as-is.

        Deep-copying severs every dict alias between the stored record and what a
        caller holds, so neither side can mutate the other. Non-audit kinds keep
        the zero-copy fast path and identity semantics unchanged.
        """
        if record.kind in self._audit_kinds:
            return record.model_copy(deep=True)
        return record

    def register(self, record: EntityRecord) -> EntityRecord:
        """Register a record. Idempotent on identical content; raises on collision.

        Record identity is ``(kind, stable_id, version)``. Re-registering byte-for-
        byte identical content (same payload + metadata) returns the stored record.
        Re-registering the SAME id with DIFFERENT payload/metadata is a true
        content-address violation and raises :class:`HimmyError` — callers who
        want to evolve content must use :meth:`new_version`.

        Audit-kind records are stored AND returned as deep copies (write-once
        guard) so the caller's dicts never alias the stored history.
        """
        with self._lock:
            existing = self._records.get(record.record_id)
            if existing is not None:
                if (
                    existing.payload != record.payload
                    or existing.metadata != record.metadata
                ):
                    raise HimmyError(
                        "Content-address violation for "
                        f"record_id={record.record_id!r} "
                        f"(kind={record.kind!r}, stable_id={record.stable_id!r}, "
                        f"version={record.version}): an existing record with "
                        "different payload/metadata is already registered. Use "
                        "new_version() to evolve content."
                    )
                return self._shield(existing)
            stored = self._shield(record)
            self._records[stored.record_id] = stored
            versions = self._by_stable_id.setdefault(stored.stable_id, [])
            versions.append(stored)
            versions.sort(key=lambda r: r.version)
            return self._shield(stored)

    def new_version(
        self,
        *,
        stable_id: str,
        kind: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> EntityRecord:
        """Create the next version of an artefact with optimistic concurrency.

        If ``expected_version`` is supplied and does not match the current latest
        version, an :class:`HimmyError` is raised. The read-latest + register pair
        runs under the registry lock, so concurrent versioning serialises instead
        of silently losing an update.
        """
        with self._lock:
            latest = self.get_latest(stable_id)
            current_version = latest.version if latest is not None else 0
            if expected_version is not None and expected_version != current_version:
                raise HimmyError(
                    f"Optimistic concurrency conflict for stable_id={stable_id!r}: "
                    f"expected version {expected_version}, found {current_version}."
                )
            record = EntityRecord.create(
                stable_id=stable_id,
                version=current_version + 1,
                kind=kind,
                payload=payload,
                metadata=metadata or {},
            )
            return self.register(record)

    def link(
        self,
        *,
        from_record_id: str,
        to_record_id: str,
        relation: str,
        metadata: dict[str, Any] | None = None,
    ) -> EntityLink:
        """Create and store a typed link between two records."""
        link = EntityLink(
            from_record_id=from_record_id,
            to_record_id=to_record_id,
            relation=relation,
            metadata=metadata or {},
        )
        with self._lock:
            self._links.append(link)
        return link

    def get(self, record_id: str) -> EntityRecord | None:
        """Return a record by its physical id, or None (audit kinds: a deep copy)."""
        with self._lock:
            record = self._records.get(record_id)
            return self._shield(record) if record is not None else None

    def get_latest(self, stable_id: str) -> EntityRecord | None:
        """Return the highest-version record for an artefact, or None."""
        with self._lock:
            versions = self._by_stable_id.get(stable_id)
            if not versions:
                return None
            return self._shield(versions[-1])

    def get_history(self, stable_id: str) -> list[EntityRecord]:
        """Return all versions of an artefact in ascending version order."""
        with self._lock:
            return [self._shield(r) for r in self._by_stable_id.get(stable_id, [])]

    def list_by_kind(self, kind: str) -> list[EntityRecord]:
        """Return every record of the given kind."""
        with self._lock:
            return [self._shield(r) for r in self._records.values() if r.kind == kind]

    def query(self, q: EntityQuery) -> list[EntityRecord]:
        """Return records matching the supplied query filter.

        ``kind`` and ``stable_id`` are optional (metadata-only / stable_id-only
        queries are valid). ``metadata_filters`` uses JSONB-containment semantics
        (``metadata_contains``), equivalent to the Postgres ``@>`` operator.
        """
        results: list[EntityRecord] = []
        with self._lock:
            for record in self._records.values():
                if q.kind is not None and record.kind != q.kind:
                    continue
                if q.stable_id is not None and record.stable_id != q.stable_id:
                    continue
                if q.metadata_filters and not metadata_contains(
                    record.metadata, q.metadata_filters
                ):
                    continue
                results.append(self._shield(record))
        return results

    def links_from(self, record_id: str) -> list[EntityLink]:
        """Return all links originating from a record."""
        with self._lock:
            return [link for link in self._links if link.from_record_id == record_id]

    def links_to(self, record_id: str) -> list[EntityLink]:
        """Return all links pointing INTO a record (the reverse of ``links_from``).

        This is the read primitive the documented "trace any output back to its
        persona/evidence" promise needs; the data was always captured, the reverse
        query was simply never exposed.
        """
        with self._lock:
            return [link for link in self._links if link.to_record_id == record_id]

    def neighbors(
        self,
        record_id: str,
        *,
        direction: LineageDirection = LineageDirection.BOTH,
        relation: str | None = None,
    ) -> list[EntityLink]:
        """Return the links incident to a record in the requested direction.

        ``OUT`` = edges leaving the record, ``IN`` = edges entering it, ``BOTH`` =
        either. When ``relation`` is given, only edges of that relation are returned.
        Self-loops are de-duplicated so ``BOTH`` never reports the same link twice.
        """
        seen: set[str] = set()
        result: list[EntityLink] = []
        candidates: list[EntityLink] = []
        if direction in (LineageDirection.OUT, LineageDirection.BOTH):
            candidates.extend(self.links_from(record_id))
        if direction in (LineageDirection.IN, LineageDirection.BOTH):
            candidates.extend(self.links_to(record_id))
        for link in candidates:
            if relation is not None and link.relation != relation:
                continue
            if link.link_id in seen:
                continue
            seen.add(link.link_id)
            result.append(link)
        return result

    def trace(
        self,
        record_id: str,
        *,
        max_depth: int = DEFAULT_TRACE_DEPTH,
        direction: LineageDirection = LineageDirection.BOTH,
        relations: Collection[str] | None = None,
    ) -> LineageGraph:
        """Walk the lineage graph from ``record_id`` and return the reached subgraph.

        Breadth-first up to ``max_depth`` hops, following edges in ``direction`` and
        (optionally) restricted to ``relations``. The returned :class:`LineageGraph`
        carries the records reached and the edges traversed; ``truncated`` is set
        when the depth budget stopped the walk while reachable nodes remained.
        """
        rel_set = set(relations) if relations is not None else None
        nodes: dict[str, EntityRecord] = {}
        edges: list[EntityLink] = []
        seen_edges: set[str] = set()
        visited: set[str] = {record_id}

        root = self.get(record_id)
        if root is not None:
            nodes[record_id] = root

        frontier = [record_id]
        for _ in range(max(0, max_depth)):
            if not frontier:
                break
            next_frontier: list[str] = []
            for rid in frontier:
                for link in self.neighbors(rid, direction=direction):
                    if rel_set is not None and link.relation not in rel_set:
                        continue
                    if link.link_id not in seen_edges:
                        seen_edges.add(link.link_id)
                        edges.append(link)
                    other = (
                        link.to_record_id
                        if link.from_record_id == rid
                        else link.from_record_id
                    )
                    if other in visited:
                        continue
                    visited.add(other)
                    record = self.get(other)
                    if record is not None:
                        nodes[other] = record
                    next_frontier.append(other)
            frontier = next_frontier

        # The walk is truncated iff a leftover frontier node still has an edge to
        # an unvisited record we never got to expand.
        truncated = False
        for rid in frontier:
            for link in self.neighbors(rid, direction=direction):
                if rel_set is not None and link.relation not in rel_set:
                    continue
                other = (
                    link.to_record_id
                    if link.from_record_id == rid
                    else link.from_record_id
                )
                if other not in visited:
                    truncated = True
                    break
            if truncated:
                break
        return LineageGraph(
            root_id=record_id, nodes=nodes, edges=edges, truncated=truncated
        )


__all__ = ["AUDIT_RECORD_KINDS", "EntityRegistry", "record_id_for"]
