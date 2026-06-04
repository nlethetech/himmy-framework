"""Entities kernel: the in-memory entity registry (lineage backbone)."""

from __future__ import annotations

from typing import Any

from opensims.core.errors import OpenSimsError
from opensims.entities.records import (
    EntityLink,
    EntityQuery,
    EntityRecord,
    metadata_contains,
    record_id_for,
)


class EntityRegistry:
    """An in-memory, versioned registry of entity records and their links.

    Records are immutable and keyed by ``record_id``; versions of the same
    artefact share a ``stable_id``. Registration is idempotent on ``record_id``.
    """

    def __init__(self) -> None:
        self._records: dict[str, EntityRecord] = {}
        self._by_stable_id: dict[str, list[EntityRecord]] = {}
        self._links: list[EntityLink] = []

    def register(self, record: EntityRecord) -> EntityRecord:
        """Register a record. Idempotent on identical content; raises on collision.

        Record identity is ``(kind, stable_id, version)``. Re-registering byte-for-
        byte identical content (same payload + metadata) returns the stored record.
        Re-registering the SAME id with DIFFERENT payload/metadata is a true
        content-address violation and raises :class:`OpenSimsError` — callers who
        want to evolve content must use :meth:`new_version`.
        """
        existing = self._records.get(record.record_id)
        if existing is not None:
            if (
                existing.payload != record.payload
                or existing.metadata != record.metadata
            ):
                raise OpenSimsError(
                    "Content-address violation for "
                    f"record_id={record.record_id!r} "
                    f"(kind={record.kind!r}, stable_id={record.stable_id!r}, "
                    f"version={record.version}): an existing record with different "
                    "payload/metadata is already registered. Use new_version() to "
                    "evolve content."
                )
            return existing
        self._records[record.record_id] = record
        versions = self._by_stable_id.setdefault(record.stable_id, [])
        versions.append(record)
        versions.sort(key=lambda r: r.version)
        return record

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
        version, an :class:`OpenSimsError` is raised.
        """
        latest = self.get_latest(stable_id)
        current_version = latest.version if latest is not None else 0
        if expected_version is not None and expected_version != current_version:
            raise OpenSimsError(
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
        self._links.append(link)
        return link

    def get(self, record_id: str) -> EntityRecord | None:
        """Return a record by its physical id, or None."""
        return self._records.get(record_id)

    def get_latest(self, stable_id: str) -> EntityRecord | None:
        """Return the highest-version record for an artefact, or None."""
        versions = self._by_stable_id.get(stable_id)
        if not versions:
            return None
        return versions[-1]

    def get_history(self, stable_id: str) -> list[EntityRecord]:
        """Return all versions of an artefact in ascending version order."""
        return list(self._by_stable_id.get(stable_id, []))

    def list_by_kind(self, kind: str) -> list[EntityRecord]:
        """Return every record of the given kind."""
        return [r for r in self._records.values() if r.kind == kind]

    def query(self, q: EntityQuery) -> list[EntityRecord]:
        """Return records matching the supplied query filter.

        ``kind`` and ``stable_id`` are optional (metadata-only / stable_id-only
        queries are valid). ``metadata_filters`` uses JSONB-containment semantics
        (``metadata_contains``), equivalent to the Postgres ``@>`` operator.
        """
        results: list[EntityRecord] = []
        for record in self._records.values():
            if q.kind is not None and record.kind != q.kind:
                continue
            if q.stable_id is not None and record.stable_id != q.stable_id:
                continue
            if q.metadata_filters and not metadata_contains(
                record.metadata, q.metadata_filters
            ):
                continue
            results.append(record)
        return results

    def links_from(self, record_id: str) -> list[EntityLink]:
        """Return all links originating from a record."""
        return [link for link in self._links if link.from_record_id == record_id]


__all__ = ["EntityRegistry", "record_id_for"]
