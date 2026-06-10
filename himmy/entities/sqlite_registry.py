"""Entities kernel: a durable, file-backed SQLite registry (offline + sync).

A middle tier between the volatile in-memory ``EntityRegistry`` and the full
``PostgresEntityRegistry``: durable across process restarts (and power cuts), but
just a stdlib ``sqlite3`` file — no server to run. Crucially it is **synchronous**,
so unlike the async Postgres registry it drops straight into the runtime (which
calls the registry synchronously) and gives a run's lineage a durable home.

API-compatible with :class:`EntityRegistry`: register / new_version / link / get /
get_latest / get_history / list_by_kind / query / links_from / links_to /
neighbors / trace. Records' integrity hashes are stored alongside, ready for the
tamper-evidence layer in :mod:`himmy.entities.integrity`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Collection
from typing import Any

from himmy.core.errors import HimmyError
from himmy.core.sqlite_util import connect_hardened
from himmy.entities.integrity import content_hash
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
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_records (
    record_id    TEXT PRIMARY KEY,
    stable_id    TEXT NOT NULL,
    version      INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    UNIQUE (stable_id, kind, version)
);
CREATE INDEX IF NOT EXISTS entity_records_stable_id_idx ON entity_records (stable_id);
CREATE INDEX IF NOT EXISTS entity_records_kind_idx ON entity_records (kind);

CREATE TABLE IF NOT EXISTS entity_links (
    link_id        TEXT PRIMARY KEY,
    from_record_id TEXT NOT NULL,
    to_record_id   TEXT NOT NULL,
    relation       TEXT NOT NULL,
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entity_links_from_idx ON entity_links (from_record_id);
CREATE INDEX IF NOT EXISTS entity_links_to_idx ON entity_links (to_record_id);
"""

_REC_COLS = "record_id, stable_id, version, kind, payload, metadata, created_at"
_LINK_COLS = "link_id, from_record_id, to_record_id, relation, metadata, created_at"


class SqliteEntityRegistry:
    """A synchronous, durable SQLite-backed registry mirroring ``EntityRegistry``."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path`` (``:memory:`` default).

        The connection is opened via :func:`connect_hardened` (WAL + busy timeout +
        ``synchronous=NORMAL``) so concurrent connections coordinate instead of
        raising ``database is locked`` immediately. A process-level write lock
        serialises writers sharing this connection across threads.
        """
        self._conn = connect_hardened(path)
        self._write_lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()

    def __enter__(self) -> SqliteEntityRegistry:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------- writes
    def _insert_record(self, record: EntityRecord) -> EntityRecord:
        """Check-then-insert a record without committing (callers own the txn)."""
        row = self._conn.execute(
            "SELECT payload, metadata FROM entity_records WHERE record_id = ?",
            (record.record_id,),
        ).fetchone()
        if row is not None:
            if json.loads(row[0]) != record.payload or json.loads(row[1]) != (
                record.metadata
            ):
                raise HimmyError(
                    "Content-address violation for "
                    f"record_id={record.record_id!r} "
                    f"(kind={record.kind!r}, stable_id={record.stable_id!r}, "
                    f"version={record.version}): an existing record with different "
                    "payload/metadata is already registered. Use new_version() to "
                    "evolve content."
                )
            stored = self.get(record.record_id)
            assert stored is not None
            return stored
        try:
            self._conn.execute(
                f"INSERT INTO entity_records ({_REC_COLS}, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.record_id,
                    record.stable_id,
                    record.version,
                    record.kind,
                    json.dumps(record.payload),
                    json.dumps(record.metadata),
                    record.created_at,
                    content_hash(record),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise HimmyError(
                f"Concurrent version conflict for stable_id={record.stable_id!r} "
                f"(kind={record.kind!r}, version={record.version}): another writer "
                "registered this version first. Retry new_version()."
            ) from exc
        return record

    def register(self, record: EntityRecord) -> EntityRecord:
        """Register a record; idempotent on identical content, raises on collision."""
        with self._write_lock:
            try:
                stored = self._insert_record(record)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return stored

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

        The read-latest + insert pair runs under the process write lock inside a
        ``BEGIN IMMEDIATE`` transaction, so concurrent versioning serialises (across
        threads sharing this registry and across other connections to the same
        file) instead of silently losing an update.
        """
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                latest = self.get_latest(stable_id)
                current_version = latest.version if latest is not None else 0
                if expected_version is not None and expected_version != (
                    current_version
                ):
                    raise HimmyError(
                        "Optimistic concurrency conflict for "
                        f"stable_id={stable_id!r}: expected version "
                        f"{expected_version}, found {current_version}."
                    )
                record = EntityRecord.create(
                    stable_id=stable_id,
                    version=current_version + 1,
                    kind=kind,
                    payload=payload,
                    metadata=metadata or {},
                )
                stored = self._insert_record(record)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return stored

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
        with self._write_lock:
            self._conn.execute(
                f"INSERT INTO entity_links ({_LINK_COLS}) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    link.link_id,
                    link.from_record_id,
                    link.to_record_id,
                    link.relation,
                    json.dumps(link.metadata),
                    link.created_at,
                ),
            )
            self._conn.commit()
        return link

    # -------------------------------------------------------------------- reads
    def get(self, record_id: str) -> EntityRecord | None:
        """Return a record by its physical id, or None."""
        row = self._conn.execute(
            f"SELECT {_REC_COLS} FROM entity_records WHERE record_id = ?", (record_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_latest(self, stable_id: str) -> EntityRecord | None:
        """Return the highest-version record for an artefact, or None."""
        row = self._conn.execute(
            f"SELECT {_REC_COLS} FROM entity_records WHERE stable_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (stable_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_history(self, stable_id: str) -> list[EntityRecord]:
        """Return all versions of an artefact in ascending version order."""
        rows = self._conn.execute(
            f"SELECT {_REC_COLS} FROM entity_records WHERE stable_id = ? "
            "ORDER BY version ASC",
            (stable_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_kind(self, kind: str) -> list[EntityRecord]:
        """Return every record of the given kind."""
        rows = self._conn.execute(
            f"SELECT {_REC_COLS} FROM entity_records WHERE kind = ?", (kind,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query(self, q: EntityQuery) -> list[EntityRecord]:
        """Return records matching the query (metadata filters applied in Python)."""
        clauses: list[str] = []
        params: list[Any] = []
        if q.kind is not None:
            clauses.append("kind = ?")
            params.append(q.kind)
        if q.stable_id is not None:
            clauses.append("stable_id = ?")
            params.append(q.stable_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT {_REC_COLS} FROM entity_records{where} ORDER BY version ASC",
            params,
        ).fetchall()
        records = [self._row_to_record(r) for r in rows]
        if q.metadata_filters:
            records = [
                r for r in records if metadata_contains(r.metadata, q.metadata_filters)
            ]
        return records

    def links_from(self, record_id: str) -> list[EntityLink]:
        """Return all links originating from a record."""
        rows = self._conn.execute(
            f"SELECT {_LINK_COLS} FROM entity_links WHERE from_record_id = ?",
            (record_id,),
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def links_to(self, record_id: str) -> list[EntityLink]:
        """Return all links pointing INTO a record (the reverse of links_from)."""
        rows = self._conn.execute(
            f"SELECT {_LINK_COLS} FROM entity_links WHERE to_record_id = ?",
            (record_id,),
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def neighbors(
        self,
        record_id: str,
        *,
        direction: LineageDirection = LineageDirection.BOTH,
        relation: str | None = None,
    ) -> list[EntityLink]:
        """Return the links incident to a record in the requested direction."""
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
        """Walk the lineage graph from ``record_id`` (BFS) and return the subgraph."""
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

    # ----------------------------------------------------------- audit helpers
    def all_records(self) -> list[EntityRecord]:
        """Every record in the store (for exporting a tamper-evidence bundle)."""
        rows = self._conn.execute(f"SELECT {_REC_COLS} FROM entity_records").fetchall()
        return [self._row_to_record(r) for r in rows]

    def all_links(self) -> list[EntityLink]:
        """Every link in the store (for exporting a tamper-evidence bundle)."""
        rows = self._conn.execute(f"SELECT {_LINK_COLS} FROM entity_links").fetchall()
        return [self._row_to_link(r) for r in rows]

    # ------------------------------------------------------------------ codecs
    @staticmethod
    def _row_to_record(row: Any) -> EntityRecord:
        return EntityRecord(
            record_id=row[0],
            stable_id=row[1],
            version=row[2],
            kind=row[3],
            payload=json.loads(row[4]),
            metadata=json.loads(row[5]),
            created_at=row[6],
        )

    @staticmethod
    def _row_to_link(row: Any) -> EntityLink:
        return EntityLink(
            link_id=row[0],
            from_record_id=row[1],
            to_record_id=row[2],
            relation=row[3],
            metadata=json.loads(row[4]),
            created_at=row[5],
        )


__all__ = ["SqliteEntityRegistry"]
