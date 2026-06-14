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
import os
import sqlite3
import threading
from collections.abc import Collection
from typing import Any

from himmy.core.errors import HimmyError
from himmy.core.sqlite_util import connect_hardened
from himmy.entities.db_chain import DbChainRow
from himmy.entities.integrity import CHAIN_GENESIS_HASH, _chain_step, content_hash
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

#: Frozen version-1 shape of the audit spine. This is the base DDL a fresh database lays
#: down; it is **never edited in place** — schema evolution rides :data:`_MIGRATIONS`
#: instead (frozen base + forward migrations), exactly like
#: :data:`himmy.services.storage.sqlite._SCHEMA`. Editing this to add a column AND
#: re-adding it via a migration would double-apply the ALTER and break fresh installs.
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

#: Schema version of the base :data:`_SCHEMA` above (``PRAGMA user_version`` after it is
#: applied). The audit spine is APPEND-ONLY and content-addressed, so it historically ran
#: ``executescript(_SCHEMA)`` once with no version stamp and could therefore never evolve a
#: column in place — yet right-to-erasure (and any future spine field, e.g. the optional
#: DB-resident hash chain) needs exactly that. This mirrors the proven storage-migrator
#: (:data:`himmy.services.storage.sqlite.SQLITE_SCHEMA_VERSION`): bump this and append to
#: :data:`_MIGRATIONS` whenever a column/index/table is added so existing ``.himmy/spine.db``
#: files upgrade forward in lock-step with the Postgres spine rather than silently diverging
#: (``CREATE ... IF NOT EXISTS`` is a no-op on an existing db, so it never reaches a column).
SPINE_SCHEMA_VERSION = 2

#: Ordered forward migrations applied after the base DDL, gated by ``PRAGMA user_version``
#: (a database steps through every migration whose version exceeds its current
#: ``user_version`` and is then stamped at the highest). Mirrors the Postgres spine's
#: :data:`himmy.entities.postgres.SPINE_MIGRATIONS`. **Frozen base + forward migrations**:
#: :data:`_SCHEMA` is the version-1 shape and is never edited; a fresh db (``user_version``
#: 0) lays down that frozen base and then runs *every* migration just like a legacy file,
#: converging to the same schema by either path. Append new entries here — never edit a
#: shipped one — and bump :data:`SPINE_SCHEMA_VERSION` to match the highest.
#:
#: * **v2** adds the OPT-IN, DB-resident hash-chain column ``prev_hash`` (S6): the
#:   predecessor's chain hash for the EXISTING :mod:`himmy.entities.integrity` chain, persisted
#:   so tamper-evidence is verifiable from the DB alone (no separately-held bundle). The column
#:   stays NULL until the chain is explicitly enabled, so a default install is unaffected
#:   beyond the column existing. ``CREATE ... IF NOT EXISTS`` never adds a column to an existing
#:   table, so this ``ALTER`` is the only way a legacy ``.himmy/spine.db`` gains it.
_MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        2,
        (
            "ALTER TABLE entity_records ADD COLUMN prev_hash TEXT",
            "CREATE INDEX IF NOT EXISTS entity_records_prev_hash_idx "
            "ON entity_records (prev_hash)",
        ),
    ),
)

_REC_COLS = "record_id, stable_id, version, kind, payload, metadata, created_at"
_LINK_COLS = "link_id, from_record_id, to_record_id, relation, metadata, created_at"

#: Env flag enabling the opt-in DB-resident hash chain (S6). Default OFF.
_DB_CHAIN_ENV = "HIMMY_SPINE_DB_CHAIN"


def _resolve_db_chain_flag(explicit: bool | None) -> bool:
    """Resolve the DB-chain flag: explicit arg wins, else ``HIMMY_SPINE_DB_CHAIN`` (default OFF)."""
    if explicit is not None:
        return explicit
    return (os.environ.get(_DB_CHAIN_ENV) or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class SqliteEntityRegistry:
    """A synchronous, durable SQLite-backed registry mirroring ``EntityRegistry``."""

    def __init__(
        self,
        path: str = ":memory:",
        *,
        register_buffer: int = 1,
        db_chain: bool | None = None,
    ) -> None:
        """Open (or create) the SQLite database at ``path`` (``:memory:`` default).

        The connection is opened via :func:`connect_hardened` (WAL + busy timeout +
        ``synchronous=NORMAL``) so concurrent connections coordinate instead of
        raising ``database is locked`` immediately. A process-level write lock
        serialises writers sharing this connection across threads.

        ``register_buffer`` (default ``1`` = commit every :meth:`register`) trades a
        little durability for throughput on a write-heavy hot path: a multi-tool agent
        run fires one :meth:`register` per ``RunEvent``, and committing each one is the
        dominant cost (measured ~50µs/event vs ~12µs/event batched). When set >1,
        :meth:`register` of a NEW record defers its commit and only flushes once
        ``register_buffer`` rows have accumulated — or sooner, whenever a read
        (:meth:`get`/:meth:`query`/…), a versioned write (:meth:`new_version`), a
        :meth:`link`, an explicit :meth:`flush`, or :meth:`close` needs the pending rows
        on disk. Content-addressing is unchanged (the records and their ``content_hash``
        are byte-identical regardless of when the commit lands), so an in-memory vs SQLite
        bundle stays parity-identical; the only window is that a hard crash can lose the
        not-yet-flushed tail of the audit trail. Default ``1`` keeps the full per-event
        durability the spine contract assumes.

        ``db_chain`` (S6) enables the OPT-IN, DB-resident tamper-evidence chain: each NEW
        :meth:`register` records the predecessor's chain hash in the ``prev_hash`` column so the
        chain is verifiable from the database ALONE (:func:`himmy.entities.db_chain.verify_db_chain`)
        — no separately-held bundle. ``None`` (default) resolves it from ``HIMMY_SPINE_DB_CHAIN``
        (default OFF), so a default install leaves ``prev_hash`` NULL and is byte-identical. The
        chain REUSES :mod:`himmy.entities.integrity` (``content_hash`` + ``_chain_step``); it is
        NOT a second mechanism.
        """
        self._conn = connect_hardened(path)
        self._write_lock = threading.Lock()
        self._register_buffer = max(1, int(register_buffer))
        self._pending = 0
        self._db_chain = _resolve_db_chain_flag(db_chain)
        self._apply_schema()
        #: In-RAM chain tip (the last committed chained row's chain_hash). Recomputed from the
        #: committed DB tail on startup so a restart resumes the existing chain; reset to the
        #: committed tail on any register/new_version rollback so a discarded batch never leaves
        #: the tip ahead of the durable tail (which would falsely report tampering — S6 must_fix).
        self._chain_tip: str = (
            self._recompute_committed_tip() if self._db_chain else CHAIN_GENESIS_HASH
        )

    def _apply_schema(self) -> None:
        """Create/upgrade the spine schema and stamp ``PRAGMA user_version`` (idempotent).

        Copies the proven storage-migrator pattern
        (:meth:`himmy.services.storage.sqlite.SqliteStorageService._apply_schema`): the base
        :data:`_SCHEMA` is ``CREATE ... IF NOT EXISTS``, so a fresh database (``user_version``
        0, no tables) gets the frozen base DDL and a legacy file (also ``user_version`` 0,
        written before versioning existed — re-running the idempotent base is safe) does too,
        and BOTH then step through every entry in :data:`_MIGRATIONS` whose version exceeds
        the stored ``user_version`` (so a fresh db converges to the same schema a legacy file
        reaches) before the highest known version is stamped. The whole upgrade runs under the
        process write lock so concurrent workers/processes sharing the file cannot
        double-apply a step.
        """
        with self._write_lock:
            current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            try:
                if current == 0:
                    # Either a brand-new database, OR a legacy file written before
                    # versioning existed (user_version defaults to 0). Re-running the
                    # idempotent base DDL is safe in both cases.
                    self._conn.executescript(_SCHEMA)
                for version, statements in sorted(_MIGRATIONS):
                    if version > current:
                        for statement in statements:
                            self._conn.execute(statement)  # noqa: S608 - constant DDL
                # Stamp the highest known version (PRAGMA cannot be parameterised).
                target = max(SPINE_SCHEMA_VERSION, current)
                self._conn.execute(f"PRAGMA user_version = {int(target)}")
                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except sqlite3.Error:  # pragma: no cover - rollback on a dead connection
                    pass
                raise

    @property
    def schema_version(self) -> int:
        """The applied schema version (``PRAGMA user_version``) of the backing file."""
        with self._write_lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    # ---------------------------------------------------------- DB chain (S6)
    def _recompute_committed_tip(self) -> str:
        """Compute the chain tip from the COMMITTED rows (resume the chain on restart).

        Walks the persisted ``prev_hash`` linked list from genesis to its end, recomputing each
        row's ``chain_hash`` via the integrity primitives, and returns the final hash. This is
        the durable tail the in-RAM tip must equal — used at startup AND to reset the tip after a
        rollback so a discarded batch never leaves the tip ahead of disk (which would falsely
        report tampering). Returns :data:`CHAIN_GENESIS_HASH` for an empty/absent chain.
        """
        rows = self._conn.execute(
            "SELECT prev_hash, " + _REC_COLS + " FROM entity_records "
            "WHERE prev_hash IS NOT NULL"
        ).fetchall()
        if not rows:
            return CHAIN_GENESIS_HASH
        by_prev: dict[str, Any] = {}
        for row in rows:
            by_prev[row[0]] = row
        cursor = CHAIN_GENESIS_HASH
        # Bound the walk by the row count so a corrupt/forked chain cannot loop forever.
        for _ in range(len(rows)):
            row = by_prev.get(cursor)
            if row is None:
                break
            record = self._row_to_record(row[1:])
            cursor = _chain_step(cursor, content_hash(record))
        return cursor

    def _commit_locked(self) -> None:
        """Commit + clear the deferred-register counter (caller holds the write lock)."""
        self._conn.commit()
        self._pending = 0

    def flush(self) -> None:
        """Commit any buffered :meth:`register` writes (no-op when nothing is pending).

        Idempotent; safe to call on a non-buffered registry. Reads and versioned writes
        call it implicitly, so a caller only needs it to force durability at a checkpoint.
        """
        with self._write_lock:
            if self._pending:
                self._commit_locked()

    def close(self) -> None:
        """Flush any buffered writes, then close the connection (idempotent)."""
        try:
            self.flush()
        except Exception:  # pragma: no cover - best-effort flush on teardown
            pass
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
        # S6: when the DB-resident chain is on, bind this row to the current tip by writing
        # its predecessor's chain hash into prev_hash, then advance the in-RAM tip. The tip is
        # only authoritative once the enclosing txn commits — a rollback resets it from disk.
        prev_hash = self._chain_tip if self._db_chain else None
        try:
            self._conn.execute(
                f"INSERT INTO entity_records ({_REC_COLS}, content_hash, prev_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.record_id,
                    record.stable_id,
                    record.version,
                    record.kind,
                    json.dumps(record.payload),
                    json.dumps(record.metadata),
                    record.created_at,
                    content_hash(record),
                    prev_hash,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise HimmyError(
                f"Concurrent version conflict for stable_id={record.stable_id!r} "
                f"(kind={record.kind!r}, version={record.version}): another writer "
                "registered this version first. Retry new_version()."
            ) from exc
        if self._db_chain and prev_hash is not None:
            self._chain_tip = _chain_step(prev_hash, content_hash(record))
        return record

    def register(self, record: EntityRecord) -> EntityRecord:
        """Register a record; idempotent on identical content, raises on collision.

        When ``register_buffer`` > 1 the commit is deferred (see :meth:`__init__`): the
        row is written but the transaction only commits once the buffer fills, so a
        write-heavy run amortises commit cost. A rollback on error still discards the
        whole pending batch, which is acceptable for a best-effort audit sink.
        """
        with self._write_lock:
            try:
                stored = self._insert_record(record)
                self._pending += 1
                if self._pending >= self._register_buffer:
                    self._commit_locked()
            except BaseException:
                self._conn.rollback()
                self._pending = 0
                # S6 must_fix: a rollback under register_buffer>1 discards the whole pending
                # batch, so any in-RAM tip advanced for the discarded rows is now ahead of the
                # committed tail. Reset it to the durable tip so the chain never falsely reports
                # tampering.
                self._reset_chain_tip_after_rollback()
                raise
        return stored

    def _reset_chain_tip_after_rollback(self) -> None:
        """Re-anchor the in-RAM chain tip to the committed DB tail (no-op when chain off)."""
        if self._db_chain:
            self._chain_tip = self._recompute_committed_tip()

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
            # Flush any deferred register writes first so the IMMEDIATE txn below opens
            # cleanly (autocommit) and reads the freshest version chain.
            if self._pending:
                self._commit_locked()
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
                # The IMMEDIATE txn discarded the new row (and any chain-tip advance it made);
                # re-anchor the tip to the committed tail so the chain stays consistent (S6).
                self._reset_chain_tip_after_rollback()
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
            # This commit also lands any deferred register writes batched on the shared
            # connection — keep the pending counter honest so flush() stays a no-op.
            self._commit_locked()
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

    @property
    def db_chain_enabled(self) -> bool:
        """Whether the opt-in DB-resident hash chain (S6) is active on this registry."""
        return self._db_chain

    def db_chain_rows(self) -> list[DbChainRow]:
        """Every record paired with its stored ``prev_hash`` (for from-DB chain verification).

        Feed the result to :func:`himmy.entities.db_chain.verify_db_chain` to detect an
        in-place row mutation from the DATABASE ALONE — no separately-held bundle. Pending
        buffered writes are flushed first so the snapshot reflects the durable tail.
        """
        self.flush()
        rows = self._conn.execute(
            f"SELECT {_REC_COLS}, prev_hash FROM entity_records"
        ).fetchall()
        return [DbChainRow(record=self._row_to_record(r), prev_hash=r[7]) for r in rows]

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


__all__ = ["SqliteEntityRegistry", "SPINE_SCHEMA_VERSION", "_DB_CHAIN_ENV"]
