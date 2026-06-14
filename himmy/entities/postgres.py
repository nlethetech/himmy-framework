"""Entities kernel: Postgres-backed registry (requires the [postgres] extra).

This module is import-safe without ``asyncpg`` or a running database. ``asyncpg``
is imported lazily inside :meth:`PostgresEntityRegistry.connect`; the DDL is a
plain string so it can be inspected offline. All data methods raise a clear
:class:`HimmyError` when no live pool is present.

Production hardening (see IMPROVEMENTS SE-3/4/5/8):

- A JSONB type codec is registered on every pooled connection so dict/metadata
  payloads bind and read back as native Python objects (no double-encoding).
- ``new_version`` runs ``SELECT ... FOR UPDATE`` (and an advisory lock on the
  stable_id) inside a transaction, and verifies a row was actually inserted, so a
  concurrent writer that would lose a version raises instead of falsely succeeding.
- ``register`` raises on a true content-address collision (same id, different
  payload/metadata), mirroring the in-memory registry.
- ``query`` pushes ``metadata_filters`` into SQL via the JSONB containment
  operator (``metadata @> $1::jsonb``) backed by a GIN index, and supports
  stable_id-only and metadata-only queries (no ``kind`` required).
- ``close()`` / async-context-manager support and connection/command timeouts.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from datetime import datetime
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
    record_id_for,
)

#: Idempotent schema for the entity registry (records + links).
ENTITY_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS entity_records (
    record_id   TEXT PRIMARY KEY,
    stable_id   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TEXT NOT NULL,
    UNIQUE (stable_id, kind, version)
);

CREATE INDEX IF NOT EXISTS entity_records_stable_id_idx
    ON entity_records (stable_id);
CREATE INDEX IF NOT EXISTS entity_records_kind_idx
    ON entity_records (kind);
-- GIN index backing the metadata @> containment filter in query().
CREATE INDEX IF NOT EXISTS entity_records_metadata_gin_idx
    ON entity_records USING GIN (metadata);

CREATE TABLE IF NOT EXISTS entity_links (
    link_id        TEXT PRIMARY KEY,
    from_record_id TEXT NOT NULL,
    to_record_id   TEXT NOT NULL,
    relation       TEXT NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS entity_links_from_idx
    ON entity_links (from_record_id);
CREATE INDEX IF NOT EXISTS entity_links_to_idx
    ON entity_links (to_record_id);
"""


#: Ordered forward migrations for the Postgres audit spine, mirroring the SQLite spine's
#: :data:`himmy.entities.sqlite_registry._MIGRATIONS` and copying the proven storage
#: migrator (:data:`himmy.services.storage.postgres.STORAGE_MIGRATIONS`). Each entry is
#: ``(version, name, [statements...])`` and is applied at most once, tracked in the
#: :data:`SPINE_MIGRATIONS_TABLE` ledger under a session-level ``pg_advisory_lock`` so two
#: booting processes serialise. Migration 1 is the frozen base DDL (``CREATE ... IF NOT
#: EXISTS``), so it is safe on a fresh or partially-migrated database. Append new entries
#: here (never edit a shipped one) to evolve the spine beyond the version-1 shape in
#: lock-step with the SQLite spine — the CI parity guard fails if one chain gains a
#: table/column without the other.
SPINE_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "base_schema",
        [ENTITY_REGISTRY_DDL],
    ),
]

#: Migration-tracking table for the spine, namespaced DISTINCTLY from the storage migrator's
#: ``schema_migrations`` (:data:`himmy.services.storage.postgres.PostgresStorageService`).
#: The advisory-lock keys already differ so the two runners never deadlock, but if an
#: operator co-locates ``HIMMY_SPINE_DATABASE_URL`` and ``HIMMY_DATABASE_URL`` in the SAME
#: Postgres database/schema, a SHARED ledger table would let whichever migrator ran first
#: insert ``version=1`` and make the other's ``migrate()`` see ``{1}`` as done and SKIP its
#: own base DDL (so the second store's tables would never be created). Owning a distinct
#: table makes the two ledgers structurally incapable of colliding even when co-located.
SPINE_MIGRATIONS_TABLE = "spine_schema_migrations"


def _spine_migration_advisory_lock_key() -> int:
    """Derive the spine migration runner's stable signed-64-bit advisory-lock key.

    DELIBERATELY hashed off a DISTINCT module name from the storage migrator's
    :func:`himmy.services.storage.postgres._migration_advisory_lock_key` so a spine
    ``migrate()`` and a storage ``migrate()`` running against the same database never
    serialise against each other (different keys = independent locks).
    """
    import hashlib

    digest = hashlib.sha256(b"himmy.entities.spine.schema_migrations").digest()[:8]
    value = int.from_bytes(digest, "big", signed=False)
    # pg_advisory_lock takes a signed bigint; fold into that range.
    return value - (1 << 63)


#: Session-level advisory-lock key serialising concurrent spine ``migrate()`` callers
#: across processes (distinct from the storage migrator's key — see above).
SPINE_MIGRATION_ADVISORY_LOCK_KEY = _spine_migration_advisory_lock_key()


def _require_asyncpg() -> Any:
    """Import asyncpg lazily, raising a clear error when the extra is missing."""
    try:
        import asyncpg  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise HimmyError(
            "PostgresEntityRegistry requires the [postgres] extra "
            "(pip install 'himmy[postgres]') and a running database."
        ) from exc
    return asyncpg


def _ts_encode(value: Any) -> Any:
    """Encode an ISO-string (or datetime) timestamp for a text-format ts column."""
    if value is None or isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _ts_decode(value: Any) -> Any:
    """Normalise Postgres' timestamp text back to canonical ISO-8601."""
    if not value:
        return value
    try:
        return datetime.fromisoformat(value).isoformat()
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return value


async def _register_codecs(conn: Any) -> None:
    """Connection ``init`` callback: JSONB + ISO-string timestamp codecs.

    JSONB binds dict/list payloads as native objects; the timestamp text codecs let
    asyncpg accept the model's ISO-8601 *string* timestamps for ``timestamp`` /
    ``timestamptz`` columns (its default codec requires ``datetime`` instances) and
    normalise reads back to canonical ISO-8601.
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    for _ts_type in ("timestamptz", "timestamp"):
        await conn.set_type_codec(
            _ts_type,
            encoder=_ts_encode,
            decoder=_ts_decode,
            schema="pg_catalog",
            format="text",
        )


def _advisory_lock_key(stable_id: str) -> int:
    """Derive a stable signed-64-bit advisory-lock key from a stable_id."""
    import hashlib

    digest = hashlib.sha256(stable_id.encode("utf-8")).digest()[:8]
    value = int.from_bytes(digest, "big", signed=False)
    # pg_advisory_xact_lock takes a signed bigint; fold into that range.
    return value - (1 << 63)


class PostgresEntityRegistry:
    """A Postgres-backed registry mirroring :class:`EntityRegistry`.

    Construct via :meth:`connect`. Without a live pool, all data methods raise
    :class:`HimmyError` — the in-memory ``EntityRegistry`` remains the default.
    Supports ``async with`` for deterministic pool teardown.
    """

    def __init__(self, pool: Any | None = None) -> None:
        self._pool = pool

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 30.0,
        command_timeout: float | None = 30.0,
        max_inactive_connection_lifetime: float = 300.0,
    ) -> PostgresEntityRegistry:
        """Create a connection pool (JSONB codec + timeouts) and wire a registry."""
        asyncpg = _require_asyncpg()
        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            command_timeout=command_timeout,
            max_inactive_connection_lifetime=max_inactive_connection_lifetime,
            init=_register_codecs,
        )
        return cls(pool=pool)

    async def close(self) -> None:
        """Close the connection pool (idempotent)."""
        if self._pool is not None:
            await self._pool.close()

    async def __aenter__(self) -> PostgresEntityRegistry:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def create_schema(self) -> None:
        """Provision/upgrade the registry schema via the governed migration runner.

        Replaces the historical bare ``execute(ENTITY_REGISTRY_DDL)`` (which could never
        evolve a column on an existing database) with :meth:`migrate`, so the Postgres
        spine gains the same forward-migration governance as the SQLite spine: a fresh
        database lays down the version-1 base DDL and an existing one steps through every
        pending :data:`SPINE_MIGRATIONS` entry, all tracked in ``schema_migrations`` under
        a session-level advisory lock.
        """
        await self.migrate()

    async def migrate(self) -> list[int]:
        """Run pending forward spine migrations; return the versions newly applied.

        Each migration in :data:`SPINE_MIGRATIONS` is applied at most once, tracked in the
        :data:`SPINE_MIGRATIONS_TABLE` (``spine_schema_migrations``) ledger inside a
        transaction. The table is namespaced DISTINCTLY from the storage migrator's
        ``schema_migrations`` so the two ledgers cannot collide even when an operator
        co-locates the spine and storage in one Postgres database. Migration 1 is the frozen
        base DDL (``CREATE ... IF NOT EXISTS``), so this is safe on a fresh or
        partially-migrated database. Concurrent migrators (two processes booting
        simultaneously) are serialised by a session-level ``pg_advisory_lock`` on
        :data:`SPINE_MIGRATION_ADVISORY_LOCK_KEY` (distinct from the storage migrator's key)
        held for the duration of the run: the loser blocks until the winner finishes,
        re-reads the ledger and applies nothing. This is a near-verbatim copy of
        :meth:`himmy.services.storage.postgres.PostgresStorageService.migrate`.
        """
        pool = self._require_pool()
        applied: list[int] = []
        # SPINE_MIGRATIONS_TABLE is a trusted module constant (never user input), so
        # interpolating it into the DDL/DML below is not an injection surface.
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT pg_advisory_lock($1)", SPINE_MIGRATION_ADVISORY_LOCK_KEY
            )
            try:
                # Bootstrap the tracking table so the first migration can record itself.
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {SPINE_MIGRATIONS_TABLE} (
                        version    INTEGER PRIMARY KEY,
                        name       TEXT NOT NULL,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                rows = await conn.fetch(
                    f"SELECT version FROM {SPINE_MIGRATIONS_TABLE}"
                )
                done = {r["version"] for r in rows}
                for version, name, statements in sorted(SPINE_MIGRATIONS):
                    if version in done:
                        continue
                    async with conn.transaction():
                        for statement in statements:
                            await conn.execute(statement)
                        await conn.execute(
                            f"INSERT INTO {SPINE_MIGRATIONS_TABLE} (version, name) "
                            "VALUES ($1, $2) ON CONFLICT (version) DO NOTHING",
                            version,
                            name,
                        )
                    applied.append(version)
            finally:
                await conn.execute(
                    "SELECT pg_advisory_unlock($1)", SPINE_MIGRATION_ADVISORY_LOCK_KEY
                )
        return applied

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise HimmyError(
                "PostgresEntityRegistry has no active pool; "
                "use PostgresEntityRegistry.connect(dsn) with the [postgres] extra."
            )
        return self._pool

    async def register(self, record: EntityRecord) -> EntityRecord:
        """Insert a record; idempotent on identical content, raises on collision.

        Record identity is ``(kind, stable_id, version)``. Re-inserting identical
        content is a no-op; the SAME id with DIFFERENT payload/metadata raises an
        :class:`HimmyError` (true content-address violation), mirroring the
        in-memory registry. The insert uses ``ON CONFLICT DO NOTHING`` and then
        reads the row back to compare content.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                inserted = await conn.fetchrow(
                    """
                    INSERT INTO entity_records
                        (record_id, stable_id, version, kind, payload, metadata,
                         created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (record_id) DO NOTHING
                    RETURNING record_id
                    """,
                    record.record_id,
                    record.stable_id,
                    record.version,
                    record.kind,
                    record.payload,
                    record.metadata,
                    record.created_at,
                )
                if inserted is not None:
                    return record
                existing = await conn.fetchrow(
                    "SELECT * FROM entity_records WHERE record_id = $1",
                    record.record_id,
                )
        existing_record = self._row_to_record(existing)
        if (
            existing_record.payload != record.payload
            or existing_record.metadata != record.metadata
        ):
            raise HimmyError(
                "Content-address violation for "
                f"record_id={record.record_id!r} "
                f"(kind={record.kind!r}, stable_id={record.stable_id!r}, "
                f"version={record.version}): an existing record with different "
                "payload/metadata is already registered. Use new_version() to "
                "evolve content."
            )
        return existing_record

    async def new_version(
        self,
        *,
        stable_id: str,
        kind: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> EntityRecord:
        """Create the next version with DB-enforced optimistic concurrency.

        Runs inside a transaction holding a transaction-scoped advisory lock on the
        stable_id plus ``SELECT ... FOR UPDATE`` on the latest row, then inserts the
        new version. The insert is NOT swallowed by ``ON CONFLICT DO NOTHING``: a
        concurrent writer that already took version N+1 makes this INSERT raise a
        unique-violation, which is surfaced as an :class:`HimmyError` so no
        update is silently lost.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Serialise concurrent new_version on the same stable_id.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock($1)",
                    _advisory_lock_key(stable_id),
                )
                latest = await conn.fetchrow(
                    """
                    SELECT version FROM entity_records
                    WHERE stable_id = $1
                    ORDER BY version DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    stable_id,
                )
                current_version = latest["version"] if latest is not None else 0
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
                try:
                    await conn.execute(
                        """
                        INSERT INTO entity_records
                            (record_id, stable_id, version, kind, payload, metadata,
                             created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        record.record_id,
                        record.stable_id,
                        record.version,
                        record.kind,
                        record.payload,
                        record.metadata,
                        record.created_at,
                    )
                except Exception as exc:  # noqa: BLE001 - normalise to HimmyError
                    if _is_unique_violation(exc):
                        raise HimmyError(
                            "Lost version for "
                            f"stable_id={stable_id!r} kind={kind!r}: a concurrent "
                            f"writer already created version {record.version}."
                        ) from exc
                    raise
        return record

    async def link(
        self,
        *,
        from_record_id: str,
        to_record_id: str,
        relation: str,
        metadata: dict[str, Any] | None = None,
    ) -> EntityLink:
        """Insert a typed link between two records."""
        pool = self._require_pool()
        link = EntityLink(
            from_record_id=from_record_id,
            to_record_id=to_record_id,
            relation=relation,
            metadata=metadata or {},
        )
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO entity_links
                    (link_id, from_record_id, to_record_id, relation, metadata,
                     created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                link.link_id,
                link.from_record_id,
                link.to_record_id,
                link.relation,
                link.metadata,
                link.created_at,
            )
        return link

    async def get(self, record_id: str) -> EntityRecord | None:
        """Return a record by physical id, or None."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entity_records WHERE record_id = $1", record_id
            )
        return self._row_to_record(row) if row else None

    async def get_latest(self, stable_id: str) -> EntityRecord | None:
        """Return the highest-version record for an artefact, or None."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM entity_records
                WHERE stable_id = $1
                ORDER BY version DESC
                LIMIT 1
                """,
                stable_id,
            )
        return self._row_to_record(row) if row else None

    async def get_history(self, stable_id: str) -> list[EntityRecord]:
        """Return all versions of an artefact in ascending version order."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM entity_records
                WHERE stable_id = $1
                ORDER BY version ASC
                """,
                stable_id,
            )
        return [self._row_to_record(r) for r in rows]

    async def list_by_kind(self, kind: str) -> list[EntityRecord]:
        """Return all records of the given kind."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM entity_records WHERE kind = $1", kind
            )
        return [self._row_to_record(r) for r in rows]

    async def query(self, q: EntityQuery) -> list[EntityRecord]:
        """Return records matching the query via indexed SQL filters.

        ``kind`` and ``stable_id`` are optional; ``metadata_filters`` is pushed into
        SQL via ``metadata @> $n::jsonb`` (GIN-indexed). A query with no predicates
        returns every record.
        """
        pool = self._require_pool()
        clauses: list[str] = []
        params: list[Any] = []
        if q.kind is not None:
            params.append(q.kind)
            clauses.append(f"kind = ${len(params)}")
        if q.stable_id is not None:
            params.append(q.stable_id)
            clauses.append(f"stable_id = ${len(params)}")
        if q.metadata_filters:
            # Bind the dict directly: the registered jsonb codec (encoder=json.dumps)
            # serializes it once. Pre-dumping to a string here double-encodes it into
            # a jsonb *string* ("{...}") that never @>-matches a jsonb *object*.
            params.append(q.metadata_filters)
            clauses.append(f"metadata @> ${len(params)}::jsonb")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM entity_records{where} ORDER BY version ASC", *params
            )
        return [self._row_to_record(r) for r in rows]

    async def links_from(self, record_id: str) -> list[EntityLink]:
        """Return all links originating from a record."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM entity_links WHERE from_record_id = $1", record_id
            )
        return [self._row_to_link(r) for r in rows]

    async def links_to(self, record_id: str) -> list[EntityLink]:
        """Return all links pointing INTO a record (reverse of ``links_from``).

        Backed by the ``entity_links_to_idx`` index, which existed in the schema but
        had no query using it until now.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM entity_links WHERE to_record_id = $1", record_id
            )
        return [self._row_to_link(r) for r in rows]

    async def neighbors(
        self,
        record_id: str,
        *,
        direction: LineageDirection = LineageDirection.BOTH,
        relation: str | None = None,
    ) -> list[EntityLink]:
        """Return the links incident to a record in the requested direction."""
        if direction is LineageDirection.OUT:
            where = "from_record_id = $1"
        elif direction is LineageDirection.IN:
            where = "to_record_id = $1"
        else:
            where = "(from_record_id = $1 OR to_record_id = $1)"
        params: list[Any] = [record_id]
        if relation is not None:
            params.append(relation)
            where += f" AND relation = ${len(params)}"
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM entity_links WHERE {where}", *params
            )
        # De-dup self-loops that match both arms of a BOTH query.
        seen: set[str] = set()
        out: list[EntityLink] = []
        for row in rows:
            link = self._row_to_link(row)
            if link.link_id not in seen:
                seen.add(link.link_id)
                out.append(link)
        return out

    async def trace(
        self,
        record_id: str,
        *,
        max_depth: int = DEFAULT_TRACE_DEPTH,
        direction: LineageDirection = LineageDirection.BOTH,
        relations: Collection[str] | None = None,
    ) -> LineageGraph:
        """Walk the lineage graph from ``record_id`` via a recursive CTE.

        A single ``WITH RECURSIVE`` query collects the reachable record ids (using
        the indexed ``from``/``to`` columns); the induced records and edges are then
        fetched in two further queries. ``truncated`` is resolved with an ``EXISTS``
        probe for an edge leaving the depth-limit frontier toward an unreached node,
        matching the in-memory registry's semantics.
        """
        rel_list = list(relations) if relations is not None else None
        depth = max(0, max_depth)

        if direction is LineageDirection.OUT:
            join_cond = "l.from_record_id = w.node_id"
            next_expr = "l.to_record_id"
        elif direction is LineageDirection.IN:
            join_cond = "l.to_record_id = w.node_id"
            next_expr = "l.from_record_id"
        else:
            join_cond = "(l.from_record_id = w.node_id OR l.to_record_id = w.node_id)"
            next_expr = (
                "CASE WHEN l.from_record_id = w.node_id "
                "THEN l.to_record_id ELSE l.from_record_id END"
            )

        walk_sql = f"""
            WITH RECURSIVE walk(node_id, depth) AS (
                SELECT $1::text, 0
              UNION
                SELECT {next_expr}, w.depth + 1
                FROM walk w
                JOIN entity_links l ON {join_cond}
                WHERE w.depth < $2
                  AND ($3::text[] IS NULL OR l.relation = ANY($3))
            )
            SELECT node_id, MIN(depth) AS depth FROM walk GROUP BY node_id
        """

        pool = self._require_pool()
        async with pool.acquire() as conn:
            walk_rows = await conn.fetch(walk_sql, record_id, depth, rel_list)
            node_ids = [r["node_id"] for r in walk_rows]
            frontier = [r["node_id"] for r in walk_rows if r["depth"] == depth]

            record_rows = await conn.fetch(
                "SELECT * FROM entity_records WHERE record_id = ANY($1)", node_ids
            )
            edge_rows = await conn.fetch(
                """
                SELECT * FROM entity_links
                WHERE from_record_id = ANY($1) AND to_record_id = ANY($1)
                  AND ($2::text[] IS NULL OR relation = ANY($2))
                """,
                node_ids,
                rel_list,
            )
            truncated = False
            if frontier:
                if direction is LineageDirection.OUT:
                    probe = "from_record_id = ANY($1) AND to_record_id <> ALL($2)"
                elif direction is LineageDirection.IN:
                    probe = "to_record_id = ANY($1) AND from_record_id <> ALL($2)"
                else:
                    probe = (
                        "((from_record_id = ANY($1) AND to_record_id <> ALL($2)) "
                        "OR (to_record_id = ANY($1) AND from_record_id <> ALL($2)))"
                    )
                truncated = bool(
                    await conn.fetchval(
                        f"""
                        SELECT EXISTS(
                            SELECT 1 FROM entity_links
                            WHERE {probe}
                              AND ($3::text[] IS NULL OR relation = ANY($3))
                        )
                        """,
                        frontier,
                        node_ids,
                        rel_list,
                    )
                )

        nodes = {r["record_id"]: self._row_to_record(r) for r in record_rows}
        edges = [self._row_to_link(r) for r in edge_rows]
        return LineageGraph(
            root_id=record_id, nodes=nodes, edges=edges, truncated=truncated
        )

    @staticmethod
    def _row_to_record(row: Any) -> EntityRecord:
        """Map a row to an EntityRecord; payload/metadata are native via the codec."""
        data = dict(row)
        for key in ("payload", "metadata"):
            value = data.get(key)
            # With the JSONB codec asyncpg returns parsed dicts; keep a str-guard
            # only as a defensive fallback for pools constructed without the codec.
            if isinstance(value, str):
                data[key] = json.loads(value)
            elif value is None:
                data[key] = {}
        return EntityRecord.model_validate(data)

    @staticmethod
    def _row_to_link(row: Any) -> EntityLink:
        data = dict(row)
        value = data.get("metadata")
        if isinstance(value, str):
            data["metadata"] = json.loads(value)
        elif value is None:
            data["metadata"] = {}
        return EntityLink.model_validate(data)


def _is_unique_violation(exc: Exception) -> bool:
    """Return True when ``exc`` is an asyncpg unique-violation error."""
    # asyncpg's UniqueViolationError carries SQLSTATE 23505; avoid importing it at
    # module top so the file stays import-safe without the extra.
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate == "23505":
        return True
    return exc.__class__.__name__ == "UniqueViolationError"


__all__ = [
    "PostgresEntityRegistry",
    "ENTITY_REGISTRY_DDL",
    "SPINE_MIGRATIONS",
    "SPINE_MIGRATIONS_TABLE",
    "SPINE_MIGRATION_ADVISORY_LOCK_KEY",
    "record_id_for",
]
