"""S1: SQLite + Postgres audit-spine migration governance + a CI parity guard.

The immutable audit spine historically ran ``executescript(_SCHEMA)`` once (SQLite) /
``execute(ENTITY_REGISTRY_DDL)`` once (Postgres) with NO version stamp, so it could never
evolve a column in place — yet right-to-erasure and any future spine field need exactly
that. This module pins the copied storage-migrator pattern on BOTH spines:

* SQLite — ``PRAGMA user_version`` + frozen base + ordered :data:`_MIGRATIONS` under the
  write lock, so a legacy ``.himmy/spine.db`` steps forward and a fresh db converges to the
  same schema by either path.
* Postgres — a ``spine_schema_migrations`` ledger (namespaced DISTINCTLY from the storage
  migrator's ``schema_migrations`` so the two cannot collide when co-located in one PG
  database) + ``pg_advisory_lock`` (driven against a fake asyncpg-shaped pool offline; the
  live DB exercise belongs in the Postgres CI lane).
* A parity guard asserting the two spines declare identical tables/columns/indexes modulo a
  single, frozen, documented allowlist of deliberate dialect divergences.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from himmy.entities import sqlite_registry as spine_mod
from himmy.entities.postgres import (
    ENTITY_REGISTRY_DDL,
    SPINE_MIGRATIONS,
    SPINE_MIGRATIONS_TABLE,
    PostgresEntityRegistry,
)
from himmy.entities.records import EntityRecord
from himmy.entities.sqlite_registry import SPINE_SCHEMA_VERSION, SqliteEntityRegistry
from tests.conftest import run_async

# --------------------------------------------------------------- SQLite governance


def test_fresh_spine_is_stamped_at_current_version(tmp_path: Path) -> None:
    reg = SqliteEntityRegistry(str(tmp_path / "spine.db"))
    assert reg.schema_version == SPINE_SCHEMA_VERSION
    reg.close()


def test_reopening_spine_keeps_version_and_records(tmp_path: Path) -> None:
    path = str(tmp_path / "spine.db")
    first = SqliteEntityRegistry(path)
    first.register(
        EntityRecord.create(stable_id="s1", version=1, kind="thing", payload={"a": 1})
    )
    first.close()
    # Reopening an existing file must not reset the version nor wipe records.
    second = SqliteEntityRegistry(path)
    assert second.schema_version == SPINE_SCHEMA_VERSION
    assert len(second.all_records()) == 1
    second.close()


def test_default_install_schema_is_frozen_base_plus_forward_migrations(
    tmp_path: Path,
) -> None:
    """The governed install lays down the frozen base then runs every forward migration.

    Before S6 the spine had zero migrations, so a default install was byte-identical to the
    bare base DDL. S6 adds the v2 ``prev_hash`` chain column via a forward migration, so the
    governed install now equals (frozen base + every migration) — deterministically. A legacy
    file that applies the SAME base then the SAME migrations converges to the identical schema,
    which is the real invariant: fresh==legacy by either path.
    """
    governed = str(tmp_path / "governed.db")
    SqliteEntityRegistry(governed).close()

    # Reconstruct the expected schema by hand: frozen base DDL, then each migration's DDL.
    legacy = str(tmp_path / "legacy.db")
    raw = sqlite3.connect(legacy)
    raw.executescript(spine_mod._SCHEMA)
    for _version, statements in sorted(spine_mod._MIGRATIONS):
        for statement in statements:
            raw.execute(statement)
    raw.commit()
    raw.close()

    def _schema(path: str) -> list[str]:
        conn = sqlite3.connect(path)
        try:
            return sorted(
                r[0]
                for r in conn.execute(
                    "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
                ).fetchall()
            )
        finally:
            conn.close()

    assert _schema(governed) == _schema(legacy)


def test_legacy_unversioned_spine_is_migrated_forward(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A pre-versioning spine file (user_version 0) steps through pending migrations."""
    path = str(tmp_path / "legacy.db")

    # Simulate an old spine: base tables present, user_version never stamped, AND
    # missing a column a later release adds.
    raw = sqlite3.connect(path)
    raw.executescript(spine_mod._SCHEMA)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    raw.commit()
    raw.close()

    # Inject a forward migration (version 2) that adds a column + bumps the target.
    monkeypatch.setattr(
        spine_mod,
        "_MIGRATIONS",
        ((2, ("ALTER TABLE entity_records ADD COLUMN tenant TEXT",)),),
    )
    monkeypatch.setattr(spine_mod, "SPINE_SCHEMA_VERSION", 2)

    reg = SqliteEntityRegistry(path)
    assert reg.schema_version == 2
    cols = {
        row[1]
        for row in reg._conn.execute("PRAGMA table_info(entity_records)").fetchall()
    }
    assert "tenant" in cols  # the migration ran against the legacy file
    # First INSERT naming the new column succeeds (no raise on the legacy file).
    reg._conn.execute(
        "INSERT INTO entity_records "
        "(record_id, stable_id, version, kind, payload, metadata, created_at, tenant) "
        "VALUES ('r1','s1',1,'k','{}','{}','2026-01-01T00:00:00+00:00','acme')"
    )
    reg._conn.commit()
    reg.close()


def test_fresh_db_converges_to_same_schema_as_migrated_legacy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Frozen-base + migration on a FRESH db == the same shape a legacy file reaches."""
    monkeypatch.setattr(
        spine_mod,
        "_MIGRATIONS",
        ((2, ("ALTER TABLE entity_records ADD COLUMN tenant TEXT",)),),
    )
    monkeypatch.setattr(spine_mod, "SPINE_SCHEMA_VERSION", 2)

    fresh = str(tmp_path / "fresh.db")
    reg = SqliteEntityRegistry(fresh)
    assert reg.schema_version == 2
    cols = {
        row[1]
        for row in reg._conn.execute("PRAGMA table_info(entity_records)").fetchall()
    }
    assert "tenant" in cols  # a fresh db (v0) runs every migration too
    reg.close()


def test_migration_runs_once_not_reapplied(tmp_path: Path, monkeypatch: Any) -> None:
    """Re-opening a spine already at the target version is idempotent (no double-apply)."""
    path = str(tmp_path / "spine.db")
    monkeypatch.setattr(
        spine_mod,
        "_MIGRATIONS",
        ((2, ("CREATE TABLE IF NOT EXISTS _probe (x TEXT)",)),),
    )
    monkeypatch.setattr(spine_mod, "SPINE_SCHEMA_VERSION", 2)

    SqliteEntityRegistry(path).close()  # build + stamp at v2
    reopened = SqliteEntityRegistry(path)  # already v2 — migration skipped
    assert reopened.schema_version == 2
    assert (
        reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE name='_probe'"
        ).fetchone()
        is not None
    )
    reopened.close()


# ------------------------------------------------------- Postgres governance (offline)


class _FakeSpineDatabase:
    """Shared state simulating the migration-relevant Postgres spine surface."""

    def __init__(self) -> None:
        self.migrations: dict[int, str] = {}
        self.applied_statements: list[str] = []
        self.all_queries: list[str] = []  # every SQL the runner executed (verbatim)
        self.advisory_lock = asyncio.Lock()
        self.lock_acquires = 0
        self.lock_releases = 0


_SPINE_MIGRATION_STATEMENTS = {s for _, _, stmts in SPINE_MIGRATIONS for s in stmts}


class _FakeTxn:
    async def __aenter__(self) -> _FakeTxn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, db: _FakeSpineDatabase) -> None:
        self._db = db

    async def execute(self, query: str, *args: Any) -> str:
        await asyncio.sleep(0)  # yield so concurrent migrators interleave
        self._db.all_queries.append(query)
        if "pg_advisory_lock" in query:
            await self._db.advisory_lock.acquire()
            self._db.lock_acquires += 1
        elif "pg_advisory_unlock" in query:
            self._db.advisory_lock.release()
            self._db.lock_releases += 1
        elif f"INSERT INTO {SPINE_MIGRATIONS_TABLE}" in query:
            version, name = args
            self._db.migrations.setdefault(version, name)
        elif query in _SPINE_MIGRATION_STATEMENTS:
            self._db.applied_statements.append(query)
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        self._db.all_queries.append(query)
        return [{"version": v} for v in self._db.migrations]

    def transaction(self) -> _FakeTxn:
        return _FakeTxn()


class _FakeAcquire:
    def __init__(self, db: _FakeSpineDatabase) -> None:
        self._db = db

    async def __aenter__(self) -> _FakeConn:
        return _FakeConn(self._db)

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakePool:
    def __init__(self, db: _FakeSpineDatabase) -> None:
        self._db = db

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._db)


def test_pg_spine_create_schema_applies_base_via_migrate() -> None:
    """create_schema routes through migrate(): the base DDL is recorded as version 1."""

    async def scenario() -> None:
        db = _FakeSpineDatabase()
        reg = PostgresEntityRegistry(pool=_FakePool(db))
        await reg.create_schema()
        # The base DDL is version 1; subsequent forward migrations (e.g. the S6 prev_hash
        # chain column at v2) are applied too — every declared SPINE_MIGRATIONS entry lands.
        assert db.migrations == {v: n for v, n, _s in SPINE_MIGRATIONS}
        assert db.applied_statements.count(ENTITY_REGISTRY_DDL) == 1
        assert db.lock_acquires == db.lock_releases == 1
        assert not db.advisory_lock.locked()

    run_async(scenario())


def test_pg_spine_concurrent_migrate_applies_each_once() -> None:
    """Two simultaneous spine migrate() calls serialise: one applies, the other no-ops."""

    async def scenario() -> None:
        db = _FakeSpineDatabase()
        first = PostgresEntityRegistry(pool=_FakePool(db))
        second = PostgresEntityRegistry(pool=_FakePool(db))
        results = await asyncio.gather(first.migrate(), second.migrate())
        all_versions = [v for v, _n, _s in SPINE_MIGRATIONS]
        assert sorted(results, key=len) == [[], all_versions]
        assert db.applied_statements.count(ENTITY_REGISTRY_DDL) == 1
        assert db.migrations == {v: n for v, n, _s in SPINE_MIGRATIONS}
        assert db.lock_acquires == db.lock_releases == 2
        assert not db.advisory_lock.locked()

    run_async(scenario())


def test_pg_spine_ledger_table_is_namespaced_distinctly_from_storage() -> None:
    """The spine ledger table is ``spine_schema_migrations``, never the storage migrator's.

    Co-locating ``HIMMY_SPINE_DATABASE_URL`` and ``HIMMY_DATABASE_URL`` in one Postgres
    database must NOT let one migrator's ``version=1`` row make the other skip its own base
    DDL. A distinct ledger table name is the structural guarantee against that collision.
    """
    db = _FakeSpineDatabase()

    async def scenario() -> None:
        reg = PostgresEntityRegistry(pool=_FakePool(db))
        await reg.migrate()

    # The spine owns a distinctly-named ledger constant.
    assert SPINE_MIGRATIONS_TABLE == "spine_schema_migrations"
    assert SPINE_MIGRATIONS_TABLE != "schema_migrations"

    run_async(scenario())

    # Every ledger statement the runner actually executed (CREATE / SELECT / INSERT) names
    # the NAMESPACED table; NONE touches the bare ``schema_migrations`` the storage migrator
    # owns. Asserting against the emitted SQL (not source strings) is robust to comments.
    ledger_stmts = [
        q
        for q in db.all_queries
        if SPINE_MIGRATIONS_TABLE in q or re.search(r"\bschema_migrations\b", q)
    ]
    assert ledger_stmts, "migrate() must touch its ledger table"
    for stmt in ledger_stmts:
        assert SPINE_MIGRATIONS_TABLE in stmt
        # No statement references the bare storage ledger name on its own.
        assert re.search(r"(?<!spine_)\bschema_migrations\b", stmt) is None
    # Concretely: the CREATE and the INSERT both target the namespaced table.
    assert any(
        f"CREATE TABLE IF NOT EXISTS {SPINE_MIGRATIONS_TABLE}" in q
        for q in db.all_queries
    )
    assert any(f"INSERT INTO {SPINE_MIGRATIONS_TABLE}" in q for q in db.all_queries)


def test_pg_spine_migrate_uses_distinct_advisory_key_from_storage() -> None:
    """The spine migration lock key must differ from the storage migration lock key."""
    from himmy.entities.postgres import SPINE_MIGRATION_ADVISORY_LOCK_KEY
    from himmy.services.storage.postgres import MIGRATION_ADVISORY_LOCK_KEY

    assert SPINE_MIGRATION_ADVISORY_LOCK_KEY != MIGRATION_ADVISORY_LOCK_KEY


# ----------------------------------------------------------------- CI parity guard
_CREATE_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", re.IGNORECASE)
_CREATE_INDEX_RE = re.compile(r"CREATE INDEX IF NOT EXISTS\s+(\w+)", re.IGNORECASE)

#: Indexes Postgres declares that the SQLite spine deliberately lacks. The GIN index backs
#: the ``metadata @> $1::jsonb`` containment filter that exists ONLY on the Postgres
#: ``query()`` path (the SQLite registry filters metadata in Python), so there is no SQLite
#: equivalent. Adding to this set is a deliberate, documented act.
_POSTGRES_ONLY_INDEXES = frozenset({"entity_records_metadata_gin_idx"})

#: Columns the SQLite spine declares that the Postgres spine deliberately lacks.
#: ``content_hash`` is materialised in the SQLite row for fast bundle export; the Postgres
#: registry recomputes it from the record on export instead of storing it.
_SQLITE_ONLY_COLUMNS = frozenset({"content_hash"})


def _live_sqlite_spine_objects() -> tuple[set[str], dict[str, set[str]], set[str]]:
    """Build a fresh durable SQLite spine and return its tables, columns, indexes."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "spine.db")
        SqliteEntityRegistry(path).close()
        conn = sqlite3.connect(path)
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            columns = {
                t: {row[1] for row in conn.execute(f"PRAGMA table_info({t})").fetchall()}
                for t in tables
            }
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='index' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
        finally:
            conn.close()
    return tables, columns, indexes


def _pg_spine_objects() -> tuple[set[str], dict[str, set[str]], set[str]]:
    """Parse the committed Postgres spine DDL/migrations for tables, columns, indexes."""
    ddl = "\n".join(s for _, _, stmts in SPINE_MIGRATIONS for s in stmts)
    tables = set(_CREATE_TABLE_RE.findall(ddl))
    indexes = set(_CREATE_INDEX_RE.findall(ddl))
    columns: dict[str, set[str]] = {}
    for match in re.finditer(
        r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\((.*?)\n\)", ddl, re.IGNORECASE | re.DOTALL
    ):
        table, body = match.group(1), match.group(2)
        cols: set[str] = set()
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            head = line.split()[0].upper()
            if head in {"UNIQUE", "PRIMARY", "FOREIGN", "CONSTRAINT", "CHECK", "--"}:
                continue
            if line.startswith("--"):
                continue
            cols.add(line.split()[0])
        columns[table] = cols
    # Pick up columns added by forward migrations (ALTER TABLE ... ADD COLUMN [IF NOT EXISTS]),
    # so a column introduced after the base DDL (e.g. the S6 prev_hash chain column) counts
    # toward parity exactly like a SQLite migration column does.
    for match in re.finditer(
        r"ALTER TABLE\s+(\w+)\s+ADD COLUMN\s+(?:IF NOT EXISTS\s+)?(\w+)",
        ddl,
        re.IGNORECASE,
    ):
        table, col = match.group(1), match.group(2)
        columns.setdefault(table, set()).add(col)
    return tables, columns, indexes


def test_spine_table_parity() -> None:
    """The SQLite and Postgres spines declare the same tables."""
    sqlite_tables, _sqlite_cols, _sqlite_idx = _live_sqlite_spine_objects()
    pg_tables, _pg_cols, _pg_idx = _pg_spine_objects()
    assert sqlite_tables == pg_tables, (
        f"spine table drift: sqlite={sorted(sqlite_tables)} pg={sorted(pg_tables)} "
        "— add the matching DDL/migration to the other spine."
    )


def test_spine_column_parity_modulo_documented_divergences() -> None:
    """Per-table columns match modulo the frozen SQLite-only allowlist."""
    _st, sqlite_cols, _si = _live_sqlite_spine_objects()
    _pt, pg_cols, _pi = _pg_spine_objects()
    for table in sqlite_cols.keys() & pg_cols.keys():
        only_sqlite = sqlite_cols[table] - pg_cols[table]
        only_pg = pg_cols[table] - sqlite_cols[table]
        assert only_sqlite <= _SQLITE_ONLY_COLUMNS, (
            f"{table}: SQLite-only columns drifted: {sorted(only_sqlite)} "
            "— mirror onto Postgres or extend _SQLITE_ONLY_COLUMNS."
        )
        assert not only_pg, (
            f"{table}: Postgres-only columns absent from the SQLite spine: "
            f"{sorted(only_pg)} — add the matching SQLite column."
        )


def test_spine_index_parity_modulo_documented_divergences() -> None:
    """Index sets match modulo the documented Postgres-only allowlist."""
    _st, _sc, sqlite_idx = _live_sqlite_spine_objects()
    _pt, _pc, pg_idx = _pg_spine_objects()
    missing_in_pg = sqlite_idx - pg_idx
    assert not missing_in_pg, (
        f"SQLite spine indexes absent from Postgres: {sorted(missing_in_pg)}"
    )
    missing_in_sqlite = pg_idx - sqlite_idx - _POSTGRES_ONLY_INDEXES
    assert not missing_in_sqlite, (
        f"Postgres spine indexes absent from SQLite: {sorted(missing_in_sqlite)} "
        "— add the matching SQLite index or extend _POSTGRES_ONLY_INDEXES."
    )


def test_documented_index_divergence_is_exact_not_stale() -> None:
    """The frozen index-divergence allowlist matches reality exactly (no stale entries)."""
    _st, _sc, sqlite_idx = _live_sqlite_spine_objects()
    _pt, _pc, pg_idx = _pg_spine_objects()
    assert (pg_idx - sqlite_idx) == _POSTGRES_ONLY_INDEXES


@pytest.mark.skip(reason="live Postgres spine parity belongs in the Postgres CI lane")
def test_live_pg_spine_structural_parity() -> None:  # pragma: no cover - CI-lane marker
    """Placeholder: the full structural diff runs against a throwaway PG in the PG lane."""
