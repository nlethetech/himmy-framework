"""Memory store: durable persistence for an agent's long-term memories.

A :class:`MemoryRecord` is one remembered fact or episode for a subject (a user, an
agent, a session). :class:`InMemoryMemoryStore` is the volatile default;
:class:`SqliteMemoryStore` persists to a stdlib-``sqlite3`` file so memories survive
across processes (mirroring :class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry`).
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.entities.records import EntityRecord

#: The Letta-style memory tiers, in promotion order (hot -> cold).
MEMORY_TIERS: tuple[str, ...] = ("core", "recall", "archival")


class MemoryRecord(BaseModel):
    """One stored memory for a subject.

    A memory is *bi-temporal*: ``created_at`` (and the transaction-time version
    chain on the spine) records WHEN the fact was ingested, while
    ``valid_from``/``valid_to`` record WHEN the fact is true in the world (Graphiti
    semantics). A fact is *invalidated*, not deleted, by stamping ``valid_to`` and
    pointing ``superseded_by`` at its replacement — so point-in-time recall
    ("what was true in March") stays answerable. ``stable_key`` is the semantic
    identity of the fact (e.g. ``"alice/home_city"``) so successive versions of the
    same fact share a spine ``stable_id``. All new fields are defaulted, so existing
    rows and zero-arg constructors keep working unchanged.
    """

    memory_id: str = Field(default_factory=new_uuid)
    subject_id: str = "default"
    kind: str = "semantic"  # semantic | episodic | ...
    text: str = ""
    metadata: dict[str, Any] = {}
    created_at: str = Field(default_factory=utc_now_iso)
    # --- tiered recall (Letta core/recall/archival) ------------------------
    tier: str = "recall"
    # --- bi-temporal validity (Graphiti invalidate-not-delete) -------------
    valid_from: str = ""  # filled from created_at in model_post_init when blank
    valid_to: str | None = None  # None = currently true
    superseded_by: str | None = None  # memory_id of the fact that replaced it
    # --- provenance --------------------------------------------------------
    confidence: float = 1.0
    source: str = "user"  # user | llm_extracted | tool | imported
    stable_key: str | None = None  # semantic identity of the FACT (vs the row id)

    def model_post_init(self, __context: Any) -> None:
        """Default ``valid_from`` to ``created_at`` so valid-time always has a start."""
        if not self.valid_from:
            self.valid_from = self.created_at

    def to_record(self, *, version: int = 1) -> EntityRecord:
        """Project this memory into its canonical ``EntityRecord`` (kind ``memory_fact``).

        Delegates to :func:`himmy.services.memory.projection.memory_to_record`; the
        import is lazy to avoid a service -> entities import cycle (mirroring
        :meth:`himmy.core.events.RunEvent.to_record`).
        """
        from himmy.services.memory.projection import memory_to_record

        return memory_to_record(self, version=version)


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence contract for memories (synchronous)."""

    def save(self, record: MemoryRecord) -> MemoryRecord: ...

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[MemoryRecord]: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def delete(self, memory_id: str) -> bool: ...


def _passes_filters(
    record: MemoryRecord, *, active_only: bool, tier: str | None
) -> bool:
    """Apply the (backend-agnostic) ``active_only``/``tier`` row filters."""
    if active_only and record.valid_to is not None:
        return False
    if tier is not None and record.tier != tier:
        return False
    return True


class InMemoryMemoryStore:
    """A volatile, process-local :class:`MemoryStore`."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.memory_id] = record
        return record

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[MemoryRecord]:
        records = [
            r
            for r in self._records.values()
            if (subject_id is None or r.subject_id == subject_id)
            and _passes_filters(r, active_only=active_only, tier=tier)
        ]
        return sorted(records, key=lambda r: r.created_at)

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id  TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_subject_idx ON memories (subject_id);
"""

#: Columns added after the original 6-column schema, with the ``ALTER TABLE``
#: definition used to backfill them on a legacy database. Each is PRAGMA-guarded
#: so :meth:`SqliteMemoryStore._migrate` is idempotent and upgrades existing
#: ``.himmy/memory.db`` files in place (no data loss; old rows read with defaults).
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("tier", "ALTER TABLE memories ADD COLUMN tier TEXT NOT NULL DEFAULT 'recall'"),
    (
        "valid_from",
        "ALTER TABLE memories ADD COLUMN valid_from TEXT NOT NULL DEFAULT ''",
    ),
    ("valid_to", "ALTER TABLE memories ADD COLUMN valid_to TEXT"),
    ("superseded_by", "ALTER TABLE memories ADD COLUMN superseded_by TEXT"),
    (
        "confidence",
        "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
    ),
    ("source", "ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'user'"),
    ("stable_key", "ALTER TABLE memories ADD COLUMN stable_key TEXT"),
)


class SqliteMemoryStore:
    """A durable, file-backed :class:`MemoryStore` (stdlib sqlite3)."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path`` and migrate it in place."""
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additively add any bi-temporal/tier columns missing on a legacy db."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        for column, ddl in _MIGRATIONS:
            if column not in existing:
                self._conn.execute(ddl)  # noqa: S608 - constant DDL, no interpolation

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self._conn.execute(
            "INSERT OR REPLACE INTO memories "
            "(memory_id, subject_id, kind, text, metadata, created_at, "
            " tier, valid_from, valid_to, superseded_by, confidence, source, "
            " stable_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.memory_id,
                record.subject_id,
                record.kind,
                record.text,
                json.dumps(record.metadata),
                record.created_at,
                record.tier,
                record.valid_from,
                record.valid_to,
                record.superseded_by,
                record.confidence,
                record.source,
                record.stable_key,
            ),
        )
        self._conn.commit()
        return record

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[MemoryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if subject_id is not None:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if active_only:
            clauses.append("valid_to IS NULL")
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM memories{where} ORDER BY created_at",  # noqa: S608
            params,
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return self._row(row) if row else None

    def delete(self, memory_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> MemoryRecord:
        keys = set(row.keys())
        # ``stable_key`` etc. may be absent only if a migration was skipped; guard
        # each new column so a partially-migrated row never raises a KeyError.
        return MemoryRecord(
            memory_id=row["memory_id"],
            subject_id=row["subject_id"],
            kind=row["kind"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            tier=row["tier"] if "tier" in keys else "recall",
            valid_from=(
                row["valid_from"]
                if "valid_from" in keys and row["valid_from"]
                else row["created_at"]
            ),
            valid_to=row["valid_to"] if "valid_to" in keys else None,
            superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
            confidence=row["confidence"] if "confidence" in keys else 1.0,
            source=row["source"] if "source" in keys else "user",
            stable_key=row["stable_key"] if "stable_key" in keys else None,
        )


__all__ = [
    "MEMORY_TIERS",
    "MemoryRecord",
    "MemoryStore",
    "InMemoryMemoryStore",
    "SqliteMemoryStore",
]
