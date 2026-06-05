"""Memory store: durable persistence for an agent's long-term memories.

A :class:`MemoryRecord` is one remembered fact or episode for a subject (a user, an
agent, a session). :class:`InMemoryMemoryStore` is the volatile default;
:class:`SqliteMemoryStore` persists to a stdlib-``sqlite3`` file so memories survive
across processes (mirroring :class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry`).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso


class MemoryRecord(BaseModel):
    """One stored memory for a subject."""

    memory_id: str = Field(default_factory=new_uuid)
    subject_id: str = "default"
    kind: str = "semantic"  # semantic | episodic | ...
    text: str = ""
    metadata: dict[str, Any] = {}
    created_at: str = Field(default_factory=utc_now_iso)


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence contract for memories (synchronous)."""

    def save(self, record: MemoryRecord) -> MemoryRecord: ...

    def list(self, subject_id: str | None = None) -> list[MemoryRecord]: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def delete(self, memory_id: str) -> bool: ...


class InMemoryMemoryStore:
    """A volatile, process-local :class:`MemoryStore`."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.memory_id] = record
        return record

    def list(self, subject_id: str | None = None) -> list[MemoryRecord]:
        records = [
            r
            for r in self._records.values()
            if subject_id is None or r.subject_id == subject_id
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


class SqliteMemoryStore:
    """A durable, file-backed :class:`MemoryStore` (stdlib sqlite3)."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path``."""
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self._conn.execute(
            "INSERT OR REPLACE INTO memories "
            "(memory_id, subject_id, kind, text, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.memory_id,
                record.subject_id,
                record.kind,
                record.text,
                json.dumps(record.metadata),
                record.created_at,
            ),
        )
        self._conn.commit()
        return record

    def list(self, subject_id: str | None = None) -> list[MemoryRecord]:
        if subject_id is None:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY created_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE subject_id = ? ORDER BY created_at",
                (subject_id,),
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
        return MemoryRecord(
            memory_id=row["memory_id"],
            subject_id=row["subject_id"],
            kind=row["kind"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
        )


__all__ = [
    "MemoryRecord",
    "MemoryStore",
    "InMemoryMemoryStore",
    "SqliteMemoryStore",
]
