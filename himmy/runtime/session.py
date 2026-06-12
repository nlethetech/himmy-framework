"""Chat session persistence: a durable thread store so conversations survive restarts.

`himmy chat --session <id>` saves the :class:`ChatThread` after each turn and reloads it
next time, so a CLI conversation continues across invocations. :class:`SqliteSessionStore`
is a thin stdlib-sqlite3 store (mirroring the other Sqlite stores) keyed by a session id;
a :class:`ChatThread` round-trips cleanly via ``model_dump_json``/``model_validate_json``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from himmy.agents.base_agent.thread import ChatThread
from himmy.core.ids import utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    thread     TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

#: Max bound parameters per ``DELETE ... WHERE session_id IN (?, …)`` when pruning. SQLite
#: caps a statement's host parameters at ``SQLITE_MAX_VARIABLE_NUMBER`` (999 on libsqlite
#: < 3.32) and raises ``sqlite3.OperationalError: too many SQL variables`` past it, so a
#: large doomed set is deleted in batches of this size rather than one oversized IN-list.
_PRUNE_DELETE_CHUNK = 900


@dataclass(frozen=True)
class SessionInfo:
    """A lightweight summary of one stored session for the ``/resume`` picker."""

    session_id: str
    updated_at: str
    message_count: int


class SqliteSessionStore:
    """A durable, file-backed store of chat threads keyed by session id."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path``."""
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, session_id: str, thread: ChatThread) -> None:
        """Upsert ``thread`` under ``session_id``."""
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, thread, updated_at) "
            "VALUES (?, ?, ?)",
            (session_id, thread.model_dump_json(), utc_now_iso()),
        )
        self._conn.commit()

    def load(self, session_id: str) -> ChatThread | None:
        """Return the saved thread for ``session_id`` (or ``None``)."""
        row = self._conn.execute(
            "SELECT thread FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return ChatThread.model_validate_json(row[0]) if row else None

    def list_sessions(self, *, limit: int | None = None) -> list[SessionInfo]:
        """Recent sessions, newest first, with each thread's message count.

        Powers the ``/resume`` picker: returns a :class:`SessionInfo` per stored
        session (id, last-active timestamp, and how many messages its thread holds).
        A thread that fails to deserialize is reported with ``message_count=0``
        rather than breaking the whole listing.
        """
        sql = "SELECT session_id, thread, updated_at FROM sessions ORDER BY updated_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql).fetchall()
        out: list[SessionInfo] = []
        for session_id, thread_json, updated_at in rows:
            try:
                count = len(ChatThread.model_validate_json(thread_json).messages)
            except Exception:  # noqa: BLE001 - a corrupt row must not break the list
                count = 0
            out.append(
                SessionInfo(
                    session_id=session_id,
                    updated_at=updated_at,
                    message_count=count,
                )
            )
        return out

    def prune(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int:
        """Delete stale sessions, bounding the session store's unbounded growth.

        Every REPL turn upserts the whole thread, so a session store grows without
        bound across a long-lived deployment. This explicit, operator-invoked call
        reclaims that space; nothing prunes automatically — callers must drive it from
        their own ops job (the bundled ``scripts/ops_prune.py`` does so for the default
        ``.himmy`` stores). Pass ``older_than_days`` to drop sessions not touched since the
        cutoff (by ``updated_at``) and/or ``keep_last`` to keep only the N most recently
        updated sessions. At least one bound must be given; a session is removed if it
        fails EITHER bound. Returns the number of sessions deleted.
        """
        if older_than_days is None and keep_last is None:
            raise ValueError("prune needs at least one of older_than_days / keep_last")
        doomed: set[str] = set()
        if older_than_days is not None:
            cutoff = (
                datetime.now(UTC) - timedelta(days=older_than_days)
            ).isoformat()
            rows = self._conn.execute(
                "SELECT session_id FROM sessions WHERE updated_at < ?", (cutoff,)
            ).fetchall()
            doomed.update(r[0] for r in rows)
        if keep_last is not None:
            keep = max(int(keep_last), 0)
            rows = self._conn.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC "
                "LIMIT -1 OFFSET ?",
                (keep,),
            ).fetchall()
            doomed.update(r[0] for r in rows)
        if not doomed:
            return 0
        # Delete in chunks so the IN-list never exceeds SQLITE_MAX_VARIABLE_NUMBER
        # ("too many SQL variables"); all chunks commit together so a mid-prune failure
        # leaves the store untouched rather than half-pruned.
        ordered = sorted(doomed)
        for start in range(0, len(ordered), _PRUNE_DELETE_CHUNK):
            batch = ordered[start : start + _PRUNE_DELETE_CHUNK]
            placeholders = ",".join("?" for _ in batch)
            self._conn.execute(
                f"DELETE FROM sessions WHERE session_id IN ({placeholders})",  # noqa: S608
                tuple(batch),
            )
        self._conn.commit()
        return len(doomed)

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()


__all__ = ["SessionInfo", "SqliteSessionStore"]
