"""Chat session persistence: a durable thread store so conversations survive restarts.

`himmy chat --session <id>` saves the :class:`ChatThread` after each turn and reloads it
next time, so a CLI conversation continues across invocations. :class:`SqliteSessionStore`
is a thin stdlib-sqlite3 store (mirroring the other Sqlite stores) keyed by a session id;
a :class:`ChatThread` round-trips cleanly via ``model_dump_json``/``model_validate_json``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from himmy.agents.base_agent.thread import ChatThread
from himmy.core.ids import utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    thread     TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


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

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()


__all__ = ["SessionInfo", "SqliteSessionStore"]
