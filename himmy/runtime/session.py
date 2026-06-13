"""CLI chat-session persistence — now a thin adapter over the unified conversation store.

`himmy chat --session <id>` saves the :class:`ChatThread` after each turn and reloads it
next time, so a CLI conversation continues across invocations. Historically this lived in a
standalone ``.himmy/sessions.db``; as of T2.3 the durable home is the ONE
``.himmy/conversations.db`` shared with Studio (:mod:`himmy.services.storage.conversations`),
so a conversation started in ``himmy chat`` is visible in the Studio Chats list and the
``/resume`` picker and Studio show the SAME threads.

:class:`SqliteSessionStore` is kept as a BACK-COMPATIBLE FACADE over
:class:`~himmy.services.storage.conversations.ConversationStore`: same constructor
(``SqliteSessionStore(path)``), same ``save``/``load``/``list_sessions``/``prune``/``close``
surface, same :class:`SessionInfo` shape, and a live ``_conn`` exposing the unified database
(over which a ``sessions`` compatibility view lets the existing retention tests keep
backdating ``updated_at`` and lowering the SQLite variable cap). A :class:`ChatThread`
round-trips losslessly because the authoritative storage is its own JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from himmy.agents.base_agent.thread import ChatThread
from himmy.services.storage.conversations import ORIGIN_CLI, ConversationStore

#: Max bound parameters per ``DELETE ... WHERE session_id IN (?, …)`` when pruning. SQLite
#: caps a statement's host parameters at ``SQLITE_MAX_VARIABLE_NUMBER`` (999 on libsqlite
#: < 3.32) and raises ``sqlite3.OperationalError: too many SQL variables`` past it, so a
#: large doomed set is deleted in batches of this size rather than one oversized IN-list.
#: (Kept at module level — the retention tests monkeypatch it.)
_PRUNE_DELETE_CHUNK = 900


@dataclass(frozen=True)
class SessionInfo:
    """A lightweight summary of one stored session for the ``/resume`` picker."""

    session_id: str
    updated_at: str
    message_count: int


class SqliteSessionStore:
    """A durable, file-backed store of chat threads keyed by session id.

    Thin facade over :class:`~himmy.services.storage.conversations.ConversationStore`: a CLI
    "session id" is used directly as the unified ``conversation_id``, so a session saved here
    is the same row Studio Chats lists. The ``_conn`` attribute is the unified store's live
    connection (the legacy ``sessions`` view + INSTEAD-OF trigger make the old retention
    tests' direct ``UPDATE sessions`` / ``setlimit`` calls keep working).
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the unified conversation database at ``path``."""
        self._store = ConversationStore(path)

    @property
    def _conn(self):  # type: ignore[no-untyped-def]
        """The unified store's live SQLite connection (back-compat for direct-SQL tests)."""
        return self._store._conn

    def save(self, session_id: str, thread: ChatThread) -> None:
        """Upsert ``thread`` under ``session_id`` as a CLI-origin conversation."""
        self._store.save_thread(session_id, thread, origin=ORIGIN_CLI)

    def load(self, session_id: str) -> ChatThread | None:
        """Return the saved thread for ``session_id`` (or ``None``)."""
        return self._store.load_thread(session_id)

    def list_sessions(self, *, limit: int | None = None) -> list[SessionInfo]:
        """Recent sessions, newest first, with each thread's message count.

        Powers the ``/resume`` picker. Returns ALL conversations (CLI- and Studio-origin)
        so a thread started in Studio is resumable from the CLI and vice versa — the whole
        point of the unified store. A thread that fails to deserialize is reported with the
        flat-projection count rather than breaking the listing.
        """
        return [
            SessionInfo(
                session_id=s.conversation_id,
                updated_at=s.updated_at,
                message_count=s.message_count,
            )
            for s in self._store.list_summaries(limit=limit)
        ]

    def prune(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int:
        """Delete stale sessions, bounding the conversation store's unbounded growth.

        Every REPL turn upserts the whole thread, so the store grows without bound across a
        long-lived deployment. This explicit, operator-invoked call reclaims that space;
        nothing prunes automatically. Pass ``older_than_days`` to drop sessions not touched
        since the cutoff and/or ``keep_last`` to keep only the N most recently updated; at
        least one bound is required and a session is removed if it fails EITHER bound.
        Deletes via the unified ``conversations`` table in chunks of
        :data:`_PRUNE_DELETE_CHUNK` so the ``IN``-list never trips the SQLite variable cap.
        Returns the number of sessions deleted.
        """
        if older_than_days is None and keep_last is None:
            raise ValueError("prune needs at least one of older_than_days / keep_last")
        conn = self._store._conn
        doomed: set[str] = set()
        if older_than_days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
            rows = conn.execute(
                "SELECT conversation_id FROM conversations WHERE updated_at < ?",
                (cutoff,),
            ).fetchall()
            doomed.update(r[0] for r in rows)
        if keep_last is not None:
            keep = max(int(keep_last), 0)
            rows = conn.execute(
                "SELECT conversation_id FROM conversations ORDER BY updated_at DESC "
                "LIMIT -1 OFFSET ?",
                (keep,),
            ).fetchall()
            doomed.update(r[0] for r in rows)
        if not doomed:
            return 0
        ordered = sorted(doomed)
        for start in range(0, len(ordered), _PRUNE_DELETE_CHUNK):
            batch = ordered[start : start + _PRUNE_DELETE_CHUNK]
            placeholders = ",".join("?" for _ in batch)
            conn.execute(
                f"DELETE FROM conversation_messages "  # noqa: S608
                f"WHERE conversation_id IN ({placeholders})",
                tuple(batch),
            )
            conn.execute(
                f"DELETE FROM conversations "  # noqa: S608
                f"WHERE conversation_id IN ({placeholders})",
                tuple(batch),
            )
        conn.commit()
        return len(doomed)

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._store.close()


__all__ = ["SessionInfo", "SqliteSessionStore"]
