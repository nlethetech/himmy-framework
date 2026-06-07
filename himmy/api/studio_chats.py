"""Studio Chats: durable, resumable chat sessions.

The Chat screen keeps its live transcript in React state; this store persists a session
so it can be reopened later. A ``ChatSession`` holds metadata (title, agent path,
provider) and an ordered list of ``ChatMessage`` rows. SQLite at ``.himmy/chats.db``
(cwd-keyed singleton), mirroring the Tasks/Notes/Calendar stores.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    agent_path  TEXT,
    provider    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages (session_id, seq);
"""


class ChatMessage(BaseModel):
    role: str  # "user" | "agent"
    text: str


class ChatSession(BaseModel):
    id: str = Field(default_factory=new_uuid)
    title: str = "New chat"
    agent_path: str | None = None
    provider: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    message_count: int = 0


class ChatSessionDetail(ChatSession):
    messages: list[ChatMessage] = []


class ChatsStore:
    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def list(self) -> list[ChatSession]:
        rows = self._conn.execute(
            """
            SELECT s.*, COUNT(m.id) AS n
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
        return [self._session(r, r["n"]) for r in rows]

    def get(self, session_id: str) -> ChatSessionDetail | None:
        row = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        msgs = self._conn.execute(
            "SELECT role, text FROM chat_messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        base = self._session(row, len(msgs))
        return ChatSessionDetail(
            **base.model_dump(),
            messages=[ChatMessage(role=m["role"], text=m["text"]) for m in msgs],
        )

    def save(
        self,
        *,
        session_id: str | None,
        title: str | None,
        agent_path: str | None,
        provider: str | None,
        messages: Sequence[ChatMessage],
    ) -> ChatSession:
        """Create or replace a session and its messages (upsert by id)."""
        now = utc_now_iso()
        sid = session_id or new_uuid()
        resolved_title = (title or "").strip() or _derive_title(messages)
        existing = self._conn.execute(
            "SELECT created_at FROM chat_sessions WHERE id = ?", (sid,)
        ).fetchone()
        created = existing["created_at"] if existing else now
        self._conn.execute(
            """
            INSERT INTO chat_sessions
                (id, title, agent_path, provider, created_at, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                agent_path=excluded.agent_path,
                provider=excluded.provider,
                updated_at=excluded.updated_at
            """,
            (sid, resolved_title, agent_path, provider, created, now),
        )
        self._conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (sid,))
        for i, m in enumerate(messages):
            self._conn.execute(
                """
                INSERT INTO chat_messages (id, session_id, role, text, seq, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (new_uuid(), sid, m.role, m.text, i, now),
            )
        self._conn.commit()
        return ChatSession(
            id=sid,
            title=resolved_title,
            agent_path=agent_path,
            provider=provider,
            created_at=created,
            updated_at=now,
            message_count=len(messages),
        )

    def rename(self, session_id: str, title: str) -> bool:
        cur = self._conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, utc_now_iso(), session_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, session_id: str) -> bool:
        self._conn.execute(
            "DELETE FROM chat_messages WHERE session_id = ?", (session_id,)
        )
        cur = self._conn.execute(
            "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _session(row: sqlite3.Row, n: int) -> ChatSession:
        return ChatSession(
            id=row["id"],
            title=row["title"],
            agent_path=row["agent_path"],
            provider=row["provider"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=n,
        )

    def close(self) -> None:
        self._conn.close()


def _derive_title(messages: Sequence[ChatMessage]) -> str:
    """Use the first user message as the session title (trimmed)."""
    for m in messages:
        if m.role == "user" and m.text.strip():
            t = m.text.strip().splitlines()[0]
            return t[:60] + ("…" if len(t) > 60 else "")
    return "New chat"


_STORE: ChatsStore | None = None
_PATH: str | None = None


def chats_db_path() -> str:
    import os

    env = os.environ.get("HIMMY_CHATS_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "chats.db")


def get_chats_store() -> ChatsStore:
    global _STORE, _PATH
    path = chats_db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        _STORE = ChatsStore(path)
        _PATH = path
    return _STORE


def reset_chats_store() -> None:
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None


__all__ = [
    "ChatMessage",
    "ChatSession",
    "ChatSessionDetail",
    "ChatsStore",
    "chats_db_path",
    "get_chats_store",
    "reset_chats_store",
]
