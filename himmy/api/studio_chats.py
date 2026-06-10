"""Studio Chats: durable, resumable chat sessions — and the Projects that group them.

The Chat screen keeps its live transcript in React state; this store persists a session
so it can be reopened later. A ``ChatSession`` holds metadata (title, agent path,
provider) and an ordered list of ``ChatMessage`` rows. SQLite at ``.himmy/chats.db``
(cwd-keyed singleton), mirroring the Tasks/Notes/Calendar stores.

Projects give sustained work a home: a named group of chats with an optional
default knowledge base and default agent. They live in the same database
(``projects`` table) and sessions carry a nullable ``project_id`` — added via
the same additive ``PRAGMA table_info`` + ``ALTER TABLE`` migration the runs
store uses, so pre-existing databases upgrade in place without touching rows.
"""

from __future__ import annotations

import builtins
import sqlite3
from collections.abc import Mapping, Sequence
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
    updated_at  TEXT NOT NULL,
    project_id  TEXT
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
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    kb_id       TEXT,
    agent_path  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

#: Columns added to ``chat_sessions`` after the original schema shipped, with the
#: ``ALTER TABLE`` declaration that backfills a pre-existing database. Mirrors
#: the migration pattern in :mod:`himmy.api.studio_runs`. Additive + nullable
#: only — existing rows are never rewritten.
_LATER_SESSION_COLUMNS: dict[str, str] = {
    "project_id": "TEXT",
}


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
    project_id: str | None = None


class ChatSessionDetail(ChatSession):
    messages: list[ChatMessage] = []


class Project(BaseModel):
    """A named home for sustained work: chats + a default KB + a default agent."""

    id: str = Field(default_factory=new_uuid)
    name: str = "New project"
    description: str = ""
    kb_id: str | None = None
    agent_path: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    chat_count: int = 0


#: The only project columns :meth:`ChatsStore.update_project` will touch.
_PROJECT_FIELDS = frozenset({"name", "description", "kb_id", "agent_path"})


class ChatsStore:
    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created (project_id)."""
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
        }
        for name, decl in _LATER_SESSION_COLUMNS.items():
            if name not in cols:
                self._conn.execute(
                    f"ALTER TABLE chat_sessions ADD COLUMN {name} {decl}"
                )

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
        project_id: str | None = None,
    ) -> ChatSession:
        """Create or replace a session and its messages (upsert by id).

        ``project_id=None`` means "leave the assignment alone" — a resave from
        the Chat screen must never silently pull a chat out of its project.
        Clearing an assignment goes through :meth:`unassign_chat`.
        """
        now = utc_now_iso()
        sid = session_id or new_uuid()
        resolved_title = (title or "").strip() or _derive_title(messages)
        existing = self._conn.execute(
            "SELECT created_at, project_id FROM chat_sessions WHERE id = ?", (sid,)
        ).fetchone()
        created = existing["created_at"] if existing else now
        effective_project = project_id or (existing["project_id"] if existing else None)
        self._conn.execute(
            """
            INSERT INTO chat_sessions
                (id, title, agent_path, provider, created_at, updated_at, project_id)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                agent_path=excluded.agent_path,
                provider=excluded.provider,
                updated_at=excluded.updated_at,
                project_id=excluded.project_id
            """,
            (
                sid,
                resolved_title,
                agent_path,
                provider,
                created,
                now,
                effective_project,
            ),
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
            project_id=effective_project,
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

    # ---- Projects ---------------------------------------------------------

    def list_projects(self) -> builtins.list[Project]:
        # (builtins.list: inside this class, bare ``list`` names the method above)
        rows = self._conn.execute(
            """
            SELECT p.*, COUNT(s.id) AS n
            FROM projects p
            LEFT JOIN chat_sessions s ON s.project_id = p.id
            GROUP BY p.id
            ORDER BY p.updated_at DESC
            """
        ).fetchall()
        return [self._project(r, r["n"]) for r in rows]

    def get_project(self, project_id: str) -> Project | None:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None
        n = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chat_sessions WHERE project_id = ?",
            (project_id,),
        ).fetchone()["n"]
        return self._project(row, n)

    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        kb_id: str | None = None,
        agent_path: str | None = None,
    ) -> Project:
        now = utc_now_iso()
        pid = new_uuid()
        self._conn.execute(
            """
            INSERT INTO projects
                (id, name, description, kb_id, agent_path, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (pid, name, description, kb_id, agent_path, now, now),
        )
        self._conn.commit()
        return Project(
            id=pid,
            name=name,
            description=description,
            kb_id=kb_id,
            agent_path=agent_path,
            created_at=now,
            updated_at=now,
            chat_count=0,
        )

    def update_project(
        self, project_id: str, changes: Mapping[str, str | None]
    ) -> Project | None:
        """Apply a partial update; unknown keys are ignored, ``None`` clears.

        Only the columns in ``_PROJECT_FIELDS`` are writable (the SQL below is
        built from that allowlist, never from caller-supplied names).
        """
        sets = {k: changes[k] for k in _PROJECT_FIELDS if k in changes}
        if "name" in sets and not (sets["name"] or "").strip():
            del sets["name"]  # a project always keeps a non-empty name
        if sets:
            assignments = ", ".join(f"{k} = ?" for k in sets)
            cur = self._conn.execute(
                # column names come from the _PROJECT_FIELDS allowlist above,
                # never from caller input — only values are interpolated via ?
                f"UPDATE projects SET {assignments}, updated_at = ? WHERE id = ?",  # noqa: S608
                (*sets.values(), utc_now_iso(), project_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        """Delete a project; its chats survive, merely ungrouped."""
        self._conn.execute(
            "UPDATE chat_sessions SET project_id = NULL WHERE project_id = ?",
            (project_id,),
        )
        cur = self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def project_chats(self, project_id: str) -> builtins.list[ChatSession]:
        """The project's sessions, newest activity first."""
        rows = self._conn.execute(
            """
            SELECT s.*, COUNT(m.id) AS n
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.project_id = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            """,
            (project_id,),
        ).fetchall()
        return [self._session(r, r["n"]) for r in rows]

    def assign_chat(self, project_id: str, chat_id: str) -> bool:
        """Put a chat into a project. False when either side is unknown."""
        if self.get_project(project_id) is None:
            return False
        cur = self._conn.execute(
            "UPDATE chat_sessions SET project_id = ? WHERE id = ?",
            (project_id, chat_id),
        )
        if cur.rowcount > 0:
            self._conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), project_id),
            )
        self._conn.commit()
        return cur.rowcount > 0

    def unassign_chat(self, chat_id: str) -> bool:
        """Pull a chat out of whatever project holds it."""
        cur = self._conn.execute(
            "UPDATE chat_sessions SET project_id = NULL "
            "WHERE id = ? AND project_id IS NOT NULL",
            (chat_id,),
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
            project_id=row["project_id"],
        )

    @staticmethod
    def _project(row: sqlite3.Row, n: int) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            kb_id=row["kb_id"],
            agent_path=row["agent_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            chat_count=n,
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
    "Project",
    "chats_db_path",
    "get_chats_store",
    "reset_chats_store",
]
