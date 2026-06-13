"""One durable conversation store: the union of CLI sessions + Studio chats (T2.3).

Historically a conversation lived in one of two disconnected databases:

* ``.himmy/sessions.db`` — the CLI's :class:`~himmy.runtime.session.SqliteSessionStore`,
  which round-trips a *rich* :class:`~himmy.agents.base_agent.thread.ChatThread` as JSON
  (every role — ``system``/``user``/``assistant``/``tool`` — plus per-message metadata).
  This is the LOSSLESS, authoritative shape.
* ``.himmy/chats.db`` — Studio's :class:`~himmy.api.studio_chats.ChatsStore`, which keeps a
  *flat* display transcript (``user``/``agent`` rows only) plus a ``projects`` grouping.

So a conversation started in ``himmy chat`` was invisible in the Studio Chats list and vice
versa. :class:`ConversationStore` collapses both into ONE durable ``.himmy/conversations.db``
whose contract is:

* the rich :class:`ChatThread` JSON is the SINGLE source of truth (``conversations.thread``);
* a regenerated FLAT ``conversation_messages`` projection (``user``/``agent`` rows) is what
  both UIs' list/get surfaces read — it is always rebuilt from the authoritative thread on
  every write, never edited independently, so the two can never drift.

Both legacy stores are re-expressed as THIN ADAPTERS over this one store (see
:mod:`himmy.runtime.session` and :mod:`himmy.api.studio_chats`) so their public APIs — and
the existing tests that reach ``store._conn`` / the literal ``chat_sessions`` table — keep
working unchanged: a backwards-compatible ``chat_sessions`` VIEW and a ``sessions`` VIEW are
created over ``conversations``, and the adapters expose the live connection as ``_conn``.

The lossy direction (a flat Studio save -> a ChatThread) is handled EXPLICITLY by
:func:`thread_from_flat`: each ``user`` row becomes a ``MessageRole.USER`` message and each
``agent`` row a ``MessageRole.ASSISTANT`` message; Studio carries no ``system``/``tool``
turns, so none are fabricated. The inverse (:func:`flat_from_thread`) is also explicit:
``user``->``user``, everything else (``assistant``/``system``/``tool``) collapses to the
display role ``agent`` (Studio's only non-user role) and empty turns are dropped.

Offline-first is preserved: stdlib ``sqlite3`` only, the canonical path resolves with zero
config (``.himmy/conversations.db``), and a one-time best-effort import folds any pre-existing
``sessions.db`` (lossless) and ``chats.db`` (flat -> thread) rows into the unified store.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
from himmy.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: ``origin`` marks which front door created a conversation. Used only for provenance /
#: filtering; both origins live in the same table and are visible to both UIs.
ORIGIN_CLI = "cli"
ORIGIN_STUDIO = "studio"

#: Studio's flat transcript knows only two display roles. Every authoritative ChatThread
#: role collapses onto one of these for the projection both UIs read.
FLAT_ROLE_USER = "user"
FLAT_ROLE_AGENT = "agent"

#: Max bound parameters per ``DELETE ... WHERE conversation_id IN (?, …)`` when pruning, so
#: a large doomed set never trips ``SQLITE_MAX_VARIABLE_NUMBER`` ("too many SQL variables").
#: Mirrors the guard in :mod:`himmy.runtime.session` / :mod:`himmy.services.storage.sqlite`.
_PRUNE_DELETE_CHUNK = 900

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    thread          TEXT NOT NULL,
    origin          TEXT NOT NULL DEFAULT 'cli',
    title           TEXT,
    agent_path      TEXT,
    provider        TEXT,
    project_id      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations (updated_at DESC);
CREATE TABLE IF NOT EXISTS conversation_messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    text            TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_conv
    ON conversation_messages (conversation_id, seq);
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

#: Back-compat VIEWs so the legacy adapters' literal SQL keeps working. ``chat_sessions``
#: is what :mod:`himmy.api.studio_chats` (and tests reaching ``store._conn``) read; the
#: column list is byte-identical to the old ``chat_sessions`` table so existing
#: ``PRAGMA table_info`` / ``SELECT *`` assertions still see the same shape. ``sessions``
#: mirrors the old CLI ``SqliteSessionStore`` table (``session_id``/``thread``/``updated_at``).
_COMPAT_VIEWS = """
CREATE VIEW IF NOT EXISTS chat_sessions AS
    SELECT conversation_id AS id,
           title,
           agent_path,
           provider,
           created_at,
           updated_at,
           project_id
    FROM conversations;
CREATE VIEW IF NOT EXISTS sessions AS
    SELECT conversation_id AS session_id,
           thread,
           updated_at
    FROM conversations;
-- The legacy retention tests backdate a row via ``UPDATE sessions SET updated_at = ?``.
-- A SQLite view is read-only without an INSTEAD-OF trigger, so install one that rewrites
-- the underlying ``conversations`` row (only ``updated_at`` is writable through the view).
CREATE TRIGGER IF NOT EXISTS sessions_set_updated_at
    INSTEAD OF UPDATE OF updated_at ON sessions
BEGIN
    UPDATE conversations SET updated_at = NEW.updated_at
    WHERE conversation_id = OLD.session_id;
END;
"""


# --------------------------------------------------------------------------- mapping
# The lossy/lossless mappings between the authoritative ChatThread and the flat projection
# are kept here as explicit module functions (reviewer must-fix: the flat->thread direction
# must be handled, not hand-waved).


def flat_role(role: MessageRole | str) -> str:
    """Collapse an authoritative thread role onto Studio's flat display role.

    ``user`` stays ``user``; ``assistant``/``system``/``tool`` all become ``agent`` —
    Studio's only non-user transcript role.
    """
    value = role.value if isinstance(role, MessageRole) else str(role)
    return FLAT_ROLE_USER if value == MessageRole.USER.value else FLAT_ROLE_AGENT


def flat_from_thread(thread: ChatThread) -> list[tuple[str, str]]:
    """Project a rich :class:`ChatThread` to ordered ``(flat_role, text)`` pairs.

    Empty-content turns (e.g. an assistant message that only carried tool calls) are
    dropped so the display transcript matches what a human saw. This is the projection
    both UIs' list/get surfaces read; it is regenerated on every write.
    """
    out: list[tuple[str, str]] = []
    for m in thread.messages:
        text = m.content or ""
        if not text.strip():
            continue
        out.append((flat_role(m.role), text))
    return out


def thread_from_flat(
    *,
    conversation_id: str,
    agent_path: str | None,
    flat_messages: Sequence[tuple[str, str]],
) -> ChatThread:
    """Reconstruct an authoritative :class:`ChatThread` from a flat Studio transcript.

    The LOSSY direction (reviewer must-fix — handled explicitly, not hand-waved): Studio
    persists only ``user``/``agent`` display rows with no per-message metadata, so the
    reconstruction is deliberately faithful-but-minimal — each ``user`` row becomes a
    :data:`MessageRole.USER` message and every other row a :data:`MessageRole.ASSISTANT`
    message. No ``system``/``tool`` turns are fabricated (Studio never recorded any), and
    the original ``message_id``/timestamps are not recoverable, so fresh ones are minted.
    The thread id is pinned to ``conversation_id`` so the round-trip is stable.
    """
    thread = ChatThread(thread_id=conversation_id, agent_id=agent_path)
    for role, text in flat_messages:
        thread.append_message(
            Message(
                role=(
                    MessageRole.USER
                    if role == FLAT_ROLE_USER
                    else MessageRole.ASSISTANT
                ),
                content=text,
            )
        )
    return thread


def _derive_title(thread: ChatThread) -> str:
    """The first non-empty user turn, trimmed, as a conversation title."""
    for m in thread.messages:
        if m.role == MessageRole.USER and (m.content or "").strip():
            line = m.content.strip().splitlines()[0]
            return line[:60] + ("…" if len(line) > 60 else "")
    return "New chat"


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class ConversationSummary:
    """A lightweight summary of one stored conversation (no message bodies)."""

    conversation_id: str
    origin: str
    title: str
    agent_path: str | None
    provider: str | None
    project_id: str | None
    created_at: str
    updated_at: str
    message_count: int


@dataclass(frozen=True)
class FlatMessage:
    """One row of the regenerated flat projection both UIs read."""

    role: str
    text: str


# --------------------------------------------------------------------------- store


class ConversationStore:
    """The ONE durable store of conversations, backing both the CLI and Studio.

    Authoritative state is the rich :class:`ChatThread` JSON in ``conversations.thread``;
    the flat ``conversation_messages`` projection is regenerated from it on every write.
    The store keys a conversation by its ``conversation_id`` — the CLI's session id
    (``"last"``, ``"work"``, …) and Studio's chat id (a UUID) are BOTH used directly as the
    conversation id, so neither namespace needs an alias table and both UIs see the same
    rows.
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the unified conversation database at ``path``."""
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.executescript(_COMPAT_VIEWS)
        self._conn.commit()

    # -- authoritative writes ------------------------------------------------

    def save_thread(
        self,
        conversation_id: str,
        thread: ChatThread,
        *,
        origin: str = ORIGIN_CLI,
        title: str | None = None,
        agent_path: str | None = None,
        provider: str | None = None,
        project_id: str | None = None,
    ) -> ConversationSummary:
        """Upsert the authoritative ``thread`` and regenerate its flat projection.

        ``title``/``agent_path``/``provider`` default to values derived from the thread
        when not given (so the lossless CLI path needs to pass nothing). ``project_id``
        follows the Studio "leave assignment alone" contract: ``None`` keeps any existing
        assignment rather than clearing it. ``created_at`` is preserved across re-saves.
        """
        now = utc_now_iso()
        existing = self._conn.execute(
            "SELECT created_at, project_id, agent_path, provider, origin "
            "FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        created = existing["created_at"] if existing else now
        resolved_title = (title or "").strip() or _derive_title(thread)
        resolved_agent = (
            agent_path
            if agent_path is not None
            else (existing["agent_path"] if existing else thread.agent_id)
        )
        resolved_provider = (
            provider if provider is not None else (existing["provider"] if existing else None)
        )
        effective_project = project_id or (existing["project_id"] if existing else None)
        resolved_origin = origin or (existing["origin"] if existing else ORIGIN_CLI)
        self._conn.execute(
            """
            INSERT INTO conversations
                (conversation_id, thread, origin, title, agent_path, provider,
                 project_id, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                thread=excluded.thread,
                title=excluded.title,
                agent_path=excluded.agent_path,
                provider=excluded.provider,
                project_id=excluded.project_id,
                updated_at=excluded.updated_at
            """,
            (
                conversation_id,
                thread.model_dump_json(),
                resolved_origin,
                resolved_title,
                resolved_agent,
                resolved_provider,
                effective_project,
                created,
                now,
            ),
        )
        self._reproject(conversation_id, thread, now)
        self._conn.commit()
        flat = flat_from_thread(thread)
        return ConversationSummary(
            conversation_id=conversation_id,
            origin=resolved_origin,
            title=resolved_title,
            agent_path=resolved_agent,
            provider=resolved_provider,
            project_id=effective_project,
            created_at=created,
            updated_at=now,
            message_count=len(flat),
        )

    def save_flat(
        self,
        *,
        conversation_id: str | None,
        title: str | None,
        agent_path: str | None,
        provider: str | None,
        flat_messages: Sequence[tuple[str, str]],
        project_id: str | None = None,
        origin: str = ORIGIN_STUDIO,
    ) -> ConversationSummary:
        """Save a FLAT Studio transcript by first lifting it to an authoritative thread.

        This is the lossy ingress (reviewer must-fix): the flat rows are mapped to a
        :class:`ChatThread` via :func:`thread_from_flat`, which then becomes the source of
        truth and is re-projected. A title is derived from the first user row when not given.
        """
        cid = conversation_id or new_uuid()
        thread = thread_from_flat(
            conversation_id=cid, agent_path=agent_path, flat_messages=flat_messages
        )
        return self.save_thread(
            cid,
            thread,
            origin=origin,
            title=title,
            agent_path=agent_path,
            provider=provider,
            project_id=project_id,
        )

    def _reproject(self, conversation_id: str, thread: ChatThread, now: str) -> None:
        """Rebuild the flat projection for one conversation from its authoritative thread."""
        self._conn.execute(
            "DELETE FROM conversation_messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        for seq, (role, text) in enumerate(flat_from_thread(thread)):
            self._conn.execute(
                """
                INSERT INTO conversation_messages
                    (id, conversation_id, role, text, seq, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (new_uuid(), conversation_id, role, text, seq, now),
            )

    # -- reads ---------------------------------------------------------------

    def load_thread(self, conversation_id: str) -> ChatThread | None:
        """Return the authoritative :class:`ChatThread` for ``conversation_id`` (or None)."""
        row = self._conn.execute(
            "SELECT thread FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return ChatThread.model_validate_json(row["thread"])
        except Exception:  # noqa: BLE001 - a corrupt row must not break the caller
            return None

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        """Return a summary (no bodies) for ``conversation_id`` (or None)."""
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return self._summary(row, self._count(conversation_id))

    def flat_messages(self, conversation_id: str) -> list[FlatMessage]:
        """The regenerated flat transcript both UIs read, ordered."""
        rows = self._conn.execute(
            "SELECT role, text FROM conversation_messages "
            "WHERE conversation_id = ? ORDER BY seq",
            (conversation_id,),
        ).fetchall()
        return [FlatMessage(role=r["role"], text=r["text"]) for r in rows]

    def list_summaries(
        self, *, limit: int | None = None, origin: str | None = None
    ) -> list[ConversationSummary]:
        """Recent conversations, newest first, with each conversation's message count.

        ``origin`` filters to one front door's conversations; omit it to see ALL (the
        union both UIs are meant to show). A row whose thread fails to deserialize is still
        listed with ``message_count`` taken from the flat projection (never raises).
        """
        sql = (
            "SELECT c.*, COUNT(m.id) AS n "
            "FROM conversations c "
            "LEFT JOIN conversation_messages m "
            "  ON m.conversation_id = c.conversation_id "
        )
        params: list[object] = []
        if origin is not None:
            sql += "WHERE c.origin = ? "
            params.append(origin)
        sql += "GROUP BY c.conversation_id ORDER BY c.updated_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._summary(r, r["n"]) for r in rows]

    def _count(self, conversation_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM conversation_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    # -- mutations -----------------------------------------------------------

    def rename(self, conversation_id: str, title: str) -> bool:
        cur = self._conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
            (title, utc_now_iso(), conversation_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, conversation_id: str) -> bool:
        self._conn.execute(
            "DELETE FROM conversation_messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        cur = self._conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def set_project(self, conversation_id: str, project_id: str | None) -> bool:
        """Assign/clear the project of a conversation (touch ``updated_at`` only on change)."""
        cur = self._conn.execute(
            "UPDATE conversations SET project_id = ? WHERE conversation_id = ?",
            (project_id, conversation_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def prune(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int:
        """Delete stale conversations, bounding otherwise-unbounded growth.

        Mirrors :meth:`himmy.runtime.session.SqliteSessionStore.prune`: at least one bound
        must be given; a conversation is removed if it fails EITHER bound. Deletes in chunks
        so the ``IN``-list never exceeds ``SQLITE_MAX_VARIABLE_NUMBER``. Returns the count
        deleted.
        """
        if older_than_days is None and keep_last is None:
            raise ValueError("prune needs at least one of older_than_days / keep_last")
        doomed: set[str] = set()
        if older_than_days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
            rows = self._conn.execute(
                "SELECT conversation_id FROM conversations WHERE updated_at < ?",
                (cutoff,),
            ).fetchall()
            doomed.update(r["conversation_id"] for r in rows)
        if keep_last is not None:
            keep = max(int(keep_last), 0)
            rows = self._conn.execute(
                "SELECT conversation_id FROM conversations ORDER BY updated_at DESC "
                "LIMIT -1 OFFSET ?",
                (keep,),
            ).fetchall()
            doomed.update(r["conversation_id"] for r in rows)
        if not doomed:
            return 0
        ordered = sorted(doomed)
        for start in range(0, len(ordered), _PRUNE_DELETE_CHUNK):
            batch = ordered[start : start + _PRUNE_DELETE_CHUNK]
            placeholders = ",".join("?" for _ in batch)
            self._conn.execute(
                f"DELETE FROM conversation_messages "  # noqa: S608
                f"WHERE conversation_id IN ({placeholders})",
                tuple(batch),
            )
            self._conn.execute(
                f"DELETE FROM conversations "  # noqa: S608
                f"WHERE conversation_id IN ({placeholders})",
                tuple(batch),
            )
        self._conn.commit()
        return len(doomed)

    # -- projects ------------------------------------------------------------
    # Projects move here unchanged from the old chats store so Studio's grouping keeps
    # working over the unified DB. They reference a conversation's project_id.

    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        kb_id: str | None = None,
        agent_path: str | None = None,
    ) -> dict[str, object]:
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
        return {
            "id": pid,
            "name": name,
            "description": description,
            "kb_id": kb_id,
            "agent_path": agent_path,
            "created_at": now,
            "updated_at": now,
            "chat_count": 0,
        }

    def get_project_row(self, project_id: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return row

    def project_chat_count(self, project_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def list_project_rows(self) -> list[tuple[sqlite3.Row, int]]:
        rows = self._conn.execute(
            """
            SELECT p.*, COUNT(c.conversation_id) AS n
            FROM projects p
            LEFT JOIN conversations c ON c.project_id = p.id
            GROUP BY p.id
            ORDER BY p.updated_at DESC
            """
        ).fetchall()
        return [(r, r["n"]) for r in rows]

    def update_project(
        self, project_id: str, assignments: dict[str, str | None]
    ) -> bool:
        """Apply pre-validated column=value assignments to a project (touch updated_at)."""
        if not assignments:
            return self.get_project_row(project_id) is not None
        cols = ", ".join(f"{k} = ?" for k in assignments)
        cur = self._conn.execute(
            f"UPDATE projects SET {cols}, updated_at = ? WHERE id = ?",  # noqa: S608
            (*assignments.values(), utc_now_iso(), project_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_project(self, project_id: str) -> bool:
        self._conn.execute(
            "UPDATE conversations SET project_id = NULL WHERE project_id = ?",
            (project_id,),
        )
        cur = self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def project_conversation_summaries(
        self, project_id: str
    ) -> list[ConversationSummary]:
        rows = self._conn.execute(
            """
            SELECT c.*, COUNT(m.id) AS n
            FROM conversations c
            LEFT JOIN conversation_messages m
                ON m.conversation_id = c.conversation_id
            WHERE c.project_id = ?
            GROUP BY c.conversation_id
            ORDER BY c.updated_at DESC
            """,
            (project_id,),
        ).fetchall()
        return [self._summary(r, r["n"]) for r in rows]

    def assign_conversation(self, project_id: str, conversation_id: str) -> bool:
        if self.get_project_row(project_id) is None:
            return False
        cur = self._conn.execute(
            "UPDATE conversations SET project_id = ? WHERE conversation_id = ?",
            (project_id, conversation_id),
        )
        if cur.rowcount > 0:
            self._conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), project_id),
            )
        self._conn.commit()
        return cur.rowcount > 0

    def unassign_conversation(self, conversation_id: str) -> bool:
        cur = self._conn.execute(
            "UPDATE conversations SET project_id = NULL "
            "WHERE conversation_id = ? AND project_id IS NOT NULL",
            (conversation_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # -- migration -----------------------------------------------------------

    def import_legacy(
        self, *, sessions_db: str | None = None, chats_db: str | None = None
    ) -> dict[str, int]:
        """Best-effort one-time fold of legacy ``sessions.db`` + ``chats.db`` into this store.

        * ``sessions.db`` (CLI) rows are LOSSLESS — the stored ChatThread JSON is loaded
          verbatim and saved as the authoritative thread under the same session id.
        * ``chats.db`` (Studio) rows are FLAT — each session's ``user``/``agent`` messages
          are lifted to a ChatThread via :func:`thread_from_flat`, and its ``projects`` +
          assignments are carried over.

        A conversation_id already present in this store is NOT overwritten (idempotent —
        re-running the import never clobbers live data). Returns per-source counts. Any
        unreadable legacy DB is skipped silently (best-effort, never fatal).
        """
        counts = {"sessions": 0, "chats": 0, "projects": 0}
        if sessions_db and Path(sessions_db).exists():
            counts["sessions"] = self._import_sessions(sessions_db)
        if chats_db and Path(chats_db).exists():
            chats, projects = self._import_chats(chats_db)
            counts["chats"] = chats
            counts["projects"] = projects
        return counts

    def _import_sessions(self, path: str) -> int:
        imported = 0
        try:
            src = sqlite3.connect(path)
            src.row_factory = sqlite3.Row
            rows = src.execute(
                "SELECT session_id, thread, updated_at FROM sessions"
            ).fetchall()
        except sqlite3.Error:
            return 0
        try:
            for r in rows:
                sid = r["session_id"]
                if self._exists(sid):
                    continue
                try:
                    thread = ChatThread.model_validate_json(r["thread"])
                except Exception:  # noqa: BLE001 - skip a corrupt legacy row
                    continue
                self.save_thread(sid, thread, origin=ORIGIN_CLI)
                imported += 1
        finally:
            src.close()
        return imported

    def _import_chats(self, path: str) -> tuple[int, int]:
        chats = 0
        projects = 0
        try:
            src = sqlite3.connect(path)
            src.row_factory = sqlite3.Row
            cols = {
                r[1]
                for r in src.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }
            if "id" not in cols:  # not a Studio chats DB shape
                src.close()
                return (0, 0)
            # ``project_id`` only exists in post-projects databases; older ones lack it.
            has_project = "project_id" in cols
            select_cols = "id, title, agent_path, provider" + (
                ", project_id" if has_project else ""
            )
            sess_rows = src.execute(
                f"SELECT {select_cols} FROM chat_sessions"  # noqa: S608
            ).fetchall()
        except sqlite3.Error:
            return (0, 0)
        try:
            # projects first so an assignment has a target
            try:
                proj_rows = src.execute("SELECT * FROM projects").fetchall()
            except sqlite3.Error:
                proj_rows = []
            for p in proj_rows:
                if self.get_project_row(p["id"]) is not None:
                    continue
                self._conn.execute(
                    """
                    INSERT INTO projects
                        (id, name, description, kb_id, agent_path, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        p["id"],
                        p["name"],
                        p["description"],
                        p["kb_id"],
                        p["agent_path"],
                        p["created_at"],
                        p["updated_at"],
                    ),
                )
                projects += 1
            for r in sess_rows:
                cid = r["id"]
                if self._exists(cid):
                    continue
                msgs = src.execute(
                    "SELECT role, text FROM chat_messages "
                    "WHERE session_id = ? ORDER BY seq",
                    (cid,),
                ).fetchall()
                self.save_flat(
                    conversation_id=cid,
                    title=r["title"],
                    agent_path=r["agent_path"],
                    provider=r["provider"],
                    flat_messages=[(m["role"], m["text"]) for m in msgs],
                    project_id=r["project_id"] if has_project else None,
                    origin=ORIGIN_STUDIO,
                )
                chats += 1
            self._conn.commit()
        finally:
            src.close()
        return (chats, projects)

    def _exists(self, conversation_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            is not None
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _summary(row: sqlite3.Row, count: int) -> ConversationSummary:
        return ConversationSummary(
            conversation_id=row["conversation_id"],
            origin=row["origin"],
            title=row["title"],
            agent_path=row["agent_path"],
            provider=row["provider"],
            project_id=row["project_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=count,
        )

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()


# --------------------------------------------------------------------------- singleton
# One process-wide ConversationStore keyed on the resolved path, mirroring the cwd-keyed
# singletons the other Studio stores use (get_chats_store / get_tasks_store). Both the CLI
# adapter and the Studio adapter resolve THROUGH here so they share one open connection
# (and one durable .himmy/conversations.db). On first open in a project, any pre-existing
# legacy sessions.db / chats.db are folded in once (best-effort, idempotent).

_STORE: ConversationStore | None = None
_PATH: str | None = None
_IMPORTED: set[str] = set()


def get_conversation_store() -> ConversationStore:
    """Return the process-wide :class:`ConversationStore` for the canonical path.

    Resolves the path via :func:`himmy.config.project.conversations_db_path` (cwd/project
    keyed) and reuses one open store across calls; a path change (e.g. a test pointing
    ``HIMMY_CONVERSATIONS_PATH`` at a tmp dir) transparently reopens. On the FIRST open of a
    given path, legacy ``sessions.db``/``chats.db`` siblings are imported once.
    """
    global _STORE, _PATH
    from himmy.config.project import conversations_db_path

    path = conversations_db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        _STORE = ConversationStore(path)
        _PATH = path
        _maybe_import_legacy(_STORE, path)
    return _STORE


def _maybe_import_legacy(store: ConversationStore, path: str) -> None:
    """One-time best-effort import of sibling legacy DBs into ``store`` (never fatal)."""
    if path in _IMPORTED:
        return
    _IMPORTED.add(path)
    try:
        directory = Path(path).resolve().parent
        sessions = directory / "sessions.db"
        chats = directory / "chats.db"
        store.import_legacy(
            sessions_db=str(sessions) if sessions.exists() else None,
            chats_db=str(chats) if chats.exists() else None,
        )
    except Exception:  # noqa: BLE001 - migration is best-effort, never blocks startup
        pass


def reset_conversation_store() -> None:
    """Close + drop the singleton (test hook; the next access reopens at the current path)."""
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None
    _IMPORTED.clear()


__all__ = [
    "FLAT_ROLE_AGENT",
    "FLAT_ROLE_USER",
    "ORIGIN_CLI",
    "ORIGIN_STUDIO",
    "ConversationStore",
    "ConversationSummary",
    "FlatMessage",
    "flat_from_thread",
    "flat_role",
    "get_conversation_store",
    "reset_conversation_store",
    "thread_from_flat",
]
