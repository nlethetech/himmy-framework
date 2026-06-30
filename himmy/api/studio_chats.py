"""Studio Chats — now a thin adapter over the ONE unified conversation store (T2.3).

The Chat screen keeps its live transcript in React state; this store persists a session so
it can be reopened later. Historically that lived in a standalone ``.himmy/chats.db`` (a
FLAT ``user``/``agent`` transcript) disconnected from the CLI's rich ``.himmy/sessions.db``.
As of T2.3 both collapse into ONE durable ``.himmy/conversations.db``
(:mod:`himmy.services.storage.conversations`): the authoritative store keeps a rich
:class:`~himmy.agents.base_agent.thread.ChatThread` JSON and regenerates the flat projection
this screen reads, so a conversation started in ``himmy chat`` shows up in the Studio Chats
list and vice versa.

:class:`ChatsStore` is kept as a BACK-COMPATIBLE FACADE: the same ``list``/``get``/``save``/
``rename``/``delete`` chat surface and the same Projects surface, returning the same
:class:`ChatSession`/:class:`ChatSessionDetail`/:class:`Project` models, with a live ``_conn``
exposing the unified database (a ``chat_sessions`` compatibility view keeps the legacy
direct-SQL inspection working). Studio's flat save is mapped to the authoritative thread via
:func:`~himmy.services.storage.conversations.thread_from_flat`.

Security: the co-mounted ``POST /api/studio/chats`` write path is single-user-gated — every
``/api/studio`` route carries the ``studio:use`` permission guard (see
:func:`himmy.api.routers.studio._studio_permission`), which is a no-op only in the zero-config
loopback default and otherwise requires the ``studio:use`` role. There is no per-tenant
workspace on a Studio conversation by design (Studio is single-user-local); the guard is the
isolation boundary.
"""

from __future__ import annotations

import builtins
import os
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso
from himmy.services.storage.conversations import (
    ORIGIN_STUDIO,
    ConversationStore,
    get_conversation_store,
    reset_conversation_store,
)


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


def _derive_title(messages: Sequence[ChatMessage]) -> str:
    """Use the first user message as the session title (trimmed)."""
    for m in messages:
        if m.role == "user" and m.text.strip():
            t = m.text.strip().splitlines()[0]
            return t[:60] + ("…" if len(t) > 60 else "")
    return "New chat"


class ChatsStore:
    """Facade over :class:`~himmy.services.storage.conversations.ConversationStore`.

    Translates the Studio chat/project surface to the unified store. A Studio chat id is
    used directly as the unified ``conversation_id`` (so a chat saved here is the same row
    the CLI ``/resume`` picker lists). When constructed with an explicit ``path`` (e.g. by a
    test pointing ``HIMMY_CHATS_PATH`` at a tmp file) it owns its own store; the singleton
    accessor reuses the process-wide one.
    """

    def __init__(self, path: str = ":memory:", *, store: Any = None) -> None:
        # ``store`` is the unified conversation store: a SQLite ``ConversationStore`` OR, under
        # a Postgres DSN (K4), the ``PostgresConversationStore`` mirror — both expose the same
        # save_flat/list_summaries/... surface this facade calls. Typed ``Any`` because the two
        # share a duck-typed surface, not a nominal base.
        self._store = store if store is not None else ConversationStore(path)

    @property
    def _conn(self) -> Any:
        """The unified store's live SQLite connection (back-compat for direct-SQL tests).

        Only the SQLite ``ConversationStore`` exposes ``_conn``; the Postgres mirror has no
        single connection, so this raises ``AttributeError`` there (the property is a
        test-only direct-SQL shim never exercised on the Postgres path)."""
        return self._store._conn

    # ---- chats ------------------------------------------------------------

    def list(
        self, *, workspace_id: str | frozenset[str] | None = None
    ) -> builtins.list[ChatSession]:
        return [
            self._to_session(s)
            for s in self._store.list_summaries(workspace_id=workspace_id)
        ]

    def get(
        self,
        session_id: str,
        *,
        workspace_id: str | frozenset[str] | None = None,
    ) -> ChatSessionDetail | None:
        summary = self._store.get_summary(session_id, workspace_id=workspace_id)
        if summary is None:
            return None
        msgs = self._store.flat_messages(session_id)
        return ChatSessionDetail(
            id=summary.conversation_id,
            title=summary.title,
            agent_path=summary.agent_path,
            provider=summary.provider,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            message_count=summary.message_count,
            project_id=summary.project_id,
            messages=[ChatMessage(role=m.role, text=m.text) for m in msgs],
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
        workspace_id: str | None = None,
    ) -> ChatSession:
        """Create or replace a session and its messages (upsert by id).

        ``project_id=None`` means "leave the assignment alone" (the unified store honours
        the same contract). The flat ``messages`` are lifted to the authoritative ChatThread
        by the store. ``workspace_id`` (tenant axis) stamps a tenant-bound Studio write;
        ``None`` (CLI / offline) writes an unscoped row — byte-unchanged.
        """
        resolved_title = (title or "").strip() or _derive_title(messages)
        summary = self._store.save_flat(
            conversation_id=session_id,
            title=resolved_title,
            agent_path=agent_path,
            provider=provider,
            flat_messages=[(m.role, m.text) for m in messages],
            project_id=project_id,
            origin=ORIGIN_STUDIO,
            workspace_id=workspace_id,
        )
        return self._to_session(summary)

    def rename(
        self,
        session_id: str,
        title: str,
        *,
        workspace_id: str | frozenset[str] | None = None,
    ) -> bool:
        return self._store.rename(session_id, title, workspace_id=workspace_id)

    def delete(
        self, session_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return self._store.delete(session_id, workspace_id=workspace_id)

    # ---- projects ---------------------------------------------------------

    def list_projects(
        self, *, workspace_id: str | frozenset[str] | None = None
    ) -> builtins.list[Project]:
        return [
            self._to_project(r, n)
            for r, n in self._store.list_project_rows(workspace_id=workspace_id)
        ]

    def get_project(
        self, project_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> Project | None:
        row = self._store.get_project_row(project_id, workspace_id=workspace_id)
        if row is None:
            return None
        return self._to_project(row, self._store.project_chat_count(project_id))

    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        kb_id: str | None = None,
        agent_path: str | None = None,
        workspace_id: str | None = None,
    ) -> Project:
        data = self._store.create_project(
            name=name,
            description=description,
            kb_id=kb_id,
            agent_path=agent_path,
            workspace_id=workspace_id,
        )
        return Project(**data)  # type: ignore[arg-type]

    def update_project(
        self,
        project_id: str,
        changes: Mapping[str, str | None],
        *,
        workspace_id: str | frozenset[str] | None = None,
    ) -> Project | None:
        """Apply a partial update; unknown keys are ignored, ``None`` clears.

        Only the columns in ``_PROJECT_FIELDS`` are writable (the assignment dict is built
        from that allowlist, never from caller-supplied names). ``workspace_id`` scopes the
        update to the caller's own project (a cross-tenant / legacy project folds to None).
        """
        sets = {k: changes[k] for k in _PROJECT_FIELDS if k in changes}
        if "name" in sets and not (sets["name"] or "").strip():
            del sets["name"]  # a project always keeps a non-empty name
        if self._store.get_project_row(project_id, workspace_id=workspace_id) is None:
            return None
        self._store.update_project(project_id, dict(sets), workspace_id=workspace_id)
        return self.get_project(project_id, workspace_id=workspace_id)

    def delete_project(
        self, project_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return self._store.delete_project(project_id, workspace_id=workspace_id)

    def project_chats(
        self, project_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> builtins.list[ChatSession]:
        return [
            self._to_session(s)
            for s in self._store.project_conversation_summaries(
                project_id, workspace_id=workspace_id
            )
        ]

    def assign_chat(
        self,
        project_id: str,
        chat_id: str,
        *,
        workspace_id: str | frozenset[str] | None = None,
    ) -> bool:
        return self._store.assign_conversation(
            project_id, chat_id, workspace_id=workspace_id
        )

    def unassign_chat(
        self, chat_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return self._store.unassign_conversation(chat_id, workspace_id=workspace_id)

    # ---- mapping ----------------------------------------------------------

    @staticmethod
    def _to_session(summary) -> ChatSession:  # type: ignore[no-untyped-def]
        return ChatSession(
            id=summary.conversation_id,
            title=summary.title,
            agent_path=summary.agent_path,
            provider=summary.provider,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            message_count=summary.message_count,
            project_id=summary.project_id,
        )

    @staticmethod
    def _to_project(row, n: int) -> Project:  # type: ignore[no-untyped-def]
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
        self._store.close()


def chats_db_path() -> str:
    """The path the Studio chats facade opens.

    Honours the legacy ``HIMMY_CHATS_PATH`` override (kept for test isolation and any
    operator who pinned it) and otherwise resolves to the canonical unified
    ``.himmy/conversations.db`` — so by default Studio and the CLI share one database.
    """
    env = os.environ.get("HIMMY_CHATS_PATH")
    if env:
        return env
    from himmy.config.project import conversations_db_path

    return conversations_db_path()


def get_chats_store() -> ChatsStore:
    """Return the process-wide Studio chats facade over the unified store.

    When ``HIMMY_CHATS_PATH`` is set the facade opens that explicit path (own connection,
    used by tests for isolation); otherwise it wraps the shared
    :func:`~himmy.services.storage.conversations.get_conversation_store` singleton so Studio
    and the CLI literally share one open store.
    """
    env = os.environ.get("HIMMY_CHATS_PATH")
    if env:
        global _OVERRIDE_STORE, _OVERRIDE_PATH
        if _OVERRIDE_STORE is None or _OVERRIDE_PATH != env:
            if _OVERRIDE_STORE is not None:
                _OVERRIDE_STORE.close()
            _OVERRIDE_STORE = ChatsStore(env)
            _OVERRIDE_PATH = env
        return _OVERRIDE_STORE
    return ChatsStore(store=get_conversation_store())


def reset_chats_store() -> None:
    """Drop both the override facade and the shared singleton (test hook)."""
    global _OVERRIDE_STORE, _OVERRIDE_PATH
    if _OVERRIDE_STORE is not None:
        _OVERRIDE_STORE.close()
    _OVERRIDE_STORE = None
    _OVERRIDE_PATH = None
    reset_conversation_store()


_OVERRIDE_STORE: ChatsStore | None = None
_OVERRIDE_PATH: str | None = None


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
