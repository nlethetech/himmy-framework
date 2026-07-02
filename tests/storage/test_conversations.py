"""Tests for the unified conversation store (T2.3).

The store collapses the CLI's rich ``sessions.db`` (lossless ChatThread JSON) and Studio's
flat ``chats.db`` transcript into ONE ``.himmy/conversations.db``. These tests pin:

* the authoritative ChatThread JSON is the source of truth and round-trips losslessly;
* the regenerated flat projection both UIs read agrees with the thread;
* the explicit flat<->thread mappings (the lossy Studio ingress and the display projection);
* the CROSS-SURFACE acceptance — a CLI conversation is visible to the Studio surface and a
  Studio conversation is visible to the CLI surface (both read one store);
* projects, prune, and the one-time legacy import of both legacy DBs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
from himmy.api import studio_chats as sc
from himmy.runtime.session import SqliteSessionStore
from himmy.services.storage.conversations import (
    FLAT_ROLE_AGENT,
    FLAT_ROLE_USER,
    ORIGIN_CLI,
    ORIGIN_STUDIO,
    ConversationStore,
    flat_from_thread,
    flat_role,
    get_conversation_store,
    reset_conversation_store,
    thread_from_flat,
)


def _rich_thread() -> ChatThread:
    """A thread exercising every role, including an empty (tool-only) assistant turn."""
    t = ChatThread(agent_id="a.yaml")
    t.append_message(Message(role=MessageRole.SYSTEM, content="be terse"))
    t.append_message(Message(role=MessageRole.USER, content="weather?"))
    t.append_message(Message(role=MessageRole.ASSISTANT, content=""))  # tool-call turn
    t.append_message(Message(role=MessageRole.TOOL, content="sunny"))
    t.append_message(Message(role=MessageRole.ASSISTANT, content="It is sunny."))
    return t


# ----------------------------------------------------------------- mappings


def test_flat_role_collapses_nonuser_to_agent() -> None:
    assert flat_role(MessageRole.USER) == FLAT_ROLE_USER
    assert flat_role(MessageRole.ASSISTANT) == FLAT_ROLE_AGENT
    assert flat_role(MessageRole.SYSTEM) == FLAT_ROLE_AGENT
    assert flat_role(MessageRole.TOOL) == FLAT_ROLE_AGENT
    assert flat_role("user") == FLAT_ROLE_USER


def test_flat_from_thread_drops_empty_turns_and_collapses_roles() -> None:
    flat = flat_from_thread(_rich_thread())
    # system+user+(empty assistant dropped)+tool+assistant -> agent,user,agent,agent
    assert flat == [
        (FLAT_ROLE_AGENT, "be terse"),
        (FLAT_ROLE_USER, "weather?"),
        (FLAT_ROLE_AGENT, "sunny"),
        (FLAT_ROLE_AGENT, "It is sunny."),
    ]


def test_thread_from_flat_is_explicit_and_stable() -> None:
    thread = thread_from_flat(
        conversation_id="c1",
        agent_path="a.yaml",
        flat_messages=[("user", "hi"), ("agent", "hello"), ("user", "thanks")],
    )
    assert thread.thread_id == "c1"
    assert thread.agent_id == "a.yaml"
    assert [m.role for m in thread.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert [m.content for m in thread.messages] == ["hi", "hello", "thanks"]


# ----------------------------------------------------------------- authoritative


def test_save_thread_is_lossless_and_projects_flat(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "c.db"))
    summary = store.save_thread("sess1", _rich_thread(), origin=ORIGIN_CLI)
    assert summary.origin == ORIGIN_CLI
    assert summary.title == "weather?"  # derived from first user turn
    assert summary.message_count == 4  # flat projection drops the empty assistant turn

    # Authoritative thread round-trips with ALL roles intact (lossless).
    loaded = store.load_thread("sess1")
    assert loaded is not None
    assert [m.role for m in loaded.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    # The flat projection both UIs read agrees with the thread.
    flat = store.flat_messages("sess1")
    assert [(m.role, m.text) for m in flat] == flat_from_thread(loaded)
    store.close()


def test_reproject_replaces_not_appends(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "c.db"))
    t1 = ChatThread(thread_id="x")
    t1.append_message(Message(role=MessageRole.USER, content="first"))
    store.save_thread("x", t1)
    t2 = ChatThread(thread_id="x")
    t2.append_message(Message(role=MessageRole.USER, content="second"))
    t2.append_message(Message(role=MessageRole.ASSISTANT, content="ok"))
    store.save_thread("x", t2)
    flat = store.flat_messages("x")
    assert [m.text for m in flat] == ["second", "ok"]  # not the stale "first"
    store.close()


# ----------------------------------------------------------------- batch persist (P2.1)


def _row_tuples(store: ConversationStore, cid: str) -> list[tuple[str, str, int, str]]:
    """The identity-bearing columns of the flat projection (everything but the random id)."""
    rows = store._conn.execute(
        "SELECT role, text, seq, created_at FROM conversation_messages "
        "WHERE conversation_id = ? ORDER BY seq",
        (cid,),
    ).fetchall()
    return [(r["role"], r["text"], r["seq"], r["created_at"]) for r in rows]


def test_batched_reproject_writes_identical_rows_to_rowwise(tmp_path: Path) -> None:
    """The batched ``executemany`` projection is byte-identical to a row-by-row re-INSERT.

    Persists the same thread through the (batched) store and through a hand-rolled row-by-row
    INSERT under the SAME ``now`` timestamp, then asserts role/text/seq/created_at match column
    for column, row for row (only the random per-row ``id`` differs — it always did).
    """
    store = ConversationStore(str(tmp_path / "c.db"))
    thread = _rich_thread()
    store.save_thread("batched", thread)
    batched = _row_tuples(store, "batched")

    # Reference: the exact row-by-row loop the batching replaced, same projection + timestamp.
    now = store._conn.execute(
        "SELECT created_at FROM conversation_messages WHERE conversation_id = ? LIMIT 1",
        ("batched",),
    ).fetchone()["created_at"]
    for seq, (role, text) in enumerate(flat_from_thread(thread)):
        store._conn.execute(
            "INSERT INTO conversation_messages "
            "(id, conversation_id, role, text, seq, created_at) VALUES (?,?,?,?,?,?)",
            (f"ref-{seq}", "rowwise", role, text, seq, now),
        )
    store._conn.commit()
    reference = _row_tuples(store, "rowwise")

    assert batched == reference
    assert len(batched) == 4  # the tool-only assistant turn is dropped, as before
    store.close()


def test_batched_reproject_handles_edited_and_regenerated_thread(tmp_path: Path) -> None:
    """An edited/regenerated thread re-projects correctly (DELETE-all + batched re-INSERT).

    Covers the shorten (fewer rows), grow (more rows) and empty-out (zero rows) cases — the
    ``executemany`` path must never leave a stale row from the previous, longer projection.
    """
    store = ConversationStore(str(tmp_path / "c.db"))

    t1 = ChatThread(thread_id="e")
    t1.append_message(Message(role=MessageRole.USER, content="a"))
    t1.append_message(Message(role=MessageRole.ASSISTANT, content="b"))
    t1.append_message(Message(role=MessageRole.USER, content="c"))
    store.save_thread("e", t1)
    assert [m.text for m in store.flat_messages("e")] == ["a", "b", "c"]

    # Regenerate SHORTER: the tail rows must be gone, not merely overwritten in place.
    t2 = ChatThread(thread_id="e")
    t2.append_message(Message(role=MessageRole.USER, content="only"))
    store.save_thread("e", t2)
    assert [m.text for m in store.flat_messages("e")] == ["only"]

    # Regenerate LONGER.
    t3 = ChatThread(thread_id="e")
    for txt in ("x", "y", "z", "w"):
        t3.append_message(Message(role=MessageRole.USER, content=txt))
    store.save_thread("e", t3)
    assert [m.text for m in store.flat_messages("e")] == ["x", "y", "z", "w"]

    # Regenerate to EMPTY (only tool-only / blank turns): zero projected rows, no leftovers.
    t4 = ChatThread(thread_id="e")
    t4.append_message(Message(role=MessageRole.ASSISTANT, content=""))
    store.save_thread("e", t4)
    assert store.flat_messages("e") == []
    assert store._count("e") == 0
    store.close()


def test_batched_persist_is_thread_safe_across_conversations(tmp_path: Path) -> None:
    """Concurrent batched saves of distinct conversations preserve each projection exactly.

    The store keeps ONE hardened sqlite connection and does not itself lock — exactly as
    before the batching (the row-by-row path raised the same ``InterfaceError`` under raw
    concurrent use of a single connection). Callers serialise writes; this drives many threads
    saving DISTINCT conversation ids through a shared lock (the real contract) and asserts every
    batched projection is intact and in order, with no cross-conversation row bleed.
    """
    import threading

    store = ConversationStore(str(tmp_path / "c.db"))
    lock = threading.Lock()

    def _save(n: int) -> None:
        t = ChatThread(thread_id=f"conv{n}")
        for i in range(5):
            t.append_message(
                Message(role=MessageRole.USER, content=f"conv{n}-msg{i}")
            )
        with lock:
            store.save_thread(f"conv{n}", t)

    threads = [threading.Thread(target=_save, args=(n,)) for n in range(16)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    for n in range(16):
        got = [m.text for m in store.flat_messages(f"conv{n}")]
        assert got == [f"conv{n}-msg{i}" for i in range(5)]
    store.close()


def test_pg_reproject_batches_inserts_with_executemany() -> None:
    """The Postgres mirror collapses the N per-row INSERTs into ONE ``executemany``.

    Drives the async reproject against a recording fake conn and asserts (1) the flat-message
    rows are written via a single ``executemany`` (not N ``execute`` calls) and (2) the batched
    argument sequence carries the same tenant / conversation / role / text / ``seq`` / timestamp
    the row-by-row path emitted (only the random per-row id differs).
    """
    import asyncio

    from himmy.services.storage.postgres_aux import _AsyncConversationStore

    class _RecordingConn:
        def __init__(self) -> None:
            self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
            self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

        async def execute(self, sql: str, *args: object) -> str:
            self.execute_calls.append((sql, args))
            return "DELETE 1"

        async def executemany(self, sql: str, args_seq: object) -> str:
            self.executemany_calls.append((sql, list(args_seq)))  # type: ignore[arg-type]
            return "INSERT 0 3"

    conn = _RecordingConn()
    store = _AsyncConversationStore.__new__(_AsyncConversationStore)
    store._tenant = "local"  # type: ignore[attr-defined]

    thread = _rich_thread()
    asyncio.run(store._reproject(conn, "cX", thread, "2026-01-01T00:00:00+00:00"))

    # One DELETE-all, then exactly one batched INSERT (never one execute per row).
    assert len(conn.executemany_calls) == 1
    assert not any("INSERT INTO aux_conversation_messages" in s for s, _ in conn.execute_calls)
    sql, batch = conn.executemany_calls[0]
    assert "INSERT INTO aux_conversation_messages" in sql
    # Identity of the batched rows: (tenant, id, conversation, role, text, seq, created_at).
    projected = [
        (row[0], row[2], row[3], row[4], row[5], row[6]) for row in batch
    ]
    assert projected == [
        ("local", "cX", "agent", "be terse", 0, "2026-01-01T00:00:00+00:00"),
        ("local", "cX", "user", "weather?", 1, "2026-01-01T00:00:00+00:00"),
        ("local", "cX", "agent", "sunny", 2, "2026-01-01T00:00:00+00:00"),
        ("local", "cX", "agent", "It is sunny.", 3, "2026-01-01T00:00:00+00:00"),
    ]


def test_save_thread_preserves_created_at_and_project(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "c.db"))
    proj = store.create_project(name="P")
    first = store.save_flat(
        conversation_id="s",
        title=None,
        agent_path=None,
        provider=None,
        flat_messages=[("user", "hi")],
        project_id=str(proj["id"]),
    )
    # A resave with project_id=None must NOT pull the chat out of its project.
    again = store.save_flat(
        conversation_id="s",
        title=None,
        agent_path=None,
        provider=None,
        flat_messages=[("user", "hi"), ("agent", "yo")],
    )
    assert again.created_at == first.created_at
    assert again.project_id == proj["id"]
    store.close()


# ----------------------------------------------------------------- cross-surface


def test_cli_conversation_visible_to_studio(tmp_path: Path) -> None:
    """A thread saved via the CLI session store appears in the Studio chats facade."""
    path = str(tmp_path / "conversations.db")
    store = ConversationStore(path)

    cli = SqliteSessionStore(path)
    t = ChatThread(agent_id="a.yaml")
    t.append_message(Message(role=MessageRole.USER, content="from the CLI"))
    t.append_message(Message(role=MessageRole.ASSISTANT, content="reply"))
    cli.save("work", t)

    studio = sc.ChatsStore(store=store)
    listed = studio.list()
    assert any(s.id == "work" for s in listed)
    detail = studio.get("work")
    assert detail is not None
    assert detail.messages[0].text == "from the CLI"
    assert detail.messages[0].role == "user"
    assert detail.messages[1].role == "agent"  # assistant collapses to agent
    cli.close()


def test_studio_conversation_visible_to_cli(tmp_path: Path) -> None:
    """A chat saved via the Studio facade is resumable from the CLI session store."""
    path = str(tmp_path / "conversations.db")
    store = ConversationStore(path)

    studio = sc.ChatsStore(store=store)
    saved = studio.save(
        session_id=None,
        title=None,
        agent_path="a.yaml",
        provider="stub",
        messages=[
            sc.ChatMessage(role="user", text="from studio"),
            sc.ChatMessage(role="agent", text="hi back"),
        ],
    )

    cli = SqliteSessionStore(path)
    infos = cli.list_sessions()
    assert any(i.session_id == saved.id for i in infos)
    loaded = cli.load(saved.id)
    assert loaded is not None
    # The lossy ingress reconstructed a faithful ChatThread (user + assistant).
    assert [m.role for m in loaded.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert loaded.messages[0].content == "from studio"
    cli.close()


def test_list_summaries_unions_origins(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "c.db"))
    store.save_thread(
        "cli1",
        ChatThread(messages=[Message(role=MessageRole.USER, content="a")]),
        origin=ORIGIN_CLI,
    )
    store.save_flat(
        conversation_id="studio1",
        title=None,
        agent_path=None,
        provider=None,
        flat_messages=[("user", "b")],
        origin=ORIGIN_STUDIO,
    )
    ids = {s.conversation_id for s in store.list_summaries()}
    assert ids == {"cli1", "studio1"}  # the union, regardless of origin
    cli_only = {s.conversation_id for s in store.list_summaries(origin=ORIGIN_CLI)}
    assert cli_only == {"cli1"}
    store.close()


# ----------------------------------------------------------------- projects


def test_projects_over_unified_store(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "c.db"))
    proj = store.create_project(name="Ledger", description="money")
    pid = str(proj["id"])
    store.save_thread(
        "c1", ChatThread(messages=[Message(role=MessageRole.USER, content="x")])
    )
    assert store.assign_conversation(pid, "c1") is True
    assert store.project_chat_count(pid) == 1
    chats = store.project_conversation_summaries(pid)
    assert [c.conversation_id for c in chats] == ["c1"]
    assert store.unassign_conversation("c1") is True
    assert store.project_chat_count(pid) == 0
    # delete keeps the chat, just ungrouped
    store.assign_conversation(pid, "c1")
    assert store.delete_project(pid) is True
    assert store.get_summary("c1") is not None
    assert store.get_summary("c1").project_id is None  # type: ignore[union-attr]
    store.close()


# ----------------------------------------------------------------- prune


def test_prune_keep_last_and_older_than(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "c.db"))
    for i in range(4):
        store.save_thread(
            f"s{i}",
            ChatThread(messages=[Message(role=MessageRole.USER, content="m")]),
        )
    assert store.prune(keep_last=2) == 2
    assert len(store.list_summaries()) == 2
    with pytest.raises(ValueError, match="at least one"):
        store.prune()
    store.close()


def test_prune_removes_flat_projection_rows(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "c.db"))
    store.save_thread(
        "doomed",
        ChatThread(messages=[Message(role=MessageRole.USER, content="bye")]),
    )
    store.prune(keep_last=0)
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM conversation_messages"
    ).fetchone()
    assert rows[0] == 0  # the projection was cleaned up too
    store.close()


# ----------------------------------------------------------------- legacy import


def test_import_legacy_sessions_is_lossless_and_idempotent(tmp_path: Path) -> None:
    legacy = tmp_path / "sessions.db"
    src = sqlite3.connect(str(legacy))
    src.executescript(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, thread TEXT NOT NULL, "
        "updated_at TEXT NOT NULL);"
    )
    thread = _rich_thread()
    src.execute(
        "INSERT INTO sessions VALUES (?,?,?)",
        ("old", thread.model_dump_json(), "2026-01-01T00:00:00+00:00"),
    )
    src.commit()
    src.close()

    store = ConversationStore(str(tmp_path / "c.db"))
    counts = store.import_legacy(sessions_db=str(legacy))
    assert counts["sessions"] == 1
    loaded = store.load_thread("old")
    assert loaded is not None
    # Lossless: every role survived the import.
    assert [m.role for m in loaded.messages] == [m.role for m in thread.messages]
    # Idempotent: re-importing does not duplicate or clobber.
    again = store.import_legacy(sessions_db=str(legacy))
    assert again["sessions"] == 0
    store.close()


def test_import_legacy_chats_flat_to_thread(tmp_path: Path) -> None:
    legacy = tmp_path / "chats.db"
    src = sqlite3.connect(str(legacy))
    src.executescript(
        """
        CREATE TABLE chat_sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL,
            agent_path TEXT, provider TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, project_id TEXT);
        CREATE TABLE chat_messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            role TEXT NOT NULL, text TEXT NOT NULL, seq INTEGER NOT NULL,
            created_at TEXT NOT NULL);
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '', kb_id TEXT, agent_path TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        """
    )
    src.execute(
        "INSERT INTO projects VALUES ('p1','Farm','',NULL,NULL,'t','t')"
    )
    src.execute(
        "INSERT INTO chat_sessions VALUES ('c1','Chat',NULL,NULL,'t','t','p1')"
    )
    src.execute("INSERT INTO chat_messages VALUES ('m1','c1','user','hi',0,'t')")
    src.execute("INSERT INTO chat_messages VALUES ('m2','c1','agent','yo',1,'t')")
    src.commit()
    src.close()

    store = ConversationStore(str(tmp_path / "c.db"))
    counts = store.import_legacy(chats_db=str(legacy))
    assert counts["chats"] == 1
    assert counts["projects"] == 1
    loaded = store.load_thread("c1")
    assert loaded is not None
    assert [m.content for m in loaded.messages] == ["hi", "yo"]
    assert store.get_summary("c1").project_id == "p1"  # type: ignore[union-attr]
    store.close()


# ----------------------------------------------------------------- singleton


def test_singleton_resolves_and_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIMMY_CONVERSATIONS_PATH", str(tmp_path / "conv.db"))
    reset_conversation_store()
    try:
        a = get_conversation_store()
        b = get_conversation_store()
        assert a is b  # one process-wide store per path
        a.save_thread(
            "s", ChatThread(messages=[Message(role=MessageRole.USER, content="hi")])
        )
        assert get_conversation_store().get_summary("s") is not None
    finally:
        reset_conversation_store()


def test_cli_chat_visible_in_studio_chats_list_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance: a conversation from ``himmy chat`` shows in the Studio Chats API.

    Drives the REAL CLI entrypoint (which now writes to the unified conversations.db) and
    then the REAL Studio ``GET /api/studio/chats`` route — both resolve the same database
    with zero config — proving the two surfaces are connected.
    """
    from fastapi.testclient import TestClient

    from himmy.api.app import create_app
    from himmy.cli.__main__ import main

    monkeypatch.chdir(tmp_path)
    # Both surfaces must resolve the SAME conversations.db. Default resolution is
    # cwd-relative .himmy/conversations.db, which both honour; pin it explicitly so the
    # Studio singleton (reset below) and the CLI agree even across the cwd change.
    conv = str(tmp_path / ".himmy" / "conversations.db")
    monkeypatch.setenv("HIMMY_CONVERSATIONS_PATH", conv)
    monkeypatch.delenv("HIMMY_CHATS_PATH", raising=False)
    reset_conversation_store()
    sc.reset_chats_store()
    try:
        code = main(
            [
                "chat",
                "--provider",
                "stub",
                "--name",
                "a",
                "--session",
                "deskwork",
                "--message",
                "plan my week",
            ]
        )
        assert code == 0

        client = TestClient(create_app())
        listed = client.get("/api/studio/chats").json()
        ids = [row["id"] for row in listed]
        assert "deskwork" in ids  # the CLI conversation is in the Studio list
        detail = client.get("/api/studio/chats/deskwork").json()
        assert any("plan my week" in m["text"] for m in detail["messages"])
    finally:
        sc.reset_chats_store()
        reset_conversation_store()


def test_singleton_imports_sibling_legacy_dbs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lay down a legacy sessions.db next to where conversations.db will live.
    (tmp_path).mkdir(exist_ok=True)
    legacy = tmp_path / "sessions.db"
    src = sqlite3.connect(str(legacy))
    src.executescript(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, thread TEXT NOT NULL, "
        "updated_at TEXT NOT NULL);"
    )
    t = ChatThread(messages=[Message(role=MessageRole.USER, content="legacy turn")])
    src.execute(
        "INSERT INTO sessions VALUES (?,?,?)",
        ("legacy", t.model_dump_json(), "2026-01-01T00:00:00+00:00"),
    )
    src.commit()
    src.close()

    monkeypatch.setenv("HIMMY_CONVERSATIONS_PATH", str(tmp_path / "conversations.db"))
    reset_conversation_store()
    try:
        store = get_conversation_store()
        assert store.get_summary("legacy") is not None  # imported on first open
    finally:
        reset_conversation_store()
