"""P2.5 storage-micro efficiency polish — pure speed/memory, output byte-identical.

Three independent micro-wins, each pinned by an identity test (the fast/light path
returns exactly what the old path did):

* ``ConversationStore.list_summaries`` selects only the 8 metadata columns ``_summary``
  reads — never the large ``thread`` blob — yet returns identical summaries;
* ``InMemoryEventLog`` keeps thread/trace indexes for O(1) candidate lookup instead of a
  linear scan; a filtered read returns the SAME events (order + membership) and the full
  audit spine is preserved (every appended event still queryable);
* :func:`connect_hardened` applies the throughput pragmas on an ON-DISK path but NOT on
  ``:memory:`` (where a page cache / mmap / temp-store-in-RAM buys nothing).
"""

from __future__ import annotations

import threading
from pathlib import Path

from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
from himmy.core.events import EventType, RunEvent
from himmy.core.sqlite_util import connect_hardened
from himmy.services.storage.conversations import ConversationStore
from himmy.services.storage.inmemory import InMemoryEventLog


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _thread_with_blob(cid: str, blob: str) -> ChatThread:
    """A thread whose body is a large blob — the payload a summary must never pull."""
    t = ChatThread(thread_id=cid, agent_id="a.yaml")
    t.append_message(Message(role=MessageRole.USER, content=blob))
    t.append_message(Message(role=MessageRole.ASSISTANT, content=blob))
    return t


# --------------------------------------------------------------------------- (a)


def test_list_summaries_identical_without_thread_blob() -> None:
    """Summaries match a from-scratch full-column recompute — the blob is not read."""
    store = ConversationStore(":memory:")
    big = "x" * 50_000  # a blob large enough that omitting it is the whole point
    saved = []
    for i in range(3):
        cid = f"c{i}"
        saved.append(
            store.save_thread(cid, _thread_with_blob(cid, big), title=f"t{i}")
        )

    got = store.list_summaries()

    # Reference: recompute summaries the OLD way (SELECT * incl. the thread blob) and
    # feed each row through the same ``_summary`` builder. The two lists must be equal.
    rows = store._conn.execute(
        "SELECT c.*, COUNT(m.id) AS n FROM conversations c "
        "LEFT JOIN conversation_messages m ON m.conversation_id = c.conversation_id "
        "GROUP BY c.conversation_id ORDER BY c.updated_at DESC"
    ).fetchall()
    reference = [store._summary(r, r["n"]) for r in rows]

    assert got == reference
    # And the message counts survived the narrower projection.
    assert all(s.message_count == 2 for s in got)
    assert {s.conversation_id for s in got} == {"c0", "c1", "c2"}


def test_list_summaries_sql_does_not_select_thread() -> None:
    """The list query never names the ``thread`` column (the large blob stays home)."""
    captured: list[str] = []
    store = ConversationStore(":memory:")
    store.save_thread("c0", _thread_with_blob("c0", "hi"), title="t")

    # ``set_trace_callback`` sees every executed statement without touching the
    # read-only ``execute`` attribute.
    store._conn.set_trace_callback(captured.append)
    try:
        store.list_summaries()
    finally:
        store._conn.set_trace_callback(None)

    list_sql = [s for s in captured if "FROM conversations c" in s]
    assert list_sql, "list_summaries should issue its SELECT"
    # The thread blob column must not appear in the projection.
    assert "thread" not in list_sql[0]


# --------------------------------------------------------------------------- (b)


def _seed_events() -> list[RunEvent]:
    """Interleaved events across two threads and two traces (insertion order)."""
    return [
        RunEvent(event_type=EventType.AGENT_RUN_STARTED, thread_id="thA", trace_id="trA"),
        RunEvent(event_type=EventType.TOOL_CALLED, thread_id="thB", trace_id="trB"),
        RunEvent(
            event_type=EventType.TOOL_COMPLETED,
            thread_id="thA",
            trace_id="trA",
            payload={"tool_name": "search"},
        ),
        RunEvent(
            event_type=EventType.TOOL_FAILED,
            thread_id="thA",
            trace_id="trX",  # same thread, different trace
            payload={"tool_name": "wire"},
        ),
        RunEvent(event_type=EventType.TOOL_COMPLETED, thread_id="thB", trace_id="trB"),
    ]


def _linear_scan(events, **filters):  # noqa: ANN001, ANN003
    """The OLD full-scan reference implementation of ``list_events``."""
    from himmy.services.storage.protocols import normalize_event_type

    thread_id = filters.get("thread_id")
    trace_id = filters.get("trace_id")
    want_type = normalize_event_type(filters.get("event_type"))
    tool_name = filters.get("tool_name")
    workspace_id = filters.get("workspace_id")
    matches = [
        e
        for e in events
        if (thread_id is None or e.thread_id == thread_id)
        and (trace_id is None or e.trace_id == trace_id)
        and (
            want_type is None
            or getattr(e.event_type, "value", str(e.event_type)) == want_type
        )
        and (tool_name is None or (e.payload or {}).get("tool_name") == tool_name)
        and (workspace_id is None or e.workspace_id == workspace_id)
    ]
    if filters.get("newest_first"):
        matches = list(reversed(matches))
    limit = filters.get("limit")
    if limit is not None:
        matches = matches[: max(0, limit)]
    return matches


def test_event_log_indexed_lookup_matches_linear_scan() -> None:
    """Every filter combination returns exactly what a full linear scan would."""
    seed = _seed_events()
    log = InMemoryEventLog()
    for e in seed:
        _run(log.append_event(e))

    cases = [
        {},
        {"thread_id": "thA"},
        {"thread_id": "thB"},
        {"thread_id": "missing"},
        {"trace_id": "trA"},
        {"trace_id": "trB"},
        {"trace_id": "trX"},
        {"thread_id": "thA", "trace_id": "trA"},
        {"thread_id": "thA", "event_type": EventType.TOOL_FAILED},
        {"thread_id": "thA", "tool_name": "search"},
        {"trace_id": "trA", "tool_name": "search"},
        {"thread_id": "thA", "newest_first": True},
        {"thread_id": "thA", "limit": 1},
        {"thread_id": "thA", "newest_first": True, "limit": 1},
        {"event_type": EventType.TOOL_COMPLETED},
    ]
    for case in cases:
        got = _run(log.list_events(**case))
        expected = _linear_scan(seed, **case)
        assert got == expected, case


def test_event_log_full_audit_spine_preserved() -> None:
    """Every appended event stays queryable — the index caps nothing."""
    log = InMemoryEventLog()
    seed = _seed_events()
    for e in seed:
        _run(log.append_event(e))
    # Unfiltered read returns the full spine in insertion order.
    assert _run(log.list_events()) == seed
    # Union of per-thread reads covers all events (no event orphaned from its index).
    a = _run(log.list_events(thread_id="thA"))
    b = _run(log.list_events(thread_id="thB"))
    assert len(a) + len(b) == len(seed)


def test_event_log_delete_keeps_indexes_consistent() -> None:
    """After erasure the indexes never hand back a dropped event (sibling buckets too)."""
    log = InMemoryEventLog()
    seed = _seed_events()
    for e in seed:
        _run(log.append_event(e))
    # Erase thread thA. Event 3 (thA/trX) is dropped for its thread, so its sibling
    # trace trX must also shed it.
    removed = log.delete_events({"thA"}, set())
    assert removed == 3
    assert _run(log.list_events(thread_id="thA")) == []
    assert _run(log.list_events(trace_id="trA")) == []
    assert _run(log.list_events(trace_id="trX")) == []  # sibling bucket cleaned
    # thB survivors are intact and still indexed.
    survivors = _run(log.list_events())
    assert all(e.thread_id == "thB" for e in survivors)
    assert _run(log.list_events(thread_id="thB")) == survivors


def test_event_log_concurrent_append_and_read() -> None:
    """Concurrent appends + reads never corrupt the stream or its indexes."""
    log = InMemoryEventLog()
    n = 200

    def _append(tid: str) -> None:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            for i in range(n):
                loop.run_until_complete(
                    log.append_event(
                        RunEvent(
                            event_type=EventType.TOOL_CALLED,
                            thread_id=tid,
                            trace_id=f"{tid}-{i}",
                        )
                    )
                )
        finally:
            loop.close()

    def _read() -> None:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            for _ in range(n):
                loop.run_until_complete(log.list_events(thread_id="w0"))
        finally:
            loop.close()

    threads = [
        threading.Thread(target=_append, args=("w0",)),
        threading.Thread(target=_append, args=("w1",)),
        threading.Thread(target=_read),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(_run(log.list_events())) == 2 * n
    assert len(_run(log.list_events(thread_id="w0"))) == n
    assert len(_run(log.list_events(thread_id="w1"))) == n


# --------------------------------------------------------------------------- (c)


def _pragma(conn, name: str):  # noqa: ANN001, ANN202
    row = conn.execute(f"PRAGMA {name}").fetchone()
    return row[0] if row is not None else None


def test_pragmas_applied_on_disk(tmp_path: Path) -> None:
    """The throughput pragmas take effect for an on-disk database."""
    db = str(tmp_path / "t.db")
    conn = connect_hardened(db)
    try:
        assert _pragma(conn, "cache_size") == -16000
        assert int(_pragma(conn, "temp_store")) == 2  # 2 == MEMORY
        assert int(_pragma(conn, "mmap_size")) == 268435456
        # Unchanged hardening still holds.
        assert str(_pragma(conn, "journal_mode")).lower() == "wal"
    finally:
        conn.close()


def test_pragmas_skipped_in_memory() -> None:
    """An in-memory DB gets no page-cache/mmap/temp-store override (default behaviour)."""
    conn = connect_hardened(":memory:")
    try:
        # Defaults: temp_store DEFAULT (0), mmap not applied (in-memory DBs return no
        # ``mmap_size`` row at all), and cache_size is NOT forced to our -16000 value.
        assert int(_pragma(conn, "temp_store")) == 0
        assert _pragma(conn, "mmap_size") is None
        assert _pragma(conn, "cache_size") != -16000
    finally:
        conn.close()
