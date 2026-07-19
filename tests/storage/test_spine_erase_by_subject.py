"""S3/S4 — spine storage ``delete_by_subject`` (chat_threads + run_events erasure).

The runtime persists every governed run's full transcript (``chat_threads``) and event
stream (``run_events``) into ``storage.db`` encrypted under the store-WIDE KEK — NOT the
per-subject key. A crypto-shred of the subject key therefore leaves them recoverable, so
right-to-erasure must HARD-DELETE them. These tests pin that the backend resolves a
subject's threads/traces from the ``runs`` table and removes exactly those rows (and only
those), idempotently, on both the in-memory facade and the durable SQLite backend.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
from himmy.core.events import RunEvent
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService


def _thread(thread_id: str, text: str) -> ChatThread:
    t = ChatThread(thread_id=thread_id)
    t.append_message(Message(role=MessageRole.USER, content=text))
    return t


def _event(thread_id: str, trace_id: str, text: str) -> RunEvent:
    return RunEvent(
        event_type="AGENT_RUN_STARTED",
        thread_id=thread_id,
        trace_id=trace_id,
        payload={"note": text},
    )


async def _seed(storage: object) -> None:
    """Two subjects: alice (t1/tr1) and the control bob (t2/tr2)."""
    s = storage
    await s.save_run(  # type: ignore[attr-defined]
        RunRecord(
            workspace_id="w",
            subject_id="alice",
            thread_id="t1",
            trace_id="tr1",
            status=RunStatus.SUCCEEDED,
        )
    )
    await s.save_thread(_thread("t1", "alice-secret"))  # type: ignore[attr-defined]
    await s.append_event(_event("t1", "tr1", "alice-evt"))  # type: ignore[attr-defined]
    await s.save_run(  # type: ignore[attr-defined]
        RunRecord(
            workspace_id="w",
            subject_id="bob",
            thread_id="t2",
            trace_id="tr2",
            status=RunStatus.SUCCEEDED,
        )
    )
    await s.save_thread(_thread("t2", "bob-data"))  # type: ignore[attr-defined]
    await s.append_event(_event("t2", "tr2", "bob-evt"))  # type: ignore[attr-defined]


def test_inmemory_delete_by_subject_removes_threads_and_events() -> None:
    s = StorageService()
    asyncio.run(_seed(s))

    deleted = s.delete_by_subject("alice")
    assert deleted == 3  # one thread + one event + the subject's own run row

    async def _check() -> None:
        assert await s.load_thread("t1") is None
        assert await s.list_events(thread_id="t1") == []
        # The control subject is untouched.
        assert await s.load_thread("t2") is not None
        assert await s.list_events(thread_id="t2")

    asyncio.run(_check())
    # Idempotent: a re-run after the subject is gone removes nothing.
    assert s.delete_by_subject("alice") == 0


def test_sqlite_delete_by_subject_removes_threads_and_events(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from himmy.services.storage.at_rest import StorePayloadCipher
    from himmy.services.storage.encryption import FieldEncryptor

    path = str(tmp_path / "storage.db")
    # Encryption ON (store-wide KEK) — the exact governed posture the finding flagged.
    cipher = StorePayloadCipher(FieldEncryptor(os.urandom(32)))
    store = SqliteStorageService(path, cipher=cipher)
    asyncio.run(_seed(store))
    asyncio.run(store.close())

    # Reopen (cross-restart) and erase alice.
    store2 = SqliteStorageService(path, cipher=cipher)
    deleted = store2.delete_by_subject("alice")
    assert deleted == 3  # one thread + one event + the subject's own run row

    async def _check() -> None:
        assert await store2.load_thread("t1") is None
        assert await store2.list_events(thread_id="t1") == []
        assert await store2.load_thread("t2") is not None
        assert await store2.list_events(thread_id="t2")

    asyncio.run(_check())
    asyncio.run(store2.close())

    # The raw rows are physically gone (count them directly).
    store3 = SqliteStorageService(path, cipher=cipher)
    n_threads = store3._fetchone(  # noqa: SLF001 - direct row count for the assertion
        "SELECT COUNT(*) AS n FROM chat_threads WHERE thread_id = 't1'"
    )["n"]
    n_events = store3._fetchone(  # noqa: SLF001
        "SELECT COUNT(*) AS n FROM run_events WHERE thread_id = 't1' OR trace_id = 'tr1'"
    )["n"]
    n_runs = store3._fetchone(  # noqa: SLF001
        "SELECT COUNT(*) AS n FROM runs WHERE subject_id = 'alice'"
    )["n"]
    assert n_threads == 0
    assert n_events == 0
    # The subject's own run row (holding cleartext output_text/output_structured) is gone.
    assert n_runs == 0
    # Idempotent re-run on the clean store.
    assert store3.delete_by_subject("alice") == 0
    asyncio.run(store3.close())
