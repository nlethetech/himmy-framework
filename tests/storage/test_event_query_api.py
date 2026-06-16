"""P1: the filtered ``list_events`` query API across backends.

The self-learning miners read recorded TOOL_FAILED/TOOL_COMPLETED signals by index:
``list_events(event_type=..., tool_name=..., limit=..., newest_first=...)``. This asserts
the new filters return the correct rows at parity on the in-memory facade and the durable
SQLite backend (including the encryption-aware tool_name path). Postgres parity is covered
by the live PG lane; here we assert the offline backends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.core.events import EventType, RunEvent
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _seed(store: object) -> None:
    """Append a deterministic mix of tool + non-tool events to a store."""
    events = [
        RunEvent(event_type=EventType.AGENT_RUN_STARTED, thread_id="th", trace_id="tr"),
        RunEvent(
            event_type=EventType.TOOL_COMPLETED,
            thread_id="th",
            trace_id="tr",
            payload={"tool_name": "search", "tool_outcome": "success"},
        ),
        RunEvent(
            event_type=EventType.TOOL_FAILED,
            thread_id="th",
            trace_id="tr",
            payload={"tool_name": "wire_money", "tool_outcome": "error"},
        ),
        RunEvent(
            event_type=EventType.TOOL_FAILED,
            thread_id="th",
            trace_id="tr",
            payload={"tool_name": "wire_money", "tool_outcome": "error"},
        ),
        RunEvent(
            event_type=EventType.TOOL_COMPLETED,
            thread_id="th",
            trace_id="tr",
            payload={"tool_name": "search", "tool_outcome": "success"},
        ),
    ]
    for e in events:
        run_async(store.append_event(e))  # type: ignore[attr-defined]


def test_inmemory_filtered_list_and_count() -> None:
    store = StorageService()
    _seed(store)

    failed = run_async(store.list_events(event_type=EventType.TOOL_FAILED))
    assert {e.payload["tool_name"] for e in failed} == {"wire_money"}
    assert len(failed) == 2

    # event_type accepts the raw string value too (normalised like append_event).
    by_str = run_async(store.list_events(event_type="TOOL_FAILED"))
    assert len(by_str) == 2

    wire_failed = run_async(
        store.list_events(event_type=EventType.TOOL_FAILED, tool_name="wire_money")
    )
    assert len(wire_failed) == 2
    search_done = run_async(
        store.list_events(event_type=EventType.TOOL_COMPLETED, tool_name="search")
    )
    assert len(search_done) == 2

    # A filter that matches nothing returns an empty list (not an error).
    none_match = run_async(
        store.list_events(event_type=EventType.TOOL_FAILED, tool_name="search")
    )
    assert none_match == []


def test_inmemory_limit_and_newest_first() -> None:
    store = StorageService()
    _seed(store)
    # The two TOOL_COMPLETED 'search' events were appended at index 1 (first) and 4 (last).
    newest = run_async(
        store.list_events(
            event_type=EventType.TOOL_COMPLETED,
            tool_name="search",
            limit=1,
            newest_first=True,
        )
    )
    assert len(newest) == 1
    oldest = run_async(
        store.list_events(
            event_type=EventType.TOOL_COMPLETED, tool_name="search", limit=1
        )
    )
    assert len(oldest) == 1
    # newest_first vs oldest pick different physical rows (distinct event_ids).
    assert newest[0].event_id != oldest[0].event_id


def test_sqlite_filtered_list_and_count(tmp_path: Path) -> None:
    store = SqliteStorageService(str(tmp_path / "s.db"))
    _seed(store)

    failed = run_async(store.list_events(event_type=EventType.TOOL_FAILED))
    assert len(failed) == 2
    wire = run_async(
        store.list_events(event_type=EventType.TOOL_FAILED, tool_name="wire_money")
    )
    assert len(wire) == 2
    search = run_async(
        store.list_events(event_type=EventType.TOOL_COMPLETED, tool_name="search")
    )
    assert len(search) == 2
    # Limit + newest_first bound and order by the insertion-order seq column.
    one = run_async(
        store.list_events(
            event_type=EventType.TOOL_FAILED,
            tool_name="wire_money",
            limit=1,
            newest_first=True,
        )
    )
    assert len(one) == 1


def test_sqlite_tool_name_filter_with_encryption(tmp_path: Path) -> None:
    """The denormalised tool_name column is captured pre-encryption, so the filter works
    even when the payload is encrypted at rest (the column is plaintext, the JSON is not).
    """
    pytest.importorskip("cryptography")
    from himmy.services.storage.at_rest import StorePayloadCipher
    from himmy.services.storage.encryption import FieldEncryptor

    cipher = StorePayloadCipher(FieldEncryptor.generate())
    store = SqliteStorageService(str(tmp_path / "enc.db"), cipher=cipher)
    _seed(store)

    wire = run_async(
        store.list_events(event_type=EventType.TOOL_FAILED, tool_name="wire_money")
    )
    assert len(wire) == 2
    # The decrypted payload still round-trips (the filter rode the plaintext column).
    assert all(e.payload["tool_name"] == "wire_money" for e in wire)
