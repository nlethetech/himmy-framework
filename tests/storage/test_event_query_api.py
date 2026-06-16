"""P1: the filtered ``list_events`` query API across backends.

The self-learning miners read recorded TOOL_FAILED/TOOL_COMPLETED signals by index:
``list_events(event_type=..., tool_name=..., limit=..., newest_first=...)``. This asserts
the new filters return the correct rows at parity on the in-memory facade and the durable
SQLite backend (including the encryption-aware tool_name path). Postgres parity is covered
by the live PG lane; here we assert the offline backends.

It also pins the load-bearing per-tenant guarantee on the DURABLE store: a reputation read
scoped to ``workspace_id`` sees only its own tenant's tool events on a SHARED SQLite backend
(plaintext AND encrypted at rest), so a regression in the denormalised ``workspace_id``
capture/filter — the actual /v1 cross-tenant boundary — fails here, not only on the
in-memory facade the dedicated learning tests reach.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.core.events import EventType, RunEvent
from himmy.services.learning.service import LearningService
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


def _seed_scoped(store: object) -> None:
    """Seed one shared store with the SAME tool under two tenants (A fails, B succeeds).

    Five ``TOOL_FAILED`` of ``wire`` under workspace ``A`` and five ``TOOL_COMPLETED`` of
    the same tool under ``B`` — the denormalised ``workspace_id`` column is what keeps the
    two tenants' reputation reads apart on the shared backend.
    """
    events = [
        RunEvent(
            event_type=EventType.TOOL_FAILED,
            payload={"tool_name": "wire", "tool_outcome": "error"},
            workspace_id="A",
        )
        for _ in range(5)
    ] + [
        RunEvent(
            event_type=EventType.TOOL_COMPLETED,
            payload={"tool_name": "wire", "tool_outcome": "success"},
            workspace_id="B",
        )
        for _ in range(5)
    ]
    for e in events:
        run_async(store.append_event(e))  # type: ignore[attr-defined]


def _assert_tenant_isolated(store: object) -> None:
    """A reputation read on ``store`` sees ONLY its own tenant's events (None aggregates).

    This is the durable-backend mirror of the in-memory
    ``test_reputation_scoped_to_workspace_isolates_tenants`` — it pins the denormalised
    ``workspace_id`` capture+filter that the in-memory facade cannot reach, so a cross-tenant
    reputation bleed on the real /v1 store would fail here even with every other test green.
    """
    rep_a = run_async(
        LearningService(store, workspace_id="A").get_tool_reputation(["wire"])
    )
    rep_b = run_async(
        LearningService(store, workspace_id="B").get_tool_reputation(["wire"])
    )
    rep_all = run_async(LearningService(store).get_tool_reputation(["wire"]))
    assert rep_a["wire"].failed == 5 and rep_a["wire"].completed == 0
    assert rep_a["wire"].score == 0.0  # only A's failures are in scope
    assert rep_b["wire"].failed == 0 and rep_b["wire"].completed == 5
    assert rep_b["wire"].score == 1.0  # only B's completions are in scope
    # Unscoped reads the whole stream — both tenants' events aggregate (back-compat).
    assert rep_all["wire"].failed == 5 and rep_all["wire"].completed == 5


def test_sqlite_reputation_scoped_to_workspace_isolates_tenants(tmp_path: Path) -> None:
    """Per-tenant reputation isolation holds on the durable SQLite store (plaintext)."""
    store = SqliteStorageService(str(tmp_path / "scoped.db"))
    _seed_scoped(store)
    _assert_tenant_isolated(store)


def test_sqlite_reputation_isolation_with_encryption(tmp_path: Path) -> None:
    """Isolation survives at-rest encryption: workspace_id is denormalised PRE-encryption.

    The /v1 security boundary lives on the SHARED encrypted store, so the scoping must ride
    the plaintext ``workspace_id`` column, not the (encrypted) payload — A's failures must
    never bleed into B's reputation even when the JSON is unreadable at rest.
    """
    pytest.importorskip("cryptography")
    from himmy.services.storage.at_rest import StorePayloadCipher
    from himmy.services.storage.encryption import FieldEncryptor

    cipher = StorePayloadCipher(FieldEncryptor.generate())
    store = SqliteStorageService(str(tmp_path / "scoped_enc.db"), cipher=cipher)
    _seed_scoped(store)
    _assert_tenant_isolated(store)
