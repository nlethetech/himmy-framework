"""Tests for the run-trace inspector: format_timeline + SqliteEventStore."""

from __future__ import annotations

from pathlib import Path

from himmy.core.events import EventType, RunEvent
from himmy.services.observability.trace import SqliteEventStore, format_timeline
from tests.conftest import run_async


def _events() -> list[RunEvent]:
    return [
        RunEvent(event_type=EventType.AGENT_RUN_STARTED, thread_id="t1", timestamp="1"),
        RunEvent(
            event_type=EventType.TOOL_CALLED,
            thread_id="t1",
            timestamp="2",
            payload={"tool_name": "calculator", "tool_args": {"expression": "2+2"}},
            latency_ms=12.0,
        ),
        RunEvent(
            event_type=EventType.AGENT_HANDOFF,
            thread_id="t1",
            timestamp="3",
            payload={"from": "a", "to": "b"},
        ),
        RunEvent(
            event_type=EventType.AGENT_RUN_FINISHED,
            thread_id="t1",
            timestamp="4",
            cost=0.01,
        ),
    ]


def test_format_timeline_renders_events() -> None:
    out = format_timeline(_events())
    assert "run started" in out
    assert "calculator" in out and "2+2" in out
    assert "a → b" in out  # handoff
    assert "4 events" in out
    assert "1 tool call" in out


def test_format_timeline_empty() -> None:
    assert format_timeline([]) == "(no events)"


def test_event_store_roundtrip_and_filter() -> None:
    store = SqliteEventStore(":memory:")
    for e in _events():
        run_async(store.append_event(e))
    # other thread
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.AGENT_RUN_STARTED, thread_id="t2", timestamp="5"
            )
        )
    )
    t1 = store.list_events(thread_id="t1")
    assert len(t1) == 4
    assert {e.event_type for e in t1} == {
        EventType.AGENT_RUN_STARTED,
        EventType.TOOL_CALLED,
        EventType.AGENT_HANDOFF,
        EventType.AGENT_RUN_FINISHED,
    }
    assert len(store.list_events()) == 5  # all threads
    store.close()


def test_event_store_recent_threads() -> None:
    store = SqliteEventStore(":memory:")
    for e in _events():
        run_async(store.append_event(e))
    runs = store.recent_threads()
    assert runs[0]["thread_id"] == "t1"
    assert runs[0]["events"] == 4
    store.close()


def test_event_store_durable(tmp_path: Path) -> None:
    db = str(tmp_path / "trace.db")
    a = SqliteEventStore(db)
    run_async(
        a.append_event(
            RunEvent(
                event_type=EventType.AGENT_RUN_STARTED, thread_id="x", timestamp="1"
            )
        )
    )
    a.close()
    b = SqliteEventStore(db)
    assert len(b.list_events(thread_id="x")) == 1
    b.close()
