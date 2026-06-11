"""Tests for the lineage / provenance CLI (`himmy lineage`)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from himmy.cli import lineage_cmd
from himmy.core.events import EventType, RunEvent
from himmy.services.observability.trace import SqliteEventStore


def _ev(et: EventType, *, ts: str, thread: str = "run-1", **kw: object) -> RunEvent:
    """A RunEvent with an explicit timestamp so ordering is deterministic."""
    return RunEvent(event_type=et, thread_id=thread, timestamp=ts, **kw)


def _seed_trace(db_path: str, thread: str = "run-1") -> None:
    """Write a small but realistic run trace: start → inference → two tool calls."""
    store = SqliteEventStore(db_path)
    events = [
        _ev(
            EventType.AGENT_RUN_STARTED,
            ts="2024-01-01T00:00:00",
            thread=thread,
            payload={"model_key": "claude-cli:sonnet", "persona_name": "himmy"},
        ),
        _ev(
            EventType.INFERENCE_REQUESTED,
            ts="2024-01-01T00:00:01",
            thread=thread,
            payload={"model_key": "claude-cli:sonnet"},
        ),
        _ev(
            EventType.TOOL_CALLED,
            ts="2024-01-01T00:00:02",
            thread=thread,
            tool_call_id="c1",
            payload={"tool_name": "web_search", "args": {"q": "permaculture"}},
        ),
        _ev(
            EventType.TOOL_COMPLETED,
            ts="2024-01-01T00:00:03",
            thread=thread,
            tool_call_id="c1",
            latency_ms=42.0,
            payload={"tool_name": "web_search", "tool_outcome": "success"},
        ),
        _ev(
            EventType.TOOL_CALLED,
            ts="2024-01-01T00:00:04",
            thread=thread,
            tool_call_id="c2",
            payload={"tool_name": "kb_search", "args": {"q": "ducks"}},
        ),
        _ev(
            EventType.TOOL_COMPLETED,
            ts="2024-01-01T00:00:05",
            thread=thread,
            tool_call_id="c2",
            payload={"tool_name": "kb_search", "tool_outcome": "success"},
        ),
        _ev(
            EventType.INFERENCE_SUCCEEDED,
            ts="2024-01-01T00:00:06",
            thread=thread,
            latency_ms=920.0,
            cost=0.0021,
            payload={"input_tokens": 100, "output_tokens": 40},
        ),
        _ev(
            EventType.AGENT_RUN_FINISHED,
            ts="2024-01-01T00:00:07",
            thread=thread,
            payload={"status": "SUCCESS"},
        ),
    ]
    for e in events:
        asyncio.run(store.append_event(e))
    store.close()


def _ns(**kw: object) -> argparse.Namespace:
    base = {"run_id": None, "dot": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_lineage_shows_tool_calls_in_order(tmp_path, monkeypatch, capsys):
    """The lineage view lists the run's tool calls in trace order."""
    monkeypatch.chdir(tmp_path)
    Path(".himmy").mkdir()
    _seed_trace(str(tmp_path / ".himmy" / "trace.db"))

    rc = lineage_cmd.cmd_lineage(_ns(run_id="run-1"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "run run-1" in out
    assert "web_search" in out
    assert "kb_search" in out
    # Order: web_search precedes kb_search in the rendered tree.
    assert out.index("web_search") < out.index("kb_search")
    # Inference + result nodes are present.
    assert "inference" in out
    assert "result" in out


def test_lineage_latest_when_no_run_id(tmp_path, monkeypatch, capsys):
    """With no run_id, the most recent traced run is used."""
    monkeypatch.chdir(tmp_path)
    Path(".himmy").mkdir()
    _seed_trace(str(tmp_path / ".himmy" / "trace.db"), thread="newest")

    rc = lineage_cmd.cmd_lineage(_ns())
    out = capsys.readouterr().out

    assert rc == 0
    assert "run newest" in out


def test_lineage_dot_emits_graphviz(tmp_path, monkeypatch, capsys):
    """--dot emits a Graphviz digraph with the run as root + tool nodes/edges."""
    monkeypatch.chdir(tmp_path)
    Path(".himmy").mkdir()
    _seed_trace(str(tmp_path / ".himmy" / "trace.db"))

    rc = lineage_cmd.cmd_lineage(_ns(run_id="run-1", dot=True))
    out = capsys.readouterr().out

    assert rc == 0
    assert out.startswith("digraph lineage {")
    assert "rankdir=LR;" in out
    assert "->" in out  # at least one edge
    assert "web_search" in out
    assert out.rstrip().endswith("}")


def test_lineage_no_trace_db(tmp_path, monkeypatch, capsys):
    """No trace.db at all → exit 1 with a helpful hint, no crash."""
    monkeypatch.chdir(tmp_path)
    rc = lineage_cmd.cmd_lineage(_ns())
    err = capsys.readouterr().err
    assert rc == 1
    assert "no traces yet" in err


def test_lineage_unknown_run_id(tmp_path, monkeypatch, capsys):
    """A run_id with no events → exit 1, not a traceback."""
    monkeypatch.chdir(tmp_path)
    Path(".himmy").mkdir()
    _seed_trace(str(tmp_path / ".himmy" / "trace.db"))

    rc = lineage_cmd.cmd_lineage(_ns(run_id="does-not-exist"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "no events" in err


def test_lineage_empty_db(tmp_path, monkeypatch, capsys):
    """A trace.db that exists but holds no runs → exit 1 cleanly."""
    monkeypatch.chdir(tmp_path)
    Path(".himmy").mkdir()
    # Create the db file with the schema but no events.
    SqliteEventStore(str(tmp_path / ".himmy" / "trace.db")).close()

    rc = lineage_cmd.cmd_lineage(_ns())
    err = capsys.readouterr().err
    assert rc == 1
    assert "no traced runs found" in err


def test_build_tree_attaches_result_under_call():
    """A tool result hangs under its matching call node (provenance edge)."""
    events = [
        _ev(
            EventType.TOOL_CALLED,
            ts="2024-01-01T00:00:00",
            tool_call_id="c1",
            payload={"tool_name": "calc", "args": {"x": 1}},
        ),
        _ev(
            EventType.TOOL_COMPLETED,
            ts="2024-01-01T00:00:01",
            tool_call_id="c1",
            payload={"tool_name": "calc", "tool_outcome": "success"},
        ),
    ]
    root = lineage_cmd.build_lineage_tree(events, "run-1")
    # one tool call under root, with the result as its child
    call_nodes = [n for n in root.children if n.label.startswith("tool ")]
    assert len(call_nodes) == 1
    assert call_nodes[0].label == "tool calc"
    assert [c.label for c in call_nodes[0].children] == ["result"]


def test_tool_failed_renders_crimson_result(tmp_path, monkeypatch, capsys):
    """A failed tool produces a 'result FAILED' node carrying the error."""
    monkeypatch.chdir(tmp_path)
    Path(".himmy").mkdir()
    store = SqliteEventStore(str(tmp_path / ".himmy" / "trace.db"))
    asyncio.run(
        store.append_event(
            _ev(
                EventType.TOOL_CALLED,
                ts="2024-01-01T00:00:00",
                tool_call_id="c1",
                payload={"tool_name": "boom", "args": {}},
            )
        )
    )
    asyncio.run(
        store.append_event(
            _ev(
                EventType.TOOL_FAILED,
                ts="2024-01-01T00:00:01",
                tool_call_id="c1",
                error="kaboom",
                payload={"tool_name": "boom"},
            )
        )
    )
    store.close()

    rc = lineage_cmd.cmd_lineage(_ns(run_id="run-1"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "result FAILED" in out
    assert "kaboom" in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
