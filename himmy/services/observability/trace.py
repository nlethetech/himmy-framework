"""Run tracing: a human-readable timeline of a run's events + a durable event log.

Every run already emits a stream of :class:`~himmy.core.events.RunEvent`s (turns,
inference, tool calls, handoffs). :func:`format_timeline` renders an ordered, indented
timeline from a list of them; :class:`SqliteEventStore` persists them (stdlib sqlite3,
mirroring the other Sqlite stores) so `himmy trace` can inspect a *past* run, not just one
streaming live. Together they make a real agent run debuggable.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from himmy.core.events import EventType, RunEvent

# How each event type renders (icon + indent depth) in the timeline.
_RENDER: dict[str, tuple[str, int]] = {
    "AGENT_RUN_STARTED": ("▶ run started", 0),
    "AGENT_RUN_FINISHED": ("■ run finished", 0),
    "AGENT_TURN_STARTED": ("○ turn", 1),
    "AGENT_TURN_COMPLETED": ("● turn done", 1),
    "INFERENCE_REQUESTED": ("→ inference", 2),
    "INFERENCE_SUCCEEDED": ("← inference ok", 2),
    "INFERENCE_FAILED": ("← inference FAILED", 2),
    "TOOL_CALLED": ("⚙ tool call", 2),
    "TOOL_COMPLETED": ("✓ tool done", 2),
    "TOOL_FAILED": ("✗ tool FAILED", 2),
    "AGENT_HANDOFF": ("⇄ handoff", 1),
    "AGENT_DELEGATED": ("⇲ delegated", 2),
    "APPROVAL_REQUIRED": ("⏸ approval required", 1),
}


def _line(event: RunEvent) -> str:
    """Render one event as an indented timeline line."""
    label, depth = _RENDER.get(event.event_type.value, (event.event_type.value, 1))
    detail = ""
    p = event.payload or {}
    if event.event_type in (
        EventType.TOOL_CALLED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
    ):
        detail = f" {p.get('tool_name', '')} {json.dumps(p.get('tool_args', {}), default=str)}"[
            :80
        ]
    elif event.event_type == EventType.AGENT_HANDOFF:
        detail = f" {p.get('from')} → {p.get('to')}"
    elif event.event_type == EventType.AGENT_DELEGATED:
        detail = f" {p.get('worker')}"
    elif event.event_type == EventType.AGENT_TURN_STARTED:
        detail = f" #{p.get('turn', '')}"
    timing = f"  ({event.latency_ms:.0f}ms)" if event.latency_ms else ""
    return f"{'  ' * depth}{label}{detail}{timing}"


def format_timeline(events: list[RunEvent]) -> str:
    """Render a chronological, indented timeline with a cost/token footer."""
    if not events:
        return "(no events)"
    ordered = sorted(events, key=lambda e: e.timestamp)
    lines = [_line(e) for e in ordered]
    total_cost = sum(e.cost or 0.0 for e in ordered)
    tools = sum(1 for e in ordered if e.event_type == EventType.TOOL_CALLED)
    lines.append("")
    lines.append(
        f"— {len(ordered)} events, {tools} tool call(s), cost ${total_cost:.4f}"
    )
    return "\n".join(lines)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    trace_id   TEXT,
    thread_id  TEXT,
    agent_id   TEXT,
    latency_ms REAL,
    cost       REAL,
    payload    TEXT NOT NULL DEFAULT '{}',
    error      TEXT,
    timestamp  TEXT NOT NULL,
    request_id   TEXT,
    tool_call_id TEXT,
    workspace_id TEXT
);
CREATE INDEX IF NOT EXISTS events_thread_idx ON events (thread_id);
CREATE INDEX IF NOT EXISTS events_trace_idx ON events (trace_id);
"""

# Columns added after the original schema shipped; ALTER them in on open so an
# existing .himmy/trace.db (created before these columns existed) doesn't break
# _row's SELECT *. CREATE TABLE IF NOT EXISTS never migrates an existing table.
_MIGRATIONS = ("request_id", "tool_call_id", "workspace_id")


class SqliteEventStore:
    """A durable event log (stdlib sqlite3) usable as a runtime event sink."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path``."""
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(events)")}
        for col in _MIGRATIONS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
        self._conn.commit()

    async def append_event(self, event: RunEvent) -> None:
        """Persist one event (async to satisfy the EventSink protocol)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO events (event_id, event_type, trace_id, thread_id, "
            "agent_id, latency_ms, cost, payload, error, timestamp, "
            "request_id, tool_call_id, workspace_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.event_type.value,
                event.trace_id,
                event.thread_id,
                event.agent_id,
                event.latency_ms,
                event.cost,
                json.dumps(event.payload, default=str),
                event.error,
                event.timestamp,
                event.request_id,
                event.tool_call_id,
                event.workspace_id,
            ),
        )
        self._conn.commit()

    def list_events(
        self, thread_id: str | None = None, trace_id: str | None = None
    ) -> list[RunEvent]:
        """Read events (optionally filtered), ordered by timestamp."""
        clauses, params = [], []
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if trace_id:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp", params
        ).fetchall()
        return [self._row(r) for r in rows]

    def recent_threads(self, limit: int = 10) -> list[dict[str, Any]]:
        """List the most recent runs (thread_id + event count + last timestamp)."""
        rows = self._conn.execute(
            "SELECT thread_id, COUNT(*) n, MAX(timestamp) last FROM events "
            "WHERE thread_id IS NOT NULL GROUP BY thread_id ORDER BY last DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"thread_id": r["thread_id"], "events": r["n"], "last": r["last"]}
            for r in rows
        ]

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> RunEvent:
        return RunEvent(
            event_id=row["event_id"],
            event_type=EventType(row["event_type"]),
            trace_id=row["trace_id"],
            thread_id=row["thread_id"],
            agent_id=row["agent_id"],
            latency_ms=row["latency_ms"],
            cost=row["cost"],
            payload=json.loads(row["payload"]),
            error=row["error"],
            timestamp=row["timestamp"],
            request_id=row["request_id"],
            tool_call_id=row["tool_call_id"],
            workspace_id=row["workspace_id"],
        )


__all__ = ["format_timeline", "SqliteEventStore"]
