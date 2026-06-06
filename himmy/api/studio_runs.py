"""Persistence for Himmy Studio runs — so the GUI can browse past conversations.

A small, self-contained SQLite store (stdlib ``sqlite3``) at ``.himmy/studio.db``
under the project root. Every run made from Studio records its agent, provider,
prompt, answer, the tools it touched, the full turn transcript, and a step-by-step
timeline — enough to list runs and replay one as a trace. Separate from the
``/v1`` run store (entity-backed, tenant-scoped); this is the local single-user GUI.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_SCHEMA = """
CREATE TABLE IF NOT EXISTS studio_runs (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    agent_name  TEXT,
    agent_path  TEXT,
    provider    TEXT,
    model       TEXT,
    prompt      TEXT,
    output      TEXT,
    status      TEXT NOT NULL DEFAULT 'ok',
    duration_ms REAL,
    thread_id   TEXT,
    tools       TEXT NOT NULL DEFAULT '[]',
    messages    TEXT NOT NULL DEFAULT '[]',
    timeline    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS studio_runs_created_idx ON studio_runs (created_at);
"""


class TimelineStep(BaseModel):
    """One step in a run's timeline (rendered as a trace in the GUI)."""

    seq: int
    type: str
    label: str
    detail: str = ""
    ts: str | None = None


class TranscriptMessage(BaseModel):
    """A message in the run's transcript."""

    role: str
    content: str


class StudioRunSummary(BaseModel):
    """List-view shape: everything but the heavy transcript/timeline."""

    id: str
    created_at: str
    agent_name: str | None = None
    agent_path: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt: str = ""
    output_preview: str = ""
    status: str = "ok"
    duration_ms: float | None = None
    tool_count: int = 0


class StudioRun(StudioRunSummary):
    """Detail-view shape: adds the full transcript, tools, and timeline."""

    output: str = ""
    thread_id: str | None = None
    tools: list[str] = []
    messages: list[TranscriptMessage] = []
    timeline: list[TimelineStep] = []


class StudioRunListResponse(BaseModel):
    """Paged list of run summaries."""

    items: list[StudioRunSummary]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


def _preview(text: str, n: int = 140) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


class StudioRunStore:
    """A durable SQLite store of Studio runs (stdlib sqlite3)."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, run: StudioRun) -> None:
        """Insert (or replace) one run record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO studio_runs (id, created_at, agent_name, "
            "agent_path, provider, model, prompt, output, status, duration_ms, "
            "thread_id, tools, messages, timeline) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run.id,
                run.created_at,
                run.agent_name,
                run.agent_path,
                run.provider,
                run.model,
                run.prompt,
                run.output,
                run.status,
                run.duration_ms,
                run.thread_id,
                json.dumps(run.tools),
                json.dumps([m.model_dump() for m in run.messages]),
                json.dumps([s.model_dump() for s in run.timeline]),
            ),
        )
        self._conn.commit()

    def count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM studio_runs").fetchone()[0]
        )

    def list(self, limit: int = 50, offset: int = 0) -> list[StudioRunSummary]:
        """List run summaries, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM studio_runs ORDER BY created_at DESC, id DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._summary(r) for r in rows]

    def get(self, run_id: str) -> StudioRun | None:
        row = self._conn.execute(
            "SELECT * FROM studio_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return self._full(row) if row else None

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _summary(row: sqlite3.Row) -> StudioRunSummary:
        return StudioRunSummary(
            id=row["id"],
            created_at=row["created_at"],
            agent_name=row["agent_name"],
            agent_path=row["agent_path"],
            provider=row["provider"],
            model=row["model"],
            prompt=_preview(row["prompt"]),
            output_preview=_preview(row["output"]),
            status=row["status"],
            duration_ms=row["duration_ms"],
            tool_count=len(json.loads(row["tools"] or "[]")),
        )

    @staticmethod
    def _full(row: sqlite3.Row) -> StudioRun:
        return StudioRun(
            id=row["id"],
            created_at=row["created_at"],
            agent_name=row["agent_name"],
            agent_path=row["agent_path"],
            provider=row["provider"],
            model=row["model"],
            prompt=row["prompt"],
            output=row["output"],
            output_preview=_preview(row["output"]),
            status=row["status"],
            duration_ms=row["duration_ms"],
            thread_id=row["thread_id"],
            tools=json.loads(row["tools"] or "[]"),
            messages=[
                TranscriptMessage(**m) for m in json.loads(row["messages"] or "[]")
            ],
            timeline=[
                TimelineStep(**s) for s in json.loads(row["timeline"] or "[]")
            ],
            tool_count=len(json.loads(row["tools"] or "[]")),
        )


def studio_db_path() -> str:
    """Path to the Studio run DB (``.himmy/studio.db``), creating ``.himmy``."""
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "studio.db")


_STORE: StudioRunStore | None = None
_STORE_PATH: str | None = None


def get_run_store() -> StudioRunStore:
    """Process-wide run store, (re)opened if the project root (cwd) changed."""
    global _STORE, _STORE_PATH
    path = studio_db_path()
    if _STORE is None or _STORE_PATH != path:
        if _STORE is not None:
            _STORE.close()
        _STORE = StudioRunStore(path)
        _STORE_PATH = path
    return _STORE


def reset_run_store() -> None:
    """Drop the cached store handle (tests change cwd between cases)."""
    global _STORE, _STORE_PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _STORE_PATH = None


__all__ = [
    "StudioRun",
    "StudioRunSummary",
    "StudioRunListResponse",
    "TimelineStep",
    "TranscriptMessage",
    "StudioRunStore",
    "get_run_store",
    "reset_run_store",
    "studio_db_path",
]
