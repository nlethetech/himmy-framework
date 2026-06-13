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

from pydantic import BaseModel

_SCHEMA = """
CREATE TABLE IF NOT EXISTS studio_runs (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    agent_name    TEXT,
    agent_path    TEXT,
    provider      TEXT,
    model         TEXT,
    prompt        TEXT,
    output        TEXT,
    status        TEXT NOT NULL DEFAULT 'ok',
    duration_ms   REAL,
    thread_id     TEXT,
    tools         TEXT NOT NULL DEFAULT '[]',
    messages      TEXT NOT NULL DEFAULT '[]',
    timeline      TEXT NOT NULL DEFAULT '[]',
    steps         TEXT NOT NULL DEFAULT '[]',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost          REAL NOT NULL DEFAULT 0,
    usage_by_model TEXT NOT NULL DEFAULT '[]',
    checkpoint_id TEXT,
    feedback_verdict TEXT,
    feedback_note TEXT,
    feedback_at TEXT
);
CREATE INDEX IF NOT EXISTS studio_runs_created_idx ON studio_runs (created_at);
"""

# Columns added after the table first shipped — applied by _migrate() with a
# matching SQL type so an existing DB is upgraded in place.
_LATER_COLUMNS = {
    "steps": "TEXT NOT NULL DEFAULT '[]'",
    "input_tokens": "INTEGER NOT NULL DEFAULT 0",
    "output_tokens": "INTEGER NOT NULL DEFAULT 0",
    "cost": "REAL NOT NULL DEFAULT 0",
    "usage_by_model": "TEXT NOT NULL DEFAULT '[]'",
    "checkpoint_id": "TEXT",
    # P0-A run feedback (human thumbs up/down + optional note). The
    # authoritative current-state copy; the immutable history lives on the
    # entity spine (kind='run_feedback'). Written only by set_feedback(), never
    # by save(), so re-saving a run (e.g. an HITL resume) never clobbers it.
    "feedback_verdict": "TEXT",
    "feedback_note": "TEXT",
    "feedback_at": "TEXT",
}

#: Allowed feedback verdicts (kept in sync with the API + entity payload).
FEEDBACK_VERDICTS: frozenset[str] = frozenset({"up", "down"})


class TimelineStep(BaseModel):
    """One step in a run's timeline (rendered as a trace in the GUI)."""

    seq: int
    type: str
    label: str
    detail: str = ""
    ts: str | None = None
    # Raw model I/O for inference steps when capture is on (the trace inspector).
    io: dict[str, Any] | None = None


class CognitionStep(BaseModel):
    """One step in the live cognition trace (think → act → observe)."""

    seq: int
    kind: str  # agent | reason | tool | delegate | handoff
    agent: str | None = None
    model: str | None = None
    # reasoning
    text: str | None = None
    # tool call
    name: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    outcome: str | None = None
    read_only: bool | None = None
    latency_ms: float | None = None
    # delegation / handoff
    worker: str | None = None
    task: str | None = None
    to: str | None = None
    # grounding (memory recall / knowledge retrieval)
    source: str | None = None  # "memory" | "knowledge"
    query: str | None = None
    citations: list[dict[str, Any]] | None = None
    # guardrails / safety
    stage: str | None = None  # "input" | "output"
    blocked: bool | None = None
    redacted: bool | None = None
    flags: list[str] | None = None
    reasons: list[str] | None = None
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
    # Usage rollup (surfaced in the list + analytics).
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    # P0-A: human feedback on this run, if any ("up" | "down").
    feedback_verdict: str | None = None
    feedback_note: str | None = None
    feedback_at: str | None = None


class ModelUsage(BaseModel):
    """Per-model token + cost rollup for one run (teams touch several models)."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    inferences: int = 0


class StudioRun(StudioRunSummary):
    """Detail-view shape: adds the full transcript, tools, and timeline."""

    output: str = ""
    thread_id: str | None = None
    tools: list[str] = []
    messages: list[TranscriptMessage] = []
    timeline: list[TimelineStep] = []
    steps: list[CognitionStep] = []  # the cognition trace (think → act → observe)
    usage_by_model: list[ModelUsage] = []
    checkpoint_id: str | None = None  # set when the run paused for HITL approval


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


def _col(row: sqlite3.Row, name: str, default: Any) -> Any:
    """Read a column that may be absent on rows written before a migration."""
    if name in row.keys():
        val = row[name]
        return default if val is None else val
    return default


def _pct(values: list[float], q: float) -> float:
    """The q-th percentile (0–1) by nearest-rank, on an unsorted list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


class ModelAnalytics(BaseModel):
    """Per-model rollup across all runs."""

    model: str
    runs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class DayAnalytics(BaseModel):
    """One day's rollup (for the cost-over-time chart)."""

    day: str  # YYYY-MM-DD
    runs: int = 0
    cost: float = 0.0
    tokens: int = 0


class RunAnalytics(BaseModel):
    """Aggregate stats across all runs — powers the analytics dashboard."""

    total_runs: int = 0
    ok_runs: int = 0
    error_runs: int = 0
    success_rate: float = 0.0  # 0–1
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    # P0-A: explicit human-feedback tallies over the window (the cheapest,
    # least-noisy learning signal — surfaced so the founder can see at a glance
    # how much rated traffic exists before any loop relies on it).
    feedback_up: int = 0
    feedback_down: int = 0
    by_model: list[ModelAnalytics] = []
    by_day: list[DayAnalytics] = []


class StudioRunStore:
    """A durable SQLite store of Studio runs (stdlib sqlite3)."""

    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created (steps, usage)."""
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(studio_runs)").fetchall()
        }
        for name, decl in _LATER_COLUMNS.items():
            if name not in cols:
                self._conn.execute(f"ALTER TABLE studio_runs ADD COLUMN {name} {decl}")

    def save(self, run: StudioRun) -> None:
        """Insert or update one run record, PRESERVING any human feedback.

        Uses ``ON CONFLICT(id) DO UPDATE`` rather than ``INSERT OR REPLACE`` so
        re-saving a run (an HITL resume persists the continuation under the same
        id) overwrites only the run-content columns — ``feedback_*`` is owned by
        :meth:`set_feedback` and is never touched here.
        """
        self._conn.execute(
            "INSERT INTO studio_runs (id, created_at, agent_name, "
            "agent_path, provider, model, prompt, output, status, duration_ms, "
            "thread_id, tools, messages, timeline, steps, input_tokens, "
            "output_tokens, cost, usage_by_model, checkpoint_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "created_at=excluded.created_at, agent_name=excluded.agent_name, "
            "agent_path=excluded.agent_path, provider=excluded.provider, "
            "model=excluded.model, prompt=excluded.prompt, output=excluded.output, "
            "status=excluded.status, duration_ms=excluded.duration_ms, "
            "thread_id=excluded.thread_id, tools=excluded.tools, "
            "messages=excluded.messages, timeline=excluded.timeline, "
            "steps=excluded.steps, input_tokens=excluded.input_tokens, "
            "output_tokens=excluded.output_tokens, cost=excluded.cost, "
            "usage_by_model=excluded.usage_by_model, "
            "checkpoint_id=excluded.checkpoint_id",
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
                json.dumps([s.model_dump() for s in run.steps]),
                run.input_tokens,
                run.output_tokens,
                run.cost,
                json.dumps([u.model_dump() for u in run.usage_by_model]),
                run.checkpoint_id,
            ),
        )
        self._conn.commit()

    def set_feedback(
        self, run_id: str, *, verdict: str, note: str | None, at: str
    ) -> bool:
        """Record human feedback on a run. Returns False when the run is unknown.

        Idempotent and re-settable: a later call overwrites the current-state
        columns (the immutable history lives on the entity spine). Raises
        ``ValueError`` on an unknown verdict so the caller can return 422.
        """
        if verdict not in FEEDBACK_VERDICTS:
            raise ValueError(
                f"unknown feedback verdict {verdict!r}; "
                f"expected one of {sorted(FEEDBACK_VERDICTS)}"
            )
        cur = self._conn.execute(
            "UPDATE studio_runs SET feedback_verdict = ?, feedback_note = ?, "
            "feedback_at = ? WHERE id = ?",
            (verdict, note, at, run_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM studio_runs").fetchone()[0])

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

    def get_by_checkpoint(self, checkpoint_id: str) -> StudioRun | None:
        """The run that paused on this checkpoint (so a resume can rebuild it)."""
        row = self._conn.execute(
            "SELECT * FROM studio_runs WHERE checkpoint_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (checkpoint_id,),
        ).fetchone()
        return self._full(row) if row else None

    def analytics(self, limit: int = 1000) -> RunAnalytics:
        """Aggregate stats over the most recent ``limit`` runs.

        Computed in Python (percentiles + per-model rollup from the usage JSON)
        over a bounded window — fine for the local single-user store.
        """
        rows = self._conn.execute(
            "SELECT created_at, status, duration_ms, cost, input_tokens, "
            "output_tokens, model, usage_by_model, feedback_verdict FROM studio_runs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return RunAnalytics()

        ok = sum(1 for r in rows if r["status"] == "ok")
        feedback_up = sum(1 for r in rows if _col(r, "feedback_verdict", None) == "up")
        feedback_down = sum(
            1 for r in rows if _col(r, "feedback_verdict", None) == "down"
        )
        latencies = [float(r["duration_ms"]) for r in rows if r["duration_ms"]]
        by_model: dict[str, ModelAnalytics] = {}
        by_day: dict[str, DayAnalytics] = {}
        total_cost = total_in = total_out = 0.0

        for r in rows:
            total_cost += float(_col(r, "cost", 0.0) or 0.0)
            total_in += int(_col(r, "input_tokens", 0) or 0)
            total_out += int(_col(r, "output_tokens", 0) or 0)

            # per-day rollup
            day = str(r["created_at"])[:10]
            d = by_day.setdefault(day, DayAnalytics(day=day))
            d.runs += 1
            d.cost += float(_col(r, "cost", 0.0) or 0.0)
            d.tokens += int(_col(r, "input_tokens", 0) or 0) + int(
                _col(r, "output_tokens", 0) or 0
            )

            # per-model rollup: prefer the detailed usage_by_model breakdown,
            # falling back to the run's headline model for older rows.
            breakdown = json.loads(_col(r, "usage_by_model", "[]") or "[]")
            if breakdown:
                for u in breakdown:
                    name = u.get("model") or r["model"] or "unknown"
                    m = by_model.setdefault(name, ModelAnalytics(model=name))
                    m.runs += 0  # counted once below
                    m.input_tokens += int(u.get("input_tokens", 0))
                    m.output_tokens += int(u.get("output_tokens", 0))
                    m.cost += float(u.get("cost", 0.0))
                for name in {
                    u.get("model") or r["model"] or "unknown" for u in breakdown
                }:
                    by_model[name].runs += 1
            else:
                name = r["model"] or "unknown"
                m = by_model.setdefault(name, ModelAnalytics(model=name))
                m.runs += 1
                m.input_tokens += int(_col(r, "input_tokens", 0) or 0)
                m.output_tokens += int(_col(r, "output_tokens", 0) or 0)
                m.cost += float(_col(r, "cost", 0.0) or 0.0)

        return RunAnalytics(
            total_runs=len(rows),
            ok_runs=ok,
            error_runs=len(rows) - ok,
            success_rate=ok / len(rows),
            total_cost=total_cost,
            total_input_tokens=int(total_in),
            total_output_tokens=int(total_out),
            avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
            p50_latency_ms=_pct(latencies, 0.50),
            p95_latency_ms=_pct(latencies, 0.95),
            feedback_up=feedback_up,
            feedback_down=feedback_down,
            by_model=sorted(by_model.values(), key=lambda m: m.cost, reverse=True),
            by_day=[by_day[k] for k in sorted(by_day)],
        )

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
            input_tokens=_col(row, "input_tokens", 0),
            output_tokens=_col(row, "output_tokens", 0),
            cost=_col(row, "cost", 0.0),
            feedback_verdict=_col(row, "feedback_verdict", None),
            feedback_note=_col(row, "feedback_note", None),
            feedback_at=_col(row, "feedback_at", None),
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
            timeline=[TimelineStep(**s) for s in json.loads(row["timeline"] or "[]")],
            steps=[
                CognitionStep(**s)
                for s in json.loads(
                    (row["steps"] if "steps" in row.keys() else None) or "[]"
                )
            ],
            input_tokens=_col(row, "input_tokens", 0),
            output_tokens=_col(row, "output_tokens", 0),
            cost=_col(row, "cost", 0.0),
            usage_by_model=[
                ModelUsage(**u)
                for u in json.loads(_col(row, "usage_by_model", "[]") or "[]")
            ],
            checkpoint_id=_col(row, "checkpoint_id", None),
            tool_count=len(json.loads(row["tools"] or "[]")),
            feedback_verdict=_col(row, "feedback_verdict", None),
            feedback_note=_col(row, "feedback_note", None),
            feedback_at=_col(row, "feedback_at", None),
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
    "CognitionStep",
    "ModelUsage",
    "RunAnalytics",
    "ModelAnalytics",
    "DayAnalytics",
    "TranscriptMessage",
    "StudioRunStore",
    "get_run_store",
    "reset_run_store",
    "studio_db_path",
]
