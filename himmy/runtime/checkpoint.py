"""Runtime kernel: durable checkpoints for human-in-the-loop pause/resume.

When an agent loop hits a tool that requires approval, the runtime suspends the
run into an :class:`AgentCheckpoint` — the full thread, the persona/task/context
needed to resume, the loop's limits, and the exact pending tool call(s) awaiting a
human decision — and persists it via a :class:`CheckpointStore`. A later
``resume_agent_loop(checkpoint_id, approved=…)`` rehydrates it and either executes
the approved action or records the rejection, then continues.

Stores: :class:`InMemoryCheckpointStore` (default, volatile) and
:class:`SqliteCheckpointStore` (durable, stdlib ``sqlite3`` — survives restarts).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from himmy.core.errors import HimmyError
from himmy.core.ids import new_uuid, utc_now_iso

#: Checkpoint lifecycle states.
AWAITING_APPROVAL = "awaiting_approval"
#: Transient claim state: a resume that won the atomic compare-and-set flips
#: ``awaiting_approval`` -> ``resolving`` BEFORE executing any pending tool, so a
#: second concurrent resume (a double-clicked Approve, two tabs, two workers on the
#: shared SQLite file) loses the claim and is a no-op — the gated tool runs once.
RESOLVING = "resolving"
APPROVED = "approved"
REJECTED = "rejected"

#: Current :class:`AgentCheckpoint` serialization format. Bump this (and add a
#: step to ``_MIGRATIONS``) whenever a persisted field changes shape.
CHECKPOINT_SCHEMA_VERSION = 2


#: Arg-key substrings whose values are masked before a pending tool call is shown to a
#: human (so an approvals inbox never surfaces a credential). The canonical home for the
#: redaction, shared by every surface's approvals view (Studio + /v1).
_SECRETY_ARG_KEYS = ("token", "password", "secret", "key", "authorization", "auth")


def redact_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Mask values whose key looks secret, so an approvals UI never shows a credential.

    The single, surface-neutral redaction for HITL pending-tool views: both the Studio
    approvals inbox and the ``/v1`` ``GET /runs/{id}/pending-approvals`` endpoint reuse it,
    so the two surfaces redact identically.
    """
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        out[k] = "••••" if any(s in k.lower() for s in _SECRETY_ARG_KEYS) else v
    return out


class PendingToolCall(BaseModel):
    """A tool call the model made that is blocked awaiting human approval."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentCheckpoint(BaseModel):
    """A suspended agent run, persisted so a human decision can resume it.

    Carries everything needed to continue in a fresh process: the serialized
    persona/task/thread/llm_config, the loop limits, how many turns had run, and the
    pending tool call(s). ``status`` moves ``awaiting_approval`` -> ``approved`` /
    ``rejected`` exactly once.
    """

    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: str = Field(default_factory=new_uuid)
    status: str = AWAITING_APPROVAL
    persona: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)
    thread: dict[str, Any] = Field(default_factory=dict)
    ctx: dict[str, Any] = Field(default_factory=dict)
    llm_config: dict[str, Any] | None = None
    max_turns: int = 6
    cost_budget: float | None = None
    turns_completed: int = 0
    cost_completed: float = 0.0
    pending_tool_calls: list[PendingToolCall] = Field(default_factory=list)
    #: Idempotency ledger for HITL resume: serialized ``ToolExecutionResult`` dumps
    #: keyed by ``tool_call_id``, written (and persisted) immediately after each
    #: approved tool executes. A repeated resume — including a retry after a crash
    #: BETWEEN executing a state-mutating tool and resolving the checkpoint —
    #: replays the recorded result instead of executing the tool a second time.
    executed_tool_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


def _migrate_v0_to_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """v0 (pre-versioning) -> v1: same field shapes, only the version stamp is new."""
    return raw


def _migrate_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """v1 -> v2: adds ``executed_tool_results`` (empty — nothing had executed)."""
    raw.setdefault("executed_tool_results", {})
    return raw


#: Stepwise upgraders keyed by the *from* version; each returns the raw dict at
#: version ``from + 1``. Every historical format must have an unbroken chain here.
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    0: _migrate_v0_to_v1,
    1: _migrate_v1_to_v2,
}


def _migrate_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw checkpoint dict to :data:`CHECKPOINT_SCHEMA_VERSION` stepwise.

    A missing ``schema_version`` means a legacy (pre-versioning) checkpoint and is
    treated as version 0. A version *newer* than this build understands raises a
    clear :class:`~himmy.core.errors.HimmyError` instead of a confusing pydantic
    validation error.
    """
    version = raw.get("schema_version", 0)
    if not isinstance(version, int) or version < 0:
        raise HimmyError(
            f"Invalid checkpoint schema_version {version!r}; expected a "
            f"non-negative int <= {CHECKPOINT_SCHEMA_VERSION}."
        )
    if version > CHECKPOINT_SCHEMA_VERSION:
        raise HimmyError(
            f"Checkpoint schema_version {version} is newer than this himmy build "
            f"supports (max {CHECKPOINT_SCHEMA_VERSION}); upgrade himmy to resume it."
        )
    while version < CHECKPOINT_SCHEMA_VERSION:
        raw = _MIGRATIONS[version](raw)
        version += 1
        raw["schema_version"] = version
    return raw


@runtime_checkable
class CheckpointStore(Protocol):
    """Persists and retrieves :class:`AgentCheckpoint`s (save upserts by id)."""

    def save(self, checkpoint: AgentCheckpoint) -> None:
        """Insert or replace a checkpoint by its id."""
        ...

    def load(self, checkpoint_id: str) -> AgentCheckpoint | None:
        """Return a checkpoint by id, or None."""
        ...

    def claim(self, checkpoint_id: str) -> bool:
        """Atomically take the resume claim; True iff we won.

        The atomic compare-and-set behind HITL exactly-once: a checkpoint in
        ``awaiting_approval`` (a fresh resume) or ``resolving`` (a crash-retry
        re-entering after a prior resume died mid-flight) is claimable and flips to
        :data:`RESOLVING`; an ``approved``/``rejected`` (or unknown) checkpoint is
        not, returning ``False``. Concurrent callers serialize on this CAS so the
        gated tool's execution stays exactly-once (the per-tool idempotency ledger
        replays an already-executed call for the crash-retry case).
        """
        ...


class InMemoryCheckpointStore:
    """A volatile, process-local :class:`CheckpointStore` (the default)."""

    def __init__(self) -> None:
        self._store: dict[str, AgentCheckpoint] = {}

    def save(self, checkpoint: AgentCheckpoint) -> None:
        """Store the checkpoint (a deep copy, so later mutation can't leak in)."""
        self._store[checkpoint.checkpoint_id] = checkpoint.model_copy(deep=True)

    def load(self, checkpoint_id: str) -> AgentCheckpoint | None:
        """Return a deep copy of the stored checkpoint, or None."""
        found = self._store.get(checkpoint_id)
        return found.model_copy(deep=True) if found is not None else None

    def claim(self, checkpoint_id: str) -> bool:
        """Compare-and-set a claimable checkpoint -> ``resolving`` (True iff we won).

        Synchronous and process-local: the check-and-set runs to completion without
        an ``await`` in between, so concurrent resume coroutines on one event loop
        cannot interleave inside it — exactly one wins. Claimable means
        ``awaiting_approval`` (fresh) or ``resolving`` (crash-retry re-entry).
        """
        found = self._store.get(checkpoint_id)
        if found is None or found.status not in (AWAITING_APPROVAL, RESOLVING):
            return False
        self._store[checkpoint_id] = found.model_copy(update={"status": RESOLVING})
        return True


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_checkpoints_status_idx
    ON agent_checkpoints (status);
"""


#: Graph-run lifecycle states (used by :class:`GraphCheckpoint`).
GRAPH_RUNNING = "running"
GRAPH_COMPLETED = "completed"
GRAPH_INTERRUPTED = "interrupted"
GRAPH_FAILED = "failed"

#: Current :class:`GraphCheckpoint` serialization format. Bump this (and add a
#: step to ``_GRAPH_MIGRATIONS``) whenever a persisted field changes shape.
GRAPH_CHECKPOINT_SCHEMA_VERSION = 1


class GraphCheckpoint(BaseModel):
    """A durable snapshot of a :class:`~himmy.orchestrators.state_graph.StateGraph` run.

    Persisted after every completed superstep so a long graph run survives a
    process restart and resumes deterministically from where it stopped. Carries
    the accumulated shared ``state``, the ``frontier`` of node names scheduled for
    the next superstep, the superstep counter, and the visit counts used by the
    loop guard. ``graph_name`` ties the checkpoint to the graph definition that
    must be re-supplied on resume (the topology itself is code, not serialized).
    """

    schema_version: int = GRAPH_CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: str = Field(default_factory=new_uuid)
    graph_name: str = ""
    status: str = GRAPH_RUNNING
    state: dict[str, Any] = Field(default_factory=dict)
    frontier: list[str] = Field(default_factory=list)
    superstep: int = 0
    visit_counts: dict[str, int] = Field(default_factory=dict)
    completed_nodes: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


def _migrate_graph_v0_to_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """v0 (pre-versioning) -> v1: same field shapes, only the version stamp is new."""
    return raw


#: Stepwise upgraders keyed by the *from* version; each returns the raw dict at
#: version ``from + 1``. Every historical format must have an unbroken chain here.
_GRAPH_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    0: _migrate_graph_v0_to_v1,
}


def _migrate_graph_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw graph checkpoint dict to the current version stepwise.

    A missing ``schema_version`` means a legacy (pre-versioning) checkpoint and is
    treated as version 0. A version *newer* than this build understands raises a
    clear :class:`~himmy.core.errors.HimmyError` instead of a confusing pydantic
    validation error — the same contract as :func:`_migrate_checkpoint`.
    """
    version = raw.get("schema_version", 0)
    if not isinstance(version, int) or version < 0:
        raise HimmyError(
            f"Invalid graph checkpoint schema_version {version!r}; expected a "
            f"non-negative int <= {GRAPH_CHECKPOINT_SCHEMA_VERSION}."
        )
    if version > GRAPH_CHECKPOINT_SCHEMA_VERSION:
        raise HimmyError(
            f"Graph checkpoint schema_version {version} is newer than this himmy "
            f"build supports (max {GRAPH_CHECKPOINT_SCHEMA_VERSION}); upgrade himmy "
            f"to resume it."
        )
    while version < GRAPH_CHECKPOINT_SCHEMA_VERSION:
        raw = _GRAPH_MIGRATIONS[version](raw)
        version += 1
        raw["schema_version"] = version
    return raw


@runtime_checkable
class GraphCheckpointStore(Protocol):
    """Persists and retrieves :class:`GraphCheckpoint`s (save upserts by id)."""

    def save(self, checkpoint: GraphCheckpoint) -> None:
        """Insert or replace a graph checkpoint by its id."""
        ...

    def load(self, checkpoint_id: str) -> GraphCheckpoint | None:
        """Return a graph checkpoint by id, or None."""
        ...


class InMemoryGraphCheckpointStore:
    """A volatile, process-local :class:`GraphCheckpointStore` (the default)."""

    def __init__(self) -> None:
        self._store: dict[str, GraphCheckpoint] = {}

    def save(self, checkpoint: GraphCheckpoint) -> None:
        """Store the checkpoint (a deep copy, so later mutation can't leak in)."""
        self._store[checkpoint.checkpoint_id] = checkpoint.model_copy(deep=True)

    def load(self, checkpoint_id: str) -> GraphCheckpoint | None:
        """Return a deep copy of the stored checkpoint, or None."""
        found = self._store.get(checkpoint_id)
        return found.model_copy(deep=True) if found is not None else None


_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    graph_name    TEXT NOT NULL,
    status        TEXT NOT NULL,
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS graph_checkpoints_status_idx
    ON graph_checkpoints (status);
"""


class SqliteGraphCheckpointStore:
    """A durable, file-backed :class:`GraphCheckpointStore` (stdlib ``sqlite3``).

    Survives process restarts so a graph interrupted mid-run (a deadline, a
    crash, or a human-in-the-loop pause) can resume from its last completed
    superstep in a fresh process — the durable-resume guarantee.
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path``."""
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.executescript(_GRAPH_SCHEMA)
        self._conn.commit()

    def save(self, checkpoint: GraphCheckpoint) -> None:
        """Upsert the checkpoint as a JSON row keyed by id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO graph_checkpoints "
            "(checkpoint_id, graph_name, status, data, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                checkpoint.checkpoint_id,
                checkpoint.graph_name,
                checkpoint.status,
                checkpoint.model_dump_json(),
                checkpoint.created_at,
                checkpoint.updated_at,
            ),
        )
        self._conn.commit()

    def load(self, checkpoint_id: str) -> GraphCheckpoint | None:
        """Read + deserialize a graph checkpoint by id, or None."""
        row = self._conn.execute(
            "SELECT data FROM graph_checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            return None
        return GraphCheckpoint.model_validate(
            _migrate_graph_checkpoint(json.loads(row[0]))
        )

    def list_by_status(self, status: str) -> list[GraphCheckpoint]:
        """All graph checkpoints with the given status, newest first."""
        rows = self._conn.execute(
            "SELECT data FROM graph_checkpoints WHERE status = ? "
            "ORDER BY updated_at DESC",
            (status,),
        ).fetchall()
        return [
            GraphCheckpoint.model_validate(_migrate_graph_checkpoint(json.loads(r[0])))
            for r in rows
        ]

    def prune_terminal(self, *, older_than_days: float = 30.0) -> int:
        """Delete terminal (completed/failed) graph checkpoints older than the cutoff.

        Like :meth:`SqliteCheckpointStore.prune_resolved`, this is an explicit
        operator-invoked retention call. ONLY ``completed``/``failed`` rows are eligible;
        a ``running``/``interrupted`` checkpoint is resumable work and is never deleted.
        Returns the number of rows removed.
        """
        cutoff = (
            datetime.now(UTC) - timedelta(days=older_than_days)
        ).isoformat()
        try:
            cur = self._conn.execute(
                "DELETE FROM graph_checkpoints "
                "WHERE status IN (?, ?) AND updated_at < ?",
                (GRAPH_COMPLETED, GRAPH_FAILED, cutoff),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return cur.rowcount

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()


class SqliteCheckpointStore:
    """A durable, file-backed :class:`CheckpointStore` (stdlib ``sqlite3``)."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path``."""
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, checkpoint: AgentCheckpoint) -> None:
        """Upsert the checkpoint as a JSON row keyed by id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_checkpoints "
            "(checkpoint_id, status, data, created_at) VALUES (?, ?, ?, ?)",
            (
                checkpoint.checkpoint_id,
                checkpoint.status,
                checkpoint.model_dump_json(),
                checkpoint.created_at,
            ),
        )
        self._conn.commit()

    def load(self, checkpoint_id: str) -> AgentCheckpoint | None:
        """Read + deserialize a checkpoint by id, or None."""
        row = self._conn.execute(
            "SELECT data FROM agent_checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            return None
        return AgentCheckpoint.model_validate(_migrate_checkpoint(json.loads(row[0])))

    def claim(self, checkpoint_id: str) -> bool:
        """Atomically take the resume claim -> ``resolving`` (True iff we won).

        A single conditional UPDATE under ``BEGIN IMMEDIATE`` (which takes the write
        lock up front) so two concurrent resumes — even across processes sharing this
        file — serialize: the first matches a claimable status and flips it (rowcount
        1, the claim); a concurrent caller's UPDATE runs after the commit and, for a
        fresh resume that had been ``awaiting_approval``, now sees ``resolving`` —
        still claimable, so the cross-process loser is gated by the in-process
        per-checkpoint lock and, failing that, by the durable idempotency ledger
        (an already-executed tool is replayed, never re-run). Claimable =
        ``awaiting_approval`` (fresh) or ``resolving`` (crash-retry). The persisted
        ``data`` JSON's ``status`` is rewritten too so a later ``load`` is consistent
        with the ``status`` column.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute(
                "UPDATE agent_checkpoints "
                "SET status = ?, "
                "    data = json_set(data, '$.status', ?) "
                "WHERE checkpoint_id = ? AND status IN (?, ?)",
                (
                    RESOLVING,
                    RESOLVING,
                    checkpoint_id,
                    AWAITING_APPROVAL,
                    RESOLVING,
                ),
            )
            won = cur.rowcount == 1
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return won

    def list_by_status(self, status: str) -> list[AgentCheckpoint]:
        """All checkpoints with the given status, newest first (for an approvals UI)."""
        rows = self._conn.execute(
            "SELECT data FROM agent_checkpoints WHERE status = ? "
            "ORDER BY created_at DESC",
            (status,),
        ).fetchall()
        return [
            AgentCheckpoint.model_validate(_migrate_checkpoint(json.loads(r[0])))
            for r in rows
        ]

    def prune_resolved(self, *, older_than_days: float = 30.0) -> int:
        """Delete resolved (approved/rejected) checkpoints older than the cutoff.

        Resolved checkpoints carry the full serialized persona/task/thread/ctx per row
        and accumulate forever, so a long-lived approvals DB grows without bound. This
        explicit, operator-invoked call reclaims that space; nothing prunes
        automatically — callers must drive it from their own ops job (the bundled
        ``scripts/ops_prune.py`` does so for the default ``.himmy`` stores). ONLY terminal
        ``approved``/``rejected`` rows are eligible — an ``awaiting_approval``/
        ``resolving`` checkpoint is live work and is never deleted regardless of age.
        Returns the number of rows removed.
        """
        cutoff = (
            datetime.now(UTC) - timedelta(days=older_than_days)
        ).isoformat()
        try:
            cur = self._conn.execute(
                "DELETE FROM agent_checkpoints "
                "WHERE status IN (?, ?) AND created_at < ?",
                (APPROVED, REJECTED, cutoff),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return cur.rowcount

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()


__all__ = [
    "AWAITING_APPROVAL",
    "RESOLVING",
    "APPROVED",
    "REJECTED",
    "CHECKPOINT_SCHEMA_VERSION",
    "redact_tool_args",
    "PendingToolCall",
    "AgentCheckpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "SqliteCheckpointStore",
    "GRAPH_RUNNING",
    "GRAPH_COMPLETED",
    "GRAPH_INTERRUPTED",
    "GRAPH_FAILED",
    "GRAPH_CHECKPOINT_SCHEMA_VERSION",
    "GraphCheckpoint",
    "GraphCheckpointStore",
    "InMemoryGraphCheckpointStore",
    "SqliteGraphCheckpointStore",
]
