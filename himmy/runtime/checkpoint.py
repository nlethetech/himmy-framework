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
import sqlite3
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

#: Checkpoint lifecycle states.
AWAITING_APPROVAL = "awaiting_approval"
APPROVED = "approved"
REJECTED = "rejected"


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
    created_at: str = Field(default_factory=utc_now_iso)


@runtime_checkable
class CheckpointStore(Protocol):
    """Persists and retrieves :class:`AgentCheckpoint`s (save upserts by id)."""

    def save(self, checkpoint: AgentCheckpoint) -> None:
        """Insert or replace a checkpoint by its id."""
        ...

    def load(self, checkpoint_id: str) -> AgentCheckpoint | None:
        """Return a checkpoint by id, or None."""
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


class SqliteCheckpointStore:
    """A durable, file-backed :class:`CheckpointStore` (stdlib ``sqlite3``)."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path``."""
        self._conn = sqlite3.connect(path, check_same_thread=False)
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
        return AgentCheckpoint.model_validate(json.loads(row[0]))

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()


__all__ = [
    "AWAITING_APPROVAL",
    "APPROVED",
    "REJECTED",
    "PendingToolCall",
    "AgentCheckpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "SqliteCheckpointStore",
]
