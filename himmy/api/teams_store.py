"""Self-contained stores for /v1 teams + workflows (T3b).

A **team** is an ordered set of STORED agent ids (the ``/v1/agents`` resource, T2e) plus
an orchestration ``kind`` (``multi_agent`` | ``group_chat`` | ``graph``). A **workflow** is
an ordered set of stored agent ids run as a fixed pipeline (each member's output feeds the
next), again workspace-scoped and referencing stored agents only.

Both are durable, workspace-scoped resources stored in their OWN SQLite files
(``.himmy/teams.db`` / ``.himmy/workflows.db``), mirroring the routines store
(``himmy/api/routines.py``) so they need no new migration in the main storage chain — the
schema is idempotent ``CREATE TABLE IF NOT EXISTS`` DDL resolved per-path, which works the
same offline (``.himmy/...``) and in a server process.

Two reviewer-grade properties are load-bearing here:

* **Referential integrity.** ``TeamsStore.find_by_agent_id`` / ``WorkflowsStore``'s
  equivalent list the team/workflow ids in a workspace that reference a stored
  ``agent_id``; these are wired into the :class:`AgentDefAppService` reference finder so
  ``DELETE /v1/agents/{id}`` of a still-referenced agent returns HTTP 409 rather than
  orphaning the binding (the production reference_finder the T2e delete-409 path needs).
* **Workspace scoping.** Every read/write is tenant-scoped on ``workspace_id`` so one
  tenant's team/workflow is invisible (404) to another in an authenticated deployment.

The member ``agent_id`` list stores ONLY ids — there is deliberately no ``agent_path``
field, exactly as ``/v1/routines`` refuses a tenant filesystem path (no FS-leak surface).
The exactly-which-agents-exist check (and same-workspace membership) is enforced at the
router/service boundary against the stored-agent store, not by a column constraint.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from himmy.core.ids import new_uuid, utc_now_iso
from himmy.services.storage.models import LOCAL_WORKSPACE

#: ``list`` is shadowed by the ``def list`` method in the store classes' scope; this alias
#: keeps the builtin generic available for type annotations on ``find_by_agent_id``.
_list = list

#: The orchestration kinds a team may declare. ``multi_agent`` is the handoff/delegation
#: orchestrator; ``group_chat`` is the selector-driven panel; ``graph`` is the durable
#: state-graph engine (long runs checkpoint to a :class:`SqliteGraphCheckpointStore`).
OrchestrationKind = Literal["multi_agent", "group_chat", "graph"]
_KINDS: frozenset[str] = frozenset({"multi_agent", "group_chat", "graph"})

#: How many members a team / workflow may bind (bounds the per-request fan-out).
MAX_MEMBERS = 24


class TeamRecord(BaseModel):
    """A workspace-scoped team: ordered stored ``agent_id`` members + an orchestration kind.

    ``members`` is the ordered list of stored ``agent_id`` references; the first is the
    orchestration ENTRY member (the agent that speaks first / the manager). ``kind`` picks
    the orchestrator. There is no ``agent_path`` field by design (no FS-leak surface).
    """

    id: str = Field(default_factory=new_uuid)
    workspace_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    kind: OrchestrationKind = "multi_agent"
    members: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @field_validator("members")
    @classmethod
    def _check_members(cls, value: list[str]) -> list[str]:
        """A team needs at least one member, at most :data:`MAX_MEMBERS`, all distinct."""
        if not value:
            raise ValueError("a team requires at least one member agent_id")
        if len(value) > MAX_MEMBERS:
            raise ValueError(f"a team may have at most {MAX_MEMBERS} members")
        if len(set(value)) != len(value):
            raise ValueError("team member agent_ids must be distinct")
        return value


class WorkflowRecord(BaseModel):
    """A workspace-scoped workflow: an ordered stored-agent pipeline.

    The members run in order — each member's text output is threaded as context into the
    next member's prompt — so a workflow is a deterministic multi-step pipeline (distinct
    from a team's dynamic handoff/selection). Same agent-id-only membership rule as a team.
    """

    id: str = Field(default_factory=new_uuid)
    workspace_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    members: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @field_validator("members")
    @classmethod
    def _check_members(cls, value: list[str]) -> list[str]:
        """A workflow needs at least one member, at most :data:`MAX_MEMBERS`, all distinct."""
        if not value:
            raise ValueError("a workflow requires at least one member agent_id")
        if len(value) > MAX_MEMBERS:
            raise ValueError(f"a workflow may have at most {MAX_MEMBERS} members")
        if len(set(value)) != len(value):
            raise ValueError("workflow member agent_ids must be distinct")
        return value


_TEAMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL DEFAULT '__local__',
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'multi_agent',
    members_json    TEXT NOT NULL DEFAULT '[]',
    idempotency_key TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS teams_workspace_id_idx ON teams (workspace_id);
"""

_WORKFLOWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL DEFAULT '__local__',
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    members_json    TEXT NOT NULL DEFAULT '[]',
    idempotency_key TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS workflows_workspace_id_idx ON workflows (workspace_id);
"""


def _row_to_team(r: sqlite3.Row) -> TeamRecord:
    return TeamRecord(
        id=r["id"],
        workspace_id=r["workspace_id"] or LOCAL_WORKSPACE,
        name=r["name"],
        description=r["description"] or "",
        kind=r["kind"],
        members=json.loads(r["members_json"] or "[]"),
        idempotency_key=r["idempotency_key"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def _row_to_workflow(r: sqlite3.Row) -> WorkflowRecord:
    return WorkflowRecord(
        id=r["id"],
        workspace_id=r["workspace_id"] or LOCAL_WORKSPACE,
        name=r["name"],
        description=r["description"] or "",
        members=json.loads(r["members_json"] or "[]"),
        idempotency_key=r["idempotency_key"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


class TeamsStore:
    """SQLite-backed team storage (same pattern as the routines/notes stores)."""

    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_TEAMS_SCHEMA)
        self._conn.commit()

    def list(self, *, workspace_id: str | None = None) -> list[TeamRecord]:
        """All teams newest-first; scoped to ``workspace_id`` when given."""
        if workspace_id is None:
            rows = self._conn.execute(
                "SELECT * FROM teams ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM teams WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        return [_row_to_team(r) for r in rows]

    def get(
        self, team_id: str, *, workspace_id: str | None = None
    ) -> TeamRecord | None:
        """Fetch one team by id; ``None`` when out-of-workspace (404) when scoped."""
        row = self._conn.execute(
            "SELECT * FROM teams WHERE id = ?", (team_id,)
        ).fetchone()
        if row is None:
            return None
        team = _row_to_team(row)
        if workspace_id is not None and team.workspace_id != workspace_id:
            return None
        return team

    def get_by_idempotency_key(
        self, key: str, *, workspace_id: str
    ) -> TeamRecord | None:
        """Return the team with ``idempotency_key`` in ``workspace_id`` (idempotent create)."""
        row = self._conn.execute(
            "SELECT * FROM teams WHERE idempotency_key = ? AND workspace_id = ?",
            (key, workspace_id),
        ).fetchone()
        return _row_to_team(row) if row is not None else None

    def upsert(self, team: TeamRecord) -> TeamRecord:
        team.updated_at = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO teams (
                id, workspace_id, name, description, kind, members_json,
                idempotency_key, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                workspace_id=excluded.workspace_id, name=excluded.name,
                description=excluded.description, kind=excluded.kind,
                members_json=excluded.members_json,
                idempotency_key=excluded.idempotency_key,
                updated_at=excluded.updated_at
            """,
            (
                team.id,
                team.workspace_id,
                team.name,
                team.description,
                team.kind,
                json.dumps(list(team.members)),
                team.idempotency_key,
                team.created_at,
                team.updated_at,
            ),
        )
        self._conn.commit()
        return team

    def delete(self, team_id: str, *, workspace_id: str | None = None) -> bool:
        """Delete a team, tenant-scoped. Returns ``True`` iff a row was removed."""
        if workspace_id is None:
            cur = self._conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))
        else:
            cur = self._conn.execute(
                "DELETE FROM teams WHERE id = ? AND workspace_id = ?",
                (team_id, workspace_id),
            )
        self._conn.commit()
        return cur.rowcount > 0

    def find_by_agent_id(self, agent_id: str, *, workspace_id: str) -> _list[str]:
        """Team ids in ``workspace_id`` that reference stored ``agent_id`` (T2e ref-finder).

        A team stores its members as a JSON array, so the membership test is done in
        Python over the workspace's teams (the row count is small + bounded).
        """
        rows = self._conn.execute(
            "SELECT id, members_json FROM teams WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchall()
        out: _list[str] = []
        for r in rows:
            try:
                members = json.loads(r["members_json"] or "[]")
            except (ValueError, TypeError):  # pragma: no cover - defensive
                members = []
            if agent_id in members:
                out.append(r["id"])
        return out

    def close(self) -> None:
        self._conn.close()


class WorkflowsStore:
    """SQLite-backed workflow storage (same pattern as :class:`TeamsStore`)."""

    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_WORKFLOWS_SCHEMA)
        self._conn.commit()

    def list(self, *, workspace_id: str | None = None) -> list[WorkflowRecord]:
        """All workflows newest-first; scoped to ``workspace_id`` when given."""
        if workspace_id is None:
            rows = self._conn.execute(
                "SELECT * FROM workflows ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM workflows WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        return [_row_to_workflow(r) for r in rows]

    def get(
        self, workflow_id: str, *, workspace_id: str | None = None
    ) -> WorkflowRecord | None:
        """Fetch one workflow by id; ``None`` when out-of-workspace (404) when scoped."""
        row = self._conn.execute(
            "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            return None
        workflow = _row_to_workflow(row)
        if workspace_id is not None and workflow.workspace_id != workspace_id:
            return None
        return workflow

    def get_by_idempotency_key(
        self, key: str, *, workspace_id: str
    ) -> WorkflowRecord | None:
        """Return the workflow with ``idempotency_key`` in ``workspace_id``."""
        row = self._conn.execute(
            "SELECT * FROM workflows WHERE idempotency_key = ? AND workspace_id = ?",
            (key, workspace_id),
        ).fetchone()
        return _row_to_workflow(row) if row is not None else None

    def upsert(self, workflow: WorkflowRecord) -> WorkflowRecord:
        workflow.updated_at = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO workflows (
                id, workspace_id, name, description, members_json,
                idempotency_key, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                workspace_id=excluded.workspace_id, name=excluded.name,
                description=excluded.description,
                members_json=excluded.members_json,
                idempotency_key=excluded.idempotency_key,
                updated_at=excluded.updated_at
            """,
            (
                workflow.id,
                workflow.workspace_id,
                workflow.name,
                workflow.description,
                json.dumps(list(workflow.members)),
                workflow.idempotency_key,
                workflow.created_at,
                workflow.updated_at,
            ),
        )
        self._conn.commit()
        return workflow

    def delete(self, workflow_id: str, *, workspace_id: str | None = None) -> bool:
        """Delete a workflow, tenant-scoped. Returns ``True`` iff a row was removed."""
        if workspace_id is None:
            cur = self._conn.execute(
                "DELETE FROM workflows WHERE id = ?", (workflow_id,)
            )
        else:
            cur = self._conn.execute(
                "DELETE FROM workflows WHERE id = ? AND workspace_id = ?",
                (workflow_id, workspace_id),
            )
        self._conn.commit()
        return cur.rowcount > 0

    def find_by_agent_id(self, agent_id: str, *, workspace_id: str) -> _list[str]:
        """Workflow ids in ``workspace_id`` that reference stored ``agent_id`` (ref-finder)."""
        rows = self._conn.execute(
            "SELECT id, members_json FROM workflows WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchall()
        out: _list[str] = []
        for r in rows:
            try:
                members = json.loads(r["members_json"] or "[]")
            except (ValueError, TypeError):  # pragma: no cover - defensive
                members = []
            if agent_id in members:
                out.append(r["id"])
        return out

    def close(self) -> None:
        self._conn.close()


# ---- process-wide singletons (path-keyed, like the routines store) ----------

_TEAMS_STORE: TeamsStore | None = None
_TEAMS_PATH: str | None = None
_WORKFLOWS_STORE: WorkflowsStore | None = None
_WORKFLOWS_PATH: str | None = None


def teams_db_path() -> str:
    """The teams DB path: ``HIMMY_TEAMS_PATH`` override else ``.himmy/teams.db``."""
    env = os.environ.get("HIMMY_TEAMS_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "teams.db")


def workflows_db_path() -> str:
    """The workflows DB path: ``HIMMY_WORKFLOWS_PATH`` override else ``.himmy/workflows.db``."""
    env = os.environ.get("HIMMY_WORKFLOWS_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "workflows.db")


def get_teams_store() -> TeamsStore:
    """Resolve the process-wide teams store, rebuilding it if the path changed."""
    global _TEAMS_STORE, _TEAMS_PATH
    path = teams_db_path()
    if _TEAMS_STORE is None or _TEAMS_PATH != path:
        if _TEAMS_STORE is not None:
            _TEAMS_STORE.close()
        _TEAMS_STORE = TeamsStore(path)
        _TEAMS_PATH = path
    return _TEAMS_STORE


def get_workflows_store() -> WorkflowsStore:
    """Resolve the process-wide workflows store, rebuilding it if the path changed."""
    global _WORKFLOWS_STORE, _WORKFLOWS_PATH
    path = workflows_db_path()
    if _WORKFLOWS_STORE is None or _WORKFLOWS_PATH != path:
        if _WORKFLOWS_STORE is not None:
            _WORKFLOWS_STORE.close()
        _WORKFLOWS_STORE = WorkflowsStore(path)
        _WORKFLOWS_PATH = path
    return _WORKFLOWS_STORE


def reset_teams_store() -> None:
    """Drop the cached teams store (tests + a path/env change)."""
    global _TEAMS_STORE, _TEAMS_PATH
    if _TEAMS_STORE is not None:
        _TEAMS_STORE.close()
    _TEAMS_STORE = None
    _TEAMS_PATH = None


def reset_workflows_store() -> None:
    """Drop the cached workflows store (tests + a path/env change)."""
    global _WORKFLOWS_STORE, _WORKFLOWS_PATH
    if _WORKFLOWS_STORE is not None:
        _WORKFLOWS_STORE.close()
    _WORKFLOWS_STORE = None
    _WORKFLOWS_PATH = None


# ---- durable graph-checkpoint store for long team/workflow runs (T3b) -------

_GRAPH_CP_STORE: object | None = None
_GRAPH_CP_PATH: str | None = None


def get_graph_checkpoint_store() -> object:
    """Resolve the durable graph-checkpoint store: file-backed in a server, RAM offline.

    In a server context (the lifespan-established durable deployment) this is a
    file-backed :class:`SqliteGraphCheckpointStore` at ``.himmy/graph_checkpoints.db``, so a
    long ``graph`` team / workflow run survives a restart and resumes from its last
    completed superstep (the T3b durable-resume acceptance). Offline (the zero-config
    default) it is an in-memory store so no file is written. Path-keyed so an env override
    rebuilds it.
    """
    global _GRAPH_CP_STORE, _GRAPH_CP_PATH
    from himmy.runtime.checkpoint import (
        InMemoryGraphCheckpointStore,
        SqliteGraphCheckpointStore,
    )
    from himmy.services.storage.factory import in_server_context

    if not in_server_context():
        if not isinstance(_GRAPH_CP_STORE, InMemoryGraphCheckpointStore):
            _GRAPH_CP_STORE = InMemoryGraphCheckpointStore()
            _GRAPH_CP_PATH = None
        return _GRAPH_CP_STORE

    from himmy.config.project import graph_checkpoints_db_path

    path = graph_checkpoints_db_path()
    if not isinstance(_GRAPH_CP_STORE, SqliteGraphCheckpointStore) or _GRAPH_CP_PATH != path:
        _GRAPH_CP_STORE = SqliteGraphCheckpointStore(path)
        _GRAPH_CP_PATH = path
    return _GRAPH_CP_STORE


def reset_graph_checkpoint_store() -> None:
    """Drop the cached graph-checkpoint store (tests + a path/env change)."""
    global _GRAPH_CP_STORE, _GRAPH_CP_PATH
    _GRAPH_CP_STORE = None
    _GRAPH_CP_PATH = None


__all__ = [
    "OrchestrationKind",
    "MAX_MEMBERS",
    "TeamRecord",
    "WorkflowRecord",
    "TeamsStore",
    "WorkflowsStore",
    "teams_db_path",
    "workflows_db_path",
    "get_teams_store",
    "get_workflows_store",
    "reset_teams_store",
    "reset_workflows_store",
    "get_graph_checkpoint_store",
    "reset_graph_checkpoint_store",
]
