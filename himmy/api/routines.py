"""Studio Routines: scheduled agent runs — the store + the scheduler.

A routine is a saved prompt against a saved agent that runs on a simple
schedule (``daily at HH:MM`` or ``every N hours``) without anyone asking.
Runs go through the exact same pipeline as an interactive Studio run
(:func:`himmy.api.studio_service.stream_agent_run`), so they land in the runs
store, show in Activity, and obey the same safety machinery.

Unattended safety rails (honest list of what the machinery does):

* **Approval-gated tools are never executed.** The run pipeline executes with
  ``hitl=True``: when a tool requires approval the loop PAUSES on a durable
  checkpoint (the Approvals inbox) instead of running it. The routine records
  ``awaiting_approval`` and a notification points the user at Approvals — the
  scheduler never auto-approves.
* **Wall-clock timeout** (default 5 minutes, ``HIMMY_ROUTINE_TIMEOUT_S``):
  a stuck run is cancelled and recorded as ``timeout``.
* **No overlapping execution** of the same routine (in-process task registry).
* **One failure never kills the loop** — every tick and every run is guarded.

Storage mirrors the other Studio stores: SQLite at ``.himmy/routines.db``,
cwd-keyed singleton (see :mod:`himmy.api.studio_tasks`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from himmy.core.ids import new_uuid, utc_now_iso
from himmy.services.storage.models import LOCAL_WORKSPACE

logger = logging.getLogger("himmy.api.routines")

#: Builtin ``list`` aliased so annotations on :class:`RoutinesStore` methods that follow
#: the ``def list`` method (which shadows the builtin name in class scope) still resolve to
#: the builtin generic for the type checker — e.g. ``find_by_agent_id -> _list[str]``.
_list = list


class _Unconditional:
    """Sentinel: ``mark_started`` should stamp the start WITHOUT a ``last_run_at`` guard.

    Distinct from a real ``last_run_at`` sentinel (a string OR ``None`` for a never-run
    routine): the conditional claim is the TICK path's anti-double-fire guard, but the
    manual ``run_now`` path must always fire (overlap is prevented by the host flock), so it
    passes :data:`UNCONDITIONAL` to skip the predicate.
    """


#: The "no ``last_run_at`` guard" marker for the manual ``run_now`` start stamp.
UNCONDITIONAL = _Unconditional()

#: A captured-sentinel type: either a real ``last_run_at`` value (str / ``None``) to gate
#: the atomic claim on, or :data:`UNCONDITIONAL` to skip the gate.
_StartAnchor = "str | None | _Unconditional"

# ---- schedule grammar -----------------------------------------------------

_AT_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

DeliverKind = Literal["none", "telegram", "email"]

#: Delivered result text is hard-capped at this many characters (Telegram's
#: message limit is 4096; email keeps the same bound for symmetry).
DELIVERY_MAX_CHARS = 4000
#: Stored preview of the last result, shown in the GUI list.
PREVIEW_MAX_CHARS = 400
#: Default per-run wall-clock budget for an unattended run.
DEFAULT_RUN_TIMEOUT_S = 300.0


class Schedule(BaseModel):
    """When a routine runs. Deliberately simple — no cron strings.

    * ``{"kind": "daily", "at": "HH:MM"}`` — once a day at that local time.
    * ``{"kind": "every", "hours": N}`` — every N hours (1..168).
    """

    kind: Literal["daily", "every"]
    at: str | None = Field(default=None, max_length=5)
    hours: int | None = Field(default=None, ge=1, le=168)

    @model_validator(mode="after")
    def _check_fields(self) -> Schedule:
        if self.kind == "daily":
            if not self.at or not _AT_RE.match(self.at):
                raise ValueError("daily schedule needs at='HH:MM' (24-hour)")
            self.hours = None
        else:  # every
            if self.hours is None:
                raise ValueError("every schedule needs hours (1..168)")
            self.at = None
        return self

    def describe(self) -> str:
        """A short human string ('daily 07:00' / 'every 6h')."""
        if self.kind == "daily":
            return f"daily {self.at}"
        return f"every {self.hours}h"


class Routine(BaseModel):
    """One scheduled routine, as stored.

    A routine binds to its agent through EXACTLY ONE of two mutually-exclusive seams:

    * ``agent_path`` — a project-relative ``agent.yaml`` on the SERVER filesystem. This
      is the single-user-local seam the Studio routines screen and ``himmy routines``
      drive (the local operator owns the filesystem). Runs execute headless through the
      Studio run pipeline and land in the ``__local__`` workspace of the canonical store.
    * ``agent_id`` — a stored, workspace-scoped :class:`AgentDefRecord` (T2e). This is
      the multi-tenant ``/v1/routines`` seam: a tenant references a stored agent by id,
      NEVER a filesystem path (a path would leak the server FS — reviewer must_fix). Runs
      execute through ``RunAppService.create_run`` under the routine's ``workspace_id``.

    ``workspace_id`` is the canonical tenant scope. It defaults to ``__local__`` so an
    existing single-user routine (created before T3c) and the CLI/Studio surfaces all
    share one scope; ``/v1`` stamps the authenticated tenant's workspace.
    """

    id: str = Field(default_factory=new_uuid)
    name: str
    workspace_id: str = LOCAL_WORKSPACE
    agent_path: str | None = None
    agent_id: str | None = None
    prompt: str
    schedule: Schedule
    provider: str | None = None
    model: str | None = None
    deliver: DeliverKind = "none"
    enabled: bool = True
    idempotency_key: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    last_run_at: str | None = None
    last_status: str | None = None  # running | ok | error | timeout | awaiting_approval
    last_preview: str = ""
    last_error: str | None = None
    last_delivery: str | None = None  # delivery failure note; None = ok / not asked

    @model_validator(mode="after")
    def _check_agent_binding(self) -> Routine:
        """Exactly one of ``agent_path`` / ``agent_id`` must identify the agent."""
        if bool(self.agent_path) == bool(self.agent_id):
            raise ValueError(
                "a routine needs exactly one of agent_path (single-user-local) or "
                "agent_id (workspace-scoped stored agent)"
            )
        return self


def _parse_iso(value: str) -> datetime | None:
    """Parse a stored ISO timestamp; naive values are assumed UTC."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        from datetime import UTC

        dt = dt.replace(tzinfo=UTC)
    return dt


def is_due(routine: Routine, now: datetime) -> bool:
    """Pure due-math: should ``routine`` fire at aware-datetime ``now``?

    The anchor is the last run (or creation, so a fresh routine never fires
    retroactively). ``daily``: due when the most recent HH:MM occurrence (in
    ``now``'s timezone) is after the anchor. ``every``: due when N hours have
    passed since the anchor.
    """
    if not routine.enabled:
        return False
    anchor = _parse_iso(routine.last_run_at or routine.created_at)
    if anchor is None:
        return False
    sched = routine.schedule
    if sched.kind == "daily":
        m = _AT_RE.match(sched.at or "")
        if not m:
            return False
        occurrence = now.replace(
            hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0
        )
        if occurrence > now:
            occurrence -= timedelta(days=1)
        return anchor < occurrence
    hours = sched.hours or 0
    if hours <= 0:
        return False
    return now >= anchor + timedelta(hours=hours)


# ---- the store --------------------------------------------------------------

#: The current (T3c) routines-table shape. ``agent_path`` is NULLABLE because a
#: workspace-scoped ``/v1`` routine binds by ``agent_id`` instead (and vice-versa); the
#: exactly-one-of invariant is enforced in the :class:`Routine` model, not by a column
#: constraint. ``workspace_id`` defaults to ``__local__`` so a routine created before
#: T3c (which had neither column) upgrades into the local scope. A fresh database lays
#: this down directly; a legacy database is migrated to it by :meth:`RoutinesStore._migrate`.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS routines (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    workspace_id   TEXT NOT NULL DEFAULT '__local__',
    agent_path     TEXT,
    agent_id       TEXT,
    idempotency_key TEXT,
    prompt         TEXT NOT NULL,
    schedule_kind  TEXT NOT NULL,
    schedule_at    TEXT,
    schedule_hours INTEGER,
    provider       TEXT,
    model          TEXT,
    deliver        TEXT NOT NULL DEFAULT 'none',
    enabled        INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    last_run_at    TEXT,
    last_status    TEXT,
    last_preview   TEXT NOT NULL DEFAULT '',
    last_error     TEXT,
    last_delivery  TEXT
);
"""


def _row_to_routine(r: sqlite3.Row) -> Routine:
    keys = r.keys()
    return Routine(
        id=r["id"],
        name=r["name"],
        workspace_id=(r["workspace_id"] if "workspace_id" in keys else None)
        or LOCAL_WORKSPACE,
        agent_path=r["agent_path"],
        agent_id=(r["agent_id"] if "agent_id" in keys else None),
        idempotency_key=(
            r["idempotency_key"] if "idempotency_key" in keys else None
        ),
        prompt=r["prompt"],
        schedule=Schedule(
            kind=r["schedule_kind"],
            at=r["schedule_at"],
            hours=r["schedule_hours"],
        ),
        provider=r["provider"],
        model=r["model"],
        deliver=r["deliver"],
        enabled=bool(r["enabled"]),
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        last_run_at=r["last_run_at"],
        last_status=r["last_status"],
        last_preview=r["last_preview"] or "",
        last_error=r["last_error"],
        last_delivery=r["last_delivery"],
    )


class RoutinesStore:
    """SQLite-backed routine storage (same pattern as the Tasks/Notes stores)."""

    def __init__(self, path: str = ":memory:") -> None:
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Forward-migrate a legacy (pre-T3c) routines table to the current shape.

        A database created before T3c lacks the ``workspace_id``/``agent_id``/
        ``idempotency_key`` columns AND carries a ``NOT NULL`` on ``agent_path`` — which a
        plain ``ALTER ADD COLUMN`` cannot drop, yet an ``agent_id``-bound ``/v1`` routine
        legitimately has a NULL ``agent_path``. So when the legacy ``NOT NULL`` is present
        we REBUILD the table into the nullable shape (copy rows → drop → rename); otherwise
        we additively add any missing columns. The introspection is idempotent, so the
        method converges to the current schema on every open regardless of starting point.
        """
        info = self._conn.execute("PRAGMA table_info(routines)").fetchall()
        cols = {row["name"] for row in info}
        # ``notnull`` is 1 for a NOT NULL column. A legacy table has agent_path NOT NULL;
        # the current shape has it nullable — that flag distinguishes the two and cannot be
        # changed by ALTER, so a rebuild is required to relax it.
        agent_path_notnull = any(
            row["name"] == "agent_path" and row["notnull"] for row in info
        )
        if agent_path_notnull:
            self._rebuild_legacy_table()
        else:
            if "workspace_id" not in cols:
                self._conn.execute(
                    "ALTER TABLE routines ADD COLUMN workspace_id TEXT "
                    "NOT NULL DEFAULT '__local__'"
                )
            if "agent_id" not in cols:
                self._conn.execute("ALTER TABLE routines ADD COLUMN agent_id TEXT")
            if "idempotency_key" not in cols:
                self._conn.execute(
                    "ALTER TABLE routines ADD COLUMN idempotency_key TEXT"
                )
        # Create the workspace index AFTER the column is guaranteed to exist (a fresh
        # database has it from _SCHEMA; a legacy one just got it above). Idempotent.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS routines_workspace_id_idx "
            "ON routines (workspace_id)"
        )

    def _rebuild_legacy_table(self) -> None:
        """Rebuild the legacy routines table into the nullable-``agent_path`` shape.

        Copies every legacy row into a fresh table that matches the current ``_SCHEMA``
        (stamping ``workspace_id='__local__'`` and leaving ``agent_id``/``idempotency_key``
        NULL), then atomically swaps it in. Wrapped in a transaction so a failure leaves the
        original table intact.
        """
        create_new = _SCHEMA.replace(
            "CREATE TABLE IF NOT EXISTS routines", "CREATE TABLE routines_new"
        )
        with self._conn:  # one transaction; commits on success, rolls back on error
            self._conn.execute(create_new)
            self._conn.execute(
                "INSERT INTO routines_new ("
                "id, name, workspace_id, agent_path, agent_id, idempotency_key, prompt, "
                "schedule_kind, schedule_at, schedule_hours, provider, model, deliver, "
                "enabled, created_at, updated_at, last_run_at, last_status, last_preview, "
                "last_error, last_delivery) "
                "SELECT id, name, '__local__', agent_path, NULL, NULL, prompt, "
                "schedule_kind, schedule_at, schedule_hours, provider, model, deliver, "
                "enabled, created_at, updated_at, last_run_at, last_status, last_preview, "
                "last_error, last_delivery FROM routines"
            )
            self._conn.execute("DROP TABLE routines")
            self._conn.execute("ALTER TABLE routines_new RENAME TO routines")

    def list(self, *, workspace_id: str | None = None) -> list[Routine]:
        """All routines newest-first; scoped to ``workspace_id`` when given.

        The scheduler tick lists across ALL workspaces (``workspace_id=None``) so one
        process fires every tenant's due routines; ``/v1`` always passes a concrete
        workspace so one tenant never sees another's routines.
        """
        if workspace_id is None:
            rows = self._conn.execute(
                "SELECT * FROM routines ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM routines WHERE workspace_id = ? "
                "ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        return [_row_to_routine(r) for r in rows]

    def get(
        self, routine_id: str, *, workspace_id: str | None = None
    ) -> Routine | None:
        """Fetch one routine by id; ``None`` when out-of-workspace (404) when scoped."""
        row = self._conn.execute(
            "SELECT * FROM routines WHERE id = ?", (routine_id,)
        ).fetchone()
        if row is None:
            return None
        routine = _row_to_routine(row)
        if workspace_id is not None and routine.workspace_id != workspace_id:
            return None
        return routine

    def upsert(self, routine: Routine) -> Routine:
        routine.updated_at = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO routines (
                id, name, workspace_id, agent_path, agent_id, idempotency_key,
                prompt, schedule_kind, schedule_at, schedule_hours, provider, model,
                deliver, enabled, created_at, updated_at, last_run_at, last_status,
                last_preview, last_error, last_delivery
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, workspace_id=excluded.workspace_id,
                agent_path=excluded.agent_path, agent_id=excluded.agent_id,
                idempotency_key=excluded.idempotency_key,
                prompt=excluded.prompt, schedule_kind=excluded.schedule_kind,
                schedule_at=excluded.schedule_at,
                schedule_hours=excluded.schedule_hours,
                provider=excluded.provider, model=excluded.model,
                deliver=excluded.deliver, enabled=excluded.enabled,
                updated_at=excluded.updated_at,
                last_run_at=excluded.last_run_at,
                last_status=excluded.last_status,
                last_preview=excluded.last_preview,
                last_error=excluded.last_error,
                last_delivery=excluded.last_delivery
            """,
            (
                routine.id,
                routine.name,
                routine.workspace_id,
                routine.agent_path,
                routine.agent_id,
                routine.idempotency_key,
                routine.prompt,
                routine.schedule.kind,
                routine.schedule.at,
                routine.schedule.hours,
                routine.provider,
                routine.model,
                routine.deliver,
                int(routine.enabled),
                routine.created_at,
                routine.updated_at,
                routine.last_run_at,
                routine.last_status,
                routine.last_preview,
                routine.last_error,
                routine.last_delivery,
            ),
        )
        self._conn.commit()
        return routine

    def delete(self, routine_id: str, *, workspace_id: str | None = None) -> bool:
        """Delete a routine, tenant-scoped. Returns ``True`` iff a row was removed."""
        if workspace_id is None:
            cur = self._conn.execute(
                "DELETE FROM routines WHERE id = ?", (routine_id,)
            )
        else:
            cur = self._conn.execute(
                "DELETE FROM routines WHERE id = ? AND workspace_id = ?",
                (routine_id, workspace_id),
            )
        self._conn.commit()
        return cur.rowcount > 0

    def find_by_agent_id(self, agent_id: str, *, workspace_id: str) -> _list[str]:
        """Routine ids in ``workspace_id`` that reference stored ``agent_id`` (T2e ref-finder).

        Used as the :class:`AgentDefAppService` reference finder so deleting a stored
        agent still referenced by a routine returns HTTP 409 rather than orphaning it.
        """
        rows = self._conn.execute(
            "SELECT id FROM routines WHERE agent_id = ? AND workspace_id = ?",
            (agent_id, workspace_id),
        ).fetchall()
        return [r["id"] for r in rows]

    def mark_started(
        self,
        routine_id: str,
        started_at: str,
        *,
        expected_last_run_at: Any = UNCONDITIONAL,
    ) -> bool:
        """Stamp the run start — also the due-math anchor, so a slow run can't
        re-trigger itself on the next tick. Returns ``True`` iff the start was claimed.

        ``expected_last_run_at`` is the value read at due-evaluation in
        :meth:`RoutineScheduler.tick`: passing it makes the stamp a CONDITIONAL UPDATE
        (``WHERE last_run_at IS ?`` — ``IS`` matches the NULL first-run case) so a routine
        due on every tick is claimed exactly once even across processes — the same contract
        the Postgres mirror gives across replicas (the SQLite box ALSO keeps the host
        flock). The manual ``run_now`` path passes :data:`UNCONDITIONAL` (the default) to
        always fire — overlap there is prevented by the flock, not the anchor.
        """
        if isinstance(expected_last_run_at, _Unconditional):
            cur = self._conn.execute(
                "UPDATE routines SET last_run_at = ?, last_status = 'running',"
                " last_error = NULL, last_delivery = NULL WHERE id = ?",
                (started_at, routine_id),
            )
        else:
            cur = self._conn.execute(
                "UPDATE routines SET last_run_at = ?, last_status = 'running',"
                " last_error = NULL, last_delivery = NULL"
                " WHERE id = ? AND last_run_at IS ?",
                (started_at, routine_id, expected_last_run_at),
            )
        self._conn.commit()
        return cur.rowcount == 1

    def record_result(
        self,
        routine_id: str,
        *,
        status: str,
        preview: str,
        error: str | None = None,
        delivery: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE routines SET last_status = ?, last_preview = ?,"
            " last_error = ?, last_delivery = ? WHERE id = ?",
            (status, preview[:PREVIEW_MAX_CHARS], error, delivery, routine_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


#: ``Any`` because under a Postgres DSN this is the K4 :class:`PostgresRoutinesStore`
#: (whose ``mark_started`` is the atomic cluster-wide claim) rather than the SQLite store.
_STORE: Any | None = None
_PATH: str | None = None


def routines_db_path() -> str:
    env = os.environ.get("HIMMY_ROUTINES_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "routines.db")


def get_routines_store() -> Any:
    """Resolve the process-wide routines store.

    Under a Postgres DSN this is the K4 Postgres mirror (``"local"`` tenant) whose
    ``mark_started`` is the atomic cluster-wide claim (a routine due on every tick fires
    exactly once across N replicas); offline the durable SQLite file store byte-for-byte,
    guarded by the host flock.
    """
    global _STORE, _PATH
    path = routines_db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        # K4: route through the one aux-store selector — the Postgres mirror is the
        # ``"local"``-tenant routines store; the SQLite builder is the offline default.
        from himmy.services.storage.aux_store_factory import select_aux_store

        def _pg() -> Any:
            from himmy.services.storage.postgres_aux import PostgresRoutinesStore

            return PostgresRoutinesStore(tenant="local")

        _STORE = select_aux_store(lambda: RoutinesStore(path), _pg)
        _PATH = path
    return _STORE


def reset_routines_store() -> None:
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None


# ---- workspace-scoped (/v1) execution seam -----------------------------------

#: Process-wide provider of the wired :class:`ApiContainer` (T3c). The app factory sets
#: this to read ``app.state.container`` so a workspace-scoped (``agent_id``) routine run
#: dispatched by the scheduler executes through the SAME ``RunAppService`` the request-
#: driven ``/v1`` surface uses — landing the run in the tenant's workspace of the
#: canonical store, under the T0.4 per-workspace quota, with the HITL/approval rails.
#: Unset outside a server process (a plain CLI single-user-local routine never needs it).
_CONTAINER_PROVIDER: Callable[[], Any] | None = None


def set_routine_container_provider(provider: Callable[[], Any] | None) -> None:
    """Install (or clear) the process-wide app-container provider for /v1 routines (T3c)."""
    global _CONTAINER_PROVIDER
    _CONTAINER_PROVIDER = provider


def resolve_routine_container() -> Any | None:
    """Resolve the wired app container, or ``None`` when not in a server process."""
    provider = _CONTAINER_PROVIDER
    if provider is None:
        return None
    try:
        return provider()
    except Exception:  # noqa: BLE001 - a bad provider must never break a run
        logger.debug("routine container provider failed", exc_info=True)
        return None


# ---- headless execution ------------------------------------------------------


def _cap(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def run_timeout_s() -> float:
    """Per-run wall-clock budget (seconds), bounded 10s..1h."""
    raw = os.environ.get("HIMMY_ROUTINE_TIMEOUT_S", "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_RUN_TIMEOUT_S
    return min(max(value, 10.0), 3600.0)


async def _run_headless(routine: Routine) -> tuple[str, str, str | None]:
    """Run one routine unattended; returns ``(status, output_text, error)``.

    Two execution seams, picked by how the routine binds its agent:

    * ``agent_id`` (workspace-scoped ``/v1`` routine) → :func:`_run_headless_agent_id`,
      which dispatches through ``RunAppService.create_run`` under the routine's
      ``workspace_id`` with the HITL/approval rails, landing the run in the TENANT's
      workspace of the canonical store (the right place for a multi-tenant run).
    * ``agent_path`` (single-user-local Studio/CLI routine) → the Studio
      ``stream_agent_run`` pipeline (below), landing in the ``__local__`` workspace.

    In both seams an approval-gated tool PAUSES the run (status ``awaiting_approval``)
    rather than executing it; the scheduler never auto-approves.
    """
    if routine.agent_id:
        return await _run_headless_agent_id(routine)

    from himmy.api import studio_service
    from himmy.api.studio_canonical import resolve_canonical_storage

    # The model invariant guarantees exactly one of agent_path/agent_id; the agent_id
    # branch returned above, so agent_path is set here.
    agent_path = routine.agent_path or ""
    canonical = resolve_canonical_storage()
    try:
        spec = studio_service.load_studio_spec(
            agent_path, provider=routine.provider, model=routine.model
        )
    except FileNotFoundError as exc:
        return "error", "", str(exc)
    except ValueError as exc:
        return "error", "", str(exc)

    output = ""
    status = "ok"
    error: str | None = None

    # Stamp the routine actor + lineage source onto the canonical run record so a
    # scheduled / run-now run is attributable to its routine in GET /v1/runs +
    # ``himmy runs`` + Studio (the durable-bridge contract). The agent_id seam carries
    # the same actor through ``create_run``; this is its agent_path equivalent.
    routine_metadata = {
        "source": "routine",
        "actor": {"source": "routine", "routine_id": routine.id},
        "routine_id": routine.id,
    }

    async def _drain() -> None:
        nonlocal output, status, error
        async for event in studio_service.stream_agent_run(
            spec,
            routine.prompt,
            provider=routine.provider,
            model=routine.model,
            agent_path=agent_path,
            canonical_storage=canonical,
            extra_metadata=routine_metadata,
        ):
            kind = event.get("type")
            if kind == "message":
                output = str(event.get("text") or output)
            elif kind == "done":
                output = str(event.get("output_text") or output)
            elif kind == "paused":
                status = "awaiting_approval"
            elif kind == "error":
                status = "error"
                error = str(event.get("message") or "run failed")

    try:
        await asyncio.wait_for(_drain(), timeout=run_timeout_s())
    except TimeoutError:
        status = "timeout"
        error = f"run exceeded {run_timeout_s():.0f}s and was cancelled"
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken run must not kill the loop
        status = "error"
        error = str(exc)
    return status, output, error


#: Canonical statuses that a polled unattended ``/v1`` run has SETTLED on — the scheduler
#: stops waiting once a dispatched run reaches one (``AWAITING_APPROVAL`` is terminal for an
#: unattended run: it paused on a gated tool and is NOT auto-approved).
_SETTLED_RUN_STATUSES = {"SUCCEEDED", "FAILED", "AWAITING_APPROVAL"}

#: Canonical → routine status string, mirroring the Studio mapping so a routine's
#: ``last_status`` reads the same vocabulary regardless of which seam ran it.
_RUN_STATUS_TO_ROUTINE = {
    "SUCCEEDED": "ok",
    "FAILED": "error",
    "AWAITING_APPROVAL": "awaiting_approval",
}


async def _run_headless_agent_id(routine: Routine) -> tuple[str, str, str | None]:
    """Dispatch a workspace-scoped (``agent_id``) routine through ``RunAppService`` (T3c).

    Resolves the wired app container, loads the routine's stored agent in its OWN
    ``workspace_id`` (404→error when missing/out-of-workspace), then launches the run via
    ``create_run`` with ``hitl=True`` so an approval-gated tool PAUSES at
    ``AWAITING_APPROVAL`` instead of firing unattended. ``create_run`` returns a QUEUED
    record and executes on a background task, so we poll the canonical run until it settles
    (or the wall-clock budget elapses). The run lands in the tenant's workspace of the ONE
    canonical store — visible in ``GET /v1/runs`` AND ``himmy runs`` AND Studio.
    """
    container = resolve_routine_container()
    if container is None:
        return (
            "error",
            "",
            "workspace-scoped routines require a running server (no app container)",
        )
    run_app = getattr(container, "run_app", None)
    agent_app = getattr(container, "agent_app", None)
    if run_app is None or agent_app is None:
        return "error", "", "app container is missing the run/agent services"

    from himmy.agents.base_agent.task import Task

    workspace_id = routine.workspace_id
    try:
        agent_def = await agent_app.get_agent_def(
            routine.agent_id, workspace_id=workspace_id
        )
        if agent_def is None:
            return "error", "", f"stored agent {routine.agent_id!r} not found"
        agent_spec = agent_def.agent_spec()
        persona = agent_spec.to_persona()
        llm_config = agent_spec.to_llm_config()
    except Exception as exc:  # noqa: BLE001 - a bad spec must not kill the loop
        return "error", "", f"could not resolve stored agent: {exc}"

    task = Task(title=routine.name, prompt=routine.prompt, context={})
    # HITL only when the stored agent actually builds a tool registry to gate; a tool-less
    # agent has nothing to pause on, so dispatching it with hitl=True is rejected by
    # create_run (HitlRequiresAgentError). Run it plainly in that case — it cannot reach a
    # gated tool anyway, so the unattended-safety contract is preserved.
    hitl = bool(agent_spec.builds_tool_registry())
    actor = {"source": "routine", "routine_id": routine.id}
    try:
        run = await run_app.create_run(
            workspace_id=workspace_id,
            subject_id=workspace_id,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            agent_def=agent_def,
            hitl=hitl,
            actor=actor,
        )
    except Exception as exc:  # noqa: BLE001 - admission/precondition failures
        return "error", "", str(exc)

    return await _await_run_settled(run_app, run.run_id, workspace_id)


async def _await_run_settled(
    run_app: Any, run_id: str, workspace_id: str
) -> tuple[str, str, str | None]:
    """Poll a dispatched ``/v1`` run until it settles or the wall-clock budget elapses."""
    deadline = run_timeout_s()
    poll = 0.1
    waited = 0.0
    while True:
        run = await run_app.get_run(run_id, workspace_id=workspace_id)
        if run is None:  # vanished (deleted/erased) — treat as an error, never hang
            return "error", "", "run record disappeared while waiting"
        status_value = run.status.value
        if status_value in _SETTLED_RUN_STATUSES:
            mapped = _RUN_STATUS_TO_ROUTINE.get(status_value, "ok")
            output = str(run.output_text or "")
            error = run.error if mapped == "error" else None
            return mapped, output, error
        if waited >= deadline:
            return (
                "timeout",
                "",
                f"run exceeded {deadline:.0f}s and was left running in the background",
            )
        await asyncio.sleep(poll)
        waited += poll
        poll = min(poll * 1.5, 2.0)


async def _deliver(routine: Routine, output: str) -> str | None:
    """Send the result through a configured connection.

    Returns ``None`` on success or a short failure note recorded on the routine —
    a delivery failure is never raised.
    """
    from himmy.api import studio_connections as conns

    try:
        status = conns.get_connection(routine.deliver)
        if status is None or not status.configured:
            return f"{routine.deliver} not configured — delivery skipped"
        text = _cap(f"{routine.name}\n\n{output}", DELIVERY_MAX_CHARS)
        if routine.deliver == "telegram":
            result = await conns.send_via_connection("telegram", {"text": text})
        else:  # email — to self (the SMTP account that sends is the owner)
            from himmy.toolkit.config import ToolkitConfig

            cfg = ToolkitConfig.from_env()
            to = cfg.smtp_user or cfg.smtp_from or ""
            if "@" not in to:
                return "email has no recipient (set the SMTP username) — skipped"
            result = await conns.send_via_connection(
                "email",
                {
                    "to": to,
                    "subject": f"Routine: {routine.name}",
                    "body": _cap(output, DELIVERY_MAX_CHARS),
                },
            )
        if not result.ok:
            return f"{routine.deliver} delivery failed: {result.detail}"
        return None
    except Exception as exc:  # noqa: BLE001 - delivery is best-effort by contract
        return f"{routine.deliver} delivery failed: {exc}"


def _notify(routine: Routine, status: str, preview: str, error: str | None) -> None:
    """Record the lifecycle notification (never raises, per the contract)."""
    from himmy.api.routers.studio_notify import record_notification

    if status == "ok":
        record_notification(
            "routine",
            f"Routine ran: {routine.name}",
            body=preview,
            link="/routines",
        )
    elif status == "awaiting_approval":
        record_notification(
            "routine",
            f"Routine needs approval: {routine.name}",
            body="An approval-gated tool paused the run — review it in Approvals.",
            link="/approvals",
        )
    else:  # error | timeout
        record_notification(
            "routine",
            f"Routine failed: {routine.name}",
            body=error or status,
            link="/routines",
        )


def routine_lock_name(routine_id: str) -> str:
    """The cross-process flock name guarding a single routine's execution (T3c)."""
    return f"routine-{routine_id}"


async def execute_routine(
    routine_id: str,
    *,
    now: Callable[[], datetime] | None = None,
    expected_last_run_at: Any = UNCONDITIONAL,
) -> Routine | None:
    """Run one routine end-to-end: run → deliver → record → notify.

    Returns the refreshed routine, or ``None`` when it no longer exists.

    Cross-process single-flight (T3c reviewer must_fix): the in-process
    :class:`RoutineScheduler` registry stops same-process overlap, but a ``himmy
    routines run-now`` runs in a SEPARATE process that cannot see that registry — so
    without a host-level guard the CLI and the Studio scheduler could fire the SAME
    routine simultaneously and double-run its gated tools / deliveries. A non-blocking
    :func:`process_lock` keyed ``routine-<id>`` is held for the whole execution; a
    concurrent attempt anywhere on the host raises :class:`RoutineBusyError` (mapped to
    a 409 / a CLI "already running" message) rather than executing twice.

    Cluster-wide single-flight (K4 reviewer must_fix): on the Postgres mirror the host
    flock does NOT span replicas, so ``expected_last_run_at`` — the ``last_run_at`` captured
    at due-evaluation in :meth:`RoutineScheduler.tick` — gates an ATOMIC start claim in
    ``mark_started``. If another replica already claimed this tick (the conditional UPDATE
    matched 0 rows), this call returns the routine unchanged WITHOUT running. ``run_now``
    passes :data:`UNCONDITIONAL` so a manual trigger always fires.
    """
    from himmy.core.process_lock import ProcessLockBusy, process_lock

    store = get_routines_store()
    routine = store.get(routine_id)
    if routine is None:
        return None
    try:
        with process_lock(routine_lock_name(routine_id)):
            return await _execute_locked(
                routine_id,
                routine,
                now=now,
                expected_last_run_at=expected_last_run_at,
            )
    except ProcessLockBusy as exc:
        raise RoutineBusyError(routine_id) from exc


async def _execute_locked(
    routine_id: str,
    routine: Routine,
    *,
    now: Callable[[], datetime] | None = None,
    expected_last_run_at: Any = UNCONDITIONAL,
) -> Routine | None:
    """The guarded body of :func:`execute_routine` (runs while the flock is held).

    The start stamp is the atomic claim: when ``expected_last_run_at`` is a captured
    sentinel (the tick path) and the conditional ``mark_started`` matches 0 rows, a peer
    replica already won this tick — return the routine WITHOUT running so the gated tools /
    deliveries fire exactly once cluster-wide.
    """
    store = get_routines_store()
    now_fn = now or _default_now
    claimed = store.mark_started(
        routine_id,
        now_fn().isoformat(),
        expected_last_run_at=expected_last_run_at,
    )
    if not claimed:
        # A peer replica won this tick's atomic claim — do not double-run.
        peer_won: Routine | None = store.get(routine_id)
        return peer_won
    try:
        status, output, error = await _run_headless(routine)
    except asyncio.CancelledError:
        store.record_result(
            routine_id,
            status="error",
            preview="",
            error="cancelled (Studio shut down mid-run)",
        )
        raise
    delivery: str | None = None
    if status == "ok" and routine.deliver in ("telegram", "email"):
        delivery = await _deliver(routine, output)
    preview = _cap(output or error or "", PREVIEW_MAX_CHARS)
    store.record_result(
        routine_id, status=status, preview=preview, error=error, delivery=delivery
    )
    _notify(routine, status, preview, error)
    refreshed: Routine | None = store.get(routine_id)
    return refreshed


# ---- the scheduler -----------------------------------------------------------


def _default_now() -> datetime:
    """Aware local time — 'daily at 07:00' means the user's 7am."""
    return datetime.now().astimezone()


class RoutineBusyError(RuntimeError):
    """The routine is already executing (overlap is refused, not queued)."""


class RoutineScheduler:
    """The asyncio loop that fires due routines. One per process.

    ``now`` is injectable so due-math and tick behavior are testable with a
    frozen clock. ``start``/``stop`` are wired to the router's startup/shutdown
    events (merged into the app lifespan by FastAPI's ``include_router``).
    """

    def __init__(
        self,
        *,
        tick_seconds: float = 30.0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._tick_seconds = tick_seconds
        self._now = now or _default_now
        self._loop_task: asyncio.Task[None] | None = None
        self._running: dict[str, asyncio.Task[Any]] = {}

    # -- lifecycle ------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True while the background tick loop is running."""
        return self._loop_task is not None and not self._loop_task.done()

    def start(self) -> None:
        """Start the tick loop (idempotent)."""
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(
                self._loop(), name="himmy-routines-scheduler"
            )

    async def stop(self) -> None:
        """Cancel the loop and any in-flight routine runs; await them."""
        tasks = [t for t in self._running.values() if not t.done()]
        if self._loop_task is not None:
            tasks.append(self._loop_task)
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._loop_task = None
        self._running.clear()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._tick_seconds)
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad tick never kills the loop
                logger.exception("routine scheduler tick failed")

    # -- ticking ----------------------------------------------------------------

    def is_running(self, routine_id: str) -> bool:
        task = self._running.get(routine_id)
        return task is not None and not task.done()

    def tick(self) -> list[str]:
        """Launch every due, enabled, not-already-running routine.

        Returns the launched routine ids (for tests/observability).
        """
        now = self._now()
        try:
            routines = get_routines_store().list()
        except Exception:  # noqa: BLE001 - a broken store must not kill the loop
            logger.exception("routine store unavailable during tick")
            return []
        launched: list[str] = []
        for routine in routines:
            if self.is_running(routine.id):
                continue
            if not is_due(routine, now):
                continue
            # Capture last_run_at AT due-evaluation (the SAME value is_due anchored on) and
            # thread it into the atomic start claim, so the cluster-wide conditional UPDATE
            # gates on the value this tick observed — not a fresh re-read that a peer replica
            # may already have moved (K4 reviewer must_fix).
            self._launch(routine.id, routine.last_run_at)
            launched.append(routine.id)
        return launched

    def _launch(
        self, routine_id: str, expected_last_run_at: Any
    ) -> asyncio.Task[Routine | None]:
        """Fire-and-forget launch for the tick loop — failures are swallowed.

        The tick path never awaits the returned task, so a genuine run failure (or
        cross-process flock contention) must not escape: :func:`_guarded_execute`
        funnels everything to a logged ``None``. ``run_now`` does NOT use this path
        precisely because it must let :class:`RoutineBusyError` surface (a 409).
        ``expected_last_run_at`` is the due-time sentinel for the atomic cluster-wide claim.
        """
        task = asyncio.create_task(
            self._guarded_execute(routine_id, expected_last_run_at),
            name=f"himmy-routine-{routine_id}",
        )
        self._track(routine_id, task)
        return task

    def _track(self, routine_id: str, task: asyncio.Task[Routine | None]) -> None:
        """Register an in-flight run so same-process :meth:`is_running` sees it."""
        self._running[routine_id] = task
        task.add_done_callback(lambda _t: self._running.pop(routine_id, None))

    async def _guarded_execute(
        self, routine_id: str, expected_last_run_at: Any
    ) -> Routine | None:
        try:
            return await execute_routine(
                routine_id,
                now=self._now,
                expected_last_run_at=expected_last_run_at,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one failed run never kills the scheduler
            logger.exception("routine %s failed", routine_id)
            return None

    # -- manual trigger -----------------------------------------------------------

    async def run_now(self, routine_id: str) -> Routine | None:
        """Run a routine immediately through the same rails; await the result.

        Raises :class:`RoutineBusyError` when the routine is already executing —
        whether in THIS process (the same-process pre-check) or in another process
        on the host (the cross-process flock inside :func:`execute_routine`). Unlike
        the tick path, ``run_now`` is awaited and surfaces a 409, so it must NOT route
        through :func:`_guarded_execute` (which swallows busy as a logged ``None``).
        Genuine run failures still settle into the routine's ``last_status`` rather
        than raising — only contention propagates.
        """
        if self.is_running(routine_id):
            raise RoutineBusyError(routine_id)
        # Track in _running (so a concurrent same-process run-now/tick sees it) but
        # await a task that lets RoutineBusyError out instead of eating it.
        task: asyncio.Task[Routine | None] = asyncio.create_task(
            execute_routine(routine_id, now=self._now),
            name=f"himmy-routine-run-now-{routine_id}",
        )
        self._track(routine_id, task)
        return await task


_SCHEDULER: RoutineScheduler | None = None


def get_scheduler() -> RoutineScheduler:
    global _SCHEDULER
    if _SCHEDULER is None:
        _SCHEDULER = RoutineScheduler()
    return _SCHEDULER


def reset_scheduler() -> None:
    """Drop the singleton (tests). The caller must have stopped it first."""
    global _SCHEDULER
    _SCHEDULER = None


__all__ = [
    "DELIVERY_MAX_CHARS",
    "LOCAL_WORKSPACE",
    "PREVIEW_MAX_CHARS",
    "Routine",
    "RoutineBusyError",
    "RoutineScheduler",
    "RoutinesStore",
    "Schedule",
    "execute_routine",
    "get_routines_store",
    "get_scheduler",
    "is_due",
    "reset_routines_store",
    "reset_scheduler",
    "resolve_routine_container",
    "routine_lock_name",
    "routines_db_path",
    "run_timeout_s",
    "set_routine_container_provider",
]
