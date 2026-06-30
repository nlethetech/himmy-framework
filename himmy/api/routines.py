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
import threading
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


#: Missed-run / catch-up policy (Phase 1). Read the doc on each member; ``coalesce`` is the
#: laptop-sleep-safe default (fire ONCE now if anything was missed).
MissedRunPolicy = Literal["coalesce", "skip", "run_each"]
#: Overlap policy (Phase 1). Only ``skip`` exists today (a routine never runs concurrently
#: with itself — the lane/flock already serialises it); the field is here for forward room.
OverlapPolicy = Literal["skip"]

#: Defaults for the ``run_each`` catch-up backfill (a storm cap + a staleness floor).
DEFAULT_MAX_CATCHUP = 100
DEFAULT_GRACE_SECONDS = 900

#: T3 per-tenant QUOTAS (additive, fail-open). A real (non-``__local__``) tenant may hold at
#: most this many ENABLED routines and backfill at most this much catch-up per routine; both
#: protect the SHARED queue from one tenant flooding it. ``0`` disables a cap. The reserved
#: ``__local__`` single-user workspace is always exempt, so the offline/single-tenant path is
#: unaffected. Operators override via ``HIMMY_TENANT_MAX_ROUTINES`` /
#: ``HIMMY_TENANT_MAX_CATCHUP``. Defaults are generous so a normal multi-tenant load is never
#: clipped — the cap only bites a runaway.
DEFAULT_TENANT_MAX_ROUTINES = 200
DEFAULT_TENANT_MAX_CATCHUP = 100


class TenantRoutineQuotaExceeded(Exception):
    """A workspace exceeded its per-tenant routine quota at create (T3 -> HTTP 429).

    Raised by the routine-create path when a tenant already holds its maximum number of
    ENABLED routines. Carries the workspace + the cap so the router returns an informative
    429 instead of letting one tenant flood the shared scheduler/queue with routines.
    """

    def __init__(self, workspace_id: str, *, cap: int, count: int) -> None:
        """Record the workspace and the routine cap it hit for the API error surface."""
        self.workspace_id = workspace_id
        self.cap = cap
        self.count = count
        super().__init__(
            f"workspace {workspace_id!r} already has {count} enabled routine(s); "
            f"the per-tenant cap is {cap}. Delete or disable a routine first."
        )


def tenant_max_routines() -> int:
    """The per-tenant ENABLED-routine cap (``HIMMY_TENANT_MAX_ROUTINES``; 0 = unlimited)."""
    raw = os.environ.get("HIMMY_TENANT_MAX_ROUTINES")
    if raw is None or not raw.strip():
        return DEFAULT_TENANT_MAX_ROUTINES
    try:
        value = int(raw.strip())
    except ValueError:
        return DEFAULT_TENANT_MAX_ROUTINES
    return value if value >= 0 else DEFAULT_TENANT_MAX_ROUTINES


def tenant_max_catchup() -> int:
    """The per-tenant per-routine catch-up ceiling (``HIMMY_TENANT_MAX_CATCHUP``; 0 = unlimited).

    A tenant-level upper bound on a routine's own ``schedule.max_catchup`` so a tenant cannot
    set an arbitrarily large backfill that storms the shared queue on launch. Clamps DOWN; the
    routine keeps its own (possibly smaller) value.
    """
    raw = os.environ.get("HIMMY_TENANT_MAX_CATCHUP")
    if raw is None or not raw.strip():
        return DEFAULT_TENANT_MAX_CATCHUP
    try:
        value = int(raw.strip())
    except ValueError:
        return DEFAULT_TENANT_MAX_CATCHUP
    return value if value >= 0 else DEFAULT_TENANT_MAX_CATCHUP


def enforce_tenant_routine_quota(
    store: RoutinesStore, *, workspace_id: str, is_new: bool
) -> None:
    """Reject a routine create that pushes a tenant past its ENABLED-routine cap (T3).

    Fail-open + additive: exempts the reserved ``__local__`` single-user workspace and a
    zero/disabled cap, so single-tenant + existing behaviour is unchanged. ``is_new`` is True
    for a create (the new routine would ADD to the count) and False for an update of an
    existing row (no net add — an update never trips the cap). A count error never blocks the
    write (the cap is a guard, not a correctness invariant).

    .. note::
       This is the legacy SOFT check (an unlocked ``count()`` round-trip) — it has a TOCTOU
       window against the later ``upsert`` under concurrent same-workspace creates. The
       create path now uses :func:`create_routine_with_quota`, which fuses the count and the
       insert into ONE atomic store op (advisory-locked on Postgres, ``BEGIN IMMEDIATE`` on
       SQLite) and is HARD under concurrency. This function is retained for back-compat /
       callers that only want the pre-check; prefer the atomic helper for creates.
    """
    if not is_new or workspace_id == LOCAL_WORKSPACE:
        return
    cap = tenant_max_routines()
    if cap <= 0:
        return
    try:
        current = store.count(workspace_id=workspace_id, enabled_only=True)
    except Exception:  # noqa: BLE001 - fail OPEN
        logger.warning(
            "tenant routine-quota count failed for workspace %s; admitting", workspace_id
        )
        return
    if current >= cap:
        raise TenantRoutineQuotaExceeded(workspace_id, cap=cap, count=current)


def create_routine_with_quota(
    store: RoutinesStore, routine: Routine, *, workspace_id: str
) -> Routine:
    """Create ``routine``, enforcing the per-tenant routine cap ATOMICALLY (T3, HARD).

    The hard replacement for the soft ``enforce_tenant_routine_quota`` + later ``upsert``
    pair: that split (count, then insert in a separate round-trip with no lock between)
    overshoots the cap under concurrent same-workspace creates. This routes the create through
    the store's atomic ``create_routine_if_under_quota`` — count and insert fused into ONE
    advisory-locked transaction (Postgres) / ``BEGIN IMMEDIATE`` (SQLite) — so concurrent
    creates serialise and land EXACTLY at the cap.

    The exempt paths short-circuit to a plain :meth:`upsert` BEFORE the atomic op so the
    single-tenant path is byte-identical (no lock taken): the reserved ``__local__``
    workspace and a zero/disabled cap. Raises :class:`TenantRoutineQuotaExceeded` (→ HTTP
    429) on a clean over-cap; FAIL-OPEN on an infra error happens INSIDE the store op (it
    falls back to a plain upsert and admits), so an infra hiccup never blocks a legitimate
    create.
    """
    if workspace_id == LOCAL_WORKSPACE:
        return store.upsert(routine)
    cap = tenant_max_routines()
    if cap <= 0:
        return store.upsert(routine)
    stored, admitted = store.create_routine_if_under_quota(routine, cap=cap)
    if not admitted:
        # ``create_routine_if_under_quota`` did NOT write the row; report the cap as the
        # count (the tenant is AT the cap). The exact pre-insert count is not surfaced —
        # the 429 message only needs the cap and that the tenant is at it.
        raise TenantRoutineQuotaExceeded(workspace_id, cap=cap, count=cap)
    return stored


def clamp_tenant_catchup(routine: Routine) -> Routine:
    """Clamp a routine's ``schedule.max_catchup`` DOWN to the tenant ceiling (T3, in place).

    Exempts ``__local__`` and a zero ceiling. A routine asking for more backfill than the
    tenant ceiling is silently clamped so one tenant's launch cannot storm the shared queue;
    a routine within the ceiling is untouched.
    """
    if routine.workspace_id == LOCAL_WORKSPACE:
        return routine
    ceiling = tenant_max_catchup()
    if ceiling <= 0:
        return routine
    if routine.schedule.max_catchup > ceiling:
        routine.schedule.max_catchup = ceiling
    return routine


class Schedule(BaseModel):
    """When a routine runs.

    The original two kinds are UNCHANGED (byte-identical behaviour):

    * ``{"kind": "daily", "at": "HH:MM"}`` — once a day at that local time.
    * ``{"kind": "every", "hours": N}`` — every N hours (1..168).

    Phase 1 adds two wall-clock-anchored kinds plus per-routine scheduling policy:

    * ``{"kind": "cron", "expr": "30 6 * * *"}`` — a 5-field cron expression, computed in
      the routine's ``timezone`` (needs the optional ``cronsim`` dependency). Cron is the
      right choice for a wall-clock-anchored cadence (e.g. 'every day at 06:30 Kathmandu')
      — unlike ``every`` it does not drift against the wall clock.
    * ``{"kind": "at", "at_datetime": "<ISO>"}`` — a ONE-SHOT run at a single instant; once
      fired it never fires again.

    ``timezone`` (IANA, e.g. ``Asia/Kathmandu``) governs the wall-clock kinds (``daily`` /
    ``cron`` / ``at``-with-naive-instant). Unset → env ``HIMMY_TZ`` → host-local. ``missed``
    / ``max_catchup`` / ``grace_seconds`` drive catch-up on launch; ``overlap`` is reserved.

    BACK-COMPAT: every new field has a default, and the ``daily`` / ``every`` validation +
    semantics are unchanged, so an existing routine (and its stored row / JSON blob)
    round-trips and behaves exactly as before.
    """

    kind: Literal["daily", "every", "cron", "at"]
    at: str | None = Field(default=None, max_length=5)
    hours: int | None = Field(default=None, ge=1, le=168)
    # -- Phase 1 additive fields (all defaulted; daily/every ignore them) --------
    expr: str | None = Field(default=None, max_length=200)
    at_datetime: str | None = Field(default=None, max_length=64)
    timezone: str | None = Field(default=None, max_length=64)
    missed: MissedRunPolicy = "coalesce"
    max_catchup: int = Field(default=DEFAULT_MAX_CATCHUP, ge=1, le=100_000)
    grace_seconds: int = Field(default=DEFAULT_GRACE_SECONDS, ge=0)
    overlap: OverlapPolicy = "skip"

    @model_validator(mode="after")
    def _check_fields(self) -> Schedule:
        if self.kind == "daily":
            if not self.at or not _AT_RE.match(self.at):
                raise ValueError("daily schedule needs at='HH:MM' (24-hour)")
            self.hours = None
        elif self.kind == "every":
            if self.hours is None:
                raise ValueError("every schedule needs hours (1..168)")
            self.at = None
        elif self.kind == "cron":
            from himmy.api.routine_schedule import validate_cron

            if not self.expr or not self.expr.strip():
                raise ValueError("cron schedule needs a non-empty expr (e.g. '30 6 * * *')")
            validate_cron(self.expr)
            self.at = None
            self.hours = None
        else:  # at — one-shot
            if not self.at_datetime or not self.at_datetime.strip():
                raise ValueError("at schedule needs at_datetime (an ISO-8601 instant)")
            if _parse_iso(self.at_datetime) is None:
                raise ValueError(
                    f"at_datetime {self.at_datetime!r} is not a valid ISO-8601 datetime"
                )
            self.at = None
            self.hours = None
        if self.timezone is not None:
            from himmy.api.routine_schedule import validate_timezone

            validate_timezone(self.timezone)
        return self

    def describe(self) -> str:
        """A short human string ('daily 07:00' / 'every 6h' / 'cron 30 6 * * *')."""
        if self.kind == "daily":
            return f"daily {self.at}"
        if self.kind == "every":
            return f"every {self.hours}h"
        if self.kind == "cron":
            return f"cron {self.expr}"
        return f"at {self.at_datetime}"


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
    #: The next planned fire instant, persisted as a UTC ISO string (Phase 1). Computed in
    #: the routine's timezone then normalised to UTC. ``None`` until first computed (lazily
    #: on the first tick) or for a daily/every routine that has not opted into the column.
    next_fire_at: str | None = None
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
    passed since the anchor. ``cron`` / ``at``: due when the planned fire (the next
    occurrence STRICTLY after the anchor, computed in the routine's timezone) is at-or-
    before ``now`` (see :func:`planned_fire_at`). daily/every math is unchanged.
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
    if sched.kind == "every":
        hours = sched.hours or 0
        if hours <= 0:
            return False
        return now >= anchor + timedelta(hours=hours)
    # cron / at — tz-aware: due when the planned fire after the anchor has arrived.
    planned = planned_fire_at(routine, anchor)
    if planned is None:
        return False
    return now >= planned


def planned_fire_at(routine: Routine, anchor: datetime) -> datetime | None:
    """The next planned fire instant (UTC, aware) STRICTLY after ``anchor`` for cron/at.

    For ``cron`` this is the next cron occurrence in the routine's timezone, normalised to
    UTC — the *planned_fire_at* the dedup key + ``create_run`` idempotency key are built on
    (so concurrent ticks/boots coalesce to one fire, and the key is the planned instant, not
    ``now()``). For a one-shot ``at`` it is the configured instant if it is still after the
    anchor (i.e. not yet fired), else ``None`` (already fired — never again). Returns
    ``None`` for daily/every (they do not use this seam) or when ``cronsim`` is unavailable.
    """
    sched = routine.schedule
    if sched.kind == "cron":
        from himmy.api.routine_schedule import (
            CronUnavailableError,
            cron_next_fire,
            resolve_zone,
        )

        if not sched.expr:
            return None
        try:
            return cron_next_fire(sched.expr, anchor, resolve_zone(sched.timezone))
        except CronUnavailableError:
            logger.warning(
                "routine %s is a cron routine but cronsim is not installed; "
                "install himmy[cron]",
                routine.id,
            )
            return None
    if sched.kind == "at":
        instant = _parse_iso(sched.at_datetime or "")
        if instant is None:
            return None
        instant = _to_utc(instant)
        # One-shot: due only while the configured instant is still strictly after the
        # anchor (a successful fire stamps last_run_at >= instant, so it never re-fires).
        return instant if instant > anchor else None
    return None


def _to_utc(dt: datetime) -> datetime:
    """Normalise an aware datetime to UTC (naive is assumed UTC)."""
    if dt.tzinfo is None:
        from datetime import UTC

        return dt.replace(tzinfo=UTC)
    from datetime import UTC

    return dt.astimezone(UTC)


def next_fire(routine: Routine, now: datetime) -> datetime | None:
    """The earliest instant ``t >= now`` at which ``routine`` becomes due, or ``None``.

    This is the *sleep-to-next-fire* seam. It is the temporal dual of :func:`is_due`:

    * If the routine is **already due** at ``now`` (``is_due(routine, now)`` is true) the
      answer is ``now`` itself — the scheduler should fire on the very next wake.
    * Otherwise it is the next future boundary the routine will cross.
    * ``None`` means *never* (disabled, malformed, a one-shot ``at`` that already fired, or
      a cron expression we cannot evaluate because ``cronsim`` is absent).

    The per-kind math mirrors :func:`is_due` exactly so there is no drift between "is it due
    now?" and "when is it due next?" — daily/every/cron/at all reuse the same anchors and the
    same timezone resolution. This function is *advisory*: oversleeping past a boundary is
    harmless because :func:`is_due` stays true on the next wake and catch-up covers misses.
    The returned datetime preserves ``now``'s timezone for the already-due case and is UTC-
    aware otherwise (the caller only ever takes ``(t - now).total_seconds()``).
    """
    if not routine.enabled:
        return None
    anchor = _parse_iso(routine.last_run_at or routine.created_at)
    if anchor is None:
        return None
    # Already past a boundary -> fire on the next tick (do not sleep past it).
    if is_due(routine, now):
        return now
    sched = routine.schedule
    if sched.kind == "daily":
        m = _AT_RE.match(sched.at or "")
        if not m:
            return None
        # The next HH:MM occurrence in ``now``'s wall clock that is strictly after ``now``.
        occurrence = now.replace(
            hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0
        )
        if occurrence <= now:
            occurrence += timedelta(days=1)
        return occurrence
    if sched.kind == "every":
        hours = sched.hours or 0
        if hours <= 0:
            return None
        # Not-yet-due (is_due was False above) -> the boundary is strictly ahead.
        return anchor + timedelta(hours=hours)
    # cron / at — the next planned instant after the anchor (UTC-aware) or None.
    return planned_fire_at(routine, anchor)


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
    schedule_expr  TEXT,
    schedule_at_datetime TEXT,
    timezone       TEXT,
    missed_policy  TEXT NOT NULL DEFAULT 'coalesce',
    max_catchup    INTEGER NOT NULL DEFAULT 100,
    grace_seconds  INTEGER NOT NULL DEFAULT 900,
    overlap_policy TEXT NOT NULL DEFAULT 'skip',
    next_fire_at   TEXT,
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

#: The Phase-1 additive columns + their ADD COLUMN DDL, applied idempotently by
#: :meth:`RoutinesStore._migrate`. Each is nullable or has a DEFAULT so an existing row
#: upgrades cleanly (an old daily/every routine keeps behaving identically — these columns
#: only matter for the new cron/at kinds + catch-up policy).
_PHASE1_COLUMNS: tuple[tuple[str, str], ...] = (
    ("schedule_expr", "ALTER TABLE routines ADD COLUMN schedule_expr TEXT"),
    (
        "schedule_at_datetime",
        "ALTER TABLE routines ADD COLUMN schedule_at_datetime TEXT",
    ),
    ("timezone", "ALTER TABLE routines ADD COLUMN timezone TEXT"),
    (
        "missed_policy",
        "ALTER TABLE routines ADD COLUMN missed_policy TEXT NOT NULL DEFAULT 'coalesce'",
    ),
    (
        "max_catchup",
        "ALTER TABLE routines ADD COLUMN max_catchup INTEGER NOT NULL DEFAULT 100",
    ),
    (
        "grace_seconds",
        "ALTER TABLE routines ADD COLUMN grace_seconds INTEGER NOT NULL DEFAULT 900",
    ),
    (
        "overlap_policy",
        "ALTER TABLE routines ADD COLUMN overlap_policy TEXT NOT NULL DEFAULT 'skip'",
    ),
    ("next_fire_at", "ALTER TABLE routines ADD COLUMN next_fire_at TEXT"),
)


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
            expr=(r["schedule_expr"] if "schedule_expr" in keys else None),
            at_datetime=(
                r["schedule_at_datetime"] if "schedule_at_datetime" in keys else None
            ),
            timezone=(r["timezone"] if "timezone" in keys else None),
            missed=(
                (r["missed_policy"] if "missed_policy" in keys else None) or "coalesce"
            ),
            max_catchup=(
                (r["max_catchup"] if "max_catchup" in keys else None)
                or DEFAULT_MAX_CATCHUP
            ),
            grace_seconds=(
                r["grace_seconds"]
                if ("grace_seconds" in keys and r["grace_seconds"] is not None)
                else DEFAULT_GRACE_SECONDS
            ),
            overlap=(
                (r["overlap_policy"] if "overlap_policy" in keys else None) or "skip"
            ),
        ),
        provider=r["provider"],
        model=r["model"],
        deliver=r["deliver"],
        enabled=bool(r["enabled"]),
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        next_fire_at=(r["next_fire_at"] if "next_fire_at" in keys else None),
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
        # The connection is ``check_same_thread=False`` (shared across the request loop +
        # ``to_thread`` workers), and this store — unlike ``SqliteStorageService`` — had NO
        # Python-level lock. The atomic per-tenant quota create
        # (:meth:`create_routine_if_under_quota`) needs the COUNT and the INSERT to be one
        # serialized critical section, so a write lock guards that ``BEGIN IMMEDIATE`` block
        # (mirroring ``SqliteStorageService._lock``). Other methods keep their existing
        # autocommit behaviour — only the count-then-create needs the up-front write lock.
        self._lock = threading.Lock()
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
        # Phase 1: additively add the cron/tz/catch-up columns. Idempotent — only adds a
        # column that is missing — so this converges whether the table came from a fresh
        # _SCHEMA (already has them), a pre-Phase-1 T3c table, or a just-rebuilt legacy one
        # (the rebuild emits the current _SCHEMA, so they are present and these are no-ops).
        post_info = self._conn.execute("PRAGMA table_info(routines)").fetchall()
        post_cols = {row["name"] for row in post_info}
        for name, ddl in _PHASE1_COLUMNS:
            if name not in post_cols:
                self._conn.execute(ddl)
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

    def count(
        self, *, workspace_id: str, enabled_only: bool = False
    ) -> int:
        """Count a workspace's routines (T3 per-tenant routine quota).

        ``enabled_only`` restricts the count to ENABLED routines (the ones that actually
        fire), which is what the per-tenant active-routine cap is enforced against.
        """
        if enabled_only:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM routines "
                "WHERE workspace_id = ? AND enabled = 1",
                (workspace_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM routines WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        return int(row["n"]) if row is not None else 0

    def create_routine_if_under_quota(
        self, routine: Routine, *, cap: int, enabled_only: bool = True
    ) -> tuple[Routine, bool]:
        """Atomically create ``routine`` IFF the tenant is under its routine cap (T3, HARD).

        Fuses the per-tenant COUNT and the INSERT into ONE ``BEGIN IMMEDIATE`` write
        transaction under :attr:`_lock`, so concurrent creates for the SAME workspace
        SERIALIZE: the write lock is taken UP FRONT (before the count), so the loser re-counts
        only after the winner's row is committed and sees the true post-insert count. The
        result is therefore EXACTLY ``cap`` admits with the rest a deterministic over-cap
        reject — never the unlocked count-then-create overshoot.

        Returns ``(routine, admitted)``. ``admitted=False`` means the tenant was at/over
        ``cap`` and NOTHING was written (cleaner than insert-then-delete). The caller is
        responsible for the exempt short-circuit (``__local__`` / ``cap <= 0``) so this is
        only reached for a genuinely capped tenant — but it defends ``cap <= 0`` anyway by
        admitting unconditionally.

        FAIL-OPEN on an infrastructure error: an unexpected DB failure during the locked
        count-then-create must not turn a quota guard into a hard outage, so it falls back to
        the plain :meth:`upsert` (the prior soft behaviour) and reports ``admitted=True``.
        Only a clean over-cap (count >= cap under the lock) is a fail-CLOSED reject.
        """
        if cap <= 0:  # defensive: an uncapped tenant is always admitted (plain upsert)
            return self.upsert(routine), True
        routine.updated_at = utc_now_iso()
        sql, params = self._upsert_sql(routine)
        enabled_clause = " AND enabled = 1" if enabled_only else ""
        try:
            with self._lock:
                try:
                    self._conn.execute("BEGIN IMMEDIATE")
                    row = self._conn.execute(
                        "SELECT COUNT(*) AS n FROM routines "  # noqa: S608
                        f"WHERE workspace_id = ?{enabled_clause}",
                        (routine.workspace_id,),
                    ).fetchone()
                    current = int(row["n"]) if row is not None else 0
                    if current >= cap:
                        self._conn.commit()  # nothing written; release the write lock
                        return routine, False
                    self._conn.execute(sql, params)
                    self._conn.commit()
                    return routine, True
                except BaseException:
                    try:
                        self._conn.rollback()
                    except sqlite3.Error:  # pragma: no cover - rollback on dead conn
                        pass
                    raise
        except Exception:  # noqa: BLE001 - fail OPEN on infra error (not over-cap)
            logger.warning(
                "atomic routine-quota create failed for workspace %s; admitting",
                routine.workspace_id,
            )
            return self.upsert(routine), True

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

    @staticmethod
    def _upsert_sql(routine: Routine) -> tuple[str, tuple[Any, ...]]:
        """The INSERT…ON CONFLICT statement + bound params for a routine upsert.

        Factored out so :meth:`upsert` and the atomic
        :meth:`create_routine_if_under_quota` write the BYTE-IDENTICAL row — the only
        difference between them is whether the write runs in autocommit (upsert) or inside
        the quota ``BEGIN IMMEDIATE`` (the atomic create).
        """
        sql = """
            INSERT INTO routines (
                id, name, workspace_id, agent_path, agent_id, idempotency_key,
                prompt, schedule_kind, schedule_at, schedule_hours,
                schedule_expr, schedule_at_datetime, timezone, missed_policy,
                max_catchup, grace_seconds, overlap_policy, next_fire_at,
                provider, model,
                deliver, enabled, created_at, updated_at, last_run_at, last_status,
                last_preview, last_error, last_delivery
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, workspace_id=excluded.workspace_id,
                agent_path=excluded.agent_path, agent_id=excluded.agent_id,
                idempotency_key=excluded.idempotency_key,
                prompt=excluded.prompt, schedule_kind=excluded.schedule_kind,
                schedule_at=excluded.schedule_at,
                schedule_hours=excluded.schedule_hours,
                schedule_expr=excluded.schedule_expr,
                schedule_at_datetime=excluded.schedule_at_datetime,
                timezone=excluded.timezone,
                missed_policy=excluded.missed_policy,
                max_catchup=excluded.max_catchup,
                grace_seconds=excluded.grace_seconds,
                overlap_policy=excluded.overlap_policy,
                next_fire_at=excluded.next_fire_at,
                provider=excluded.provider, model=excluded.model,
                deliver=excluded.deliver, enabled=excluded.enabled,
                updated_at=excluded.updated_at,
                last_run_at=excluded.last_run_at,
                last_status=excluded.last_status,
                last_preview=excluded.last_preview,
                last_error=excluded.last_error,
                last_delivery=excluded.last_delivery
            """
        params = (
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
            routine.schedule.expr,
            routine.schedule.at_datetime,
            routine.schedule.timezone,
            routine.schedule.missed,
            routine.schedule.max_catchup,
            routine.schedule.grace_seconds,
            routine.schedule.overlap,
            routine.next_fire_at,
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
        )
        return sql, params

    def upsert(self, routine: Routine) -> Routine:
        routine.updated_at = utc_now_iso()
        sql, params = self._upsert_sql(routine)
        self._conn.execute(sql, params)
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

    def advance_next_fire(self, routine_id: str, next_fire_at: str | None) -> None:
        """Persist the recomputed ``next_fire_at`` after a fire (Phase 1).

        Idempotent against the same planned instant: writing the same value twice (e.g. a
        crash-replayed advance) is a no-op-equivalent UPDATE. The dedup claim — keyed on the
        planned fire — is the crash backstop that guarantees at-most-once even if the advance
        never lands; this column is the observable hint, not the safety primitive.
        """
        self._conn.execute(
            "UPDATE routines SET next_fire_at = ? WHERE id = ?",
            (next_fire_at, routine_id),
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


def active_access_policy() -> Any | None:
    """The active RBAC :class:`AccessPolicy`, or ``None`` when auth is not configured.

    centralize-tool-gate: the off-request background seams (the ``agent_path`` routine fire,
    a background Mission, a Telegram-triggered run) run off any HTTP request, so they cannot
    read ``app.state.access_policy``; they read it from the wired run service instead (the
    SAME policy the ``/v1`` paths use, set in ``create_app`` ONLY when an authenticator is
    configured). ``None`` offline (no server / no authenticator) so the caller's ambient
    gate is inert — byte-unchanged. Shared by routines / missions / studio_telegram via the
    :func:`himmy.api.auth.service_principal.bind_service_authorizer` seam.
    """
    container = resolve_routine_container()
    if container is None:
        return None
    run_app = getattr(container, "run_app", None)
    return getattr(run_app, "_access_policy", None)


def _routine_access_policy() -> Any | None:
    """Back-compat alias for :func:`active_access_policy` (the routine drain's gate policy)."""
    return active_access_policy()


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

    # centralize-tool-gate: the scheduler runs on a background loop with NO HTTP request,
    # so it must bind the tool-capability gate ambiently itself (the request-boundary seam
    # cannot reach here). Under a CONFIGURED authenticator the gate is the routine's
    # least-privilege SERVICE principal (the SAME identity the ``agent_id`` seam stamps as
    # its run actor), so the routine's tools are deny-by-default to that service role rather
    # than running with the agent's full authority. Offline (no server / no authenticator)
    # ``_routine_access_policy`` is ``None`` → the ambient gate is ``None`` and inert,
    # byte-unchanged. Were this binding ever forgotten under configured auth, the
    # chokepoint's process-level fail-closed default would deny every tool anyway.
    from himmy.services.tools.ambient import _active_authorizer
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    _routine_policy = _routine_access_policy()
    if _routine_policy is not None:
        from himmy.api.auth.service_principal import routine_service_principal

        routine_authorizer = ToolCapabilityAuthorizer.from_principal(
            routine_service_principal(workspace_id=routine.workspace_id or "__local__"),
            _routine_policy,
        )
    else:
        routine_authorizer = None

    async def _drain() -> None:
        nonlocal output, status, error
        # Bind the routine gate ambiently for the whole drain via set/reset; reset in
        # ``finally`` so the binding is strictly scoped to this drain.
        _routine_authz_token = _active_authorizer.set(routine_authorizer)
        try:
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
        finally:
            _active_authorizer.reset(_routine_authz_token)

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
    # P0 #2: stamp a least-privilege SERVICE principal as the run's audit actor + identity.
    # ``actor_metadata`` carries the routine's roles + (for a tenant-bound principal under a
    # configured authenticator) the ``tool_authz_enforce`` flag, so the routine's tools are
    # gated deny-by-default to its allow-list roles instead of running with the agent's full
    # authority. Offline (no authenticator) the flag is inert — byte-unchanged. The routine
    # source/id is preserved alongside for routine<->run lineage.
    from himmy.api.auth.service_principal import routine_service_principal

    principal = routine_service_principal(workspace_id=workspace_id)
    actor = {
        **principal.actor_metadata(),
        "source": "routine",
        "routine_id": routine.id,
    }
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
    """Record the lifecycle notification (never raises, per the contract).

    The notification is stamped with the routine's OWNING workspace so a tenant-bound
    Studio reader only sees the bell (and the routine NAME / live LLM output preview) for
    its OWN routines. The reserved ``__local__`` single-tenant workspace maps to ``None``
    so the offline/single-box path stays byte-unchanged (NULL-owned = visible to all, the
    intended shared behaviour). Mirrors the missions lane (missions.py:_run).
    """
    from himmy.api.routers.studio_notify import record_notification

    ws = None if routine.workspace_id == LOCAL_WORKSPACE else routine.workspace_id

    if status == "ok":
        record_notification(
            "routine",
            f"Routine ran: {routine.name}",
            body=preview,
            link="/routines",
            workspace_id=ws,
        )
    elif status == "awaiting_approval":
        record_notification(
            "routine",
            f"Routine needs approval: {routine.name}",
            body="An approval-gated tool paused the run — review it in Approvals.",
            link="/approvals",
            workspace_id=ws,
        )
    else:  # error | timeout
        record_notification(
            "routine",
            f"Routine failed: {routine.name}",
            body=error or status,
            link="/routines",
            workspace_id=ws,
        )


def routine_lock_name(routine_id: str) -> str:
    """The cross-process flock name guarding a single routine's execution (T3c)."""
    return f"routine-{routine_id}"


#: The dedup scope for scheduled routine fires (namespaces the at-most-once keys so a
#: routine fire never collides with a webhook delivery id).
SCHEDULE_DEDUP_SCOPE = "schedule"
#: Floor on the dedup completed-TTL for a single fire. The TTL must be >= the routine's
#: cadence so two ticks/boots within one interval coalesce to one fire; a short-cadence
#: cron (e.g. every minute) sets a small interval, but never below this floor so a slow
#: run + a fast next tick still dedupe.
_DEDUP_TTL_FLOOR_SECONDS = 120.0


def schedule_dedup_key(routine_id: str, planned_fire_at_iso: str) -> str:
    """The at-most-once dedup key for a single planned fire (Phase 1).

    Keyed on the PLANNED fire instant (UTC ISO), never ``now()``, so concurrent ticks /
    boots that observe the same due fire coalesce to exactly one claim → one ``create_run``.
    """
    return f"{routine_id}:{planned_fire_at_iso}"


def _dedup_ttl_for(routine: Routine) -> float:
    """The completed-TTL (seconds) for a fire's dedup row: >= the routine cadence.

    For ``every`` it is N hours; for ``cron`` it is the gap to the NEXT occurrence after the
    one we are firing (so two fires of a short cadence never share a TTL window); for
    daily/at a day. Always at least :data:`_DEDUP_TTL_FLOOR_SECONDS`.
    """
    sched = routine.schedule
    if sched.kind == "every" and sched.hours:
        return max(sched.hours * 3600.0, _DEDUP_TTL_FLOOR_SECONDS)
    if sched.kind == "cron" and sched.expr:
        from himmy.api.routine_schedule import (
            CronUnavailableError,
            cron_next_fire,
            resolve_zone,
        )

        anchor = _parse_iso(routine.next_fire_at or "") or _parse_iso(
            routine.last_run_at or routine.created_at
        )
        if anchor is not None:
            try:
                zone = resolve_zone(sched.timezone)
                nxt = cron_next_fire(sched.expr, anchor, zone)
                gap = (nxt - _to_utc(anchor)).total_seconds()
                return max(gap, _DEDUP_TTL_FLOOR_SECONDS)
            except (CronUnavailableError, StopIteration):
                pass
    # daily / at / fallback — a day comfortably covers a once-a-day cadence.
    return max(86_400.0, _DEDUP_TTL_FLOOR_SECONDS)


async def execute_routine(
    routine_id: str,
    *,
    now: Callable[[], datetime] | None = None,
    expected_last_run_at: Any = UNCONDITIONAL,
    planned_fire_at_iso: str | None = None,
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
                planned_fire_at_iso=planned_fire_at_iso,
            )
    except ProcessLockBusy as exc:
        raise RoutineBusyError(routine_id) from exc


async def _try_schedule_dedup_claim(
    routine: Routine, planned_fire_at_iso: str
) -> bool | None:
    """Durable at-most-once claim for one planned fire (Phase 1 backstop).

    Returns ``True`` when THIS caller won the claim (run), ``False`` when the same planned
    fire was already claimed/completed (a concurrent tick/boot — do NOT run), or ``None``
    when no durable dedup store is available (offline in-memory) — the caller then relies on
    the flock + cluster CAS alone (single-box-safe; documented).

    Keyed on ``routine_id:planned_fire_at`` (the planned instant, NOT now), so two ticks or
    a tick + a boot-catch-up that resolve to the SAME planned fire coalesce to one run. The
    completed-TTL is the routine cadence (>= the interval) so a short-cadence routine's
    consecutive fires never share a window. We do NOT call ``dedup_complete`` here — the
    in-flight lease is left to EXPIRE so a crash mid-run is reclaimable (the routine's own
    ``last_run_at`` advance + the cluster CAS prevent a same-tick re-fire; the dedup row's
    role is the cross-boot coalesce of the SAME planned instant within the TTL window).
    """
    storage = _resolve_dedup_storage()
    if storage is None:
        return None
    from himmy.services.storage.trigger_dedup import CLAIM_WON

    key = schedule_dedup_key(routine.id, planned_fire_at_iso)
    ttl = _dedup_ttl_for(routine)
    try:
        claim = await storage.dedup_try_claim(
            SCHEDULE_DEDUP_SCOPE, key, lease_seconds=ttl
        )
    except Exception:  # noqa: BLE001 - a dedup-store failure must not block the fire
        logger.warning(
            "schedule dedup claim failed for %s; proceeding on flock/CAS only",
            routine.id,
            exc_info=True,
        )
        return None
    if claim.outcome is CLAIM_WON:
        # Mark the planned fire COMPLETED immediately (its result is irrelevant — the run's
        # own record carries the outcome). Completing now upgrades the short in-flight lease
        # to the full cadence TTL so the SAME planned instant cannot be re-claimed by a later
        # boot within the window, which is exactly the cross-boot coalesce we want.
        try:
            await storage.dedup_complete(
                SCHEDULE_DEDUP_SCOPE, key, result="fired", ttl_seconds=ttl
            )
        except Exception:  # noqa: BLE001 - completion is best-effort; lease still expires
            logger.debug("schedule dedup complete failed for %s", routine.id)
        return True
    # CLAIM_DONE / CLAIM_IN_FLIGHT — this planned fire is already handled elsewhere.
    return False


def _resolve_dedup_storage() -> Any | None:
    """Resolve the process-wide durable StorageService for dedup, or ``None`` offline.

    Uses the same server-storage seam the run substrate publishes (``set_server_storage``).
    A plain CLI/in-memory deployment has no durable dedup store; the flock + cluster CAS are
    the at-most-once guarantee there (single-box-safe — documented; cross-node single-fire
    holds only on a shared Postgres dedup table).
    """
    try:
        from himmy.services.storage.factory import server_storage
    except Exception:  # noqa: BLE001 - defensive
        return None
    try:
        storage = server_storage()
    except Exception:  # noqa: BLE001 - no server storage published
        return None
    if storage is None:
        return None
    # Only durable backends carry the dedup table; the in-memory facade also implements it
    # but its rows die with the process (no cross-boot value), so treat it as present only
    # when it exposes the dedup surface.
    return storage if hasattr(storage, "dedup_try_claim") else None


async def _execute_locked(
    routine_id: str,
    routine: Routine,
    *,
    now: Callable[[], datetime] | None = None,
    expected_last_run_at: Any = UNCONDITIONAL,
    planned_fire_at_iso: str | None = None,
) -> Routine | None:
    """The guarded body of :func:`execute_routine` (runs while the flock is held).

    COMMIT ORDER (Phase 1 red-team gate — a crash between steps must not double-fire):

      1. **dedup CLAIM_WON** — the durable at-most-once claim keyed on the PLANNED fire
         (``routine_id:planned_fire_at``). If this planned fire was already claimed (a
         concurrent tick/boot), return WITHOUT running. Offline (no durable store) this is a
         no-op and we fall through to the flock + CAS.
      2. **mark_started (cluster CAS)** — the conditional start stamp. On the Postgres mirror
         this is the cross-replica claim; on SQLite it is the local anchor advance under the
         flock. A 0-row match means a peer won — return without running.
      3. **run** (``_run_headless``) → record_result.
      4. **advance next_fire_at** — recompute and persist the next planned fire LAST. A crash
         before this leaves ``next_fire_at`` stale, but the dedup claim (1) + the advanced
         ``last_run_at`` (2) prevent re-firing the SAME planned instant; the next tick simply
         recomputes from ``last_run_at``. The advance is idempotent vs the same instant.
    """
    store = get_routines_store()
    now_fn = now or _default_now

    # 1) Durable at-most-once claim on the planned fire (the crash backstop / cross-boot
    #    coalesce). Only applies to a tick fire that carries a planned instant.
    if planned_fire_at_iso is not None:
        won = await _try_schedule_dedup_claim(routine, planned_fire_at_iso)
        if won is False:
            # This exact planned fire was already claimed by another tick/boot.
            already: Routine | None = store.get(routine_id)
            return already

    # 2) Cluster-wide / local atomic start claim.
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
    # 4) Advance next_fire_at LAST (idempotent vs the same planned instant). Best-effort:
    #    the dedup claim + advanced last_run_at are the at-most-once guarantees, not this.
    _advance_next_fire(store, routine_id, now=now_fn)
    _notify(routine, status, preview, error)
    refreshed: Routine | None = store.get(routine_id)
    return refreshed


def _advance_next_fire(
    store: Any, routine_id: str, *, now: Callable[[], datetime]
) -> None:
    """Recompute + persist ``next_fire_at`` after a fire (cron/at only; safe no-op else)."""
    if not hasattr(store, "advance_next_fire"):
        return
    routine = store.get(routine_id)
    if routine is None or routine.schedule.kind not in ("cron", "at"):
        return
    anchor = _parse_iso(routine.last_run_at or "") or now()
    nxt = planned_fire_at(routine, anchor)
    try:
        store.advance_next_fire(routine_id, nxt.isoformat() if nxt else None)
    except Exception:  # noqa: BLE001 - the column is a hint, not the safety primitive
        logger.debug("advance_next_fire failed for %s", routine_id, exc_info=True)


# ---- the scheduler -----------------------------------------------------------


def _default_now() -> datetime:
    """Aware local time — 'daily at 07:00' means the user's 7am."""
    return datetime.now().astimezone()


#: Default cap on a single scheduler sleep (seconds). The loop sleeps to the next computed
#: fire boundary but never longer than this, so a wall-clock jump / laptop suspend that
#: swallows a boundary is bounded to one cap-length of latency before the next re-evaluation.
_SCHEDULER_DEFAULT_MAX_SLEEP = 30.0
#: Floor on a single sleep so a due-but-unclaimable routine cannot spin the loop at 100% CPU.
_SCHEDULER_MIN_SLEEP = 0.05


def _scheduler_max_sleep_seconds() -> float:
    """The per-sleep cap from ``HIMMY_SCHEDULER_MAX_SLEEP`` (seconds), else the default.

    An unset / blank / unparseable / non-positive value falls back to
    :data:`_SCHEDULER_DEFAULT_MAX_SLEEP` so a typo never wedges the scheduler. Mirrors the
    parse-with-fallback contract of ``dispatch_env_overrides`` in ``runtime_bootstrap``.
    """
    raw = os.environ.get("HIMMY_SCHEDULER_MAX_SLEEP")
    if raw is None or not raw.strip():
        return _SCHEDULER_DEFAULT_MAX_SLEEP
    try:
        value = float(raw.strip())
    except ValueError:
        return _SCHEDULER_DEFAULT_MAX_SLEEP
    return value if value > 0 else _SCHEDULER_DEFAULT_MAX_SLEEP


def _tick_planned_fire_iso(routine: Routine) -> str | None:
    """The planned-fire UTC ISO for a due cron/at routine (the dedup key), else ``None``.

    For cron/at this is the next occurrence STRICTLY after the anchor (the instant is_due
    just crossed). For daily/every it is ``None`` — they keep their existing flock + CAS
    at-most-once with no planned-instant dedup (byte-identical behaviour).
    """
    if routine.schedule.kind not in ("cron", "at"):
        return None
    anchor = _parse_iso(routine.last_run_at or routine.created_at)
    if anchor is None:
        return None
    planned = planned_fire_at(routine, anchor)
    return planned.isoformat() if planned is not None else None


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
        tick_seconds: float | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        # ``tick_seconds`` is the *cap* on a single sleep, not a fixed cadence: the loop
        # sleeps until the next computed fire boundary but never longer than this, so a
        # clock-jump / laptop-sleep that swallows a boundary is bounded to one cap-length
        # of latency (catch-up + a re-evaluation on the next wake cover the actual miss).
        # Env-configurable via ``HIMMY_SCHEDULER_MAX_SLEEP`` (seconds); a typo / non-positive
        # value falls back to the 30s default so a misconfig never wedges the scheduler.
        self._max_sleep = (
            _scheduler_max_sleep_seconds() if tick_seconds is None else tick_seconds
        )
        self._now = now or _default_now
        self._loop_task: asyncio.Task[None] | None = None
        self._running: dict[str, asyncio.Task[Any]] = {}
        # Set on routine CRUD (add/update/delete/enable/run_now) so the loop wakes
        # immediately to re-plan instead of oversleeping a freshly-changed schedule. The
        # Event is created lazily-bound to the running loop in :meth:`start`.
        self._wakeup: asyncio.Event = asyncio.Event()

    @property
    def tick_seconds(self) -> float:
        """The maximum single-sleep cap (back-compat alias; was the fixed tick)."""
        return self._max_sleep

    # -- lifecycle ------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True while the background tick loop is running."""
        return self._loop_task is not None and not self._loop_task.done()

    def start(self) -> None:
        """Start the tick loop (idempotent)."""
        if self._loop_task is None or self._loop_task.done():
            # Re-bind the wakeup Event to the *currently running* loop. An Event created in
            # __init__ may be bound to a different (or no) loop; rebinding here guarantees
            # ``set()`` from a request handler and ``wait()`` in the tick loop share one loop.
            self._wakeup = asyncio.Event()
            self._loop_task = asyncio.create_task(
                self._loop(), name="himmy-routines-scheduler"
            )

    def notify_change(self) -> None:
        """Wake the tick loop now (call after any routine CRUD / enable / run_now).

        Fail-open: if the Event is bound to a different loop (e.g. called from a thread
        without a running loop) we swallow the error — the worst case is the loop wakes on
        its own sleep cap instead of instantly, which is still correct, just less prompt.
        """
        with contextlib.suppress(RuntimeError):
            self._wakeup.set()

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
        # Tick once immediately on start (covers a routine already due at boot before the
        # first sleep), then sleep-to-next-fire: compute the soonest boundary across all
        # enabled routines, sleep until then (capped), and wake early on CRUD.
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad tick never kills the loop
                logger.exception("routine scheduler tick failed")
            try:
                delay = self._sleep_delay()
            except Exception:  # noqa: BLE001 - planning failure -> fall back to the cap
                logger.exception("routine scheduler sleep-planning failed")
                delay = self._max_sleep
            self._wakeup.clear()
            try:
                # Wake on the soonest of: the computed boundary (capped) or a CRUD event.
                await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
            except TimeoutError:
                # Normal path: the planned sleep elapsed; loop around and tick.
                pass

    def _sleep_delay(self) -> float:
        """Seconds to sleep before the next tick: time-to-soonest-fire, capped, never <0.

        A value at-or-below zero (something is already due, or a boundary is in the past)
        collapses to a tiny floor so the loop ticks promptly without busy-spinning. The
        cap (``_max_sleep``) bounds clock-jump / laptop-sleep latency. With no enabled
        routines the loop simply sleeps the full cap (no busy-spin) and re-checks.
        """
        now = self._now()
        soonest: float | None = None
        try:
            routines = get_routines_store().list()
        except Exception:  # noqa: BLE001 - a broken store must not pin the loop
            logger.exception("routine store unavailable during sleep-planning")
            return self._max_sleep
        for routine in routines:
            # A routine that is already running has no *next* fire we must wake for until it
            # finishes (overlap is skipped/queued anyway); skip it so it cannot peg delay=0.
            if self.is_running(routine.id):
                continue
            nxt = next_fire(routine, now)
            if nxt is None:
                continue
            secs = (nxt - now).total_seconds()
            if soonest is None or secs < soonest:
                soonest = secs
        if soonest is None:
            return self._max_sleep
        # Floor at a small positive value: a past/now boundary should tick almost
        # immediately, but never 0.0 (which could spin if a due routine cannot be claimed).
        delay = max(soonest, _SCHEDULER_MIN_SLEEP)
        return min(delay, self._max_sleep)

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
            #
            # For cron/at also compute the PLANNED fire (the instant is_due crossed) and
            # thread it as the dedup key — so two ticks/boots on the same planned fire
            # coalesce to one run. daily/every pass None (no dedup-by-planned-instant): their
            # at-most-once stays the existing flock + CAS, byte-identical to before.
            planned_iso = _tick_planned_fire_iso(routine)
            self._launch(routine.id, routine.last_run_at, planned_iso)
            launched.append(routine.id)
        return launched

    def _launch(
        self,
        routine_id: str,
        expected_last_run_at: Any,
        planned_fire_at_iso: str | None = None,
    ) -> asyncio.Task[Routine | None]:
        """Fire-and-forget launch for the tick loop — failures are swallowed.

        The tick path never awaits the returned task, so a genuine run failure (or
        cross-process flock contention) must not escape: :func:`_guarded_execute`
        funnels everything to a logged ``None``. ``run_now`` does NOT use this path
        precisely because it must let :class:`RoutineBusyError` surface (a 409).
        ``expected_last_run_at`` is the due-time sentinel for the atomic cluster-wide claim;
        ``planned_fire_at_iso`` is the dedup key for cron/at (``None`` for daily/every).
        """
        task = asyncio.create_task(
            self._guarded_execute(
                routine_id, expected_last_run_at, planned_fire_at_iso
            ),
            name=f"himmy-routine-{routine_id}",
        )
        self._track(routine_id, task)
        return task

    def _track(self, routine_id: str, task: asyncio.Task[Routine | None]) -> None:
        """Register an in-flight run so same-process :meth:`is_running` sees it."""
        self._running[routine_id] = task
        task.add_done_callback(lambda _t: self._running.pop(routine_id, None))

    async def _guarded_execute(
        self,
        routine_id: str,
        expected_last_run_at: Any,
        planned_fire_at_iso: str | None = None,
    ) -> Routine | None:
        try:
            return await execute_routine(
                routine_id,
                now=self._now,
                expected_last_run_at=expected_last_run_at,
                planned_fire_at_iso=planned_fire_at_iso,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one failed run never kills the scheduler
            logger.exception("routine %s failed", routine_id)
            return None

    # -- catch-up on launch -------------------------------------------------------

    async def catch_up_on_launch(self) -> dict[str, str]:
        """Apply each routine's missed-run policy on worker/scheduler startup (Phase 1).

        Reads the AUTHORITATIVE anchor (``last_run_at``), NOT a wall-clock delta, so it is
        clock-skew safe. Per routine:

        * ``coalesce`` (default): do nothing here — a missed occurrence leaves ``is_due``
          true, so the very next normal tick fires it EXACTLY ONCE. (Laptop-sleep-safe.)
        * ``skip``: drop the gap — advance ``last_run_at`` / ``next_fire_at`` to the latest
          MISSED occurrence WITHOUT running, so the next tick only fires the next FUTURE one.
        * ``run_each``: backfill each missed occurrence, oldest-first, HARD-CAPPED by
          ``max_catchup`` and dropping occurrences older than ``grace_seconds`` (the storm
          cap is enforced AT ENQUEUE — never an unbounded backlog after a long sleep).

        Returns ``{routine_id: action}`` for observability (``coalesce`` / ``skip:N`` /
        ``run_each:N`` / ``none``). Only cron/at/daily routines with a real gap are touched;
        ``every`` keeps its interval-anchored behaviour (it drifts vs wall-clock by design —
        cron is the recommended wall-clock cadence).
        """
        now = self._now()
        try:
            routines = get_routines_store().list()
        except Exception:  # noqa: BLE001 - a broken store must not break startup
            logger.exception("routine store unavailable during catch-up")
            return {}
        actions: dict[str, str] = {}
        for routine in routines:
            try:
                action = await self._catch_up_one(routine, now)
            except Exception:  # noqa: BLE001 - one bad routine never breaks the sweep
                logger.exception("catch-up failed for routine %s", routine.id)
                action = "error"
            if action != "none":
                actions[routine.id] = action
        return actions

    async def _catch_up_one(self, routine: Routine, now: datetime) -> str:
        """Apply one routine's missed-run policy; return the action taken."""
        if not routine.enabled:
            return "none"
        if routine.schedule.kind == "every":
            return "none"  # interval-anchored; no wall-clock catch-up
        if not is_due(routine, now):
            return "none"  # nothing missed
        policy = routine.schedule.missed
        if policy == "coalesce":
            return "coalesce"  # the next normal tick fires it once
        if policy == "skip":
            return self._catch_up_skip(routine, now)
        return await self._catch_up_run_each(routine, now)

    def _catch_up_skip(self, routine: Routine, now: datetime) -> str:
        """``skip``: advance past every missed occurrence WITHOUT firing."""
        store = get_routines_store()
        # Stamp last_run_at to NOW so is_due no longer sees the missed occurrence; recompute
        # next_fire_at from there. A plain (unconditional) stamp is fine — we deliberately
        # drop the gap and do not run.
        stamped = now.isoformat()
        try:
            store.mark_started(routine.id, stamped)  # stamps last_run_at + 'running'
            # Immediately settle the status back so a skipped routine isn't left 'running'.
            store.record_result(
                routine.id, status="skipped", preview="(missed run skipped)", error=None
            )
            _advance_next_fire(store, routine.id, now=self._now)
        except Exception:  # noqa: BLE001 - skip is best-effort housekeeping
            logger.debug("skip catch-up stamp failed for %s", routine.id, exc_info=True)
        return "skip:1"

    async def _catch_up_run_each(self, routine: Routine, now: datetime) -> str:
        """``run_each``: backfill missed occurrences (cap + grace), oldest-first."""
        occurrences = self._missed_occurrences(routine, now)
        if not occurrences:
            return "none"
        fired = 0
        for planned in occurrences:
            if self.is_running(routine.id):
                break
            # Fire through the SAME rails with the planned instant as the dedup key, so a
            # backfilled occurrence is at-most-once just like a live tick. Awaited so the
            # backfill is sequential (overlap=skip) and the cap is honoured at enqueue.
            await self._guarded_execute(
                routine.id, routine.last_run_at, planned.isoformat()
            )
            fired += 1
            refreshed = get_routines_store().get(routine.id)
            if refreshed is None:
                break
            routine = refreshed
        return f"run_each:{fired}"

    def _missed_occurrences(self, routine: Routine, now: datetime) -> list[datetime]:
        """Missed occurrences to backfill for ``run_each`` (cap + grace applied)."""
        anchor = _parse_iso(routine.last_run_at or routine.created_at)
        if anchor is None:
            return []
        grace = float(routine.schedule.grace_seconds)
        cap = routine.schedule.max_catchup
        cutoff = now - timedelta(seconds=grace)
        sched = routine.schedule
        if sched.kind == "cron" and sched.expr:
            from himmy.api.routine_schedule import (
                CronUnavailableError,
                cron_occurrences_between,
                resolve_zone,
            )

            try:
                occ = cron_occurrences_between(
                    sched.expr, anchor, now, resolve_zone(sched.timezone), limit=cap
                )
            except CronUnavailableError:
                return []
        elif sched.kind == "daily":
            occ = self._daily_occurrences_between(routine, anchor, now, cap)
        elif sched.kind == "at":
            planned = planned_fire_at(routine, anchor)
            occ = [planned] if planned is not None and planned <= now else []
        else:
            return []
        # grace: drop occurrences older than the cutoff (a long sleep doesn't replay ancient
        # fires); the cap was already applied at enumeration for cron, apply it here for all.
        kept = [o for o in occ if o >= cutoff]
        return kept[-cap:] if len(kept) > cap else kept

    def _daily_occurrences_between(
        self, routine: Routine, anchor: datetime, now: datetime, cap: int
    ) -> list[datetime]:
        """Daily HH:MM occurrences in ``(anchor, now]`` in the routine timezone, capped."""
        from himmy.api.routine_schedule import resolve_zone

        m = _AT_RE.match(routine.schedule.at or "")
        if not m:
            return []
        zone = resolve_zone(routine.schedule.timezone)
        anchor_local = _to_utc(anchor).astimezone(zone)
        now_local = now.astimezone(zone)
        out: list[datetime] = []
        cursor = anchor_local.replace(
            hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0
        )
        if cursor <= anchor_local:
            cursor += timedelta(days=1)
        while cursor <= now_local and len(out) < cap * 2 + 2:
            out.append(_to_utc(cursor))
            cursor += timedelta(days=1)
        return out[-cap:] if len(out) > cap else out

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
    "DEFAULT_GRACE_SECONDS",
    "DEFAULT_MAX_CATCHUP",
    "DEFAULT_TENANT_MAX_CATCHUP",
    "DEFAULT_TENANT_MAX_ROUTINES",
    "DELIVERY_MAX_CHARS",
    "LOCAL_WORKSPACE",
    "PREVIEW_MAX_CHARS",
    "SCHEDULE_DEDUP_SCOPE",
    "MissedRunPolicy",
    "OverlapPolicy",
    "Routine",
    "RoutineBusyError",
    "RoutineScheduler",
    "RoutinesStore",
    "Schedule",
    "TenantRoutineQuotaExceeded",
    "clamp_tenant_catchup",
    "enforce_tenant_routine_quota",
    "execute_routine",
    "get_routines_store",
    "get_scheduler",
    "is_due",
    "planned_fire_at",
    "reset_routines_store",
    "reset_scheduler",
    "resolve_routine_container",
    "routine_lock_name",
    "routines_db_path",
    "run_timeout_s",
    "schedule_dedup_key",
    "set_routine_container_provider",
    "tenant_max_catchup",
    "tenant_max_routines",
]
