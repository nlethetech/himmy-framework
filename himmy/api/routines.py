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

logger = logging.getLogger("himmy.api.routines")

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
    """One scheduled routine, as stored."""

    id: str = Field(default_factory=new_uuid)
    name: str
    agent_path: str
    prompt: str
    schedule: Schedule
    provider: str | None = None
    model: str | None = None
    deliver: DeliverKind = "none"
    enabled: bool = True
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    last_run_at: str | None = None
    last_status: str | None = None  # running | ok | error | timeout | awaiting_approval
    last_preview: str = ""
    last_error: str | None = None
    last_delivery: str | None = None  # delivery failure note; None = ok / not asked


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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routines (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    agent_path     TEXT NOT NULL,
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
    return Routine(
        id=r["id"],
        name=r["name"],
        agent_path=r["agent_path"],
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
        self._conn.commit()

    def list(self) -> list[Routine]:
        rows = self._conn.execute(
            "SELECT * FROM routines ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_routine(r) for r in rows]

    def get(self, routine_id: str) -> Routine | None:
        row = self._conn.execute(
            "SELECT * FROM routines WHERE id = ?", (routine_id,)
        ).fetchone()
        return _row_to_routine(row) if row else None

    def upsert(self, routine: Routine) -> Routine:
        routine.updated_at = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO routines (
                id, name, agent_path, prompt, schedule_kind, schedule_at,
                schedule_hours, provider, model, deliver, enabled, created_at,
                updated_at, last_run_at, last_status, last_preview, last_error,
                last_delivery
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, agent_path=excluded.agent_path,
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
                routine.agent_path,
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

    def delete(self, routine_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM routines WHERE id = ?", (routine_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def mark_started(self, routine_id: str, started_at: str) -> None:
        """Stamp the run start — also the due-math anchor, so a slow run can't
        re-trigger itself on the next tick."""
        self._conn.execute(
            "UPDATE routines SET last_run_at = ?, last_status = 'running',"
            " last_error = NULL, last_delivery = NULL WHERE id = ?",
            (started_at, routine_id),
        )
        self._conn.commit()

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


_STORE: RoutinesStore | None = None
_PATH: str | None = None


def routines_db_path() -> str:
    env = os.environ.get("HIMMY_ROUTINES_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "routines.db")


def get_routines_store() -> RoutinesStore:
    global _STORE, _PATH
    path = routines_db_path()
    if _STORE is None or _PATH != path:
        if _STORE is not None:
            _STORE.close()
        _STORE = RoutinesStore(path)
        _PATH = path
    return _STORE


def reset_routines_store() -> None:
    global _STORE, _PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _PATH = None


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
    """Run one routine through the normal Studio run pipeline, unattended.

    Returns ``(status, output_text, error)``. ``stream_agent_run`` persists the
    run to the runs store itself; an approval-gated tool pauses the run on a
    durable checkpoint (status ``awaiting_approval``) — it is NOT executed. The run
    is mirrored into the canonical run store (T3c) so a scheduled run is visible in
    ``himmy runs`` and ``/v1`` too, not just the local studio.db.
    """
    from himmy.api import studio_service
    from himmy.api.studio_canonical import resolve_canonical_storage

    canonical = resolve_canonical_storage()
    try:
        spec = studio_service.load_studio_spec(
            routine.agent_path, provider=routine.provider, model=routine.model
        )
    except FileNotFoundError as exc:
        return "error", "", str(exc)
    except ValueError as exc:
        return "error", "", str(exc)

    output = ""
    status = "ok"
    error: str | None = None

    async def _drain() -> None:
        nonlocal output, status, error
        async for event in studio_service.stream_agent_run(
            spec,
            routine.prompt,
            provider=routine.provider,
            model=routine.model,
            agent_path=routine.agent_path,
            canonical_storage=canonical,
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


async def execute_routine(
    routine_id: str, *, now: Callable[[], datetime] | None = None
) -> Routine | None:
    """Run one routine end-to-end: run → deliver → record → notify.

    Returns the refreshed routine, or ``None`` when it no longer exists.
    """
    store = get_routines_store()
    routine = store.get(routine_id)
    if routine is None:
        return None
    now_fn = now or _default_now
    store.mark_started(routine_id, now_fn().isoformat())
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
    return store.get(routine_id)


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
            self._launch(routine.id)
            launched.append(routine.id)
        return launched

    def _launch(self, routine_id: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(
            self._guarded_execute(routine_id), name=f"himmy-routine-{routine_id}"
        )
        self._running[routine_id] = task
        task.add_done_callback(lambda _t: self._running.pop(routine_id, None))
        return task

    async def _guarded_execute(self, routine_id: str) -> Routine | None:
        try:
            return await execute_routine(routine_id, now=self._now)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one failed run never kills the scheduler
            logger.exception("routine %s failed", routine_id)
            return None

    # -- manual trigger -----------------------------------------------------------

    async def run_now(self, routine_id: str) -> Routine | None:
        """Run a routine immediately through the same rails; await the result.

        Raises :class:`RoutineBusyError` if it is already executing.
        """
        if self.is_running(routine_id):
            raise RoutineBusyError(routine_id)
        result: Routine | None = await self._launch(routine_id)
        return result


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
    "routines_db_path",
    "run_timeout_s",
]
