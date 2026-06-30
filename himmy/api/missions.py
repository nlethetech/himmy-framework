"""Studio Missions: background agent runs with steering, replayable streams.

A *mission* is an agent run executed in a server-side ``asyncio.Task`` instead of
inside the caller's HTTP stream: the GUI can fire one off, navigate away, steer it
between turns, interrupt it, and reconnect to its event stream at any time. Every
frame the existing :func:`himmy.api.studio_service.stream_agent_run` generator
yields is appended to a bounded per-mission ring buffer, so a (re)connecting
client first replays the buffered frames and then follows live.

Durability contract (be honest about it everywhere):

* FINISHED missions persist exactly like foreground runs — ``stream_agent_run``
  already records them via ``_record_run`` (one history, not two; the mission's
  ``run_id`` links to the Runs browser).
* The registry itself is PROCESS-LOCAL: a server restart loses RUNNING missions
  (their partial frame buffers included). Finished ones live on in the runs store.

Concurrency is capped (:data:`MAX_RUNNING_MISSIONS`) so a misbehaving client
can't fork unbounded model loops; the router maps the cap to a clear 409.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import queue
from dataclasses import dataclass, field
from typing import Any

from himmy.core.ids import new_uuid, utc_now_iso

#: Per-mission frame ring buffer bound — old frames fall off, the stream stays sane.
FRAME_BUFFER_MAX = 2000
#: How many missions may RUN concurrently (the router turns the breach into a 409).
MAX_RUNNING_MISSIONS = 3
#: How many finished missions the in-memory registry retains (newest kept).
FINISHED_RETAIN_MAX = 200
#: Result preview cap (the full output lives on the persisted run).
RESULT_PREVIEW_MAX = 400
#: Steer text bound (mirrored by the router's request model).
STEER_TEXT_MAX = 4000

_RUNNING = "running"
_PAUSED = "paused"
_DONE = "done"
_ERROR = "error"


class MissionError(Exception):
    """Base class for mission-registry domain errors."""


class MissionNotFoundError(MissionError):
    """No mission with that id in this process's registry."""


class MissionLimitError(MissionError):
    """The running-mission concurrency cap was reached."""


class MissionNotRunningError(MissionError):
    """The operation needs a RUNNING mission (steer/interrupt on a finished one)."""


def _cap(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


@dataclass
class Mission:
    """One background agent run: identity, live status, and its frame buffer.

    ``workspace_id`` / ``subject_id`` are the OWNING tenant + data subject, stamped at
    :meth:`MissionRegistry.start_mission` time from the verified principal (never from
    client input — the router resolves them via ``resolve_workspace`` / ``get_principal``).
    They carry the tenant/subject axes the process-local registry previously lacked, so a
    Studio read can refuse a mission belonging to another tenant/subject (BOLA/IDOR) — the
    by-construction drain of this surface from the tenant-scope coverage gate's pending
    allow-list. ``None`` means "no owning tenant/subject" (the offline / ``all_tenants``
    single-box default), in which case every reader sees it — byte-unchanged.
    """

    id: str
    agent: str
    agent_path: str
    prompt: str
    provider: str | None
    model: str | None
    plan_mode: bool
    created_at: str
    workspace_id: str | None = None
    subject_id: str | None = None
    #: The within-tenant USER axis (P1 tenancy, subject axis) the mission's TOOL STORES
    #: (memory/KB/tasks/notes) namespace by — set from the launching ``subject_scoped``
    #: principal at start time (NOT from ``subject_id``, which is attribution only). ``None``
    #: keeps the tool stores tenant-only (the offline / ``all_tenants`` / non-subject-scoped
    #: tenant default) so the byte-unchanged shared-tenant path is preserved. This is the
    #: missing axis the interactive Studio path threads via ``owner_subject_scope`` — without
    #: it a subject_scoped user's background-mission memory/KB/tasks collapse to the shared
    #: tenant namespace (a cross-USER, same-tenant BOLA).
    subject_scope: str | None = None
    status: str = _RUNNING
    finished_at: str | None = None
    result_preview: str = ""
    error: str | None = None
    run_id: str | None = None
    checkpoint_id: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    #: How many old frames fell off the ring buffer (absolute-cursor arithmetic).
    frames_dropped: int = 0
    #: Between-turns steering seam — thread-safe, drained by the runtime loop.
    steer_queue: queue.Queue[str] = field(default_factory=queue.Queue)
    #: Wakes stream followers on every append / status change.
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None

    @property
    def frame_count(self) -> int:
        return self.frames_dropped + len(self.frames)

    def summary(self) -> dict[str, Any]:
        """The list-row projection (no frames, no queue internals)."""
        return {
            "id": self.id,
            "agent": self.agent,
            "agent_path": self.agent_path,
            "prompt": _cap(self.prompt, 300),
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result_preview": self.result_preview,
            "error": self.error,
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "plan_mode": self.plan_mode,
            "frame_count": self.frame_count,
        }


class MissionRegistry:
    """Process-local registry of missions (held on ``app.state``).

    All registry methods run on the server's event loop (FastAPI handlers + the
    mission tasks themselves); the only cross-thread surface is each mission's
    ``steer_queue``, which is a thread-safe ``queue.Queue`` by design.
    """

    def __init__(
        self,
        *,
        max_running: int = MAX_RUNNING_MISSIONS,
        frame_buffer: int = FRAME_BUFFER_MAX,
    ) -> None:
        self._max_running = max_running
        self._frame_buffer = max(1, frame_buffer)
        self._missions: dict[str, Mission] = {}

    # ------------------------------------------------------------- queries
    def list(self) -> builtins.list[Mission]:
        """Every known mission, newest first."""
        return sorted(self._missions.values(), key=lambda m: m.created_at, reverse=True)

    def get(self, mission_id: str) -> Mission:
        mission = self._missions.get(mission_id)
        if mission is None:
            raise MissionNotFoundError(f"unknown mission {mission_id!r}")
        return mission

    def running_count(self) -> int:
        return sum(1 for m in self._missions.values() if m.status == _RUNNING)

    @property
    def max_running(self) -> int:
        """The concurrency cap (surfaced so the GUI can explain a 409)."""
        return self._max_running

    # -------------------------------------------------------------- start
    def start_mission(
        self,
        *,
        agent_path: str,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        history: builtins.list[dict[str, Any]] | None = None,
        plan_mode: bool = False,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        subject_scope: str | None = None,
    ) -> Mission:
        """Spawn the existing stream-run generator inside a server-side task.

        Raises :class:`MissionLimitError` past the concurrency cap, and lets the
        spec loader's ``FileNotFoundError`` / ``ValueError`` propagate so the
        router can map them to clean 404/400s BEFORE anything is registered.

        ``workspace_id`` / ``subject_id`` are the resolved-from-principal owning tenant +
        subject (the router derives them from the verified caller, never from the request
        body), stamped on the :class:`Mission` so a Studio read can refuse it to a
        different tenant/subject. ``None`` (the offline / ``all_tenants`` default) leaves
        the mission unowned and visible to every reader — byte-unchanged.
        """
        from himmy.api import studio_service

        if self.running_count() >= self._max_running:
            raise MissionLimitError(
                f"mission limit reached ({self._max_running} running) — "
                "interrupt one or wait for it to finish"
            )
        spec = studio_service.load_studio_spec(
            agent_path, provider=provider, model=model
        )
        mission = Mission(
            id=new_uuid(),
            agent=spec.name,
            agent_path=agent_path,
            prompt=prompt,
            provider=provider,
            model=model,
            plan_mode=plan_mode,
            created_at=utc_now_iso(),
            workspace_id=workspace_id,
            subject_id=subject_id,
            subject_scope=subject_scope,
            history=list(history or []),
        )
        self._missions[mission.id] = mission
        self._prune_finished()
        mission.task = asyncio.create_task(
            self._run_mission(mission, spec), name=f"mission:{mission.id}"
        )
        return mission

    def _prune_finished(self) -> None:
        """Drop the oldest finished missions past the retention bound."""
        finished = [m for m in self.list() if m.status != _RUNNING]
        for stale in finished[FINISHED_RETAIN_MAX:]:
            self._missions.pop(stale.id, None)

    # ------------------------------------------------------------- runner
    async def _append(self, mission: Mission, frame: dict[str, Any]) -> None:
        """Append one frame to the ring buffer and wake every follower."""
        async with mission.cond:
            mission.frames.append(frame)
            if len(mission.frames) > self._frame_buffer:
                overflow = len(mission.frames) - self._frame_buffer
                del mission.frames[:overflow]
                mission.frames_dropped += overflow
            mission.cond.notify_all()

    async def _finish(self, mission: Mission, status: str) -> None:
        """Flip the mission to a terminal/paused status and wake followers."""
        async with mission.cond:
            mission.status = status
            mission.finished_at = utc_now_iso()
            mission.cond.notify_all()

    async def _run_mission(self, mission: Mission, spec: Any) -> None:
        """Drive the run generator to completion, mirroring frames into the buffer.

        Persistence rides the existing ``stream_agent_run`` path (``_record_run``)
        — the mission layer never writes a second history. A cooperative cancel
        (interrupt) is recorded honestly as an error with an explicit message.
        """
        from himmy.api import studio_service
        from himmy.api.auth.service_principal import (
            bind_service_authorizer,
            connector_service_principal,
        )
        from himmy.api.routers.studio_notify import record_notification
        from himmy.api.routines import active_access_policy
        from himmy.api.studio_canonical import resolve_canonical_storage
        from himmy.services.storage.models import LOCAL_WORKSPACE

        # centralize-tool-gate: a background Mission drives the agent off any HTTP request,
        # so it binds the tool-capability gate AMBIENTLY itself (the request boundary cannot
        # reach this task). Under a CONFIGURED authenticator the gate is a least-privilege
        # SERVICE principal (deny-by-default to a named service role), so a mission's tools
        # are gated rather than running with the agent's full authority — AND rather than
        # being denied wholesale by the chokepoint's fail-closed default. Offline (no
        # server / no authenticator) ``active_access_policy()`` is ``None`` → the binding is
        # inert and the path is byte-unchanged. A mission is single-box / ``__local__``, like
        # the routine ``agent_path`` seam, so it reuses the connector service principal shape.
        status = _DONE
        cancelled = False
        try:
            with bind_service_authorizer(
                connector_service_principal(workspace_id=LOCAL_WORKSPACE),
                active_access_policy(),
            ):
                async for frame in studio_service.stream_agent_run(
                    spec,
                    mission.prompt,
                    history=mission.history,
                    provider=mission.provider,
                    model=mission.model,
                    agent_path=mission.agent_path,
                    steer_queue=mission.steer_queue,
                    plan_mode=mission.plan_mode,
                    canonical_storage=resolve_canonical_storage(),
                    # scope-r4: stamp the canonical RunRecord with the LAUNCHING tenant +
                    # subject (resolved from the verified principal at start_mission time)
                    # so the run lands in that partition, not the global ``__local__``
                    # bucket. ``None`` (offline / unowned) keeps the byte-unchanged default.
                    owner_workspace_id=mission.workspace_id,
                    owner_subject_id=mission.subject_id,
                    # rbac-harden(mopup-r3-2): thread the within-tenant USER axis into the
                    # mission's TOOL STORES so a subject_scoped user's background-mission
                    # memory/KB/tasks/notes namespace by ``t:<tenant>:s:<subject>`` — exactly
                    # like the interactive Studio path (studio.py owner_subject_scope). Without
                    # this the packs collapse to the shared ``t:<tenant>`` namespace and two
                    # users of one tenant read each other's mission memory (cross-USER BOLA).
                    # ``None`` (offline / non-subject-scoped tenant) stays tenant-only,
                    # byte-unchanged. NOTE: owner_subject_id alone only stamps the RunRecord
                    # (attribution); it does NOT namespace the tool stores — only this does.
                    owner_subject_scope=mission.subject_scope,
                ):
                    await self._append(mission, frame)
                    ftype = frame.get("type")
                    if ftype == "done":
                        status = _DONE
                        mission.run_id = (
                            str(frame.get("run_id") or "") or mission.run_id
                        )
                        mission.result_preview = _cap(
                            str(frame.get("output_text") or ""), RESULT_PREVIEW_MAX
                        )
                    elif ftype == "error":
                        status = _ERROR
                        mission.error = str(frame.get("message") or "run failed")
                        mission.run_id = (
                            str(frame.get("run_id") or "") or mission.run_id
                        )
                    elif ftype == "paused":
                        status = _PAUSED
                        mission.checkpoint_id = frame.get("checkpoint_id")
                        mission.run_id = (
                            str(frame.get("run_id") or "") or mission.run_id
                        )
                        mission.result_preview = (
                            "paused at an approval gate — resume from Approvals"
                        )
        except asyncio.CancelledError:
            cancelled = True
            status = _ERROR
            mission.error = (
                "interrupted by user — cooperative cancel (this run had no "
                "approval checkpoint to pause at, so it cannot be resumed)"
            )
            await self._append(mission, {"type": "error", "message": mission.error})
        except Exception as exc:  # noqa: BLE001 - a broken run must mark, not crash
            status = _ERROR
            mission.error = str(exc)
            await self._append(mission, {"type": "error", "message": str(exc)})
        finally:
            await self._finish(mission, status)
            try:
                # Thread the mission's OWNING workspace so a tenant-bound Studio reader only
                # sees the bell for its own missions (NULL/None offline, byte-unchanged).
                ws = mission.workspace_id
                if status == _DONE:
                    record_notification(
                        "mission",
                        f"Mission finished — {mission.agent}",
                        body=mission.result_preview or _cap(mission.prompt, 200),
                        link="/missions",
                        workspace_id=ws,
                    )
                elif status == _PAUSED:
                    record_notification(
                        "mission",
                        f"Mission paused — {mission.agent} needs approval",
                        body=_cap(mission.prompt, 200),
                        link="/approvals",
                        workspace_id=ws,
                    )
                else:
                    record_notification(
                        "mission",
                        f"Mission failed — {mission.agent}",
                        body=mission.error or "",
                        link="/missions",
                        workspace_id=ws,
                    )
            except Exception:  # noqa: BLE001 - notifying must never mask the run
                pass
        if cancelled:
            raise asyncio.CancelledError()

    # ----------------------------------------------------------- commands
    async def steer(self, mission_id: str, text: str) -> Mission:
        """Queue between-turns guidance for a RUNNING mission.

        The text is drained by the runtime loop at the top of the next
        continuation turn (appended as a USER message); a ``steer`` frame is
        appended to the buffer immediately so every stream shows the guidance the
        moment it was given.
        """
        mission = self.get(mission_id)
        if mission.status != _RUNNING:
            raise MissionNotRunningError(
                f"mission is {mission.status} — steering needs a running mission"
            )
        clean = _cap(text, STEER_TEXT_MAX)
        if not clean:
            raise MissionError("steer text is empty")
        mission.steer_queue.put(clean)
        await self._append(
            mission, {"type": "steer", "text": clean, "ts": utc_now_iso()}
        )
        return mission

    async def interrupt(self, mission_id: str) -> dict[str, Any]:
        """Stop a mission, honestly reporting WHICH stop actually happened.

        * already ``paused`` (approval-gate checkpoint) → nothing to do; report
          ``mode="checkpoint"`` (resumable from Approvals);
        * ``running`` → cooperative cancel of the server-side task; the loop has
          no mid-turn checkpoint, so this is reported as ``mode="cancelled"`` and
          the mission lands in ``error`` with an explicit interrupted message.
        """
        mission = self.get(mission_id)
        if mission.status == _PAUSED:
            return {
                "status": mission.status,
                "mode": "checkpoint",
                "detail": "already paused at an approval checkpoint — resume or "
                "reject it from Approvals",
            }
        if mission.status != _RUNNING or mission.task is None:
            raise MissionNotRunningError(f"mission already finished ({mission.status})")
        mission.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(asyncio.shield(mission.task), timeout=10.0)
        return {
            "status": mission.status,
            "mode": "cancelled",
            "detail": "cooperatively cancelled — the loop had no checkpoint to "
            "pause at, so the mission cannot be resumed",
        }

    # -------------------------------------------------------------- stream
    async def stream(self, mission_id: str) -> Any:
        """Async-iterate the mission's frames: replay the buffer, then follow live.

        Reconnect-safe by construction — every consumer starts from the oldest
        buffered frame (absolute cursor) and the generator ends once the mission
        has left ``running`` AND everything buffered has been delivered.
        """
        mission = self.get(mission_id)

        async def _gen() -> Any:
            cursor = mission.frames_dropped  # absolute index of the next frame
            while True:
                async with mission.cond:
                    total = mission.frames_dropped + len(mission.frames)
                    if cursor < mission.frames_dropped:
                        cursor = mission.frames_dropped  # fell off the ring
                    if cursor < total:
                        batch = list(mission.frames[cursor - mission.frames_dropped :])
                        cursor = total
                    elif mission.status != _RUNNING:
                        return
                    else:
                        await mission.cond.wait()
                        continue
                for frame in batch:
                    yield frame

        return _gen()


def get_registry(app: Any) -> MissionRegistry:
    """The per-process registry, lazily created on ``app.state``."""
    registry = getattr(app.state, "mission_registry", None)
    if not isinstance(registry, MissionRegistry):
        registry = MissionRegistry()
        app.state.mission_registry = registry
    return registry


__all__ = [
    "FINISHED_RETAIN_MAX",
    "FRAME_BUFFER_MAX",
    "MAX_RUNNING_MISSIONS",
    "RESULT_PREVIEW_MAX",
    "STEER_TEXT_MAX",
    "Mission",
    "MissionError",
    "MissionLimitError",
    "MissionNotFoundError",
    "MissionNotRunningError",
    "MissionRegistry",
    "get_registry",
]
