"""Tests for Studio Routines: due-math, store, CRUD, run-now, and the scheduler.

Fast tests run the real pipeline against the stub provider (in-process app).
Due-math and tick behavior use an injected frozen clock. An optional live test
(skipped when Ollama is unreachable) asserts unattended-run MECHANICS — events
persisted, routine state recorded — never model prose quality.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from himmy.api import routines as svc
from himmy.api.app import create_app
from himmy.api.studio_runs import reset_run_store

_OLLAMA_URL = os.environ.get("HIMMY_OLLAMA_URL", "http://localhost:11434")
_LIVE_MODEL = "qwen2.5:7b-instruct"


def _ollama_available() -> bool:
    try:
        return httpx.get(f"{_OLLAMA_URL}/api/tags", timeout=3.0).status_code == 200
    except Exception:  # noqa: BLE001 - unreachable → skip the live test
        return False


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A project root with an agent spec + fresh cwd-keyed stores."""
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    # The background loop is exercised explicitly; keep app startups quiet.
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    svc.reset_routines_store()
    svc.reset_scheduler()
    reset_run_store()
    yield tmp_path
    svc.reset_routines_store()
    svc.reset_scheduler()


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app())


def _mk(
    *,
    schedule: svc.Schedule,
    enabled: bool = True,
    created_at: str = "2026-06-09T10:00:00+00:00",
    last_run_at: str | None = None,
) -> svc.Routine:
    return svc.Routine(
        name="r",
        agent_path="agent.yaml",
        prompt="hi",
        schedule=schedule,
        enabled=enabled,
        created_at=created_at,
        last_run_at=last_run_at,
    )


def _at(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


# ---- due-math (frozen clock) ------------------------------------------------


def test_daily_never_fires_retroactively_on_creation() -> None:
    r = _mk(schedule=svc.Schedule(kind="daily", at="07:00"))
    # Created 10:00; today's 07:00 occurrence predates creation → not due.
    assert not svc.is_due(r, _at("2026-06-09T12:00:00+00:00"))


def test_daily_fires_once_after_its_time_then_waits_a_day() -> None:
    r = _mk(schedule=svc.Schedule(kind="daily", at="07:00"))
    assert not svc.is_due(r, _at("2026-06-10T06:59:00+00:00"))
    assert svc.is_due(r, _at("2026-06-10T07:00:00+00:00"))
    r.last_run_at = "2026-06-10T07:00:10+00:00"
    assert not svc.is_due(r, _at("2026-06-10T23:00:00+00:00"))
    assert svc.is_due(r, _at("2026-06-11T07:00:30+00:00"))


def test_every_n_hours_anchors_on_last_run() -> None:
    r = _mk(schedule=svc.Schedule(kind="every", hours=2))
    assert not svc.is_due(r, _at("2026-06-09T11:59:00+00:00"))
    assert svc.is_due(r, _at("2026-06-09T12:00:00+00:00"))
    r.last_run_at = "2026-06-09T12:00:00+00:00"
    assert not svc.is_due(r, _at("2026-06-09T13:59:00+00:00"))
    assert svc.is_due(r, _at("2026-06-09T14:00:00+00:00"))


def test_disabled_or_corrupt_anchor_is_never_due() -> None:
    r = _mk(schedule=svc.Schedule(kind="every", hours=1), enabled=False)
    assert not svc.is_due(r, _at("2027-01-01T00:00:00+00:00"))
    r2 = _mk(schedule=svc.Schedule(kind="every", hours=1), created_at="not-a-date")
    assert not svc.is_due(r2, _at("2027-01-01T00:00:00+00:00"))


def test_schedule_grammar_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        svc.Schedule(kind="daily")  # missing at
    with pytest.raises(ValueError):
        svc.Schedule(kind="daily", at="25:00")
    with pytest.raises(ValueError):
        svc.Schedule(kind="every")  # missing hours
    with pytest.raises(ValueError):
        svc.Schedule(kind="every", hours=0)
    with pytest.raises(ValueError):
        svc.Schedule(kind="every", hours=169)


# ---- CRUD ---------------------------------------------------------------------


def _body(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "Morning brief",
        "agent_path": "agent.yaml",
        "prompt": "check my calendar and tasks, write a brief",
        "schedule": {"kind": "daily", "at": "07:00"},
        "deliver": "none",
    }
    base.update(over)
    return base


def test_crud_roundtrip(client: TestClient) -> None:
    created = client.post("/api/studio/routines", json=_body())
    assert created.status_code == 200
    rid = created.json()["id"]
    # Phase 1 adds additive, defaulted schedule fields; the daily semantics are unchanged.
    assert created.json()["schedule"] == {
        "kind": "daily",
        "at": "07:00",
        "hours": None,
        "expr": None,
        "at_datetime": None,
        "timezone": None,
        "missed": "coalesce",
        "max_catchup": 100,
        "grace_seconds": 900,
        "overlap": "skip",
    }
    assert created.json()["enabled"] is True
    assert created.json()["last_run_at"] is None

    lst = client.get("/api/studio/routines").json()
    assert [r["id"] for r in lst] == [rid]

    patched = client.patch(
        f"/api/studio/routines/{rid}",
        json={"enabled": False, "schedule": {"kind": "every", "hours": 6}},
    )
    assert patched.status_code == 200
    assert patched.json()["enabled"] is False
    assert patched.json()["schedule"]["kind"] == "every"
    assert patched.json()["schedule"]["hours"] == 6

    got = client.get(f"/api/studio/routines/{rid}").json()
    assert got["enabled"] is False

    assert client.delete(f"/api/studio/routines/{rid}").json() == {"ok": True}
    assert client.get(f"/api/studio/routines/{rid}").status_code == 404
    assert client.delete(f"/api/studio/routines/{rid}").status_code == 404
    assert client.patch(f"/api/studio/routines/{rid}", json={}).status_code == 404


def test_create_validation(client: TestClient) -> None:
    # schedule grammar enforced server-side
    bad = [
        _body(schedule={"kind": "daily"}),
        _body(schedule={"kind": "daily", "at": "7am"}),
        _body(schedule={"kind": "every"}),
        _body(schedule={"kind": "every", "hours": 0}),
        _body(schedule={"kind": "every", "hours": 999}),
        _body(schedule={"kind": "cron", "at": "* * * * *"}),
        _body(name=""),
        _body(prompt=""),
        _body(deliver="pigeon"),
    ]
    for body in bad:
        assert client.post("/api/studio/routines", json=body).status_code == 422
    # agent path must exist…
    assert (
        client.post(
            "/api/studio/routines", json=_body(agent_path="nope.yaml")
        ).status_code
        == 404
    )
    # …and must not escape the project root
    assert (
        client.post(
            "/api/studio/routines", json=_body(agent_path="../evil.yaml")
        ).status_code
        == 400
    )


# ---- run-now through the stub provider ----------------------------------------


def test_run_now_records_run_routine_and_notification(client: TestClient) -> None:
    rid = client.post("/api/studio/routines", json=_body(provider="stub")).json()["id"]

    res = client.post(f"/api/studio/routines/{rid}/run-now")
    assert res.status_code == 200
    body = res.json()
    assert body["last_status"] == "ok"
    assert body["last_run_at"] is not None
    assert body["last_preview"] != ""
    assert body["last_error"] is None

    # the run landed in the runs store like any Studio run
    runs = client.get("/api/studio/runs").json()
    assert runs["total"] == 1
    assert runs["items"][0]["agent_name"] == "helper"
    assert runs["items"][0]["status"] == "ok"

    # and the lifecycle notification was recorded
    notes = client.get("/api/studio/notify").json()["items"]
    assert any(n["kind"] == "routine" and "Morning brief" in n["title"] for n in notes)


def test_run_now_unknown_routine_404(client: TestClient) -> None:
    assert client.post("/api/studio/routines/nope/run-now").status_code == 404


def test_run_now_is_409_when_routine_flock_held(client: TestClient) -> None:
    """Studio run-now surfaces 409 (not 404) when the routine's flock is held elsewhere.

    Reviewer must_fix: the cross-process busy case was being swallowed on the run_now path
    (``RoutineScheduler._guarded_execute`` ate ``RoutineBusyError`` → ``None`` → router 404),
    making the Studio ``409`` handler dead code. Hold the flock (standing in for a CLI
    ``himmy routines run-now`` owning it) and assert the Studio endpoint refuses with 409
    and never starts a second run.
    """
    from himmy.core.process_lock import process_lock

    rid = client.post("/api/studio/routines", json=_body(provider="stub")).json()["id"]

    with process_lock(svc.routine_lock_name(rid)):
        res = client.post(f"/api/studio/routines/{rid}/run-now")
    assert res.status_code == 409, res.text
    assert "already running" in res.json()["detail"]
    # The guarded body never ran → no Studio run row was recorded.
    assert client.get("/api/studio/runs").json()["total"] == 0


# ---- unattended rails (frozen pipeline) -----------------------------------------


def _store_routine(**over: Any) -> svc.Routine:
    routine = _mk(schedule=svc.Schedule(kind="every", hours=1))
    for k, v in over.items():
        setattr(routine, k, v)
    return svc.get_routines_store().upsert(routine)


@pytest.mark.asyncio
async def test_timeout_rail_cancels_and_records(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from himmy.api import studio_service

    async def _slow(*a: Any, **k: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "start"}
        await asyncio.sleep(30)
        yield {"type": "done", "output_text": "late"}

    monkeypatch.setattr(studio_service, "stream_agent_run", _slow)
    monkeypatch.setattr(svc, "run_timeout_s", lambda: 0.2)
    routine = _store_routine()

    out = await svc.execute_routine(routine.id)
    assert out is not None
    assert out.last_status == "timeout"
    assert out.last_error is not None and "cancelled" in out.last_error


@pytest.mark.asyncio
async def test_approval_pause_recorded_honestly(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An approval-gated tool pauses the run; the scheduler never executes it."""
    from himmy.api import studio_service
    from himmy.api.routers import studio_notify

    async def _paused(*a: Any, **k: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "start"}
        yield {"type": "paused", "checkpoint_id": "cp1", "run_id": "r1"}

    monkeypatch.setattr(studio_service, "stream_agent_run", _paused)
    routine = _store_routine()

    out = await svc.execute_routine(routine.id)
    assert out is not None
    assert out.last_status == "awaiting_approval"

    notes = studio_notify._NOTIFICATIONS
    assert any(n["kind"] == "routine" and n["link"] == "/approvals" for n in notes)


@pytest.mark.asyncio
async def test_delivery_truncates_and_records_failure_without_raising(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from himmy.api import studio_connections as conns
    from himmy.api import studio_service

    long_text = "x" * 6000

    async def _ok(*a: Any, **k: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "done", "output_text": long_text}

    sent: list[dict[str, Any]] = []

    async def _send(ctype: str, payload: dict[str, Any]) -> conns.SendResult:
        sent.append({"ctype": ctype, **payload})
        return conns.SendResult(ok=True, detail="sent")

    class _Configured:
        configured = True

    monkeypatch.setattr(studio_service, "stream_agent_run", _ok)
    monkeypatch.setattr(conns, "get_connection", lambda _t: _Configured())
    monkeypatch.setattr(conns, "send_via_connection", _send)
    routine = _store_routine(deliver="telegram")

    out = await svc.execute_routine(routine.id)
    assert out is not None
    assert out.last_status == "ok"
    assert out.last_delivery is None  # delivered fine
    assert sent and sent[0]["ctype"] == "telegram"
    assert len(sent[0]["text"]) <= svc.DELIVERY_MAX_CHARS

    # a failing sender is recorded on the routine, never raised
    async def _fail(ctype: str, payload: dict[str, Any]) -> conns.SendResult:
        return conns.SendResult(ok=False, detail="boom")

    monkeypatch.setattr(conns, "send_via_connection", _fail)
    out2 = await svc.execute_routine(routine.id)
    assert out2 is not None
    assert out2.last_status == "ok"  # the run itself succeeded
    assert out2.last_delivery is not None and "boom" in out2.last_delivery


@pytest.mark.asyncio
async def test_delivery_skipped_when_connection_unconfigured(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from himmy.api import studio_connections as conns
    from himmy.api import studio_service

    async def _ok(*a: Any, **k: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "done", "output_text": "result"}

    class _Unconfigured:
        configured = False

    monkeypatch.setattr(studio_service, "stream_agent_run", _ok)
    monkeypatch.setattr(conns, "get_connection", lambda _t: _Unconfigured())
    routine = _store_routine(deliver="email")

    out = await svc.execute_routine(routine.id)
    assert out is not None
    assert out.last_status == "ok"
    assert out.last_delivery is not None and "not configured" in out.last_delivery


# ---- scheduler tick: enabled + overlap ------------------------------------------


@pytest.mark.asyncio
async def test_tick_skips_disabled_and_never_overlaps(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    calls: list[str] = []

    async def _fake_execute(routine_id: str, **_k: Any) -> None:
        calls.append(routine_id)
        await gate.wait()

    monkeypatch.setattr(svc, "execute_routine", _fake_execute)
    active = _store_routine()
    disabled = _store_routine(enabled=False)

    frozen = lambda: _at("2027-01-01T00:00:00+00:00")  # noqa: E731 - injected clock
    sched = svc.RoutineScheduler(now=frozen)

    launched = sched.tick()
    assert launched == [active.id]
    assert sched.is_running(active.id)
    assert not sched.is_running(disabled.id)

    await asyncio.sleep(0)  # let the run task start and block on the gate
    assert sched.tick() == []  # overlapping execution refused
    # run-now of a running routine is refused too
    with pytest.raises(svc.RoutineBusyError):
        await sched.run_now(active.id)

    gate.set()
    await asyncio.sleep(0.01)
    assert not sched.is_running(active.id)
    # fake executor never stamped last_run_at → still due → fires again
    assert sched.tick() == [active.id]
    await asyncio.sleep(0.01)  # let the second run start (gate already open)
    assert calls == [active.id, active.id]
    await sched.stop()


@pytest.mark.asyncio
async def test_tick_failure_never_kills_the_loop(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(routine_id: str, **_k: Any) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(svc, "execute_routine", _boom)
    routine = _store_routine()
    sched = svc.RoutineScheduler(now=lambda: _at("2027-01-01T00:00:00+00:00"))

    assert sched.tick() == [routine.id]
    await asyncio.sleep(0.01)  # the failing run is swallowed + logged
    assert not sched.is_running(routine.id)
    assert sched.tick() == [routine.id]  # the scheduler keeps scheduling
    await sched.stop()


def test_app_lifespan_starts_and_stops_the_loop(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "on")
    with TestClient(create_app()) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert svc.get_scheduler().active
    assert not svc.get_scheduler().active


def test_scheduler_env_off_disables_the_loop(workspace: Path) -> None:
    with TestClient(create_app()):
        assert not svc.get_scheduler().active


# ---- next_fire math (sleep-to-next-fire seam) -----------------------------------


def test_next_fire_already_due_returns_now() -> None:
    """A routine due at ``now`` returns ``now`` (fire on the next wake, no oversleep)."""
    now = _at("2026-06-10T07:00:00+00:00")
    r = _mk(schedule=svc.Schedule(kind="daily", at="07:00"))
    assert svc.is_due(r, now)
    assert svc.next_fire(r, now) == now


def test_next_fire_daily_points_at_the_next_hhmm() -> None:
    r = _mk(schedule=svc.Schedule(kind="daily", at="07:00"))
    # Not yet due (anchored on creation 10:00 the prior day); next boundary is tomorrow 07:00.
    now = _at("2026-06-10T06:00:00+00:00")
    assert not svc.is_due(r, now)
    nxt = svc.next_fire(r, now)
    assert nxt == _at("2026-06-10T07:00:00+00:00")


def test_next_fire_daily_rolls_to_tomorrow_after_its_time() -> None:
    r = _mk(
        schedule=svc.Schedule(kind="daily", at="07:00"),
        last_run_at="2026-06-10T07:00:05+00:00",
    )
    now = _at("2026-06-10T08:00:00+00:00")
    assert not svc.is_due(r, now)
    assert svc.next_fire(r, now) == _at("2026-06-11T07:00:00+00:00")


def test_next_fire_every_anchors_on_last_run() -> None:
    r = _mk(
        schedule=svc.Schedule(kind="every", hours=2),
        last_run_at="2026-06-09T12:00:00+00:00",
    )
    now = _at("2026-06-09T13:00:00+00:00")
    assert not svc.is_due(r, now)
    assert svc.next_fire(r, now) == _at("2026-06-09T14:00:00+00:00")


def test_next_fire_disabled_is_none() -> None:
    r = _mk(schedule=svc.Schedule(kind="every", hours=1), enabled=False)
    assert svc.next_fire(r, _at("2027-01-01T00:00:00+00:00")) is None


def test_next_fire_one_shot_at_after_firing_is_none() -> None:
    r = _mk(
        schedule=svc.Schedule(kind="at", at_datetime="2026-06-10T07:00:00+00:00"),
        last_run_at="2026-06-10T07:00:01+00:00",
    )
    # Already fired -> never again.
    now = _at("2026-06-10T08:00:00+00:00")
    assert not svc.is_due(r, now)
    assert svc.next_fire(r, now) is None


# ---- sleep-to-next-fire loop (no fixed 30s tick) --------------------------------


def test_max_sleep_cap_is_env_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_SCHEDULER_MAX_SLEEP", "5")
    assert svc.RoutineScheduler().tick_seconds == 5.0
    # A typo / non-positive value falls back to the 30s default (never wedges).
    monkeypatch.setenv("HIMMY_SCHEDULER_MAX_SLEEP", "nonsense")
    assert svc.RoutineScheduler().tick_seconds == 30.0
    monkeypatch.setenv("HIMMY_SCHEDULER_MAX_SLEEP", "-1")
    assert svc.RoutineScheduler().tick_seconds == 30.0
    # An explicit constructor value still wins (back-compat for tests).
    assert svc.RoutineScheduler(tick_seconds=2.0).tick_seconds == 2.0


def test_sleep_delay_is_time_to_soonest_boundary_capped(
    workspace: Path,
) -> None:
    # Cap at 100s; a routine due in ~3600s should plan a sleep of exactly the cap.
    sched = svc.RoutineScheduler(
        tick_seconds=100.0, now=lambda: _at("2026-06-10T06:00:00+00:00")
    )
    _store_routine(
        schedule=svc.Schedule(kind="daily", at="07:00"),
        last_run_at="2026-06-09T07:00:05+00:00",
    )
    # next boundary is 07:00 (3600s away) -> capped to 100s.
    assert sched._sleep_delay() == 100.0


def test_sleep_delay_no_routines_sleeps_the_full_cap(workspace: Path) -> None:
    sched = svc.RoutineScheduler(tick_seconds=42.0)
    assert sched._sleep_delay() == 42.0


def test_sleep_delay_due_routine_floors_not_zero(workspace: Path) -> None:
    # Already due -> tick promptly but never busy-spin at 0.0.
    sched = svc.RoutineScheduler(
        tick_seconds=30.0, now=lambda: _at("2027-01-01T00:00:00+00:00")
    )
    _store_routine()  # every-1h, created in the past -> due
    delay = sched._sleep_delay()
    assert 0 < delay <= 0.1


@pytest.mark.asyncio
async def test_loop_fires_within_a_second_of_a_boundary(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a routine that becomes due ~0.3s from now fires within ~1s.

    Uses real wall-clock + a small cap; the sleep-to-next-fire loop must plan a short
    sleep (not the old fixed 30s) and tick right after the boundary crosses.
    """
    fired = asyncio.Event()

    async def _fake_execute(routine_id: str, **_k: Any) -> None:
        fired.set()

    monkeypatch.setattr(svc, "execute_routine", _fake_execute)

    from datetime import UTC, datetime, timedelta

    boundary = datetime.now(UTC) + timedelta(seconds=0.3)
    # An 'at' one-shot that fires at the boundary; anchor (creation) is just before it.
    _store_routine(
        schedule=svc.Schedule(
            kind="at", at_datetime=boundary.isoformat()
        ),
        created_at=(boundary - timedelta(seconds=1)).isoformat(),
        last_run_at=None,
    )
    sched = svc.RoutineScheduler(tick_seconds=30.0)  # cap is high; planning must be tight
    sched.start()
    try:
        await asyncio.wait_for(fired.wait(), timeout=1.2)
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_loop_wakes_immediately_on_crud(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """notify_change() wakes the loop now instead of waiting out the (long) cap."""
    fired = asyncio.Event()

    async def _fake_execute(routine_id: str, **_k: Any) -> None:
        fired.set()

    monkeypatch.setattr(svc, "execute_routine", _fake_execute)
    # Huge cap: without wake-on-CRUD this test would hang ~999s.
    sched = svc.RoutineScheduler(
        tick_seconds=999.0, now=lambda: _at("2027-01-01T00:00:00+00:00")
    )
    sched.start()
    await asyncio.sleep(0.05)  # let the loop reach its long sleep
    try:
        # Add a routine that is immediately due, then nudge the loop.
        _store_routine()  # every-1h, created in the past -> due at the frozen now
        sched.notify_change()
        await asyncio.wait_for(fired.wait(), timeout=1.0)
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_loop_does_not_busy_spin_with_no_due_routines(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing due, tick() is called at most a couple of times over a short window."""
    ticks = 0
    real_tick = svc.RoutineScheduler.tick

    def _counting_tick(self: svc.RoutineScheduler) -> list[str]:
        nonlocal ticks
        ticks += 1
        return real_tick(self)

    monkeypatch.setattr(svc.RoutineScheduler, "tick", _counting_tick)
    # Routine far in the future; small cap so we'd notice a spin quickly.
    sched = svc.RoutineScheduler(
        tick_seconds=0.2, now=lambda: _at("2026-06-10T06:00:00+00:00")
    )
    _store_routine(
        schedule=svc.Schedule(kind="daily", at="07:00"),
        last_run_at="2026-06-09T07:00:05+00:00",
    )
    sched.start()
    try:
        await asyncio.sleep(0.5)  # ~2-3 cap-length sleeps, not thousands of spins
        assert ticks <= 5
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_crud_endpoint_wakes_the_running_scheduler(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creating a routine via the router sets the scheduler wakeup Event."""
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "on")
    monkeypatch.setenv("HIMMY_SCHEDULER_MAX_SLEEP", "999")

    woke = asyncio.Event()
    real_notify = svc.RoutineScheduler.notify_change

    def _spy(self: svc.RoutineScheduler) -> None:
        woke.set()
        real_notify(self)

    monkeypatch.setattr(svc.RoutineScheduler, "notify_change", _spy)

    # Start the scheduler in-loop so it is active when the endpoint fires.
    sched = svc.get_scheduler()
    sched.start()
    assert sched.active
    try:
        # The sync TestClient shares this running event loop's stores; drive the create
        # endpoint off-thread so the handler's _wake_scheduler() hits our spy.
        c = TestClient(create_app())
        resp = await asyncio.to_thread(
            c.post, "/api/studio/routines", json=_body()
        )
        assert resp.status_code == 200
        assert woke.is_set()
    finally:
        await sched.stop()


# ---- live (optional): real local model through the same rails -------------------


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable")
def test_run_now_live_ollama_mechanics(client: TestClient) -> None:
    """MECHANICS only: routine state + run persistence with a real local model."""
    rid = client.post(
        "/api/studio/routines",
        json=_body(
            name="live ping",
            prompt="Reply with exactly one short sentence.",
            provider="ollama",
            model=_LIVE_MODEL,
        ),
    ).json()["id"]

    res = client.post(f"/api/studio/routines/{rid}/run-now")
    assert res.status_code == 200
    body = res.json()
    assert body["last_status"] in ("ok", "error", "timeout")
    assert body["last_run_at"] is not None
    if body["last_status"] == "ok":
        assert body["last_preview"].strip() != ""
        runs = client.get("/api/studio/runs").json()
        assert runs["total"] >= 1
        assert runs["items"][0]["status"] == "ok"


# ---- rbac-harden(mopup-r1): routine lifecycle notifications carry the OWNING workspace


def test_notify_stamps_routine_owner_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_notify`` must stamp each notification with the routine's owning workspace.

    Regression for the cross-tenant notify leak: the routines notifier called
    ``record_notification`` with NO ``workspace_id``, so a NULL-stamped item (title = victim's
    routine NAME, body = victim's live LLM output/error) was visible to EVERY tenant. Assert
    the owning workspace is now threaded for ok / awaiting_approval / error.
    """
    captured: list[dict[str, Any]] = []

    import himmy.api.routers.studio_notify as sn

    def _spy(kind: str, title: str, **kw: Any) -> None:
        captured.append({"kind": kind, "title": title, **kw})

    monkeypatch.setattr(sn, "record_notification", _spy)

    r = svc.Routine(
        name="Pull Q3 revenue figures",
        workspace_id="tenant-A",
        agent_path="agent.yaml",
        prompt="...",
        schedule=svc.Schedule(kind="daily", at="07:00"),
    )
    svc._notify(r, "ok", "A's private agent output", None)
    svc._notify(r, "awaiting_approval", "", None)
    svc._notify(r, "error", "", "A's private error trace")

    assert captured, "no notifications recorded"
    for item in captured:
        assert item.get("workspace_id") == "tenant-A", (
            f"routine notification leaked NULL-tenant (visible to all): {item}"
        )


def test_notify_offline_local_workspace_is_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-box ``__local__`` workspace maps to None (NULL = shared, byte-unchanged)."""
    captured: list[dict[str, Any]] = []

    import himmy.api.routers.studio_notify as sn

    monkeypatch.setattr(
        sn,
        "record_notification",
        lambda kind, title, **kw: captured.append(kw),
    )

    r = svc.Routine(
        name="local routine",
        agent_path="agent.yaml",
        prompt="...",
        schedule=svc.Schedule(kind="daily", at="07:00"),
    )  # workspace_id defaults to LOCAL_WORKSPACE
    svc._notify(r, "ok", "ordinary output", None)
    assert captured and captured[0].get("workspace_id") is None
