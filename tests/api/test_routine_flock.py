"""T3c — cross-process single-flight for routines (the flock reviewer must_fix).

``RoutineScheduler._running`` only guards same-process overlap; a ``himmy routines
run-now`` runs in a SEPARATE process that cannot see that registry. These tests assert
the host-level ``flock`` guard so the CLI and the Studio scheduler can never double-
execute one routine:

* the :func:`process_lock` primitive refuses a second concurrent holder of one name;
* ``execute_routine`` (the single execution choke point every seam funnels through) holds
  that lock for the whole run, so a concurrent attempt raises ``RoutineBusyError`` and the
  routine's side-effecting body runs exactly ONCE.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from himmy.api import routines as svc
from himmy.core.process_lock import ProcessLockBusy, process_lock


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    svc.reset_routines_store()
    svc.reset_scheduler()
    yield tmp_path
    svc.reset_routines_store()
    svc.reset_scheduler()


def test_process_lock_refuses_a_second_holder(tmp_path: Path) -> None:
    """A second acquire of the same name (even in-process) is refused immediately."""
    with process_lock("dup", base=tmp_path):
        with pytest.raises(ProcessLockBusy):
            with process_lock("dup", base=tmp_path):
                pass
    # Released — re-acquirable.
    with process_lock("dup", base=tmp_path):
        pass


def test_process_lock_distinct_names_coexist(tmp_path: Path) -> None:
    """Different lock names do not contend."""
    with process_lock("a", base=tmp_path):
        with process_lock("b", base=tmp_path):
            pass


def _store_local_routine(project: Path) -> svc.Routine:
    (project / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    routine = svc.Routine(
        name="r",
        agent_path="agent.yaml",
        prompt="hi",
        schedule=svc.Schedule(kind="every", hours=1),
        provider="stub",
    )
    return svc.get_routines_store().upsert(routine)


@pytest.mark.asyncio
async def test_execute_routine_refuses_when_lock_already_held(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent run-now while the routine's flock is held raises RoutineBusyError.

    Simulating "another process is running it" by holding the same-named lock makes the
    guarded ``execute_routine`` refuse rather than execute the run a second time.
    """
    routine = _store_local_routine(project)

    ran: list[str] = []

    async def _fake_headless(_r: svc.Routine) -> tuple[str, str, str | None]:
        ran.append(_r.id)
        return "ok", "done", None

    monkeypatch.setattr(svc, "_run_headless", _fake_headless)

    # Hold the routine's lock (stands in for the Studio scheduler process owning it).
    with process_lock(svc.routine_lock_name(routine.id)):
        with pytest.raises(svc.RoutineBusyError):
            await svc.execute_routine(routine.id)
    # The guarded body never ran.
    assert ran == []

    # Once released, a run-now proceeds and runs the body exactly once.
    out = await svc.execute_routine(routine.id)
    assert out is not None and out.last_status == "ok"
    assert ran == [routine.id]


@pytest.mark.asyncio
async def test_two_concurrent_executes_run_body_once(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent execute_routine calls: one runs, the other is refused (no double)."""
    routine = _store_local_routine(project)
    started = asyncio.Event()
    release = asyncio.Event()
    ran: list[str] = []

    async def _slow_headless(_r: svc.Routine) -> tuple[str, str, str | None]:
        ran.append(_r.id)
        started.set()
        await release.wait()
        return "ok", "done", None

    monkeypatch.setattr(svc, "_run_headless", _slow_headless)

    first = asyncio.create_task(svc.execute_routine(routine.id))
    await started.wait()  # the first run holds the flock and is mid-body
    # A second concurrent attempt is refused by the cross-process flock.
    with pytest.raises(svc.RoutineBusyError):
        await svc.execute_routine(routine.id)
    release.set()
    out = await first
    assert out is not None and out.last_status == "ok"
    assert ran == [routine.id]  # ran exactly once
