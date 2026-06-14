"""Q3: leased run-queue dispatcher — enqueue/claim/recover/retry/health-gate (offline).

Drives the dispatcher against a DURABLE SQLite run store (so input_blob round-trips + a run
survives a "restart" = a fresh store on the same file) with the deterministic
:class:`StubClientManager`. Covers:

* ENQUEUE-not-launch: in dispatch mode ``create_run`` leaves the run QUEUED with recoverable
  input and does NOT fire a background task — a crash before dispatch leaves it recoverable.
* CLAIM + EXECUTE: the dispatcher claims a QUEUED run and drives it to SUCCEEDED, from its
  persisted input alone (a fresh RunAppService — the "fresh process" case).
* REAPER: a lease-expired RUNNING run is re-queued; a LIVE peer (future lease) is never reaped.
* RETRY/BACKOFF/PARK: a transient failure re-queues with backoff; an exhausted budget PARKs;
  a permanent failure is left FAILED.
* HEALTH GATE: with the local probe DOWN the local lane is excluded from claims (its run
  waits, not fails), and resumes when the probe passes.
* NO-DISPATCHER FALLBACK: an inline RunAppService (no dispatcher) runs the create immediately.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.application import RecommendationAppService, RunAppService
from himmy.application.dispatcher import LocalProviderProbe, RunDispatcher
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.run_lane import LANE_CLOUD, LANE_LOCAL
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _durable_stack(db_path: str) -> tuple[SqliteStorageService, RunAppService]:
    """A RunAppService over a DURABLE SQLite store with the deterministic stub runtime.

    ``SqliteStorageService`` is structurally a ``StorageService`` (same async store surface);
    the casts keep the test mypy-clean against the constructors' nominal ``StorageService``
    annotation without weakening the production types.
    """
    storage = SqliteStorageService(db_path)
    typed = cast(StorageService, storage)
    registry = EntityRegistry()
    context = ContextService(storage_service=typed, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )
    run_app = RunAppService(
        runtime=runtime,
        storage=typed,
        entity_registry=registry,
        recommendation_app=RecommendationAppService(storage=typed),
    )
    return storage, run_app


def test_dispatch_mode_enqueues_without_launching() -> None:
    """In dispatch mode create_run persists a QUEUED run with input but DOES NOT run it."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, run_app = _durable_stack(db)
    run_app.enable_dispatch()

    async def go() -> RunRecord | None:
        run = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=Persona(name="A"),
            task=Task(title="t", prompt="hi"),
        )
        assert run.status == RunStatus.QUEUED
        assert run.input_blob  # recoverable input was persisted
        assert run.max_attempts >= 1
        # No dispatcher running -> it stays QUEUED (recoverable), never auto-executed.
        return await storage.get_run(run.run_id)

    after = run_async(go())
    assert after is not None and after.status == RunStatus.QUEUED


def test_dispatcher_claims_and_executes_from_persisted_input() -> None:
    """A QUEUED run is claimed + driven to SUCCEEDED by a FRESH-process dispatcher."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    # Process 1: enqueue only (dispatch mode, no dispatcher running).
    _s1, enqueuer = _durable_stack(db)
    enqueuer.enable_dispatch()

    async def enqueue() -> str:
        run = await enqueuer.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=Persona(name="A"),
            task=Task(title="t", prompt="hello there"),
        )
        return run.run_id

    run_id = run_async(enqueue())

    # Process 2 (fresh store + service on the SAME file): the dispatcher recovers + runs it.
    storage2, run_app2 = _durable_stack(db)
    # Force the local probe ON so the (default-lane) run is claimable in tests.
    dispatcher = RunDispatcher(
        run_app2, probe=LocalProviderProbe(probe=lambda: True)
    )

    async def drive() -> RunRecord | None:
        dispatcher.start()
        try:
            terminal = await run_app2.await_run(run_id, timeout=5.0)
        finally:
            await dispatcher.stop()
        return terminal

    terminal = run_async(drive())
    assert terminal is not None
    assert terminal.status == RunStatus.SUCCEEDED
    assert terminal.output_text
    assert terminal.attempt == 1  # claimed exactly once


def test_reaper_requeues_expired_lease_never_a_live_peer() -> None:
    """requeue_expired_leases re-queues an expired RUNNING lease; a live peer is untouched."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, _run_app = _durable_stack(db)

    async def go() -> None:
        # Two QUEUED runs.
        dead_id, live_id = "dead", "live"
        for rid in (dead_id, live_id):
            await storage.save_run(
                RunRecord(
                    run_id=rid, workspace_id="w", subject_id="s",
                    status=RunStatus.QUEUED,
                )
            )
        # Claim ``dead`` with a lease that is already expired (negative seconds).
        claimed_dead = await storage.claim_next_queued_run("worker-dead", -10)
        assert claimed_dead is not None
        # Claim ``live`` with a long lease (a live peer).
        claimed_live = await storage.claim_next_queued_run("worker-live", 300)
        assert claimed_live is not None
        requeued = await storage.requeue_expired_leases()
        # Only the expired lease is re-queued; the live peer is left RUNNING.
        running = {r.run_id for r in await storage.list_runs(status=RunStatus.RUNNING)}
        queued = {r.run_id for r in await storage.list_runs(status=RunStatus.QUEUED)}
        assert claimed_dead.run_id in requeued
        assert claimed_live.run_id not in requeued
        assert claimed_live.run_id in running
        assert claimed_dead.run_id in queued

    run_async(go())


def test_retry_policy_requeues_transient_then_parks() -> None:
    """A transient FAILED run re-queues with backoff; an exhausted budget PARKs it."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, run_app = _durable_stack(db)
    run_app.enable_dispatch()

    async def go() -> None:
        # attempt 1 of 2, transient error -> re-queued with a future next_attempt_at.
        run = RunRecord(
            run_id="r1", workspace_id="w", subject_id="s",
            status=RunStatus.FAILED, error="connection timeout to provider",
            attempt=1, max_attempts=2,
        )
        await storage.save_run(run)
        await run_app._apply_retry_policy("r1")
        back = await storage.get_run("r1")
        assert back is not None and back.status == RunStatus.QUEUED
        assert back.last_error and "timeout" in back.last_error
        assert back.error is None
        assert back.next_attempt_at is not None

        # attempt 2 of 2 (budget exhausted), transient -> PARKED, last_error preserved.
        run2 = await storage.get_run("r1")
        assert run2 is not None
        run2.status = RunStatus.FAILED
        run2.error = "connection timeout to provider"
        run2.attempt = 2
        await storage.save_run(run2)
        await run_app._apply_retry_policy("r1")
        parked = await storage.get_run("r1")
        assert parked is not None and parked.status == RunStatus.PARKED
        assert parked.last_error and "timeout" in parked.last_error

    run_async(go())


def test_retry_policy_leaves_permanent_failure_failed() -> None:
    """A PERMANENT failure (not transient) is left FAILED, never re-queued/parked."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, run_app = _durable_stack(db)
    run_app.enable_dispatch()

    async def go() -> None:
        run = RunRecord(
            run_id="perm", workspace_id="w", subject_id="s",
            status=RunStatus.FAILED, error="schema validation failed: bad field",
            attempt=1, max_attempts=3,
        )
        await storage.save_run(run)
        await run_app._apply_retry_policy("perm")
        after = await storage.get_run("perm")
        assert after is not None and after.status == RunStatus.FAILED

    run_async(go())


def test_health_gate_excludes_local_lane_when_probe_down() -> None:
    """With the local probe DOWN the dispatcher claims cloud/default but leaves local QUEUED."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, run_app = _durable_stack(db)
    run_app.enable_dispatch()

    async def go() -> None:
        # A local-lane run + a cloud-lane run, both QUEUED.
        await storage.save_run(
            RunRecord(
                run_id="loc", workspace_id="w", subject_id="s",
                status=RunStatus.QUEUED, lane_key=LANE_LOCAL,
                model_key="ollama:llama3",
            )
        )
        await storage.save_run(
            RunRecord(
                run_id="cld", workspace_id="w", subject_id="s",
                status=RunStatus.QUEUED, lane_key=LANE_CLOUD,
                model_key="anthropic:sonnet",
            )
        )
        # Probe DOWN -> claimable lanes exclude LOCAL.
        dispatcher = RunDispatcher(
            run_app, probe=LocalProviderProbe(probe=lambda: False)
        )
        lanes = await dispatcher._claimable_lanes()
        assert lanes is not None and LANE_LOCAL not in lanes
        # A claim under those lanes grabs the cloud run, never the local one.
        claimed = await storage.claim_next_queued_run("w0", 300, lanes=lanes)
        assert claimed is not None and claimed.run_id == "cld"
        # The local run is still QUEUED (waiting, not failed).
        loc = await storage.get_run("loc")
        assert loc is not None and loc.status == RunStatus.QUEUED

        # Probe back UP -> all lanes claimable -> the local run is now claimable.
        dispatcher2 = RunDispatcher(
            run_app, probe=LocalProviderProbe(probe=lambda: True)
        )
        assert await dispatcher2._claimable_lanes() is None

    run_async(go())


def test_orchestration_run_is_claimable_under_default_lane_when_probe_down() -> None:
    """An enqueued ORCHESTRATION run drains through the neutral DEFAULT lane (probe down).

    Regression: create_orchestration_run must stamp ``lane_key = LANE_DEFAULT`` (not NULL).
    The claim filter is ``lane_key IN (...)`` and SQL NULL never matches an IN/ANY list, so a
    NULL-lane orchestration run would sit QUEUED forever exactly when the local probe gates out
    the local lane — the laptop transient the health gate is meant to drain through. This drives
    the REAL enqueue path (create_orchestration_run) and then claims under the probe-down lanes.
    """
    from himmy.services.storage.models import AgentDefRecord
    from himmy.services.storage.run_lane import LANE_DEFAULT

    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, run_app = _durable_stack(db)
    run_app.enable_dispatch()

    async def go() -> None:
        member = AgentDefRecord(workspace_id="w", name="analyst")
        run = await run_app.create_orchestration_run(
            workspace_id="w",
            subject_id="s",
            kind="multi_agent",
            members=[member],
            prompt="diagnose the outage",
            resource_kind="team",
            resource_id="team-1",
        )
        # Enqueued (recoverable), and stamped onto the always-included neutral lane.
        assert run.status == RunStatus.QUEUED
        assert run.lane_key == LANE_DEFAULT

        # Probe DOWN -> claimable lanes are [CLOUD, DEFAULT]; the orchestration run MUST be
        # claimable (it would be invisible to ``lane_key IN (...)`` had the lane stayed NULL).
        dispatcher = RunDispatcher(
            run_app, probe=LocalProviderProbe(probe=lambda: False)
        )
        lanes = await dispatcher._claimable_lanes()
        assert lanes is not None and LANE_DEFAULT in lanes
        claimed = await storage.claim_next_queued_run("w0", 300, lanes=lanes)
        assert claimed is not None and claimed.run_id == run.run_id

    run_async(go())


def test_inline_mode_runs_immediately_without_a_dispatcher() -> None:
    """No-dispatcher fallback: an INLINE RunAppService executes the create right away."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    _storage, run_app = _durable_stack(db)
    # dispatch NOT enabled -> inline fire-and-forget (the bare TestClient/CLI path).

    async def go() -> RunRecord | None:
        run = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=Persona(name="A"),
            task=Task(title="t", prompt="hi"),
        )
        assert run.status == RunStatus.QUEUED
        # The inline background task drives it to terminal with no dispatcher present.
        return await run_app.await_run(run.run_id, timeout=5.0)

    terminal = run_async(go())
    assert terminal is not None and terminal.status == RunStatus.SUCCEEDED


def test_run_with_no_recoverable_input_fails_loud_on_dispatch() -> None:
    """A claimed run with no input_blob fails with a clear reason (never silently dropped)."""
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, run_app = _durable_stack(db)
    run_app.enable_dispatch()

    async def go() -> None:
        run = RunRecord(
            run_id="noinput", workspace_id="w", subject_id="s",
            status=RunStatus.RUNNING, attempt=1, max_attempts=1, owner_id="w0",
        )
        await storage.save_run(run)
        await run_app.dispatch_claimed_run(run)
        after = await storage.get_run("noinput")
        assert after is not None and after.status == RunStatus.FAILED
        assert after.error and "recoverable input" in after.error

    run_async(go())


def test_local_probe_caches_result() -> None:
    """The local-provider probe caches its result for the cache window."""
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return True

    p = LocalProviderProbe(probe=probe, cache_seconds=1000.0)

    async def go() -> None:
        assert await p.local_available() is True
        assert await p.local_available() is True  # cached: probe not called again

    run_async(go())
    assert calls == [1]
