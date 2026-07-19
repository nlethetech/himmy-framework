"""The leased run-queue dispatcher (Q3): worker-pull execution of QUEUED runs.

Before Q3 every run was fire-and-forgotten via ``asyncio.create_task`` the moment it was
created, so a process crash between create and completion lost the work, and the startup
sweep mass-FAILed every QUEUED/RUNNING run (reaping live peers under multi-worker). Q3
splits creation (ENQUEUE — persist a QUEUED run carrying its recoverable input) from
execution (this DISPATCHER claims + runs it). The result: a crash leaves a run QUEUED
(recoverable), a lease-expired RUNNING run is re-queued (never a live peer), and the #1
laptop transient — the local model not yet loaded — pauses claiming of the LOCAL lane
without starving cloud runs.

The dispatcher is a single in-lifespan task. It is started ONLY when the durable run store
is active (the queue's whole value requires durability — an in-memory run store loses
everything on exit, so there is nothing to recover). When it is NOT started — a bare
``create_app`` / CLI / ``TestClient`` that never enters the server lifespan, or the degraded
in-memory fallback — the :class:`~himmy.application.services.RunAppService` stays in INLINE
mode and runs every create on its own background task exactly as before, so a no-dispatcher
deployment never leaves a run stuck QUEUED forever.

Design (reusing the verified Q2 storage seams):

* CLAIM — ``claim_next_queued_run`` (the atomic CAS twin of ``claim_run_for_resume``) flips
  the oldest READY run to RUNNING with a lease, scoped to the currently-claimable LANES.
* HEALTH GATE — a cached liveness probe (Ollama ``/api/tags`` + the ``claude`` CLI presence)
  decides whether the LOCAL lane is claimable; when the local model is unreachable the
  dispatcher SLEEPS that lane (does NOT PARK its runs) and keeps draining cloud/default,
  resuming the local lane the instant the probe passes.
* EXECUTE — each claimed run runs on its own worker task via
  ``RunAppService.dispatch_claimed_run`` (rehydrate input -> _execute_on_runtime ->
  retry/backoff/PARK), with a sibling HEARTBEAT task renewing the lease so a live run is
  never reaped mid-flight.
* REAP — a periodic ``requeue_expired_leases`` re-queues a crashed worker's expired lease
  (only lease-expired RUNNING runs; never QUEUED, AWAITING_APPROVAL, RESOLVING, or a live
  peer), so a worker that dies mid-run is recovered without operator action.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from himmy.services.storage.run_lane import LANE_CLOUD, LANE_DEFAULT, LANE_LOCAL

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.application.services import RunAppService

logger = logging.getLogger("himmy.application.dispatcher")


def _metrics_registry() -> Any:
    """The process-wide metrics registry (resolved per call so test resets apply)."""
    from himmy.services.observability.metrics import get_registry

    return get_registry()

#: How long a cached local-provider liveness result is trusted before the next probe (s).
#: Short enough that "the model finished loading" is noticed within a claim cycle, long
#: enough that the probe is not hammered on every poll.
_HEALTH_CACHE_SECONDS = 10.0

#: The dispatcher's idle poll interval (s): how long it sleeps when there is nothing to claim
#: (or the only ready runs are in a gated lane). Small so a freshly-enqueued run starts
#: promptly; not zero, so an empty queue does not spin the loop.
_IDLE_POLL_SECONDS = 0.25

#: How often the reaper runs ``requeue_expired_leases`` (s) to recover a crashed worker's
#: lease. Independent of the claim poll so a busy dispatcher still reaps on schedule.
_REAP_INTERVAL_SECONDS = 15.0

#: The lease-heartbeat interval as a FRACTION of the lease TTL: a live worker renews its
#: lease this often, so several heartbeats land inside one lease window and a single missed
#: beat (a GC pause) does not lose the lease.
_HEARTBEAT_FRACTION = 0.3


class LocalProviderProbe:
    """A cached liveness check for the LOCAL model providers (Ollama + the ``claude`` CLI).

    The backend-health gate the dispatcher keys on: when the local model daemon is
    unreachable the LOCAL lane is not claimed (its runs wait, they are NOT parked), so a
    laptop whose Ollama is not yet up does not churn its queued local runs into failures.
    The result is cached for :data:`_HEALTH_CACHE_SECONDS` so the probe is cheap to consult
    every poll. A custom ``probe`` (sync ``() -> bool``) is injectable for tests.
    """

    def __init__(
        self,
        *,
        cache_seconds: float = _HEALTH_CACHE_SECONDS,
        probe: Any = None,
        clock: Any = None,
    ) -> None:
        self._cache_seconds = cache_seconds
        self._probe = probe
        self._clock = clock or time.monotonic
        self._cached: bool | None = None
        self._cached_at = 0.0

    async def local_available(self) -> bool:
        """Whether a LOCAL model provider is reachable right now (cached)."""
        now = self._clock()
        if self._cached is not None and (now - self._cached_at) < self._cache_seconds:
            return self._cached
        available = await self._do_probe()
        self._cached = available
        self._cached_at = now
        return available

    async def _do_probe(self) -> bool:
        if self._probe is not None:
            result = self._probe()
            if asyncio.iscoroutine(result):
                return bool(await result)
            return bool(result)
        # Default: the local model is "available" if Ollama answers /api/tags OR the claude
        # CLI is on PATH (Claude Max). A widen of the verified shutil.which presence check to
        # a live liveness GET, so "installed but daemon down" reads as unavailable.
        import shutil

        if shutil.which("claude") is not None:
            return True
        try:
            from himmy.api import studio_bench

            models = await studio_bench._ollama_models(1)
            return bool(models)
        except Exception:  # noqa: BLE001 - any probe failure = treat local as down
            return False


class RunDispatcher:
    """Worker-pull dispatcher for the leased run queue (Q3).

    Owns a claim loop + a reaper loop on the server event loop. Start it in the lifespan
    AFTER the durable store is wired (so the run store actually persists QUEUED runs); call
    :meth:`stop` on shutdown to drain. Construction does NOT start anything — :meth:`start`
    does, and it flips the :class:`RunAppService` into dispatch mode so new creates enqueue.
    """

    def __init__(
        self,
        run_app: RunAppService,
        *,
        owner_id: str | None = None,
        max_concurrency: int = 8,
        probe: LocalProviderProbe | None = None,
        reap_interval: float = _REAP_INTERVAL_SECONDS,
    ) -> None:
        """Wire the run service + concurrency/owner/health knobs (does not start)."""
        import os
        import uuid

        self._run_app = run_app
        self._owner_id = owner_id or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._max_concurrency = max(1, int(max_concurrency))
        self._probe = probe or LocalProviderProbe()
        self._reap_interval = reap_interval
        self._claim_task: asyncio.Task[Any] | None = None
        self._reap_task: asyncio.Task[Any] | None = None
        self._workers: set[asyncio.Task[Any]] = set()
        self._sem = asyncio.Semaphore(self._max_concurrency)
        self._stop = asyncio.Event()

    @property
    def owner_id(self) -> str:
        """This dispatcher's stable owner id (the lease holder stamp)."""
        return self._owner_id

    def start(self) -> None:
        """Flip the run service into dispatch mode + launch the claim + reaper loops.

        Idempotent. After this, ``RunAppService.create_run`` (and friends) ENQUEUE rather than
        fire-and-forget, and this dispatcher claims + executes the queued runs.
        """
        if self._claim_task is not None:
            return
        self._run_app.enable_dispatch()
        self._stop.clear()
        self._claim_task = asyncio.create_task(self._claim_loop(), name="himmy-dispatch")
        self._reap_task = asyncio.create_task(self._reap_loop(), name="himmy-reaper")

    async def stop(self, *, timeout: float = 30.0) -> None:
        """Signal stop, cancel the loops, and drain in-flight worker tasks (bounded)."""
        self._stop.set()
        for task in (self._claim_task, self._reap_task):
            if task is not None:
                task.cancel()
        loops = [t for t in (self._claim_task, self._reap_task) if t is not None]
        if loops:
            await asyncio.gather(*loops, return_exceptions=True)
        self._claim_task = None
        self._reap_task = None
        workers = list(self._workers)
        if workers:
            # Let in-flight runs finish (they renew their lease meanwhile); cancel on timeout.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*workers, return_exceptions=True), timeout=timeout
                )
            except TimeoutError:  # pragma: no cover - shutdown best-effort
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

    async def _claimable_lanes(self) -> list[str] | None:
        """The lanes the dispatcher may currently claim (the health gate; None = all).

        The CLOUD + DEFAULT lanes are always claimable (a hosted provider's reachability is
        not a local daemon's problem). The LOCAL lane is included ONLY when the local-provider
        probe passes; when it fails the lane is excluded, so a queued local run waits (it is
        NOT failed) until the daemon comes back.
        """
        if await self._probe.local_available():
            return None  # all lanes claimable
        return [LANE_CLOUD, LANE_DEFAULT]

    async def _claim_loop(self) -> None:
        """The poll-claim-dispatch loop: claim ready runs (respecting the health gate)."""
        while not self._stop.is_set():
            try:
                claimed_any = await self._claim_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a claim error must not kill the loop
                logger.warning("dispatcher claim cycle failed", exc_info=True)
                claimed_any = False
            if not claimed_any:
                # Nothing ready (or everything is in a gated lane / at capacity): idle briefly.
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=_IDLE_POLL_SECONDS
                    )
                except TimeoutError:
                    pass

    async def _claim_once(self) -> bool:
        """Claim up to the free capacity of ready runs; spawn a worker per claim.

        Returns True if at least one run was claimed (so the loop polls again immediately
        rather than idling), False when the queue is drained / gated / at capacity.
        """
        lanes = await self._claimable_lanes()
        claimed_any = False
        # Claim while there is free capacity AND ready work in a claimable lane.
        while not self._stop.is_set() and not self._sem.locked():
            # Acquire a concurrency slot up front; release it if nothing was claimable
            # (or if the claim raises) so a DB error can't permanently leak the permit.
            await self._sem.acquire()
            spawned = False
            try:
                run = await self._run_app.storage.claim_next_queued_run(
                    self._owner_id,
                    self._run_app.lease_seconds,
                    lanes=lanes,
                    fairness=self._run_app.dispatch_fairness,
                    workspace_concurrency=(
                        self._run_app.workspace_concurrency
                        if self._run_app.dispatch_fairness
                        else 0
                    ),
                )
                if run is None:
                    break
                claimed_any = True
                _metrics_registry().dispatcher_claims_total.inc()
                worker = asyncio.create_task(self._run_worker(run))
                self._workers.add(worker)
                worker.add_done_callback(self._on_worker_done)
                # The permit is now owned by the worker; it releases via _on_worker_done.
                spawned = True
            finally:
                if not spawned:
                    self._sem.release()
        return claimed_any

    def _on_worker_done(self, task: asyncio.Task[Any]) -> None:
        """Release the concurrency slot + drop the worker ref when a run finishes."""
        self._workers.discard(task)
        self._sem.release()

    async def _run_worker(self, run: Any) -> None:
        """Execute one claimed run while a sibling heartbeat renews its lease."""
        run_id = run.run_id
        heartbeat = asyncio.create_task(self._heartbeat(run_id))
        registry = _metrics_registry()
        registry.dispatcher_in_flight.inc()
        started = time.monotonic()
        try:
            await self._run_app.dispatch_claimed_run(run)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a worker crash must not kill the dispatcher
            logger.warning("dispatched run %s crashed", run_id, exc_info=True)
        finally:
            registry.dispatcher_in_flight.dec()
            registry.dispatcher_run_duration_seconds.observe(time.monotonic() - started)
            heartbeat.cancel()
            try:
                await heartbeat
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _heartbeat(self, run_id: str) -> None:
        """Periodically renew the run's lease so a LIVE run is never reaped mid-flight.

        Renews at :data:`_HEARTBEAT_FRACTION` of the lease TTL. If a renewal returns False the
        worker has LOST its lease (a reaper re-queued the run, another worker re-claimed it),
        so the heartbeat stops — the at-most-once guarantee then rests on the claim CAS, and
        this worker's terminal write will simply not match the new owner.
        """
        interval = max(1.0, self._run_app.lease_seconds * _HEARTBEAT_FRACTION)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return  # stop signalled
            except TimeoutError:
                pass
            try:
                renewed = await self._run_app.storage.renew_lease(
                    run_id, self._owner_id, self._run_app.lease_seconds
                )
            except Exception:  # noqa: BLE001 - a transient renew failure: try again next beat
                continue
            if not renewed:
                return  # lost the lease — stop heartbeating

    async def _reap_loop(self) -> None:
        """Periodically re-queue lease-expired RUNNING runs (crashed-worker recovery)."""
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._reap_interval)
                return
            except TimeoutError:
                pass
            try:
                requeued = await self._run_app.storage.requeue_expired_leases()
                if requeued:
                    _metrics_registry().dispatcher_reaps_total.inc(
                        (), float(len(requeued))
                    )
                    logger.info(
                        "reaper re-queued %d expired-lease run(s)", len(requeued)
                    )
            except Exception:  # noqa: BLE001 - a reaper error must not kill the loop
                logger.warning("dispatcher reaper cycle failed", exc_info=True)


__all__ = ["LocalProviderProbe", "RunDispatcher", "LANE_LOCAL"]
