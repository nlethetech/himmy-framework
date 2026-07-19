"""Startup/shutdown recovery lifecycle for :class:`~himmy.application.services.RunAppService`.

Extracted from :mod:`himmy.application.services` as the ``RunRecovery`` collaborator in the
staged decomposition of ``RunAppService`` (LANE runapp, step 6 — recovery): it owns the small
cohesive band of process-lifecycle methods that cancel/await in-flight work on shutdown and
recover runs a dead process left non-terminal on the next startup.

Behaviour is BYTE-IDENTICAL to the former inline methods:

- :meth:`drain` cancels + awaits EVERY in-flight background task through the SINGLE shared
  ``_tasks`` set on the :class:`_RunContext` (never a private set — a private set would miss
  the resume/execute/orchestration tasks and hang shutdown while losing the cancel->FAILED
  transition), bounded by ``timeout`` so shutdown cannot hang forever,
- :meth:`sweep_stuck_runs` re-queues ONLY lease-expired RUNNING runs in DISPATCH mode (via the
  Q2 reaper) and otherwise fails-loud any abandoned QUEUED/RUNNING run in INLINE mode, in both
  cases preserving AWAITING_APPROVAL / RESOLVING,
- :meth:`reconcile_resolving_runs` re-drives a crashed HITL resume stranded at RESOLVING by
  re-opening it to AWAITING_APPROVAL and delegating back through ``resume_run`` (the SAME proven
  claim path), skipping any run without a recorded decision,
- the collaborator reads its ``storage`` / ``tasks`` / ``dispatch_enabled`` handles LIVE through
  the shared :class:`_RunContext` (never a construction-time snapshot), so a re-pointed
  ``storage`` and the current dispatch mode are observed at once.

``RunAppService``'s former methods (``drain`` / ``sweep_stuck_runs`` /
``reconcile_resolving_runs``) delegate here as thin shims, so every FastAPI lifespan hook,
internal caller, and test poke stays byte-identical.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from himmy.application.services import (
    RunNotApprovableError,
    _now,
    logger,
)
from himmy.services.storage.models import RunStatus

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from collections.abc import Awaitable, Callable

    from himmy.application.run_context import _RunContext
    from himmy.services.storage.models import RunRecord


class RunRecovery:
    """Process-lifecycle recovery (shutdown drain + startup sweep/reconcile).

    Holds no state of its own beyond the shared context handle and a back-reference to
    ``resume_run`` (which :meth:`reconcile_resolving_runs` re-drives through). Reads
    ``storage`` / ``tasks`` / ``dispatch_enabled`` live from the context so behaviour matches
    the former inline implementation byte-for-byte.
    """

    def __init__(
        self,
        *,
        context: _RunContext,
        resume_run: Callable[..., Awaitable[RunRecord]],
    ) -> None:
        """Wire the shared context (live handles) and the ``resume_run`` re-drive callback."""
        self._ctx = context
        self._resume_run = resume_run

    async def drain(self, *, timeout: float = 30.0) -> None:
        """Cancel + await all in-flight background runs (FastAPI shutdown hook).

        Each task is cancelled and awaited; the per-task cancel handler records
        the run FAILED('run cancelled'). Bounded by ``timeout`` so shutdown cannot
        hang forever.
        """
        tasks = list(self._ctx.tasks)
        if not tasks:
            return
        for t in tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except TimeoutError:  # pragma: no cover - shutdown best-effort
            logger.warning(
                "drain timed out after %.0fs with tasks still running", timeout
            )

    async def sweep_stuck_runs(self, *, ttl_seconds: float = 0.0) -> list[str]:
        """Recover runs left non-terminal by a dead process (AAEO-1; Q3 rewrite).

        Intended for startup. The behaviour now depends on whether the leased dispatcher owns
        execution:

        * DISPATCH mode (the durable store + lifespan dispatcher): a crashed worker's RUNNING
          run holds an EXPIRED lease, so it is RE-QUEUED (not failed) via the Q2 reaper
          ``requeue_expired_leases`` — the dispatcher then re-claims and re-executes it from
          its recoverable input. A QUEUED run is NOT touched (it is already recoverable — the
          dispatcher will claim it); a LIVE peer's RUNNING run (lease not yet expired) is NEVER
          re-queued; AWAITING_APPROVAL / RESOLVING are preserved (HITL pauses + re-drivable
          resumes). This is the fix for the old "crash -> mass-FAIL" and "reap a live peer"
          behaviours. Returns the re-queued run ids.

        * INLINE mode (no dispatcher — a bare ``create_app`` / CLI / TestClient): there is
          nothing to re-claim a QUEUED/RUNNING run, so the old fail-loud sweep is preserved
          byte-for-byte — those runs are transitioned to FAILED so they do not hang forever.
          ``ttl_seconds=0`` sweeps all; AWAITING_APPROVAL / RESOLVING are still excluded.
        """
        if self._ctx.dispatch_enabled:
            # Re-queue ONLY lease-expired RUNNING runs; a live peer (future lease) and every
            # QUEUED/AWAITING/RESOLVING run is left untouched by the lease predicate.
            requeued = await self._ctx.storage.requeue_expired_leases()
            if requeued:
                logger.info(
                    "re-queued %d lease-expired run(s) on startup", len(requeued)
                )
            return requeued

        from himmy.application.services import _parse_iso_epoch

        swept: list[str] = []
        runs = await self._ctx.storage.list_runs()
        now = time.time()
        for run in runs:
            # Only QUEUED/RUNNING are "abandoned"; SUCCEEDED/FAILED/PARKED are terminal,
            # AWAITING_APPROVAL is intentionally paused, and RESOLVING is a re-drivable
            # crashed resume (reconcile_resolving_runs) — never sweep any of those.
            if run.status not in (RunStatus.QUEUED, RunStatus.RUNNING):
                continue
            if (
                ttl_seconds > 0
                and (now - _parse_iso_epoch(run.updated_at)) < ttl_seconds
            ):
                continue
            run.status = RunStatus.FAILED
            run.error = "run abandoned (process restart); swept to terminal state"
            run.updated_at = _now()
            await self._ctx.storage.save_run(run)
            swept.append(run.run_id)
        return swept

    async def reconcile_resolving_runs(self) -> list[str]:
        """Re-drive HITL resumes left at RESOLVING by a crash, exactly-once (UNIT 1d).

        A resume that wins the run-level claim flips the run to RESOLVING and persists the
        approve/reject decision BEFORE launching its background task. If the process dies
        between the claim and the run reaching a terminal/AWAITING-again state, the run is
        stranded at RESOLVING — but it is RE-DRIVABLE: the member checkpoint's own
        ``claim()`` accepts a RESOLVING checkpoint on retry, and the per-tool idempotency
        ledger replays an already-executed gated tool, so re-driving the recorded decision
        executes the side effect at most once total (across the crashed attempt + this
        recovery). Intended for startup, after :meth:`sweep_stuck_runs` (which never reaps
        a RESOLVING run). Returns the re-driven run ids.

        This is DISTINCT from a concurrent second approve: that loses the run-level CAS at
        the inbox and 409s without launching anything (no RESOLVING strand to recover);
        only an in-flight crash leaves a RESOLVING run for this method to pick up.
        """
        redriven: list[str] = []
        runs = await self._ctx.storage.list_runs(status=RunStatus.RESOLVING)
        for run in runs:
            decision = (run.metadata or {}).get("resume_decision")
            if decision not in ("approved", "rejected"):
                # No recorded decision — cannot faithfully re-drive; leave it for an
                # operator (never guess approve, which would run the gated tool).
                continue
            actor = (run.metadata or {}).get("resume_actor") or "recovery"
            # Re-open to AWAITING_APPROVAL so resume_run re-claims it through the SAME
            # proven path. Startup recovery is single-threaded (no concurrent approves
            # yet), so the brief re-open carries no double-resume risk.
            run.status = RunStatus.AWAITING_APPROVAL
            run.updated_at = _now()
            await self._ctx.storage.save_run(run)
            try:
                await self._resume_run(
                    run.run_id,
                    approved=decision == "approved",
                    workspace_id=run.workspace_id,
                    actor=actor,
                )
            except RunNotApprovableError:  # pragma: no cover - lost a concurrent re-claim
                continue
            redriven.append(run.run_id)
        return redriven
