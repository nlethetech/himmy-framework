"""Leased-dispatch retry/backoff/PARK policy for :class:`~himmy.application.services.RunAppService`.

Extracted from :mod:`himmy.application.services` as the ``RetryPolicyEngine`` collaborator in
the staged decomposition of ``RunAppService`` (LANE runapp, step 5 — retry): it owns the
near-pure Q3 policy that decides, AFTER a leased-dispatch run reaches its terminal state,
whether a FAILED run is a TRANSIENT blip worth a backed-off re-queue, an exhausted-budget PARK,
or a PERMANENT failure left untouched.

Behaviour is BYTE-IDENTICAL to the former inline implementation:

- :func:`is_transient_run_error` prefers the structured ``error_code`` (decisive in BOTH
  directions — a retryable code re-queues, a non-retryable code stays PERMANENT with no
  fall-through) and only falls back to the case-insensitive substring markers when no code is
  recorded, exactly as before,
- :meth:`RetryPolicyEngine.apply_retry_policy` reads the run's terminal state, classifies it,
  and RE-QUEUES with the SAME ``base * 2**(attempt-1)`` exponential backoff (capped) while
  attempts AND age remain — writing ``next_attempt_at`` in the future so the claim CAS honours
  the delay across a restart — else PARKS it, preserving the last error; a permanent failure is
  left FAILED untouched,
- the engine reads its storage handle LIVE through the shared :class:`_RunContext` (never a
  construction-time snapshot), so a re-pointed ``storage`` is observed at once.

``RunAppService``'s former private method (``_apply_retry_policy``) delegates here as a thin
shim, so every internal caller and test poke stays byte-identical. The module-level
``_is_transient_run_error`` / ``_TRANSIENT_ERROR_MARKERS`` / ``DEFAULT_QUEUE_MAX_AGE_SECONDS`` /
``_QUEUE_BACKOFF_*`` names remain importable from :mod:`himmy.application.services` (re-exported
there), so any existing import path is unchanged.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from himmy.application.services import (
    _iso_plus_seconds,
    _now,
    logger,
)
from himmy.services.inference.models import RETRYABLE_ERROR_CODES
from himmy.services.storage.models import RunStatus

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.application.run_context import _RunContext


#: Backoff schedule for a transient-failed run's re-queue (Q3): ``base * 2**(attempt-1)``
#: seconds, capped at ``max``. Mirrors the :class:`~himmy.connectors.sdk.RetryPolicy`
#: exponential discipline, but expressed as a DB ``next_attempt_at`` delay so the backoff
#: survives a process restart (it is the gate the claim CAS already honours).
_QUEUE_BACKOFF_BASE_SECONDS = 2.0
_QUEUE_BACKOFF_MAX_SECONDS = 300.0

#: Hard age ceiling (seconds) for a run in the leased queue (Q3): once a run has been in
#: flight (created_at -> now) longer than this it is PARKED even if attempts remain, so a
#: run can't be re-queued forever on a persistently-flapping backend.
DEFAULT_QUEUE_MAX_AGE_SECONDS = 3600.0


#: Substrings that mark a run failure as TRANSIENT (worth a re-queue), matched case-
#: insensitively against the recorded ``error`` (Q3). These are the recoverable conditions a
#: laptop/offline deployment hits — the provider was briefly unreachable, a request timed out,
#: the local model was still loading. Everything NOT matching is treated as PERMANENT (a
#: validation/build/tool error that re-running cannot fix), so the dispatcher does not churn on
#: a genuinely-broken run. Conservative on purpose: it is safe to leave an ambiguous failure
#: PERMANENT (the operator can redrive) — re-queuing a truly-permanent one wastes attempts.
_TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "connect error",
    "temporarily unavailable",
    "unreachable",
    "rate limit",
    "429",
    "503",
    "502",
    "504",
    "overloaded",
    "provider",
    "econnreset",
    "broken pipe",
    "model not found",  # ollama: the model is not (yet) pulled/loaded
    "no route to host",
)


def is_transient_run_error(error: str, error_code: str | None = None) -> bool:
    """Whether a recorded run failure looks TRANSIENT (re-queuable) vs PERMANENT (Q3).

    Prefers the structured inference ``error_code`` when one was recorded — it is the
    authoritative taxonomy and avoids the free-text substring traps the marker list has
    (``'provider'`` falsely matching the permanent ``'no provider configured'``, or
    ``'Temporary failure in name resolution'`` going unmatched). A structured code is
    decisive in BOTH directions: a retryable code re-queues, a non-retryable code is left
    PERMANENT (no fall-through to the over-broad markers). The substring match is kept only
    as the last-resort fallback for failures that carry no code (the runtime-build/timeout/
    cancel paths record free text but no structured code).
    """
    if error_code:
        return error_code in {code.value for code in RETRYABLE_ERROR_CODES}
    if not error:
        return False
    low = error.lower()
    return any(marker in low for marker in _TRANSIENT_ERROR_MARKERS)


class RetryPolicyEngine:
    """Post-terminal-state Q3 retry/backoff/PARK policy for a leased-dispatch run.

    Holds no state of its own beyond the shared context handle; reads ``storage`` live from
    the context so behaviour matches the former inline implementation byte-for-byte.
    """

    def __init__(self, *, context: _RunContext) -> None:
        """Wire the shared run-lifecycle context (source of the live storage handle)."""
        self._ctx = context

    async def apply_retry_policy(self, run_id: str) -> None:
        """Re-queue a transient-failed run with backoff, else PARK it (Q3).

        Reads the run's terminal state after :meth:`_execute_on_runtime`. Only a FAILED run is
        considered — SUCCEEDED is done; AWAITING_APPROVAL/RESOLVING are paused (NOT failures).
        A FAILED run is classified transient (provider/timeout/connection blip) vs permanent
        (validation, build, unknown-tool). A transient failure with attempts AND age remaining
        is RE-QUEUED with exponential backoff (``next_attempt_at`` in the future, so the claim
        CAS leaves it until then — the backoff survives a restart); otherwise it is PARKED
        (terminal-but-redrivable) so an operator can intervene, distinct from a clean FAILED.
        A permanent failure is left FAILED untouched.
        """
        run = await self._ctx.storage.get_run(run_id)
        if run is None or run.status != RunStatus.FAILED:
            return
        error = run.error or ""
        error_code = (run.metadata or {}).get("error_code")
        if not is_transient_run_error(error, error_code):
            return  # permanent failure: leave it FAILED for the operator/caller.
        from himmy.application.services import _parse_iso_epoch

        age = time.time() - _parse_iso_epoch(run.created_at)
        attempts_left = run.attempt < max(1, run.max_attempts)
        if attempts_left and age < DEFAULT_QUEUE_MAX_AGE_SECONDS:
            delay = min(
                _QUEUE_BACKOFF_BASE_SECONDS * (2.0 ** max(0, run.attempt - 1)),
                _QUEUE_BACKOFF_MAX_SECONDS,
            )
            now_iso = _now()
            run.status = RunStatus.QUEUED
            run.owner_id = None
            run.lease_expires_at = None
            run.last_error = error
            run.error = None
            run.next_attempt_at = _iso_plus_seconds(now_iso, delay)
            run.updated_at = now_iso
            await self._ctx.storage.save_run(run)
            logger.info(
                "re-queued transient-failed run %s (attempt %d/%d) in %.0fs",
                run_id,
                run.attempt,
                run.max_attempts,
                delay,
            )
            return
        # Budget exhausted: PARK (terminal-but-redrivable), preserving the last error.
        run.status = RunStatus.PARKED
        run.last_error = error
        run.owner_id = None
        run.lease_expires_at = None
        run.updated_at = _now()
        await self._ctx.storage.save_run(run)
        logger.info(
            "parked run %s after %d attempt(s) of transient failure", run_id, run.attempt
        )
