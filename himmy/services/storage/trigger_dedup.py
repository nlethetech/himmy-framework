"""Durable inbound-trigger de-duplication backed by the ``trigger_dedup`` table (Q4).

The connector layer ships a process-local :class:`~himmy.connectors.sdk.IdempotencyStore`
(a ``threading.Lock`` over two in-RAM dicts) so a re-delivered webhook fires the agent at
most once. That guarantee evaporates on restart: the dict is gone, so a re-delivery that
arrives after a crash/redeploy fires the agent AGAIN — the cross-restart double-fire the
Q4 plan item closes. This module backs the same dedup contract with the durable
``trigger_dedup`` table (created by the Q1 migration on both SQLite and Postgres) so a
delivery id seen before a restart is still deduped after it, while a TTL-expired key is
allowed to fire again.

THE MARK-BEFORE-RUN HOLE (and why a naive durable store is WORSE, not better). The original
webhook path recorded the delivery id as consumed BEFORE running the agent. With an in-RAM
store a crash loses the mark AND the run together, so a redelivery cleanly re-runs. Make
that mark durable naively and a crash AFTER the mark but BEFORE the agent finishes turns a
transient lost delivery into a PERMANENT one: the mark survives, the work never happened,
and every redelivery is deduped into oblivion. The fix is a two-phase, *mark-after-success*
protocol with an in-flight LEASE:

1. :meth:`TriggerDedupStore.try_claim` atomically either
   (a) inserts a fresh in-flight row (lease = ``now + lease_seconds``) and returns
       :data:`CLAIM_WON` — the caller runs the handler, OR
   (b) finds a COMPLETED row (a real result within its TTL) and returns
       :data:`CLAIM_DONE` with the stored result — a genuine duplicate, do not re-run, OR
   (c) finds a LIVE in-flight lease held by another worker and returns
       :data:`CLAIM_IN_FLIGHT` — a concurrent duplicate, do not re-run.
   An EXPIRED in-flight lease (a worker that crashed mid-run) is reclaimable: it is treated
   exactly like an absent row, so the work is retried rather than lost forever.
2. On success the caller calls :meth:`TriggerDedupStore.complete`, which upgrades the
   in-flight row to a COMPLETED one (stores the result, sets the real ``expires_at`` TTL).
3. On FAILURE the caller calls :meth:`TriggerDedupStore.release`, deleting the in-flight
   row so the next redelivery re-claims and retries (at-least-once for the failed case —
   the safe direction).

A COMPLETED row is distinguished from an in-flight one by a sentinel ``result`` marker:
``None`` while in-flight, a (possibly empty) string once completed. The lease TTL on an
in-flight row is short (the handler's expected duration); the COMPLETED TTL is the
caller's dedup window.

:class:`DurableIdempotencyStore` adapts a :class:`TriggerDedupStore` to the connector
layer's sync :class:`~himmy.connectors.sdk.IdempotencyStore` interface AND adds the new
async :meth:`run_once_async` the (async) webhook handler actually calls. The sync
``seen``/``run_once`` methods are bridged onto the storage backend through the shared
aux-store loop so a legacy sync caller still works; the async path drives the storage
coroutines directly with no thread hop.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from himmy.connectors.sdk import ConnectorError, IdempotencyStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

#: The default in-flight lease for a single agent turn: long enough that a normal handler
#: finishes well inside it, short enough that a crashed worker's claim is reclaimable
#: promptly. Overridable per-store.
DEFAULT_INFLIGHT_LEASE_SECONDS = 300.0

#: The default dedup window a COMPLETED delivery is remembered for. After this a re-delivery
#: of the SAME id is allowed to fire again (the operator's "how long could a duplicate
#: realistically arrive" window).
DEFAULT_COMPLETED_TTL_SECONDS = 86_400.0


class ClaimOutcome(Enum):
    """The result of a :meth:`TriggerDedupStore.try_claim` attempt."""

    #: We inserted a fresh in-flight lease (or reclaimed an expired one); RUN the handler.
    WON = "won"
    #: A COMPLETED result exists within its TTL; this is a duplicate — return the result.
    DONE = "done"
    #: Another worker holds a LIVE in-flight lease; this is a concurrent duplicate.
    IN_FLIGHT = "in_flight"


CLAIM_WON = ClaimOutcome.WON
CLAIM_DONE = ClaimOutcome.DONE
CLAIM_IN_FLIGHT = ClaimOutcome.IN_FLIGHT


class DedupClaim:
    """The outcome of a claim attempt + the stored result when the row is COMPLETED."""

    __slots__ = ("outcome", "result")

    def __init__(self, outcome: ClaimOutcome, result: str | None = None) -> None:
        self.outcome = outcome
        #: The stored result string for a :data:`CLAIM_DONE` outcome; ``None`` otherwise.
        self.result = result

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DedupClaim(outcome={self.outcome.value!r}, result={self.result!r})"


@runtime_checkable
class TriggerDedupStore(Protocol):
    """The durable TTL-CAS surface a storage backend exposes for inbound dedup (Q4).

    All three durable-capable backends (SQLite, Postgres, in-memory) implement this on the
    :class:`~himmy.services.storage.service.StorageService` facade so the
    :class:`DurableIdempotencyStore` is backend-agnostic.
    """

    async def dedup_try_claim(
        self,
        scope: str,
        key: str,
        *,
        lease_seconds: float,
        now: str | None = None,
    ) -> DedupClaim:
        """Atomically claim ``(scope, key)`` for execution, or report a duplicate."""
        ...

    async def dedup_complete(
        self,
        scope: str,
        key: str,
        *,
        result: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> None:
        """Upgrade a won in-flight claim to COMPLETED with ``result`` + a real TTL."""
        ...

    async def dedup_release(
        self, scope: str, key: str, *, now: str | None = None
    ) -> None:
        """Drop a won-but-failed in-flight claim so a redelivery re-runs (at-least-once)."""
        ...

    async def dedup_sweep(self, *, now: str | None = None) -> int:
        """Delete expired dedup rows (lazy GC); return the number removed."""
        ...


class DurableIdempotencyStore(IdempotencyStore):
    """A durable :class:`IdempotencyStore` whose dedup survives a process restart (Q4).

    Subclasses the connector layer's in-RAM store so it drops into the ``idempotency=``
    seam unchanged, but overrides every method to drive a durable
    :class:`TriggerDedupStore` instead of the process-local dicts. Adds
    :meth:`run_once_async` — the *primary* entry point for the async webhook handler —
    which implements the mark-after-success protocol with an in-flight lease, closing the
    cross-restart double-fire AND the mark-before-run permanent-lost-delivery hole.

    ``scope`` namespaces the keys so two connectors (or two deployments sharing a database)
    cannot collide on a delivery id. ``runner`` is the shared aux-store loop used to bridge
    the legacy SYNC ``seen``/``run_once`` calls onto the async storage; the async path needs
    no bridge.
    """

    def __init__(
        self,
        store: TriggerDedupStore,
        *,
        scope: str = "webhook",
        inflight_lease_seconds: float = DEFAULT_INFLIGHT_LEASE_SECONDS,
        completed_ttl_seconds: float = DEFAULT_COMPLETED_TTL_SECONDS,
        runner: Callable[[Awaitable[Any]], Any] | None = None,
    ) -> None:
        """Wire the durable backend, the key namespace, and the TTL/lease budgets."""
        super().__init__()
        self._store = store
        self._scope = scope
        self._inflight_lease = float(inflight_lease_seconds)
        self._completed_ttl = float(completed_ttl_seconds)
        self._runner = runner

    # -------------------------------------------------------------- async (primary)
    async def run_once_async(
        self, key: str, call: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Run the async ``call`` at most once per ``key``, durably + crash-safe.

        The mark-AFTER-success protocol: claim an in-flight lease, run ``call``, then record
        the result. A genuine duplicate (a COMPLETED row within TTL) returns the stored
        result WITHOUT re-running; a concurrent duplicate (a live in-flight lease) raises
        :class:`ConnectorError` so the two callers cannot both fire. A FAILURE releases the
        claim so the next redelivery retries — at-least-once for the failed case, which is
        the safe direction (a dropped delivery is worse than a re-run of an idempotent
        handler). A crash mid-run leaves an in-flight lease that EXPIRES and is reclaimed.
        """
        claim = await self._store.dedup_try_claim(
            self._scope, key, lease_seconds=self._inflight_lease
        )
        if claim.outcome is CLAIM_DONE:
            return claim.result
        if claim.outcome is CLAIM_IN_FLIGHT:
            raise ConnectorError(
                f"idempotency key {key!r} is already in flight (duplicate request)"
            )
        # CLAIM_WON: we hold the in-flight lease — run, then mark-after-success.
        try:
            result = await call()
        except BaseException:
            # Failure (incl. cancellation): release the claim so a redelivery re-runs.
            await self._store.dedup_release(self._scope, key)
            raise
        stored = result if isinstance(result, str) else ""
        await self._store.dedup_complete(
            self._scope, key, result=stored, ttl_seconds=self._completed_ttl
        )
        return result

    async def seen_async(self, key: str) -> bool:
        """Whether ``key`` already COMPLETED durably (the async dedup pre-check)."""
        claim = await self._store.dedup_try_claim(
            self._scope, key, lease_seconds=0.0
        )
        # A zero-second lease never wins a fresh claim usefully: an absent key is reclaimed
        # then immediately expires, so only an existing COMPLETED row reports True. Release
        # any in-flight row we momentarily created so the pre-check has no side effect.
        if claim.outcome is CLAIM_WON:
            await self._store.dedup_release(self._scope, key)
            return False
        return claim.outcome is CLAIM_DONE

    # ----------------------------------------------------------- sync (bridged)
    def run_once(self, key: str, call: Callable[[], Any]) -> Any:
        """Sync mark-after-success bridged onto the async store via the aux loop.

        Kept for any sync caller of the connector idempotency interface. The (async)
        webhook handler uses :meth:`run_once_async` instead, with no thread hop.
        """
        run = self._require_runner()

        async def _async_call() -> Any:
            return call()

        return run(self.run_once_async(key, _async_call))

    def seen(self, key: str) -> bool:
        """Sync ``seen`` bridged onto the async store via the aux loop."""
        run = self._require_runner()
        return bool(run(self.seen_async(key)))

    def _require_runner(self) -> Callable[[Awaitable[Any]], Any]:
        """Resolve the sync->async bridge runner (the shared aux loop by default)."""
        if self._runner is not None:
            return self._runner
        from himmy.services.storage.aux_store_factory import aux_loop

        return aux_loop().run


__all__ = [
    "CLAIM_DONE",
    "CLAIM_IN_FLIGHT",
    "CLAIM_WON",
    "DEFAULT_COMPLETED_TTL_SECONDS",
    "DEFAULT_INFLIGHT_LEASE_SECONDS",
    "ClaimOutcome",
    "DedupClaim",
    "DurableIdempotencyStore",
    "TriggerDedupStore",
]
