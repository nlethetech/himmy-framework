"""Scheduler-tier leader election + the safe-by-default multi-node topology guard.

The durable run-queue is already multi-node-safe: many ``himmy worker`` processes can
drain it concurrently via ``SELECT ... FOR UPDATE SKIP LOCKED`` (each worker is a distinct
lease owner; the reaper recovers a crashed worker's runs). That half needs NO coordination.

The SCHEDULER tier is different. Exactly ONE node may *decide fires* (tick + the maintenance
sweeps: catch-up, dedup GC) — if N nodes all tick, they all race to fire the same routine.
The per-fire dedup-CAS (``trigger_dedup`` schedule-scope claim + the conditional
``mark_started``) is the LEADER-FREE backstop that keeps that race at-most-once even with a
brief double-leader, and it stays exactly as-is. But relying on it ALONE means N nodes do N×
the wasted tick work and, critically, on **SQLite** the CAS lives in a *per-node file* and
cannot coordinate across hosts at all — two hosts on separate SQLite files would each fire.

So this module adds two things, both additive / opt-in / fail-open:

* **Leader election** (Postgres): a session-scoped ``pg_try_advisory_lock`` lease. The holder
  is the leader and ticks; a non-holder runs worker-only. The lease auto-releases when the
  holder's connection drops (process death) → clean FAILOVER: another node's next acquire
  succeeds, with no permanently-missed fire (catch-up-on-promotion covers the gap) and no
  double-fire (the dedup-CAS guards the overlap window).

* **Topology guard** (SQLite / memory): there is no shared lock, so cross-host coordination
  is impossible. A same-HOST second scheduler is refused via a ``flock`` (the existing
  same-host primitive); a cross-host SQLite deployment is loudly warned as unsafe. On
  Postgres the guard simply blesses multi-worker (the lease does the real work).

The public entry point is :func:`acquire_scheduler_leadership`, returning a
:class:`SchedulerLeadership` the caller holds and releases on shutdown. ``is_leader`` tells
the worker whether to start the scheduler; :meth:`refresh` lets a worker-only node re-contend
(promote on the previous leader's death) and lets a leader detect a dropped lease (demote).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from himmy.api.runtime_bootstrap import active_backend_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.api.deps import ApiContainer
    from himmy.core.process_lock import ProcessLockHandle

logger = logging.getLogger("himmy.api.scheduler_leader")

#: flock name guarding a single-host scheduler when the backend cannot coordinate
#: cross-node (SQLite / in-memory). Same-host only — see the module docstring.
_SCHEDULER_HOST_LOCK = "himmy-scheduler"


@dataclass
class SchedulerLeadership:
    """The outcome of a leadership bid, plus the handles needed to refresh / release it.

    ``is_leader`` is the single bit the caller acts on: start the scheduler iff True. ``mode``
    names how leadership was decided (``postgres-lease`` / ``host-flock`` / ``single-node`` /
    ``follower``) for honest logs + diagnostics. ``reason`` explains a non-leader outcome.

    A Postgres node that LOST the election is still constructed (``is_leader=False``) and can
    be promoted later via :meth:`refresh` — that is how a worker-only node takes over when the
    previous leader dies. The held handles (``_pg_handle`` / ``_flock``) are internal.
    """

    is_leader: bool
    mode: str
    backend: str
    reason: str = ""
    _storage: Any | None = field(default=None, repr=False)
    _pg_handle: Any | None = field(default=None, repr=False)
    _flock: ProcessLockHandle | None = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)

    async def refresh(self) -> bool:
        """Re-evaluate leadership; return the (possibly changed) ``is_leader``.

        On Postgres:

        * a current leader confirms its lease is still live (a dropped session demotes it so
          it stops ticking without a lock rather than racing the new leader);
        * a current follower RE-CONTENDS for the lease (``pg_try_advisory_lock``) so it can be
          promoted the moment the previous leader's lease frees — this is the failover step.

        Fail-open: any error leaves the prior decision unchanged (never crashes the loop). On
        non-Postgres backends leadership is fixed at acquire time, so this is a no-op getter.
        """
        if self._released:
            return False
        if self.mode not in ("postgres-lease", "follower") or self._storage is None:
            return self.is_leader
        try:
            if self.is_leader:
                # Confirm the lease is still held; demote on a dropped session.
                still = await self._storage.scheduler_leader_is_held(self._pg_handle)
                if not still:
                    logger.warning(
                        "scheduler leader lease lost (connection dropped); demoting "
                        "to follower — another node will take over"
                    )
                    # Best-effort release of the (likely dead) handle, then re-contend below.
                    await self._safe_release_pg()
                    self.is_leader = False
                    self.mode = "follower"
                    self.reason = "lease lost; re-contending"
                    # fall through to a re-contend attempt this same cycle
                else:
                    return True
            # Follower (or just-demoted) path: try to (re)acquire the lease.
            handle = await self._storage.try_acquire_scheduler_leader()
            if handle is not None:
                self._pg_handle = handle
                self.is_leader = True
                self.mode = "postgres-lease"
                self.reason = ""
                logger.info("scheduler leadership ACQUIRED (postgres advisory lease)")
                return True
            self.is_leader = False
            self.mode = "follower"
            return False
        except Exception:  # noqa: BLE001 - leadership refresh must never kill the loop
            logger.warning("scheduler leadership refresh failed", exc_info=True)
            return self.is_leader

    async def release(self) -> None:
        """Release any held lease / flock (idempotent). Call on shutdown."""
        if self._released:
            return
        self._released = True
        await self._safe_release_pg()
        if self._flock is not None:
            try:
                self._flock.release()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                pass
            self._flock = None

    async def _safe_release_pg(self) -> None:
        if self._pg_handle is not None and self._storage is not None:
            try:
                await self._storage.release_scheduler_leader(self._pg_handle)
            except Exception:  # noqa: BLE001 - shutdown best-effort
                logger.debug("scheduler leader release failed", exc_info=True)
            self._pg_handle = None


async def acquire_scheduler_leadership(
    container: ApiContainer,
    *,
    require_single_scheduler_ack: bool = False,
    lock_base: Any | None = None,
) -> SchedulerLeadership:
    """Decide whether THIS process may run the scheduler, applying the topology guard.

    Backend-aware, safe-by-default:

    * **Postgres** — bless multi-worker. Bid for the cross-node advisory lease: the winner is
      the leader (``is_leader=True``), every other node is a follower that can be promoted on
      failover via :meth:`SchedulerLeadership.refresh`. The per-fire dedup-CAS remains the
      single-fire backstop for the brief overlap window.

    * **SQLite** — the run store is a per-node file with NO cross-host coordination. Guard:
      take a same-HOST ``flock`` so a second scheduler ON THIS HOST is refused (returns a
      non-leader with a clear reason). A CROSS-host SQLite deployment cannot be detected or
      coordinated, so we LOUDLY WARN it is unsafe (and, by default, still grant THIS host's
      single scheduler — the common single-box case stays working). ``require_single_scheduler_ack``
      makes the warning a hard refusal unless the operator has acknowledged single-node intent
      via ``HIMMY_SCHEDULER_SINGLE_NODE_ACK`` (belt-and-suspenders for a clustered SQLite
      misconfig).

    * **in-memory** — single-process by definition; grant leadership (``single-node``).

    Fail-open: any error in the bid logs and grants leadership (a missed scheduler is worse
    than a redundant one when the dedup-CAS is the real guard) — except the SQLite same-host
    flock, which is a genuine refusal (a second scheduler on one box is never wanted).
    """
    backend = active_backend_name(container)
    storage = getattr(container, "storage", None)

    if backend == "postgres":
        return await _acquire_postgres(storage, backend)
    if backend == "sqlite":
        return _acquire_sqlite(
            backend,
            require_single_scheduler_ack=require_single_scheduler_ack,
            lock_base=lock_base,
        )
    # in-memory / none: a single process, nothing to coordinate.
    return SchedulerLeadership(is_leader=True, mode="single-node", backend=backend)


async def _acquire_postgres(storage: Any, backend: str) -> SchedulerLeadership:
    """Bid for the Postgres advisory lease; winner leads, loser is a promotable follower."""
    if storage is None or not hasattr(storage, "try_acquire_scheduler_leader"):
        # Defensive: a Postgres backend without the lease API — fail open to leader so the
        # scheduler still runs (the dedup-CAS keeps it single-fire on shared Postgres).
        logger.warning(
            "postgres backend without scheduler-lease support; running scheduler "
            "WITHOUT leader election (dedup-CAS remains the single-fire guard)"
        )
        return SchedulerLeadership(
            is_leader=True, mode="single-node", backend=backend,
            reason="no lease API",
        )
    try:
        handle = await storage.try_acquire_scheduler_leader()
    except Exception:  # noqa: BLE001 - a lease bid error must not block the scheduler
        logger.warning(
            "scheduler leader bid failed; running scheduler WITHOUT a lease "
            "(dedup-CAS remains the single-fire guard)",
            exc_info=True,
        )
        return SchedulerLeadership(
            is_leader=True, mode="single-node", backend=backend,
            reason="lease bid error",
        )
    if handle is not None:
        logger.info("scheduler leadership ACQUIRED (postgres advisory lease)")
        return SchedulerLeadership(
            is_leader=True, mode="postgres-lease", backend=backend,
            _storage=storage, _pg_handle=handle,
        )
    logger.info(
        "scheduler leadership held by another node; this node runs worker-only "
        "(will take over on the leader's failover)"
    )
    return SchedulerLeadership(
        is_leader=False, mode="follower", backend=backend,
        reason="another node holds the lease", _storage=storage,
    )


def _single_node_acked() -> bool:
    """True when the operator acknowledged single-node SQLite intent via env."""
    from himmy.config.flags import env_truthy

    return env_truthy("HIMMY_SCHEDULER_SINGLE_NODE_ACK")


def _acquire_sqlite(
    backend: str,
    *,
    require_single_scheduler_ack: bool,
    lock_base: Any | None,
) -> SchedulerLeadership:
    """SQLite topology guard: same-host single-scheduler flock + cross-host warning."""
    from himmy.core.process_lock import ProcessLockBusy, acquire_process_lock

    try:
        flock = acquire_process_lock(_SCHEDULER_HOST_LOCK, base=lock_base)
    except ProcessLockBusy:
        # A second scheduler on THIS host — always refused (SQLite cannot coordinate them).
        logger.error(
            "refusing to start a SECOND scheduler on this host: the SQLite run store "
            "cannot coordinate multiple schedulers. Run extra nodes as queue-workers "
            "(`himmy worker --no-scheduler`), or switch to Postgres for multi-node "
            "scheduling."
        )
        return SchedulerLeadership(
            is_leader=False, mode="refused", backend=backend,
            reason="another scheduler already running on this host (SQLite)",
        )

    # We hold this host's single-scheduler lock. SQLite still cannot coordinate ACROSS hosts.
    if require_single_scheduler_ack and not _single_node_acked():
        # Hard guard for a clustered-SQLite misconfig: refuse without an explicit ack.
        flock.release()
        logger.error(
            "refusing to start the scheduler on a SQLite run store without a single-node "
            "acknowledgement: SQLite cannot coordinate schedulers across hosts (silent "
            "double-fire risk). Set HIMMY_SCHEDULER_SINGLE_NODE_ACK=1 if this is a "
            "single-box deployment, or use Postgres for real multi-node."
        )
        return SchedulerLeadership(
            is_leader=False, mode="refused", backend=backend,
            reason="SQLite multi-node not acknowledged",
        )

    logger.warning(
        "scheduler running on a SQLite run store (single-host only): SQLite cannot "
        "coordinate schedulers across machines. Safe for a single-box deployment; for "
        "multi-node use Postgres (HIMMY_DATABASE_URL) so the advisory-lease leader "
        "election + per-fire dedup-CAS coordinate fires across nodes."
    )
    return SchedulerLeadership(
        is_leader=True, mode="host-flock", backend=backend,
        reason="single-host SQLite (cross-host uncoordinated)", _flock=flock,
    )


__all__ = ["SchedulerLeadership", "acquire_scheduler_leadership"]
