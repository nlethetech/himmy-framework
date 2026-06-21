"""Shared run-substrate bootstrap — the ONE wiring the FastAPI lifespan and the
standalone ``himmy worker`` both drive (Phase 0).

The FastAPI lifespan (:func:`himmy.api.app._build_lifespan`) and the offline
``himmy worker`` command must stand up the *exact same* durable run substrate:

  1. mark the process as a server context (so any in-process agent resolves the
     durable store, not the one-shot in-memory one);
  2. select / build a DURABLE container (Postgres / file-backed SQLite);
  3. publish that store process-wide (``set_server_storage``);
  4. when the run store is durable, construct a :class:`RunDispatcher`, call
     ``run_app.enable_dispatch(...)`` **before** the recovery sweep (so the sweep
     re-queues lease-expired runs instead of mass-failing them), run the startup
     sweep + HITL reconcile, then ``dispatcher.start()`` **after** recovery;
  5. start the durable dedup-ledger GC loop;
  6. on shutdown, reverse the order: stop the dispatcher (drains in-flight claimed
     runs), stop the dedup loop, drain inline tasks, close the container, and clear
     every process-wide provider / context flag.

Before Phase 0 this sequence lived inline in the lifespan, which meant a CLI /
desktop user with no FastAPI server got NONE of it — ``create_run`` stayed in
INLINE fire-and-forget mode and no dispatcher drained the queue. Factoring it here
lets the worker reproduce the lifespan byte-for-byte without uvicorn or a FastAPI
app, while the lifespan keeps its app-state / readiness-truth responsibilities.

The two process-wide providers ``create_app`` installs in its BODY
(``set_canonical_storage_provider`` + ``set_routine_container_provider``) are wired
HERE too, against the resolved container, because the worker has no ``create_app``
body to install them — and ``resolve_routine_container()`` returning ``None`` is
exactly why agent_id routines silently fail offline.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.api.deps import ApiContainer
    from himmy.application.dispatcher import RunDispatcher

logger = logging.getLogger("himmy.api.bootstrap")

#: How often the maintenance loop GC's the durable trigger-dedup ledger (seconds).
#: Shared with the lifespan so both entrypoints sweep on the same cadence.
DEDUP_SWEEP_INTERVAL_SECONDS = 3600.0


def wants_durable_store() -> bool:
    """True when this process should wire the durable store instead of in-memory.

    Opt-in so the zero-config offline path stays in-memory: a durable backend is
    selected when a database DSN is configured (``HIMMY_DATABASE_URL``) or when the
    operator explicitly asks for it (``HIMMY_DURABLE_STORAGE=1``). The actual backend
    (Postgres for a ``postgres://`` DSN, file-backed SQLite otherwise) is resolved by
    :class:`StoreFactory`; this only decides *whether* to use it.
    """
    from himmy.config.secrets import get_secret

    if (get_secret("HIMMY_DATABASE_URL") or "").strip():
        return True
    return os.environ.get("HIMMY_DURABLE_STORAGE", "").lower() in ("1", "true", "yes")


def run_store_is_durable(container: ApiContainer) -> bool:
    """True when the container's run store is a DURABLE backend (SQLite / Postgres).

    The leased dispatcher's whole value is recovering runs across a restart, which
    only a durable run store provides — the in-memory store loses every run on exit,
    so a dispatcher there would leave runs stuck QUEUED with nothing to recover them.
    """
    storage = getattr(container, "storage", None)
    if storage is None:
        return False
    from himmy.services.storage.postgres import PostgresStorageService
    from himmy.services.storage.sqlite import SqliteStorageService

    return isinstance(storage, (SqliteStorageService, PostgresStorageService))


def active_backend_name(container: ApiContainer) -> str:
    """Name the container's storage backend: ``postgres`` / ``sqlite`` / ``memory`` / ``none``."""
    storage = getattr(container, "storage", None)
    if storage is None:
        return "none"
    from himmy.services.storage.postgres import PostgresStorageService
    from himmy.services.storage.sqlite import SqliteStorageService

    if isinstance(storage, PostgresStorageService):
        return "postgres"
    if isinstance(storage, SqliteStorageService):
        return "sqlite"
    return "memory"


def dispatch_env_overrides(
    *,
    default_concurrency: int,
    default_timeout_seconds: float,
    default_max_attempts: int,
) -> tuple[int, float, int]:
    """Resolve the dispatcher's tunable knobs from the environment (dx).

    ``HIMMY_DISPATCH_CONCURRENCY`` (worker fan-out), ``HIMMY_RUN_TIMEOUT_SECONDS``
    (each run's wall clock — also the lease TTL basis), and
    ``HIMMY_DISPATCH_MAX_ATTEMPTS`` (the enqueue retry budget). Each defaults to the
    value passed in; an unparseable or non-positive value falls back to that default
    so a typo never breaks startup.
    """

    def _positive_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw.strip())
        except ValueError:
            return default
        return value if value > 0 else default

    def _positive_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw.strip())
        except ValueError:
            return default
        return value if value > 0 else default

    return (
        _positive_int("HIMMY_DISPATCH_CONCURRENCY", default_concurrency),
        _positive_float("HIMMY_RUN_TIMEOUT_SECONDS", default_timeout_seconds),
        _positive_int("HIMMY_DISPATCH_MAX_ATTEMPTS", default_max_attempts),
    )


async def _dedup_sweep_loop(
    storage: Any,
    *,
    interval: float = DEDUP_SWEEP_INTERVAL_SECONDS,
    stop: asyncio.Event | None = None,
) -> None:
    """Periodically GC the durable trigger-dedup ledger until cancelled / ``stop`` set.

    A sweep error must not kill the loop — the next tick retries. Returns promptly when
    ``stop`` is signalled so shutdown does not wait a full interval.
    """
    stop = stop or asyncio.Event()
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            pass
        try:
            removed = await storage.dedup_sweep()
            if removed:
                logger.info("swept %d expired dedup row(s)", removed)
        except Exception:  # noqa: BLE001 - a sweep error must not kill the loop
            logger.warning("dedup sweep cycle failed", exc_info=True)


@dataclass
class RunSubstrate:
    """The resolved durable run substrate a started bootstrap owns.

    ``active`` is the container every request-time / scheduler reader should use
    (it may differ from the one passed in if a durable store was built). ``dispatch_on``
    records whether the leased dispatcher actually engaged (durable store + dispatcher
    started). The remaining fields are internal handles teardown needs.
    """

    active: ApiContainer
    dispatch_on: bool
    backend: str
    dispatcher: RunDispatcher | None = None
    run_app: Any | None = None
    _dedup_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _dedup_stop: asyncio.Event | None = field(default=None, repr=False)


async def build_durable_container() -> tuple[ApiContainer, bool]:
    """Resolve the container the run substrate sits on: durable when requested.

    Returns ``(container, built_durable)``. When :func:`wants_durable_store` is true a
    durable container is built via :meth:`ApiContainer.build_default_async` (Postgres /
    file-backed SQLite). Otherwise — and on a durable-build failure — a sync default
    container is built UNDER the server context so its spine resolves the server backend
    while the run store stays in-memory (a degraded but live fallback). The caller MUST
    have set the server context before calling this (so the sync rebuild picks the
    server spine).
    """
    from himmy.api.deps import ApiContainer

    if wants_durable_store():
        try:
            active = await ApiContainer.build_default_async()
            logger.info("durable storage wired for the worker entrypoint")
            return active, True
        except Exception:  # pragma: no cover - defensive: stay up on store failure
            logger.warning(
                "durable storage requested but could not be wired; "
                "falling back to in-memory storage",
                exc_info=True,
            )
    # Rebuild under the (now-active) server context so the spine resolves the SERVER
    # backend even on the zero-DSN path.
    return ApiContainer.build_default(), False


async def start_run_substrate(
    container: ApiContainer, *, install_providers: bool = True
) -> RunSubstrate:
    """Bring up the leased dispatcher + recovery for a resolved container.

    Reproduces the lifespan's startup ordering EXACTLY (the ordering is load-bearing):
    publish the store process-wide, enable dispatch BEFORE the recovery sweep, run the
    sweep + HITL reconcile, then start the dispatcher loops AFTER recovery, then start
    the dedup-ledger GC loop. Installs the two process-wide providers (canonical store +
    routine container) the FastAPI ``create_app`` body installs, so an offline worker's
    agent_id routines actually resolve a container instead of erroring.

    ``install_providers`` is False for the FastAPI lifespan, which installs its OWN
    providers that read ``app.state.container`` dynamically (so a later rebind is
    reflected) — re-installing fixed-container lambdas here would shadow that. The
    standalone worker leaves it True (it never rebinds; the captured container is final).

    The caller owns the server-context token and is responsible for clearing it on
    teardown (so the token's lifetime can span the whole process, not just this call).
    """
    from himmy.services.storage.factory import set_server_storage

    backend = active_backend_name(container)

    # K1: publish the RESOLVED durable storage process-wide so any in-process agent
    # wiring resolves THIS backend instead of opening a second store.
    set_server_storage(getattr(container, "storage", None))

    if install_providers:
        from himmy.api.routines import set_routine_container_provider
        from himmy.api.studio_canonical import set_canonical_storage_provider

        # Install the two providers create_app's BODY installs (app.py:780,794). Without
        # the routine container provider, resolve_routine_container() returns None and
        # agent_id routines fail with "no app container". A plain lambda over the captured
        # container mirrors the app.state.container dynamic the lifespan uses.
        set_canonical_storage_provider(lambda: getattr(container, "storage", None))
        set_routine_container_provider(lambda: container)

    run_app = getattr(container, "run_app", None)
    dispatcher: RunDispatcher | None = None

    durable = run_app is not None and run_store_is_durable(container)
    if durable and run_app is not None:
        from himmy.application.dispatcher import RunDispatcher

        concurrency, run_timeout_seconds, max_attempts = dispatch_env_overrides(
            default_concurrency=8,
            default_timeout_seconds=run_app.run_timeout_seconds,
            default_max_attempts=run_app.default_max_attempts,
        )
        dispatcher = RunDispatcher(run_app, max_concurrency=concurrency)
        # Enable dispatch BEFORE the sweep so the sweep re-queues lease-expired runs
        # instead of mass-failing them.
        run_app.enable_dispatch(
            max_attempts=max_attempts,
            run_timeout_seconds=run_timeout_seconds,
        )

    if run_app is not None:
        try:
            swept = await run_app.sweep_stuck_runs()
            if swept:
                logger.info(
                    "recovered %d stuck run(s) on startup "
                    "(re-queued under dispatch, else failed)",
                    len(swept),
                )
        except Exception:  # pragma: no cover - startup sweep is best-effort
            logger.warning("startup run sweep failed", exc_info=True)
        try:
            redriven = await run_app.reconcile_resolving_runs()
            if redriven:
                logger.info(
                    "re-drove %d crashed HITL resume(s) on startup", len(redriven)
                )
        except Exception:  # pragma: no cover - startup reconcile is best-effort
            logger.warning("startup resume reconcile failed", exc_info=True)

    dispatch_on = False
    if dispatcher is not None:
        try:
            dispatcher.start()
            dispatch_on = True
            logger.info("leased run dispatcher started (owner=%s)", dispatcher.owner_id)
        except Exception:  # pragma: no cover - dispatcher start is best-effort
            logger.warning("run dispatcher failed to start", exc_info=True)
            dispatcher = None

    dedup_task: asyncio.Task[None] | None = None
    dedup_stop: asyncio.Event | None = None
    storage = getattr(container, "storage", None)
    if storage is not None and durable:
        try:
            dedup_stop = asyncio.Event()
            dedup_task = asyncio.create_task(
                _dedup_sweep_loop(
                    storage,
                    interval=DEDUP_SWEEP_INTERVAL_SECONDS,
                    stop=dedup_stop,
                ),
                name="himmy-dedup-sweep",
            )
            logger.info("durable dedup-ledger sweep loop started")
        except Exception:  # pragma: no cover - sweep start is best-effort
            logger.warning("dedup sweep loop failed to start", exc_info=True)
            dedup_task = None
            dedup_stop = None

    # Materialize Studio "connection" non-secret fields into the process env so tool
    # config picks them up without a restart (the lifespan does this too).
    try:
        from himmy.api.studio_connections import apply_connections_to_env

        apply_connections_to_env()
    except Exception:  # pragma: no cover - connections are best-effort
        logger.warning("applying studio connections to env failed", exc_info=True)

    return RunSubstrate(
        active=container,
        dispatch_on=dispatch_on,
        backend=backend,
        dispatcher=dispatcher,
        run_app=run_app,
        _dedup_task=dedup_task,
        _dedup_stop=dedup_stop,
    )


async def stop_run_substrate(
    substrate: RunSubstrate,
    *,
    original_container: ApiContainer | None = None,
    clear_providers: bool = True,
) -> None:
    """Tear the run substrate down in the exact reverse order of startup.

    Mirrors the lifespan finally-block: stop the dispatcher (drains in-flight claimed
    runs), stop the dedup loop, drain inline tasks, close the container (and the
    original if it was swapped for a durable build), then clear every process-wide
    provider / aux loop. ``clear_providers`` is False only when another owner (the
    FastAPI lifespan) will clear the context flag itself.
    """
    if substrate.dispatcher is not None:
        try:
            await substrate.dispatcher.stop()
        except Exception:  # pragma: no cover - shutdown best-effort
            logger.warning("run dispatcher stop failed", exc_info=True)
    if substrate._dedup_task is not None:
        if substrate._dedup_stop is not None:
            substrate._dedup_stop.set()
        substrate._dedup_task.cancel()
        try:
            await substrate._dedup_task
        except (asyncio.CancelledError, Exception):  # pragma: no cover
            pass
    if substrate.run_app is not None:
        try:
            await substrate.run_app.drain()
        except Exception:  # pragma: no cover - shutdown best-effort
            logger.warning("run drain failed on shutdown", exc_info=True)
    try:
        await substrate.active.aclose()
    except Exception:  # pragma: no cover - shutdown best-effort
        pass
    if original_container is not None and original_container is not substrate.active:
        try:
            await original_container.aclose()
        except Exception:  # pragma: no cover - shutdown best-effort
            pass
    if not clear_providers:
        return
    try:
        from himmy.api.studio_canonical import set_canonical_storage_provider

        set_canonical_storage_provider(None)
    except Exception:  # pragma: no cover - shutdown best-effort
        pass
    try:
        from himmy.api.routines import set_routine_container_provider

        set_routine_container_provider(None)
    except Exception:  # pragma: no cover - shutdown best-effort
        pass
    try:
        from himmy.services.storage.aux_store_factory import reset_aux_store_factory

        reset_aux_store_factory()
    except Exception:  # pragma: no cover - shutdown best-effort
        pass
    try:
        from himmy.services.storage.factory import reset_server_storage

        reset_server_storage()
    except Exception:  # pragma: no cover - shutdown best-effort
        pass
