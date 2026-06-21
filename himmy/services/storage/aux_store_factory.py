"""Storage kernel: the shared aux-store backend selector (K2).

himmy ships ~10 *auxiliary* singleton stores beyond the core run/spine surface — Studio
approvals, teams, workflows, graph checkpoints, conversations, routines, tasks, notes,
calendar, cookbook, and the memory store. Each historically built a file-backed SQLite
database via :func:`himmy.core.sqlite_util.connect_hardened` with **no Postgres branch**,
so a Postgres deployment (``HIMMY_DATABASE_URL`` pointing at Postgres) was silently
HYBRID: the runs/spine went to Postgres while every aux store kept writing local SQLite
sidecars on the same box. The actual Postgres MIRRORS of those stores land in K3/K4/K5;
this module is the **single choke point** they all route through so the
``HIMMY_DATABASE_URL``-selects-the-backend decision lives in ONE place (mirroring
:class:`~himmy.services.storage.factory.StoreFactory` and
:class:`~himmy.entities.spine_factory.SpineFactory`), and so a future aux Postgres store
shares ONE background event loop + reuses the server's existing asyncpg pool rather than
opening N independent pools/loops (the reviewer's smell).

Design (the seam K3/K4/K5 plug into):

* :func:`aux_backend` — the one decision: ``"postgres"`` when ``HIMMY_DATABASE_URL`` is a
  ``postgres://`` DSN, else ``"sqlite"``. Every ``get_*_store`` consults this.
* :func:`select_aux_store` — the routing helper a ``get_*_store`` calls: it returns the
  Postgres variant when the backend is Postgres *and* a Postgres builder is supplied, else
  the SQLite store. Today every caller supplies ``None`` for the Postgres builder (the
  mirrors do not exist yet), so the behaviour is byte-identical to before — durable SQLite
  — while the choke point is already wired for K3/K4/K5 to drop their builder in.
* :func:`aux_loop` — a process-wide, lazily-created singleton ``_LoopThread`` (the SAME
  dedicated-event-loop facade the Postgres spine uses) shared by ALL aux Postgres stores,
  so the synchronous ``get_*_store`` call sites can drive an async asyncpg store without
  each store spinning up its own loop.
* :func:`aux_pool` — the published server storage's asyncpg pool when reachable (set by the
  K1 lifespan publish), so an aux Postgres store reuses the ONE pool the server already
  owns instead of creating a second one. ``None`` outside a live Postgres server, in which
  case a future aux store falls back to its own connection.

Offline-first is preserved byte-for-byte: with no DSN, :func:`aux_backend` is ``"sqlite"``,
:func:`select_aux_store` returns the SQLite store, and nothing imports ``asyncpg``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

from himmy.config.secrets import get_secret

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.sync_facade import _LoopThread

AuxBackend = Literal["postgres", "sqlite"]

_T = TypeVar("_T")

#: Process-wide singleton background loop shared by every aux Postgres store. Lazily
#: created on first use, guarded by ``_AUX_LOOP_LOCK``; ``None`` until then and after a
#: :func:`reset_aux_store_factory`. Deliberately ONE loop for all aux stores (not N) so a
#: Postgres deployment does not accumulate a thread per sidecar.
_AUX_LOOP: _LoopThread | None = None
_AUX_LOOP_LOCK = threading.Lock()

#: Owned asyncpg pools opened ON the aux loop by aux Postgres stores (see
#: ``_AuxPgPool._resolve_async``). Tracked so :func:`reset_aux_store_factory` can close them
#: ON the aux loop before stopping it — an asyncpg pool is loop-bound and must be closed on
#: the loop it was created on. Guarded by ``_AUX_LOOP_LOCK``.
_AUX_OWNED_POOLS: list[object] = []


def register_aux_owned_pool(pool: object) -> None:
    """Record an aux-loop-bound owned pool for orderly close at reset/shutdown."""
    with _AUX_LOOP_LOCK:
        _AUX_OWNED_POOLS.append(pool)


def _is_postgres_dsn(dsn: str | None) -> bool:
    """True when ``dsn`` names a Postgres database (``postgres://``/``postgresql://``)."""
    if not dsn:
        return False
    lowered = dsn.strip().lower()
    return lowered.startswith(("postgres://", "postgresql://", "postgresql+"))


def aux_backend() -> AuxBackend:
    """Return the selected aux-store backend for this process (the single decision).

    ``"postgres"`` when ``HIMMY_DATABASE_URL`` is a ``postgres://`` DSN, else ``"sqlite"``
    (the zero-config offline default). Every ``get_*_store`` singleton consults this so the
    backend choice lives in ONE place — the same key (``HIMMY_DATABASE_URL``) that selects
    the core run store via :class:`~himmy.services.storage.factory.StoreFactory`.
    """
    return "postgres" if _is_postgres_dsn(get_secret("HIMMY_DATABASE_URL")) else "sqlite"


def aux_postgres_enabled() -> bool:
    """True when the aux stores should route to Postgres (``HIMMY_DATABASE_URL`` is PG)."""
    return aux_backend() == "postgres"


def select_aux_store(
    sqlite_builder: Callable[[], _T],
    postgres_builder: Callable[[], _T] | None = None,
) -> _T:
    """Return the right aux store backend (the routing helper every ``get_*_store`` calls).

    Returns ``postgres_builder()`` when the aux backend is Postgres AND a Postgres builder
    is supplied; otherwise ``sqlite_builder()``. The K3/K4/K5 mirrors pass their Postgres
    builder; until then every caller passes ``None`` and the SQLite store is returned —
    byte-identical to the pre-K2 behaviour, with the choke point already in place.

    The Postgres builder is invoked lazily (only on the Postgres path) so the offline
    default never imports ``asyncpg`` or touches the network.
    """
    if postgres_builder is not None and aux_postgres_enabled():
        return postgres_builder()
    return sqlite_builder()


def aux_loop() -> _LoopThread:
    """Return the process-wide shared background loop for aux Postgres stores (lazy).

    All aux Postgres stores drive their async asyncpg work on THIS one loop, so a Postgres
    deployment runs a single ``himmy-aux-loop`` daemon thread rather than one per sidecar.
    Created on first call; reused thereafter until :func:`reset_aux_store_factory`.
    """
    global _AUX_LOOP
    with _AUX_LOOP_LOCK:
        if _AUX_LOOP is None:
            from himmy.entities.sync_facade import _LoopThread

            _AUX_LOOP = _LoopThread(name="himmy-aux-loop")
        return _AUX_LOOP


def aux_pool() -> object | None:
    """Return the server's published asyncpg pool to reuse, or ``None``.

    Reuses the ONE pool the K1 lifespan published (via the resolved
    :class:`~himmy.services.storage.postgres.PostgresStorageService`) so an aux Postgres
    store does not open a second pool. ``None`` when no Postgres server storage is
    published (CLI/offline, or a SQLite server), in which case a future aux store falls
    back to its own connection. Null-safe: never raises if the published store has no pool.
    """
    from himmy.services.storage.factory import server_storage

    storage = server_storage()
    if storage is None:
        return None
    # The PostgresStorageService holds the pool privately; expose it without forcing every
    # storage backend to grow the attribute (SQLite/in-memory simply lack it → None).
    return getattr(storage, "_pool", None)


def reset_aux_store_factory() -> None:
    """Tear down the shared aux loop (lifespan shutdown / test isolation). Idempotent.

    Closes any aux-loop-bound owned pools ON the aux loop FIRST (a loop-bound pool must be
    closed on its own loop), then stops the loop thread. Best-effort: a close failure is
    swallowed so shutdown always completes.
    """
    global _AUX_LOOP
    with _AUX_LOOP_LOCK:
        loop = _AUX_LOOP
        _AUX_LOOP = None
        pools = list(_AUX_OWNED_POOLS)
        _AUX_OWNED_POOLS.clear()
    if loop is not None:
        for pool in pools:
            try:
                loop.run(pool.close())  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        loop.close()


__all__ = [
    "AuxBackend",
    "aux_backend",
    "aux_loop",
    "aux_pool",
    "aux_postgres_enabled",
    "register_aux_owned_pool",
    "reset_aux_store_factory",
    "select_aux_store",
]
