"""Storage kernel: the durable-default store factory.

Server/multi-worker entrypoints should persist runs, idempotency, and lineage across
restarts and span workers, while a one-shot CLI invocation (``himmy run`` / ``himmy
chat``) wants zero setup — an in-memory store that needs no file, no key, no network.
:class:`StoreFactory` is the single place that resolves which backend each front end
gets:

* :meth:`StoreFactory.for_cli` — ALWAYS the in-memory
  :class:`~himmy.services.storage.service.StorageService`; it ignores
  ``HIMMY_DATABASE_URL`` so a developer's exported DSN never silently changes a quick
  one-shot run.
* :meth:`StoreFactory.for_server` — DURABLE: a Postgres-backed store when
  ``HIMMY_DATABASE_URL`` is a ``postgres://`` DSN, otherwise a file-backed
  :class:`~himmy.services.storage.sqlite.SqliteStorageService` at
  :data:`HIMMY_STORE_PATH` (default ``.himmy/storage.db``).

A process-wide :data:`_SERVER_CONTEXT` contextvar lets the shared
``build_runtime_for_spec`` wiring (used by both the CLI and Himmy Studio's API) pick
the right backend without a parameter threaded through every call site: the FastAPI app
factory sets it true for the lifetime of the app (cleared on lifespan shutdown), so an
agent wired *inside the server* gets durable storage while the same wiring on the CLI
path stays in-memory. Offline-first is preserved: SQLite is stdlib, and Postgres is
imported lazily only when a DSN is actually present.
"""

from __future__ import annotations

from contextvars import ContextVar

from himmy.config.secrets import get_secret
from himmy.services.storage.service import StorageService

#: Default file location for the durable SQLite store when no ``HIMMY_DATABASE_URL`` is
#: set. Relative to the process CWD; override with ``HIMMY_STORE_PATH``.
HIMMY_STORE_PATH = ".himmy/storage.db"

#: Process-wide flag: True while running inside a server/multi-worker entrypoint (set by
#: the FastAPI app factory). The shared spec-wiring consults it to decide whether an
#: agent's storage should be durable (server) or in-memory (one-shot CLI). Defaults
#: False so every non-server path — including the entire existing test suite — keeps the
#: in-memory behaviour unchanged.
_SERVER_CONTEXT: ContextVar[bool] = ContextVar("himmy_server_context", default=False)


def set_server_context(active: bool) -> object:
    """Set the server-context flag; returns a token to restore it later.

    Pass the returned token to :func:`reset_server_context` (or
    ``_SERVER_CONTEXT.reset(token)``) to undo the change — used by the app factory's
    lifespan to clear the flag on shutdown.
    """
    return _SERVER_CONTEXT.set(active)


def reset_server_context(token: object) -> None:
    """Restore the server-context flag to the value captured in ``token``."""
    _SERVER_CONTEXT.reset(token)  # type: ignore[arg-type]


def in_server_context() -> bool:
    """True when the current process is running inside a server entrypoint."""
    return _SERVER_CONTEXT.get()


def _is_postgres_dsn(dsn: str | None) -> bool:
    """True when ``dsn`` names a Postgres database (``postgres://``/``postgresql://``)."""
    if not dsn:
        return False
    lowered = dsn.strip().lower()
    return lowered.startswith(("postgres://", "postgresql://", "postgresql+"))


class StoreFactory:
    """Resolves the right storage backend for each front end (CLI vs server).

    All methods are stateless classmethods so the factory can be called from anywhere
    (the API container, the CLI builder, the shared spec wiring) without construction.
    """

    @classmethod
    def for_cli(cls) -> StorageService:
        """Return the in-memory store for a one-shot CLI run (zero setup, always).

        Deliberately ignores ``HIMMY_DATABASE_URL`` / ``HIMMY_STORE_PATH``: a developer
        with an exported production DSN still gets a clean ephemeral store for a quick
        ``himmy run``, so a one-shot can never accidentally write to (or read stale data
        from) a durable backend.
        """
        return StorageService()

    @classmethod
    async def for_server(cls) -> object:
        """Return the DURABLE store for a server/multi-worker entrypoint.

        Postgres when ``HIMMY_DATABASE_URL`` is a ``postgres://`` DSN (the pool is
        created with the JSONB codec + timeouts, then migrated); otherwise a file-backed
        :class:`~himmy.services.storage.sqlite.SqliteStorageService` at
        :data:`HIMMY_STORE_PATH` (override with ``HIMMY_STORE_PATH``). Returns a backend
        that mirrors the in-memory ``StorageService`` API 1:1, so callers are agnostic.
        """
        dsn = get_secret("HIMMY_DATABASE_URL")
        if _is_postgres_dsn(dsn):
            from himmy.services.storage.postgres import PostgresStorageService

            storage = await PostgresStorageService.connect(dsn)  # type: ignore[arg-type]
            await storage.migrate()
            return storage
        return cls._sqlite_store()

    @classmethod
    def for_context(cls, server: bool | None = None) -> object:
        """Return a store chosen by the current :data:`_SERVER_CONTEXT` (sync).

        ``server`` overrides the context flag explicitly (the flag is set in the
        app lifespan task and does not survive every task/thread boundary —
        callers that KNOW they are the server pass ``True``).

        Used by the shared ``build_runtime_for_spec`` wiring, which is synchronous and so
        cannot ``await`` a Postgres pool. In a server context it returns the file-backed
        SQLite store at :data:`HIMMY_STORE_PATH` (durable, no event loop required); on the
        CLI path it returns the in-memory store. The async, Postgres-capable
        :meth:`for_server` is used by the API container where an event loop is available.
        """
        if server if server is not None else in_server_context():
            return cls._sqlite_store()
        return cls.for_cli()

    @classmethod
    def _sqlite_store(cls) -> object:
        """Construct the file-backed SQLite store at the configured path."""
        from himmy.services.storage.sqlite import SqliteStorageService

        path = get_secret("HIMMY_STORE_PATH") or HIMMY_STORE_PATH
        return SqliteStorageService(path)


__all__ = [
    "HIMMY_STORE_PATH",
    "StoreFactory",
    "set_server_context",
    "reset_server_context",
    "in_server_context",
]
