"""Entities kernel: the durable-default spine factory (T2.1).

The *spine* is the entity registry that records every run's provenance, lineage, and
security/audit events. Historically there were three disconnected spines: the CLI's
durable ``.himmy/spine.db`` (``himmy seclog`` / ``himmy consent``), a bare in-memory
``EntityRegistry()`` built per process at ``api/deps.py`` (Studio + ``/v1`` share it, and
it evaporates on restart), and another bare in-memory registry wired by the shared
``build_runtime_for_spec`` for every CLI run — so a CLI run's lineage never reached the
durable spine its own ``seclog`` reads. :class:`SpineFactory` is the single decision
point that collapses those into ONE durable spine, mirroring
:class:`~himmy.services.storage.factory.StoreFactory`:

* :meth:`SpineFactory.for_cli` — the DURABLE file-backed
  :class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry` at the canonical
  :func:`~himmy.config.project.spine_db_path` (``.himmy/spine.db``). Unlike the run STORE
  (whose CLI variant is in-memory so a one-shot leaves no trace), the spine is durable on
  the CLI too: a security deny or a run's provenance must survive across invocations — that
  is the whole point of ``himmy seclog``.
* :meth:`SpineFactory.for_server` — DURABLE: a Postgres-backed
  :class:`~himmy.entities.postgres_registry.PostgresEntityRegistry` ONLY when
  ``HIMMY_SPINE_DATABASE_URL`` is set (opt-in; wrapped in a sync facade since the registry
  surface is synchronous), otherwise the same durable SQLite spine at the canonical path.
  NOTE: the default Postgres *storage* deployment keeps a durable SQLite *spine* — the two
  are independent, and a tenant pointing ``HIMMY_DATABASE_URL`` at Postgres does not force
  an async-spine rewrite.
* :meth:`SpineFactory.for_context` — chooses by the process-wide
  :data:`~himmy.services.storage.factory._SERVER_CONTEXT` flag (the same one the shared
  spec wiring already consults). Both branches resolve to the SAME canonical
  ``.himmy/spine.db``, so the CLI-run path and the server share one durable spine — the
  ``server`` parameter only governs whether the (opt-in) Postgres spine is considered.

Offline-first is preserved: SQLite is stdlib, the canonical path resolves with zero
config, and the Postgres registry is imported lazily only when its DSN is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.config.project import spine_db_path
from himmy.config.secrets import get_secret
from himmy.services.storage.factory import in_server_context

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.protocol import EntityRegistryProtocol

#: Opt-in DSN that points the SERVER spine at Postgres instead of the durable SQLite
#: default. Deliberately a DISTINCT key from ``HIMMY_DATABASE_URL`` (the storage DSN): the
#: default Postgres-storage deployment keeps a durable SQLite spine, and a Postgres spine
#: is only wired when an operator explicitly asks for it here.
HIMMY_SPINE_DATABASE_URL_ENV = "HIMMY_SPINE_DATABASE_URL"

#: Optional override for the deferred-commit batch size of the durable SQLite spine. A
#: write-heavy multi-tool run fires one ``register`` per ``RunEvent``; setting this >1
#: amortises the per-event commit cost at the price of a small crash window. Default 1
#: (commit every event) keeps the full audit-spine durability contract.
HIMMY_SPINE_BUFFER_ENV = "HIMMY_SPINE_BUFFER"


def _is_postgres_dsn(dsn: str | None) -> bool:
    """True when ``dsn`` names a Postgres database (``postgres://``/``postgresql://``)."""
    if not dsn:
        return False
    lowered = dsn.strip().lower()
    return lowered.startswith(("postgres://", "postgresql://", "postgresql+"))


def _register_buffer() -> int:
    """Resolve the deferred-commit batch size from ``HIMMY_SPINE_BUFFER`` (default 1)."""
    raw = get_secret(HIMMY_SPINE_BUFFER_ENV)
    if not raw or not str(raw).strip():
        return 1
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return 1


class SpineFactory:
    """Resolves the right durable entity-registry backend for each front end.

    All methods are stateless classmethods so the factory can be called from anywhere
    (the API container, the CLI adapters, the shared spec wiring) without construction —
    the same shape as :class:`~himmy.services.storage.factory.StoreFactory`.
    """

    @classmethod
    def for_cli(cls) -> EntityRegistryProtocol:
        """Return the DURABLE SQLite spine for the CLI (``.himmy/spine.db``).

        Always durable: a security event or a run's provenance recorded by one
        ``himmy run`` must be visible to a later ``himmy seclog`` / lineage walk. Resolves
        the canonical path via :func:`~himmy.config.project.spine_db_path` so it is the
        SAME database a co-located server writes.
        """
        return cls._sqlite_spine()

    @classmethod
    def for_server(cls) -> EntityRegistryProtocol:
        """Return the DURABLE spine for a server entrypoint.

        Postgres ONLY when ``HIMMY_SPINE_DATABASE_URL`` is set (opt-in, wrapped in a
        sync facade); otherwise the canonical durable SQLite spine. The registry surface
        is synchronous (the runtime calls it inline), so even the Postgres branch is
        exposed through a synchronous adapter — no event loop is required at the call
        sites.
        """
        dsn = get_secret(HIMMY_SPINE_DATABASE_URL_ENV)
        if _is_postgres_dsn(dsn):
            return cls._postgres_spine(dsn)  # type: ignore[arg-type]
        return cls._sqlite_spine()

    @classmethod
    def for_context(cls, server: bool | None = None) -> EntityRegistryProtocol:
        """Return a spine chosen by the current server-context flag (sync).

        ``server`` overrides the :data:`_SERVER_CONTEXT` flag explicitly (the flag is set
        in the app lifespan task and does not survive every task/thread boundary — a caller
        that KNOWS it is the server passes ``True``). Both branches resolve to the SAME
        canonical ``.himmy/spine.db`` by default; the only effect of ``server`` is whether
        the opt-in Postgres spine (``HIMMY_SPINE_DATABASE_URL``) is considered.
        """
        is_server = server if server is not None else in_server_context()
        return cls.for_server() if is_server else cls.for_cli()

    # ----------------------------------------------------------------- builders
    @classmethod
    def _sqlite_spine(cls) -> EntityRegistryProtocol:
        """Construct the durable SQLite spine at the canonical project path."""
        from himmy.entities.sqlite_registry import SqliteEntityRegistry

        return SqliteEntityRegistry(spine_db_path(), register_buffer=_register_buffer())

    @classmethod
    def _postgres_spine(cls, dsn: str) -> EntityRegistryProtocol:
        """Construct the opt-in Postgres spine behind a synchronous facade.

        The :class:`~himmy.entities.postgres_registry.PostgresEntityRegistry` is async,
        but every spine consumer (the runtime ``_emit`` hot path, ``SecurityAuditLog``)
        calls the registry synchronously. :class:`SyncEntityRegistryFacade` bridges the
        two by driving the async registry on a dedicated event loop, so the Postgres spine
        drops into the same synchronous call sites as the SQLite one.
        """
        from himmy.entities.sync_facade import SyncPostgresEntityRegistry

        return SyncPostgresEntityRegistry.connect(dsn)


__all__ = [
    "HIMMY_SPINE_BUFFER_ENV",
    "HIMMY_SPINE_DATABASE_URL_ENV",
    "SpineFactory",
]
