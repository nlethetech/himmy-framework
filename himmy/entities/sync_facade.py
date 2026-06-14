"""Entities kernel: a synchronous facade over the async Postgres spine (T2.1).

The entity-registry surface every spine consumer uses is SYNCHRONOUS — the runtime's
``_emit`` hot path calls ``registry.register(...)`` inline, ``SecurityAuditLog`` records
synchronously, and the lineage walks are sync. The durable SQLite spine is already
synchronous, but :class:`~himmy.entities.postgres.PostgresEntityRegistry` is async. To let
an operator point the SERVER spine at Postgres (opt-in via ``HIMMY_SPINE_DATABASE_URL``)
WITHOUT rewriting every call site to be async, :class:`SyncPostgresEntityRegistry` wraps
the async registry and drives it on a dedicated, long-lived background event loop thread,
exposing exactly the synchronous :class:`~himmy.entities.protocol.EntityRegistryProtocol`.

This is deliberately a thin, blocking bridge: each method submits the coroutine to the
background loop and blocks for the result. It is the SAME pattern the rest of himmy uses
to call async stores from sync code, scoped here to the (opt-in, low-volume relative to a
DB round-trip) spine. The default deployment never constructs this — it keeps the durable
SQLite spine — so the offline/default path imports nothing from ``asyncpg``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Collection
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from himmy.entities.lineage import DEFAULT_TRACE_DEPTH, LineageDirection

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.lineage import LineageGraph
    from himmy.entities.postgres import PostgresEntityRegistry
    from himmy.entities.records import (
        EntityLink,
        EntityQuery,
        EntityRecord,
    )


class _LoopThread:
    """A dedicated daemon thread running one asyncio event loop for blocking calls.

    ``name`` labels the thread (default the spine loop) so a second, distinct loop —
    e.g. the shared aux-store loop in
    :mod:`himmy.services.storage.aux_store_factory` — is distinguishable in a stack
    dump without forking this class.
    """

    def __init__(self, name: str = "himmy-spine-loop") -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any) -> Any:
        """Submit ``coro`` to the loop and block for its result."""
        fut: Future[Any] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def close(self) -> None:
        """Stop the loop and join the thread (idempotent, best-effort)."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
        except Exception:  # pragma: no cover - best-effort teardown
            pass


class SyncPostgresEntityRegistry:
    """Synchronous :class:`EntityRegistryProtocol` over the async Postgres registry.

    Build via :meth:`connect`. Every method blocks on a dedicated background event loop,
    so it drops into the runtime's synchronous spine call sites unchanged.
    """

    def __init__(self, registry: PostgresEntityRegistry, loop: _LoopThread) -> None:
        self._registry = registry
        self._loop = loop

    @classmethod
    def connect(cls, dsn: str) -> SyncPostgresEntityRegistry:
        """Create the pool + schema on a background loop and return the sync facade."""
        from himmy.entities.postgres import PostgresEntityRegistry

        loop = _LoopThread()

        async def _setup() -> PostgresEntityRegistry:
            reg = await PostgresEntityRegistry.connect(dsn)
            await reg.create_schema()
            return reg

        registry = loop.run(_setup())
        return cls(registry, loop)

    def close(self) -> None:
        """Close the pool, then stop the background loop (idempotent)."""
        try:
            self._loop.run(self._registry.close())
        finally:
            self._loop.close()

    def __enter__(self) -> SyncPostgresEntityRegistry:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- writes
    def register(self, record: EntityRecord) -> EntityRecord:
        """Register a record (blocking)."""
        return self._loop.run(self._registry.register(record))  # type: ignore[no-any-return]

    def new_version(
        self,
        *,
        stable_id: str,
        kind: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> EntityRecord:
        """Create the next version of an artefact (blocking)."""
        return self._loop.run(  # type: ignore[no-any-return]
            self._registry.new_version(
                stable_id=stable_id,
                kind=kind,
                payload=payload,
                metadata=metadata,
                expected_version=expected_version,
            )
        )

    def link(
        self,
        *,
        from_record_id: str,
        to_record_id: str,
        relation: str,
        metadata: dict[str, Any] | None = None,
    ) -> EntityLink:
        """Create a typed link between two records (blocking)."""
        return self._loop.run(  # type: ignore[no-any-return]
            self._registry.link(
                from_record_id=from_record_id,
                to_record_id=to_record_id,
                relation=relation,
                metadata=metadata,
            )
        )

    # ------------------------------------------------------------------ reads
    def get(self, record_id: str) -> EntityRecord | None:
        """Return a record by id, or None (blocking)."""
        return self._loop.run(self._registry.get(record_id))  # type: ignore[no-any-return]

    def get_latest(self, stable_id: str) -> EntityRecord | None:
        """Return the highest-version record, or None (blocking)."""
        return self._loop.run(self._registry.get_latest(stable_id))  # type: ignore[no-any-return]

    def get_history(self, stable_id: str) -> list[EntityRecord]:
        """Return all versions of an artefact (blocking)."""
        return self._loop.run(self._registry.get_history(stable_id))  # type: ignore[no-any-return]

    def list_by_kind(self, kind: str) -> list[EntityRecord]:
        """Return every record of a kind (blocking)."""
        return self._loop.run(self._registry.list_by_kind(kind))  # type: ignore[no-any-return]

    def query(self, q: EntityQuery) -> list[EntityRecord]:
        """Return records matching a query (blocking)."""
        return self._loop.run(self._registry.query(q))  # type: ignore[no-any-return]

    # --------------------------------------------------------------- lineage
    def links_from(self, record_id: str) -> list[EntityLink]:
        """Return links originating from a record (blocking)."""
        return self._loop.run(self._registry.links_from(record_id))  # type: ignore[no-any-return]

    def links_to(self, record_id: str) -> list[EntityLink]:
        """Return links pointing INTO a record (blocking)."""
        return self._loop.run(self._registry.links_to(record_id))  # type: ignore[no-any-return]

    def neighbors(
        self,
        record_id: str,
        *,
        direction: LineageDirection = LineageDirection.BOTH,
        relation: str | None = None,
    ) -> list[EntityLink]:
        """Return links incident to a record (blocking)."""
        return self._loop.run(  # type: ignore[no-any-return]
            self._registry.neighbors(
                record_id, direction=direction, relation=relation
            )
        )

    def trace(
        self,
        record_id: str,
        *,
        max_depth: int = DEFAULT_TRACE_DEPTH,
        direction: LineageDirection = LineageDirection.BOTH,
        relations: Collection[str] | None = None,
    ) -> LineageGraph:
        """Walk the lineage graph from a record (blocking)."""
        return self._loop.run(  # type: ignore[no-any-return]
            self._registry.trace(
                record_id,
                max_depth=max_depth,
                direction=direction,
                relations=relations,
            )
        )


__all__ = ["SyncPostgresEntityRegistry", "_LoopThread"]
