"""Storage kernel: Postgres mirrors of the auxiliary stores (K3 + K4).

himmy's auxiliary stores (the HITL approval checkpoints, the graph-run checkpoints, the
unified conversation store, routines, teams, and workflows) historically wrote file-backed
SQLite sidecars (``.himmy/*.db``) with no Postgres branch, so a ``HIMMY_DATABASE_URL``
Postgres deployment was silently HYBRID — runs/spine in Postgres, these on local SQLite. K3
(the two HITL checkpoint stores) and K4 (conversations + routines + teams + workflows)
route these into Postgres through the K2 :mod:`himmy.services.storage.aux_store_factory`
choke point, under the ONE ``schema_migrations`` ledger (the ``aux_*`` tables in
:data:`himmy.services.storage.postgres.STORAGE_MIGRATIONS`).

Each mirror is the SYNCHRONOUS twin of its SQLite store (the callers — the runtime's
``_emit`` hot path, the Studio approvals router, the routine scheduler — call these
synchronously), so each is a thin facade that drives async asyncpg work on the SHARED aux
event loop (:func:`himmy.services.storage.aux_store_factory.aux_loop`) and reuses the ONE
asyncpg pool the K1 lifespan published (:func:`aux_pool`) rather than opening N pools. The
async bodies live on the matching ``_Async*`` classes; the sync facades just ``loop.run``
them.

Two reviewer-grade properties from the plan are load-bearing here:

* **Tenant discriminator (K3 must_fix).** Studio's ``approvals.db`` and /v1's
  ``v1_approvals.db`` are DELIBERATELY separate surfaces; collapsing them into one
  undifferentiated Postgres table would let a /v1 tenant read Studio's (or another
  tenant's) checkpoints. Every aux table carries a ``tenant`` column and EVERY read/write
  is scoped on it, so the two HITL inboxes stay isolated. The tenant is bound at store
  construction (``"studio"`` for Studio, ``"v1"`` for /v1, ``"local"`` for the
  shared conversation/teams/workflows/graph singletons).
* **Per-record JSON schema migration on read (K3 must_fix).** A checkpoint's
  ``schema_version`` is a ROW-LEVEL blob migration (``_migrate_checkpoint`` /
  ``_migrate_graph_checkpoint``), NOT table DDL. The Postgres path reuses the SAME
  upgraders on read, so a legacy-version blob written by an older build is upgraded
  identically whether it came from SQLite or Postgres.

Offline-first is preserved byte-for-byte: nothing here is imported unless
``HIMMY_DATABASE_URL`` is a ``postgres://`` DSN AND a Postgres aux pool is reachable; the
zero-config default keeps using the durable SQLite (server) / in-memory (CLI) stores.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError
from himmy.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from himmy.agents.base_agent.thread import ChatThread
    from himmy.entities.sync_facade import _LoopThread
    from himmy.runtime.checkpoint import (
        AgentCheckpoint,
        GraphCheckpoint,
    )
    from himmy.services.storage.conversations import (
        ConversationSummary,
        FlatMessage,
    )

#: Builtin ``list`` aliased so annotations on methods that follow a ``def list`` (which
#: shadows the builtin name in class scope) still resolve to the builtin generic — the
#: same pattern the SQLite stores use.
_list = list


# --------------------------------------------------------------------------- pool plumbing


class _AuxPgPool:
    """A lazily-resolved asyncpg pool for the aux Postgres stores.

    Prefers the ONE pool the K1 lifespan published (via
    :func:`himmy.services.storage.aux_store_factory.aux_pool`) so an aux store reuses the
    server's existing pool instead of opening a second one. When no pool is published (a
    direct/programmatic construction with a DSN), it opens its OWN asyncpg pool on the
    shared aux loop and owns its teardown. ``acquire()`` returns the resolved pool's
    connection context manager; the JSONB codec is registered on every pooled connection so
    ``data`` round-trips as native dict/list.
    """

    def __init__(self, loop: _LoopThread, dsn: str | None) -> None:
        self._loop = loop
        self._dsn = dsn
        self._owned_pool: Any | None = None
        # Guards lazy owned-pool creation against two loop coroutines racing to open it.
        # Created lazily inside :meth:`_resolve_async` so it always binds to the aux loop.
        self._create_lock: asyncio.Lock | None = None

    async def _resolve_async(self) -> Any:
        """Return the pool to use, awaiting owned-pool creation INLINE on the loop.

        The aux store's async bodies run ON the shared aux loop, so pool resolution must
        ``await`` — NEVER block on a nested :meth:`_LoopThread.run`. A nested ``run`` would
        submit a coroutine to the very loop whose only thread is already parked in
        ``Future.result()``, deadlocking it.

        LOOP AFFINITY (the cross-loop fix): an asyncpg pool is bound to the event loop it was
        created on, and a connection from it can only be acquired/used on THAT loop. The K1
        lifespan's server pool is created on the process MAIN loop, but every aux store drives
        its work on the dedicated aux loop — so reusing the server pool here raised
        ``RuntimeError: ... attached to a different loop`` on EVERY aux Postgres operation
        (the routine scheduler's tick/catch-up/sleep-planning all failed silently → no
        routine ever fired on Postgres). The correct, loop-safe choice is to open the aux
        store's OWN pool ON the aux loop, from the configured DSN. This is one extra small
        pool per process (min 1 / max 5), bound to the right loop — not the N-pools smell the
        K2 reuse was avoiding, since all aux stores still share this single aux-loop pool path.
        """
        if self._owned_pool is not None:
            return self._owned_pool
        # No explicit DSN (the get_*_store singletons construct with dsn=None and historically
        # leaned on the published pool): fall back to the process DSN so we can open an
        # aux-loop-bound pool. Resolved lazily here (on the aux loop) so the offline/no-DSN
        # path never reaches this branch.
        if not self._dsn:
            from himmy.config.secrets import get_secret

            self._dsn = (get_secret("HIMMY_DATABASE_URL") or "").strip() or None
        if not self._dsn:
            raise HimmyError(
                "an aux Postgres store needs a DSN (HIMMY_DATABASE_URL) to open its "
                "aux-loop pool; none is available."
            )
        # Lazily bind the lock to the running (aux) loop; the check-then-set has no await,
        # so it is atomic on the single-threaded loop.
        if self._create_lock is None:
            self._create_lock = asyncio.Lock()
        async with self._create_lock:
            if self._owned_pool is None:
                self._owned_pool = await self._create_owned_pool(self._dsn)
                # Register for orderly close on the aux loop at reset/shutdown — the pool is
                # bound to this loop, so it MUST be closed on it (a main-loop close would
                # hit the same loop-affinity error this fix exists to avoid).
                from himmy.services.storage.aux_store_factory import (
                    register_aux_owned_pool,
                )

                register_aux_owned_pool(self._owned_pool)
        return self._owned_pool

    async def _create_owned_pool(self, dsn: str) -> Any:
        """Create an owned asyncpg pool with the shared JSONB codec init."""
        from himmy.services.storage.postgres import _register_codecs, _require_asyncpg

        asyncpg = _require_asyncpg()
        return await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=5,
            timeout=30.0,
            command_timeout=30.0,
            init=_register_codecs,
        )

    def run(self, coro: Any) -> Any:
        """Drive ``coro`` to completion on the shared aux loop and return the result."""
        return self._loop.run(coro)

    def acquire(self) -> _AuxAcquire:
        """Return an async context manager yielding a pooled connection.

        Resolution is deferred to ``__aenter__`` so the (possibly owned-pool-creating)
        ``await`` happens INLINE on the aux loop — see :meth:`_resolve_async` for why a
        synchronous resolve here would deadlock the loop.
        """
        return _AuxAcquire(self)

    def close(self) -> None:
        """Close the OWNED pool only (never the server's published pool). Idempotent."""
        if self._owned_pool is not None:
            owned = self._owned_pool
            self._owned_pool = None
            try:
                self._loop.run(owned.close())
            except Exception:  # pragma: no cover - best-effort teardown
                pass


class _AuxAcquire:
    """Async context manager that resolves the aux pool, then leases a connection.

    Resolving on ``__aenter__`` keeps owned-pool creation an ``await`` on the running aux
    loop instead of a nested blocking :meth:`_LoopThread.run`, which would deadlock the
    single-threaded loop. Delegates to the underlying asyncpg pool's own acquire context
    manager so connection release is unchanged.
    """

    __slots__ = ("_aux", "_cm", "_conn")

    def __init__(self, aux: _AuxPgPool) -> None:
        self._aux = aux
        self._cm: Any = None
        self._conn: Any = None

    async def __aenter__(self) -> Any:
        pool = await self._aux._resolve_async()
        self._cm = pool.acquire()
        self._conn = await self._cm.__aenter__()
        return self._conn

    async def __aexit__(self, *exc_info: Any) -> Any:
        if self._cm is not None:
            return await self._cm.__aexit__(*exc_info)
        return None


def _new_aux_pool(dsn: str | None = None) -> _AuxPgPool:
    """Build an :class:`_AuxPgPool` bound to the process-wide shared aux loop."""
    from himmy.services.storage.aux_store_factory import aux_loop

    return _AuxPgPool(aux_loop(), dsn)


#: Fixed first key of the two-int advisory-lock namespace for the per-WORKSPACE routine-count
#: quota (T3). A transaction-scoped ``pg_advisory_xact_lock(NS, ws_hash)`` serialises the
#: count-then-create for ONE workspace across processes (different workspaces use different
#: second keys and never contend), making the per-tenant routine cap a HARD cap rather than a
#: racy unlocked count-then-upsert. DISTINCT from ``FAIR_CLAIM_ADVISORY_NAMESPACE`` (the runs
#: budget) so routine-create and run-admit never cross-contend, and distinct from the
#: migration/leader 1-int (int8) keys so neither family collides.
ROUTINE_QUOTA_ADVISORY_NAMESPACE = 0x5251  # 'RQ'

#: Statement timeout (ms) bounding the BLOCKING advisory-lock wait in the atomic quota create,
#: so a pathologically stuck lock fails the statement (→ fail-OPEN admit) instead of hanging
#: the aux pool forever. Generous enough that normal same-workspace contention drains first.
_QUOTA_LOCK_STATEMENT_TIMEOUT_MS = 5000


def _routine_quota_advisory_key(workspace_id: str) -> int:
    """Derive the stable signed-32-bit second advisory key for a workspace (routine quota).

    Mirrors :func:`himmy.services.storage.postgres._workspace_advisory_key` (sha256 of the
    workspace id, first 4 bytes, recentered into signed int4) so the two-int
    ``pg_advisory_xact_lock(int4, int4)`` form applies; the DISTINCT namespace constant keeps
    it from colliding with the run-budget claim lock.
    """
    import hashlib

    digest = hashlib.sha256(workspace_id.encode("utf-8")).digest()[:4]
    value = int.from_bytes(digest, "big", signed=False)
    return value - (1 << 31)


# ============================================================ K3: HITL checkpoint mirrors


class _AsyncCheckpointStore:
    """Async body of the Postgres HITL approval checkpoint store (tenant-scoped)."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_agent_checkpoints "
                "(tenant, checkpoint_id, status, data, created_at) "
                "VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (tenant, checkpoint_id) DO UPDATE SET "
                "status = EXCLUDED.status, data = EXCLUDED.data",
                self._tenant,
                checkpoint.checkpoint_id,
                checkpoint.status,
                json.loads(checkpoint.model_dump_json()),
                checkpoint.created_at,
            )

    async def load(self, checkpoint_id: str) -> AgentCheckpoint | None:
        from himmy.runtime.checkpoint import AgentCheckpoint, _migrate_checkpoint

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM aux_agent_checkpoints "
                "WHERE tenant = $1 AND checkpoint_id = $2",
                self._tenant,
                checkpoint_id,
            )
        if row is None:
            return None
        return AgentCheckpoint.model_validate(_migrate_checkpoint(dict(row["data"])))

    async def claim(self, checkpoint_id: str) -> bool:
        """Atomic claimable-status CAS -> ``resolving`` (the exactly-once HITL gate).

        The structural twin of :meth:`SqliteCheckpointStore.claim`: a single conditional
        UPDATE under Postgres' row lock flips a claimable (``awaiting_approval`` /
        ``resolving``) row to ``resolving`` and rewrites the JSON blob's ``status`` so a
        later load is consistent. Two concurrent resumes (across replicas) serialise on the
        row lock; exactly one sees ``rowcount == 1``.
        """
        from himmy.runtime.checkpoint import (
            AWAITING_APPROVAL,
            RESOLVING,
        )

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE aux_agent_checkpoints "
                "SET status = $1, data = jsonb_set(data, '{status}', to_jsonb($1::text)) "
                "WHERE tenant = $2 AND checkpoint_id = $3 "
                "AND status IN ($4, $5)",
                RESOLVING,
                self._tenant,
                checkpoint_id,
                AWAITING_APPROVAL,
                RESOLVING,
            )
        return _rowcount(result) == 1

    async def list_by_status(self, status: str) -> list[AgentCheckpoint]:
        from himmy.runtime.checkpoint import AgentCheckpoint, _migrate_checkpoint

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM aux_agent_checkpoints "
                "WHERE tenant = $1 AND status = $2 ORDER BY created_at DESC",
                self._tenant,
                status,
            )
        return [
            AgentCheckpoint.model_validate(_migrate_checkpoint(dict(r["data"])))
            for r in rows
        ]

    async def prune_resolved(self, older_than_days: float) -> int:
        from datetime import UTC, datetime, timedelta

        from himmy.runtime.checkpoint import APPROVED, REJECTED

        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM aux_agent_checkpoints "
                "WHERE tenant = $1 AND status IN ($2, $3) AND created_at < $4",
                self._tenant,
                APPROVED,
                REJECTED,
                cutoff,
            )
        return _rowcount(result)


class PostgresCheckpointStore:
    """Synchronous, tenant-scoped Postgres mirror of :class:`SqliteCheckpointStore` (K3).

    Drop-in for the SQLite checkpoint store: same ``save``/``load``/``claim``/
    ``list_by_status``/``prune_resolved`` surface, every method blocking on the shared aux
    loop. The ``tenant`` discriminator keeps Studio's inbox isolated from each /v1
    workspace's.
    """

    def __init__(self, *, tenant: str, dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncCheckpointStore(self._pool, tenant)

    def save(self, checkpoint: AgentCheckpoint) -> None:
        self._pool.run(self._async.save(checkpoint))

    def load(self, checkpoint_id: str) -> AgentCheckpoint | None:
        return self._pool.run(self._async.load(checkpoint_id))  # type: ignore[no-any-return]

    def claim(self, checkpoint_id: str) -> bool:
        return bool(self._pool.run(self._async.claim(checkpoint_id)))

    def list_by_status(self, status: str) -> list[AgentCheckpoint]:
        return self._pool.run(self._async.list_by_status(status))  # type: ignore[no-any-return]

    def prune_resolved(self, *, older_than_days: float = 30.0) -> int:
        return int(self._pool.run(self._async.prune_resolved(older_than_days)))

    def close(self) -> None:
        self._pool.close()


class _AsyncGraphCheckpointStore:
    """Async body of the Postgres durable graph-run checkpoint store (tenant-scoped)."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def save(self, checkpoint: GraphCheckpoint) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_graph_checkpoints "
                "(tenant, checkpoint_id, graph_name, status, data, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7) "
                "ON CONFLICT (tenant, checkpoint_id) DO UPDATE SET "
                "graph_name = EXCLUDED.graph_name, status = EXCLUDED.status, "
                "data = EXCLUDED.data, updated_at = EXCLUDED.updated_at",
                self._tenant,
                checkpoint.checkpoint_id,
                checkpoint.graph_name,
                checkpoint.status,
                json.loads(checkpoint.model_dump_json()),
                checkpoint.created_at,
                checkpoint.updated_at,
            )

    async def load(self, checkpoint_id: str) -> GraphCheckpoint | None:
        from himmy.runtime.checkpoint import (
            GraphCheckpoint,
            _migrate_graph_checkpoint,
        )

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM aux_graph_checkpoints "
                "WHERE tenant = $1 AND checkpoint_id = $2",
                self._tenant,
                checkpoint_id,
            )
        if row is None:
            return None
        return GraphCheckpoint.model_validate(
            _migrate_graph_checkpoint(dict(row["data"]))
        )

    async def list_by_status(self, status: str) -> list[GraphCheckpoint]:
        from himmy.runtime.checkpoint import (
            GraphCheckpoint,
            _migrate_graph_checkpoint,
        )

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM aux_graph_checkpoints "
                "WHERE tenant = $1 AND status = $2 ORDER BY updated_at DESC",
                self._tenant,
                status,
            )
        return [
            GraphCheckpoint.model_validate(_migrate_graph_checkpoint(dict(r["data"])))
            for r in rows
        ]

    async def prune_terminal(self, older_than_days: float) -> int:
        from datetime import UTC, datetime, timedelta

        from himmy.runtime.checkpoint import GRAPH_COMPLETED, GRAPH_FAILED

        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM aux_graph_checkpoints "
                "WHERE tenant = $1 AND status IN ($2, $3) AND updated_at < $4",
                self._tenant,
                GRAPH_COMPLETED,
                GRAPH_FAILED,
                cutoff,
            )
        return _rowcount(result)


class PostgresGraphCheckpointStore:
    """Synchronous, tenant-scoped Postgres mirror of :class:`SqliteGraphCheckpointStore`."""

    def __init__(self, *, tenant: str, dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncGraphCheckpointStore(self._pool, tenant)

    def save(self, checkpoint: GraphCheckpoint) -> None:
        self._pool.run(self._async.save(checkpoint))

    def load(self, checkpoint_id: str) -> GraphCheckpoint | None:
        return self._pool.run(self._async.load(checkpoint_id))  # type: ignore[no-any-return]

    def list_by_status(self, status: str) -> list[GraphCheckpoint]:
        return self._pool.run(self._async.list_by_status(status))  # type: ignore[no-any-return]

    def prune_terminal(self, *, older_than_days: float = 30.0) -> int:
        return int(self._pool.run(self._async.prune_terminal(older_than_days)))

    def close(self) -> None:
        self._pool.close()


# ============================================================ K4: conversation mirror


class _AsyncConversationStore:
    """Async body of the Postgres unified-conversation store (tenant-scoped)."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def save_thread(
        self,
        conversation_id: str,
        thread: ChatThread,
        *,
        origin: str,
        title: str | None,
        agent_path: str | None,
        provider: str | None,
        project_id: str | None,
        subject_id: str | None = None,
    ) -> ConversationSummary:
        from himmy.services.storage.conversations import (
            ORIGIN_CLI,
            ConversationSummary,
            _derive_title,
            flat_from_thread,
        )

        now = utc_now_iso()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchrow(
                    "SELECT created_at, project_id, agent_path, provider, origin, "
                    "subject_id "
                    "FROM aux_conversations WHERE tenant = $1 AND conversation_id = $2",
                    self._tenant,
                    conversation_id,
                )
                created = existing["created_at"] if existing else now
                resolved_title = (title or "").strip() or _derive_title(thread)
                resolved_agent = (
                    agent_path
                    if agent_path is not None
                    else (existing["agent_path"] if existing else thread.agent_id)
                )
                resolved_provider = (
                    provider
                    if provider is not None
                    else (existing["provider"] if existing else None)
                )
                effective_project = project_id or (
                    existing["project_id"] if existing else None
                )
                resolved_subject = subject_id or (
                    existing["subject_id"] if existing else None
                )
                resolved_origin = origin or (
                    existing["origin"] if existing else ORIGIN_CLI
                )
                await conn.execute(
                    "INSERT INTO aux_conversations "
                    "(tenant, conversation_id, thread, origin, title, agent_path, "
                    "provider, project_id, subject_id, created_at, updated_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) "
                    "ON CONFLICT (tenant, conversation_id) DO UPDATE SET "
                    "thread = EXCLUDED.thread, title = EXCLUDED.title, "
                    "agent_path = EXCLUDED.agent_path, provider = EXCLUDED.provider, "
                    "project_id = EXCLUDED.project_id, "
                    "subject_id = EXCLUDED.subject_id, "
                    "updated_at = EXCLUDED.updated_at",
                    self._tenant,
                    conversation_id,
                    json.loads(thread.model_dump_json()),
                    resolved_origin,
                    resolved_title,
                    resolved_agent,
                    resolved_provider,
                    effective_project,
                    resolved_subject,
                    _norm_ts(created),
                    now,
                )
                await self._reproject(conn, conversation_id, thread, now)
        flat = flat_from_thread(thread)
        return ConversationSummary(
            conversation_id=conversation_id,
            origin=resolved_origin,
            title=resolved_title,
            agent_path=resolved_agent,
            provider=resolved_provider,
            project_id=effective_project,
            created_at=_norm_ts(created),
            updated_at=now,
            message_count=len(flat),
        )

    async def _reproject(
        self, conn: Any, conversation_id: str, thread: ChatThread, now: str
    ) -> None:
        from himmy.services.storage.conversations import flat_from_thread

        await conn.execute(
            "DELETE FROM aux_conversation_messages "
            "WHERE tenant = $1 AND conversation_id = $2",
            self._tenant,
            conversation_id,
        )
        for seq, (role, text) in enumerate(flat_from_thread(thread)):
            await conn.execute(
                "INSERT INTO aux_conversation_messages "
                "(tenant, id, conversation_id, role, text, seq, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                self._tenant,
                new_uuid(),
                conversation_id,
                role,
                text,
                seq,
                now,
            )

    async def load_thread(self, conversation_id: str) -> ChatThread | None:
        from himmy.agents.base_agent.thread import ChatThread

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT thread FROM aux_conversations "
                "WHERE tenant = $1 AND conversation_id = $2",
                self._tenant,
                conversation_id,
            )
        if row is None:
            return None
        try:
            return ChatThread.model_validate(dict(row["thread"]))
        except Exception:  # noqa: BLE001 - a corrupt row must not break the caller
            return None

    async def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT *, ("
                "  SELECT COUNT(*) FROM aux_conversation_messages m "
                "  WHERE m.tenant = c.tenant "
                "  AND m.conversation_id = c.conversation_id) AS n "
                "FROM aux_conversations c "
                "WHERE c.tenant = $1 AND c.conversation_id = $2",
                self._tenant,
                conversation_id,
            )
        if row is None:
            return None
        return _row_to_summary(row, int(row["n"]))

    async def flat_messages(self, conversation_id: str) -> list[FlatMessage]:
        from himmy.services.storage.conversations import FlatMessage

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, text FROM aux_conversation_messages "
                "WHERE tenant = $1 AND conversation_id = $2 ORDER BY seq",
                self._tenant,
                conversation_id,
            )
        return [FlatMessage(role=r["role"], text=r["text"]) for r in rows]

    async def list_summaries(
        self, *, limit: int | None, origin: str | None
    ) -> list[ConversationSummary]:
        sql = (
            "SELECT c.*, ("
            "  SELECT COUNT(*) FROM aux_conversation_messages m "
            "  WHERE m.tenant = c.tenant "
            "  AND m.conversation_id = c.conversation_id) AS n "
            "FROM aux_conversations c WHERE c.tenant = $1"
        )
        params: list[Any] = [self._tenant]
        if origin is not None:
            params.append(origin)
            sql += f" AND c.origin = ${len(params)}"
        sql += " ORDER BY c.updated_at DESC"
        if limit is not None:
            params.append(int(limit))
            sql += f" LIMIT ${len(params)}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_summary(r, int(r["n"])) for r in rows]

    async def rename(self, conversation_id: str, title: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE aux_conversations SET title = $1, updated_at = $2 "
                "WHERE tenant = $3 AND conversation_id = $4",
                title,
                utc_now_iso(),
                self._tenant,
                conversation_id,
            )
        return _rowcount(result) > 0

    async def delete(self, conversation_id: str) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM aux_conversation_messages "
                    "WHERE tenant = $1 AND conversation_id = $2",
                    self._tenant,
                    conversation_id,
                )
                result = await conn.execute(
                    "DELETE FROM aux_conversations "
                    "WHERE tenant = $1 AND conversation_id = $2",
                    self._tenant,
                    conversation_id,
                )
        return _rowcount(result) > 0

    async def subject_of(self, conversation_id: str) -> str | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT subject_id FROM aux_conversations "
                "WHERE tenant = $1 AND conversation_id = $2",
                self._tenant,
                conversation_id,
            )
        return row["subject_id"] if row and row["subject_id"] else None

    async def conversation_ids_for_subject(self, subject_id: str) -> list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT conversation_id FROM aux_conversations "
                "WHERE tenant = $1 AND subject_id = $2",
                self._tenant,
                subject_id,
            )
        return [r["conversation_id"] for r in rows]

    async def delete_by_subject(self, subject_id: str) -> int:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM aux_conversation_messages WHERE tenant = $1 "
                    "AND conversation_id IN ("
                    "  SELECT conversation_id FROM aux_conversations "
                    "  WHERE tenant = $1 AND subject_id = $2)",
                    self._tenant,
                    subject_id,
                )
                result = await conn.execute(
                    "DELETE FROM aux_conversations "
                    "WHERE tenant = $1 AND subject_id = $2",
                    self._tenant,
                    subject_id,
                )
        return _rowcount(result)

    async def set_project(
        self, conversation_id: str, project_id: str | None
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE aux_conversations SET project_id = $1 "
                "WHERE tenant = $2 AND conversation_id = $3",
                project_id,
                self._tenant,
                conversation_id,
            )
        return _rowcount(result) > 0

    # -- projects ------------------------------------------------------------

    async def create_project(
        self,
        *,
        name: str,
        description: str,
        kb_id: str | None,
        agent_path: str | None,
    ) -> dict[str, object]:
        now = utc_now_iso()
        pid = new_uuid()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_projects "
                "(tenant, id, name, description, kb_id, agent_path, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                self._tenant,
                pid,
                name,
                description,
                kb_id,
                agent_path,
                now,
                now,
            )
        return {
            "id": pid,
            "name": name,
            "description": description,
            "kb_id": kb_id,
            "agent_path": agent_path,
            "created_at": now,
            "updated_at": now,
            "chat_count": 0,
        }

    async def get_project_row(self, project_id: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM aux_projects WHERE tenant = $1 AND id = $2",
                self._tenant,
                project_id,
            )
        return _project_dict(row) if row is not None else None

    async def project_chat_count(self, project_id: str) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM aux_conversations "
                "WHERE tenant = $1 AND project_id = $2",
                self._tenant,
                project_id,
            )
        return int(row["n"]) if row else 0

    async def list_project_rows(self) -> list[tuple[dict[str, Any], int]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT p.*, ("
                "  SELECT COUNT(*) FROM aux_conversations c "
                "  WHERE c.tenant = p.tenant AND c.project_id = p.id) AS n "
                "FROM aux_projects p WHERE p.tenant = $1 ORDER BY p.updated_at DESC",
                self._tenant,
            )
        return [(_project_dict(r), int(r["n"])) for r in rows]

    async def update_project(
        self, project_id: str, assignments: dict[str, str | None]
    ) -> bool:
        if not assignments:
            return (await self.get_project_row(project_id)) is not None
        cols = ", ".join(f"{k} = ${i + 1}" for i, k in enumerate(assignments))
        params: list[Any] = list(assignments.values())
        params.append(utc_now_iso())
        params.append(self._tenant)
        params.append(project_id)
        sql = (
            f"UPDATE aux_projects SET {cols}, updated_at = ${len(params) - 2} "  # noqa: S608
            f"WHERE tenant = ${len(params) - 1} AND id = ${len(params)}"
        )
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params)
        return _rowcount(result) > 0

    async def delete_project(self, project_id: str) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE aux_conversations SET project_id = NULL "
                    "WHERE tenant = $1 AND project_id = $2",
                    self._tenant,
                    project_id,
                )
                result = await conn.execute(
                    "DELETE FROM aux_projects WHERE tenant = $1 AND id = $2",
                    self._tenant,
                    project_id,
                )
        return _rowcount(result) > 0

    async def project_conversation_summaries(
        self, project_id: str
    ) -> list[ConversationSummary]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT c.*, ("
                "  SELECT COUNT(*) FROM aux_conversation_messages m "
                "  WHERE m.tenant = c.tenant "
                "  AND m.conversation_id = c.conversation_id) AS n "
                "FROM aux_conversations c "
                "WHERE c.tenant = $1 AND c.project_id = $2 "
                "ORDER BY c.updated_at DESC",
                self._tenant,
                project_id,
            )
        return [_row_to_summary(r, int(r["n"])) for r in rows]

    async def assign_conversation(
        self, project_id: str, conversation_id: str
    ) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                exists = await conn.fetchrow(
                    "SELECT 1 FROM aux_projects WHERE tenant = $1 AND id = $2",
                    self._tenant,
                    project_id,
                )
                if exists is None:
                    return False
                result = await conn.execute(
                    "UPDATE aux_conversations SET project_id = $1 "
                    "WHERE tenant = $2 AND conversation_id = $3",
                    project_id,
                    self._tenant,
                    conversation_id,
                )
                if _rowcount(result) > 0:
                    await conn.execute(
                        "UPDATE aux_projects SET updated_at = $1 "
                        "WHERE tenant = $2 AND id = $3",
                        utc_now_iso(),
                        self._tenant,
                        project_id,
                    )
        return _rowcount(result) > 0

    async def unassign_conversation(self, conversation_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE aux_conversations SET project_id = NULL "
                "WHERE tenant = $1 AND conversation_id = $2 AND project_id IS NOT NULL",
                self._tenant,
                conversation_id,
            )
        return _rowcount(result) > 0


class PostgresConversationStore:
    """Synchronous Postgres mirror of :class:`ConversationStore` (K4).

    The full ConversationStore surface (authoritative thread + regenerated flat projection
    + projects), every method blocking on the shared aux loop. Tenant-scoped on a single
    ``tenant`` value (``"local"`` for the shared CLI/Studio/v1 singleton). ``save_flat`` and
    ``import_legacy`` are implemented in Python over the sync surface, mirroring the SQLite
    store's own composition.
    """

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncConversationStore(self._pool, tenant)

    def save_thread(
        self,
        conversation_id: str,
        thread: ChatThread,
        *,
        origin: str | None = None,
        title: str | None = None,
        agent_path: str | None = None,
        provider: str | None = None,
        project_id: str | None = None,
        subject_id: str | None = None,
    ) -> ConversationSummary:
        from himmy.services.storage.conversations import ORIGIN_CLI

        return self._pool.run(  # type: ignore[no-any-return]
            self._async.save_thread(
                conversation_id,
                thread,
                origin=origin or ORIGIN_CLI,
                title=title,
                agent_path=agent_path,
                provider=provider,
                project_id=project_id,
                subject_id=subject_id,
            )
        )

    def save_flat(
        self,
        *,
        conversation_id: str | None,
        title: str | None,
        agent_path: str | None,
        provider: str | None,
        flat_messages: Sequence[tuple[str, str]],
        project_id: str | None = None,
        origin: str | None = None,
    ) -> ConversationSummary:
        from himmy.services.storage.conversations import (
            ORIGIN_STUDIO,
            thread_from_flat,
        )

        cid = conversation_id or new_uuid()
        thread = thread_from_flat(
            conversation_id=cid, agent_path=agent_path, flat_messages=flat_messages
        )
        return self.save_thread(
            cid,
            thread,
            origin=origin or ORIGIN_STUDIO,
            title=title,
            agent_path=agent_path,
            provider=provider,
            project_id=project_id,
        )

    def load_thread(self, conversation_id: str) -> ChatThread | None:
        return self._pool.run(self._async.load_thread(conversation_id))  # type: ignore[no-any-return]

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        return self._pool.run(self._async.get_summary(conversation_id))  # type: ignore[no-any-return]

    def flat_messages(self, conversation_id: str) -> list[FlatMessage]:
        return self._pool.run(self._async.flat_messages(conversation_id))  # type: ignore[no-any-return]

    def list_summaries(
        self, *, limit: int | None = None, origin: str | None = None
    ) -> list[ConversationSummary]:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.list_summaries(limit=limit, origin=origin)
        )

    def rename(self, conversation_id: str, title: str) -> bool:
        return bool(self._pool.run(self._async.rename(conversation_id, title)))

    def delete(self, conversation_id: str) -> bool:
        return bool(self._pool.run(self._async.delete(conversation_id)))

    def subject_of(self, conversation_id: str) -> str | None:
        return self._pool.run(self._async.subject_of(conversation_id))  # type: ignore[no-any-return]

    def conversation_ids_for_subject(self, subject_id: str) -> list[str]:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.conversation_ids_for_subject(subject_id)
        )

    def delete_by_subject(self, subject_id: str) -> int:
        return int(self._pool.run(self._async.delete_by_subject(subject_id)))

    def set_project(self, conversation_id: str, project_id: str | None) -> bool:
        return bool(
            self._pool.run(self._async.set_project(conversation_id, project_id))
        )

    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        kb_id: str | None = None,
        agent_path: str | None = None,
    ) -> dict[str, object]:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.create_project(
                name=name,
                description=description,
                kb_id=kb_id,
                agent_path=agent_path,
            )
        )

    def get_project_row(self, project_id: str) -> dict[str, Any] | None:
        return self._pool.run(self._async.get_project_row(project_id))  # type: ignore[no-any-return]

    def project_chat_count(self, project_id: str) -> int:
        return int(self._pool.run(self._async.project_chat_count(project_id)))

    def list_project_rows(self) -> list[tuple[dict[str, Any], int]]:
        return self._pool.run(self._async.list_project_rows())  # type: ignore[no-any-return]

    def update_project(
        self, project_id: str, assignments: dict[str, str | None]
    ) -> bool:
        return bool(
            self._pool.run(self._async.update_project(project_id, assignments))
        )

    def delete_project(self, project_id: str) -> bool:
        return bool(self._pool.run(self._async.delete_project(project_id)))

    def project_conversation_summaries(
        self, project_id: str
    ) -> list[ConversationSummary]:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.project_conversation_summaries(project_id)
        )

    def assign_conversation(self, project_id: str, conversation_id: str) -> bool:
        return bool(
            self._pool.run(
                self._async.assign_conversation(project_id, conversation_id)
            )
        )

    def unassign_conversation(self, conversation_id: str) -> bool:
        return bool(self._pool.run(self._async.unassign_conversation(conversation_id)))

    def close(self) -> None:
        self._pool.close()


def _row_to_summary(row: Any, count: int) -> ConversationSummary:
    from himmy.services.storage.conversations import ConversationSummary

    return ConversationSummary(
        conversation_id=row["conversation_id"],
        origin=row["origin"],
        title=row["title"],
        agent_path=row["agent_path"],
        provider=row["provider"],
        project_id=row["project_id"],
        created_at=_norm_ts(row["created_at"]),
        updated_at=_norm_ts(row["updated_at"]),
        message_count=count,
    )


def _project_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "kb_id": row["kb_id"],
        "agent_path": row["agent_path"],
        "created_at": _norm_ts(row["created_at"]),
        "updated_at": _norm_ts(row["updated_at"]),
    }


# ============================================================ K4: routines mirror


class _AsyncRoutinesStore:
    """Async body of the Postgres routines store with the atomic cluster-wide tick CAS."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def list(self, workspace_id: str | None) -> list[Any]:
        from himmy.api.routines import Routine

        if workspace_id is None:
            sql = (
                "SELECT data FROM aux_routines WHERE tenant = $1 "
                "ORDER BY created_at DESC"
            )
            params: list[Any] = [self._tenant]
        else:
            sql = (
                "SELECT data FROM aux_routines WHERE tenant = $1 AND workspace_id = $2 "
                "ORDER BY created_at DESC"
            )
            params = [self._tenant, workspace_id]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [Routine.model_validate(dict(r["data"])) for r in rows]

    async def get(self, routine_id: str, workspace_id: str | None) -> Any | None:
        from himmy.api.routines import Routine

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM aux_routines WHERE tenant = $1 AND id = $2",
                self._tenant,
                routine_id,
            )
        if row is None:
            return None
        routine = Routine.model_validate(dict(row["data"]))
        if workspace_id is not None and routine.workspace_id != workspace_id:
            return None
        return routine

    async def count(self, workspace_id: str, enabled_only: bool) -> int:
        """Count a workspace's routines (per-tenant routine quota, T3).

        ``enabled_only`` restricts to ENABLED routines (the ones that actually fire), which is
        what the per-tenant active-routine cap is enforced against. ``enabled`` lives inside
        the JSON blob (the PG mirror stores the whole routine as ``data``), so it is read with
        ``(data->>'enabled')::boolean`` — mirroring the SQLite store's ``enabled = 1`` filter.
        """
        async with self._pool.acquire() as conn:
            if enabled_only:
                n = await conn.fetchval(
                    "SELECT COUNT(*) FROM aux_routines "
                    "WHERE tenant = $1 AND workspace_id = $2 "
                    "AND COALESCE((data->>'enabled')::boolean, true) IS TRUE",
                    self._tenant,
                    workspace_id,
                )
            else:
                n = await conn.fetchval(
                    "SELECT COUNT(*) FROM aux_routines "
                    "WHERE tenant = $1 AND workspace_id = $2",
                    self._tenant,
                    workspace_id,
                )
        return int(n or 0)

    async def _upsert_on_conn(self, conn: Any, routine: Any) -> None:
        """Run the routine INSERT…ON CONFLICT on an explicit ``conn`` (no acquire).

        Factored out so :meth:`upsert` (own connection) and
        :meth:`create_routine_if_under_quota` (the advisory-locked transaction's connection)
        write the BYTE-IDENTICAL row; the caller stamps ``updated_at`` first.
        """
        await conn.execute(
            "INSERT INTO aux_routines "
            "(tenant, id, workspace_id, data, last_run_at, agent_id, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) "
            "ON CONFLICT (tenant, id) DO UPDATE SET "
            "workspace_id = EXCLUDED.workspace_id, data = EXCLUDED.data, "
            "last_run_at = EXCLUDED.last_run_at, agent_id = EXCLUDED.agent_id",
            self._tenant,
            routine.id,
            routine.workspace_id,
            json.loads(routine.model_dump_json()),
            routine.last_run_at,
            routine.agent_id,
            _norm_ts(routine.created_at),
        )

    async def upsert(self, routine: Any) -> Any:
        routine.updated_at = utc_now_iso()
        async with self._pool.acquire() as conn:
            await self._upsert_on_conn(conn, routine)
        return routine

    async def create_routine_if_under_quota(
        self, routine: Any, *, cap: int, enabled_only: bool = True
    ) -> tuple[Any, bool]:
        """Atomically create ``routine`` IFF the tenant is under its routine cap (T3, HARD).

        Fuses the per-tenant COUNT and the INSERT into ONE transaction guarded by a
        TRANSACTION-scoped per-workspace BLOCKING advisory lock
        (``pg_advisory_xact_lock(ROUTINE_QUOTA_ADVISORY_NAMESPACE, ws_hash)``), so concurrent
        creates for the SAME workspace SERIALISE across processes: a create cannot
        skip-on-contention like the fair-CLAIM (that would be a spurious 429), so it BLOCKS,
        and the loser then re-counts under the lock and sees the winner's committed row — the
        result is EXACTLY ``cap`` admits with the rest a deterministic over-cap reject, never
        the unlocked count-then-upsert overshoot. Different workspaces hash to different keys
        and never block each other; the xact-scoped lock auto-releases on commit/rollback (no
        leak, even on a crash). ``SET LOCAL statement_timeout`` bounds the lock wait so a
        genuinely stuck lock fails the statement (→ fail-OPEN) instead of hanging the pool.

        Returns ``(routine, admitted)``. ``admitted=False`` ⇒ at/over ``cap``, nothing
        written. FAIL-OPEN on an infrastructure error (lock/count/db/timeout): fall back to
        the plain :meth:`upsert` and report ``admitted=True`` so an infra hiccup never turns a
        quota guard into a hard outage; only a clean over-cap is a fail-CLOSED reject.
        """
        if cap <= 0:  # defensive: an uncapped tenant is always admitted (plain upsert)
            return await self.upsert(routine), True
        routine.updated_at = utc_now_iso()
        enabled_clause = (
            " AND COALESCE((data->>'enabled')::boolean, true) IS TRUE"
            if enabled_only
            else ""
        )
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                # Bound the BLOCKING lock wait so a stuck lock cannot hang the aux loop /
                # exhaust the pool; a trip raises → caught below → fail-OPEN admit.
                await conn.execute(
                    f"SET LOCAL statement_timeout = {_QUOTA_LOCK_STATEMENT_TIMEOUT_MS}"
                )
                await conn.execute(
                    "SELECT pg_advisory_xact_lock($1, $2)",
                    ROUTINE_QUOTA_ADVISORY_NAMESPACE,
                    _routine_quota_advisory_key(routine.workspace_id),
                )
                current = await conn.fetchval(
                    "SELECT COUNT(*) FROM aux_routines "  # noqa: S608
                    "WHERE tenant = $1 AND workspace_id = $2" + enabled_clause,
                    self._tenant,
                    routine.workspace_id,
                )
                if int(current or 0) >= cap:
                    return routine, False  # over cap: commit (no insert), lock releases
                await self._upsert_on_conn(conn, routine)
                return routine, True
        except Exception:  # noqa: BLE001 - fail OPEN on infra error (not over-cap)
            import logging

            logging.getLogger("himmy.services.storage.postgres_aux").warning(
                "atomic routine-quota create failed for workspace %s; admitting",
                routine.workspace_id,
            )
            return await self.upsert(routine), True

    async def delete(self, routine_id: str, workspace_id: str | None) -> bool:
        async with self._pool.acquire() as conn:
            if workspace_id is None:
                result = await conn.execute(
                    "DELETE FROM aux_routines WHERE tenant = $1 AND id = $2",
                    self._tenant,
                    routine_id,
                )
            else:
                result = await conn.execute(
                    "DELETE FROM aux_routines "
                    "WHERE tenant = $1 AND id = $2 AND workspace_id = $3",
                    self._tenant,
                    routine_id,
                    workspace_id,
                )
        return _rowcount(result) > 0

    async def find_by_agent_id(
        self, agent_id: str, workspace_id: str
    ) -> _list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM aux_routines "
                "WHERE tenant = $1 AND agent_id = $2 AND workspace_id = $3",
                self._tenant,
                agent_id,
                workspace_id,
            )
        return [r["id"] for r in rows]

    async def mark_started(
        self,
        routine_id: str,
        started_at: str,
        expected_last_run_at: Any,
        *,
        conditional: bool,
    ) -> bool:
        """Atomic cluster-wide start claim: stamp the run start IFF ``last_run_at`` is
        still the value read at due-evaluation (the conditional-UPDATE tick).

        The cluster-wide double-execution fix (K4 reviewer must_fix): a routine due on
        every tick is claimed by the FIRST replica whose ``last_run_at`` still matches the
        sentinel captured at due-time. ``IS NOT DISTINCT FROM`` handles the NULL first-run
        case (a fresh routine's ``last_run_at`` is NULL). The claim is a single
        ``SELECT … FOR UPDATE`` + UPDATE in one transaction, so two racing replicas
        serialise on the row lock and exactly one sees the row; the loser's predicate no
        longer matches (the winner moved ``last_run_at``) and it returns ``False``.

        ``conditional`` is ``False`` for the manual ``run_now`` path (always fire — overlap
        is prevented by the host flock), ``True`` for the scheduler tick. The JSON blob's
        ``last_run_at`` / ``last_status`` / ``last_error`` / ``last_delivery`` are kept
        consistent with the indexed column via a read-modify-write of the claimed row, so a
        later load reflects the claim.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if conditional:
                    row = await conn.fetchrow(
                        "SELECT data FROM aux_routines "
                        "WHERE tenant = $1 AND id = $2 "
                        "AND last_run_at IS NOT DISTINCT FROM $3 "
                        "FOR UPDATE",
                        self._tenant,
                        routine_id,
                        expected_last_run_at,
                    )
                else:
                    row = await conn.fetchrow(
                        "SELECT data FROM aux_routines "
                        "WHERE tenant = $1 AND id = $2 FOR UPDATE",
                        self._tenant,
                        routine_id,
                    )
                if row is None:
                    return False
                data = dict(row["data"])
                data["last_run_at"] = started_at
                data["last_status"] = "running"
                data["last_error"] = None
                data["last_delivery"] = None
                await conn.execute(
                    "UPDATE aux_routines SET last_run_at = $1, data = $2 "
                    "WHERE tenant = $3 AND id = $4",
                    started_at,
                    data,
                    self._tenant,
                    routine_id,
                )
        return True

    async def record_result(
        self,
        routine_id: str,
        *,
        status: str,
        preview: str,
        error: str | None,
        delivery: str | None,
    ) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT data FROM aux_routines "
                    "WHERE tenant = $1 AND id = $2 FOR UPDATE",
                    self._tenant,
                    routine_id,
                )
                if row is None:
                    return
                data = dict(row["data"])
                data["last_status"] = status
                data["last_preview"] = preview
                data["last_error"] = error
                data["last_delivery"] = delivery
                await conn.execute(
                    "UPDATE aux_routines SET data = $1 WHERE tenant = $2 AND id = $3",
                    data,
                    self._tenant,
                    routine_id,
                )

    async def advance_next_fire(
        self, routine_id: str, next_fire_at: str | None
    ) -> None:
        """Persist the recomputed ``next_fire_at`` into the JSON blob (Phase 1).

        Read-modify-write of the routine's ``data`` so the new pydantic field round-trips;
        idempotent against the same planned instant. The dedup claim keyed on the planned
        fire is the at-most-once backstop — this is the observable hint, not the primitive.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT data FROM aux_routines "
                    "WHERE tenant = $1 AND id = $2 FOR UPDATE",
                    self._tenant,
                    routine_id,
                )
                if row is None:
                    return
                data = dict(row["data"])
                data["next_fire_at"] = next_fire_at
                await conn.execute(
                    "UPDATE aux_routines SET data = $1 WHERE tenant = $2 AND id = $3",
                    data,
                    self._tenant,
                    routine_id,
                )


class PostgresRoutinesStore:
    """Synchronous Postgres mirror of :class:`RoutinesStore` (K4) with the atomic tick.

    Same surface as the SQLite store EXCEPT ``mark_started``, which gains an
    ``expected_last_run_at`` sentinel and returns whether THIS replica won the claim — the
    conditional UPDATE that makes a routine fire exactly once across N replicas. The SQLite
    store keeps its host ``flock`` instead (single box).
    """

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncRoutinesStore(self._pool, tenant)

    def list(self, *, workspace_id: str | None = None) -> _list[Any]:
        return self._pool.run(self._async.list(workspace_id))  # type: ignore[no-any-return]

    def get(self, routine_id: str, *, workspace_id: str | None = None) -> Any | None:
        return self._pool.run(self._async.get(routine_id, workspace_id))  # type: ignore[no-any-return]

    def count(self, *, workspace_id: str, enabled_only: bool = False) -> int:
        """Count a workspace's routines (T3 per-tenant routine quota; PG mirror of SQLite)."""
        return int(self._pool.run(self._async.count(workspace_id, enabled_only)))

    def upsert(self, routine: Any) -> Any:
        return self._pool.run(self._async.upsert(routine))

    def create_routine_if_under_quota(
        self, routine: Any, *, cap: int, enabled_only: bool = True
    ) -> tuple[Any, bool]:
        """Atomic per-tenant routine-quota create (PG mirror of the SQLite store).

        Drives :meth:`_AsyncRoutinesStore.create_routine_if_under_quota` — one
        advisory-locked transaction that fuses count+insert so concurrent same-workspace
        creates serialise and land EXACTLY at ``cap`` across worker processes. Returns
        ``(routine, admitted)``.
        """
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.create_routine_if_under_quota(
                routine, cap=cap, enabled_only=enabled_only
            )
        )

    def delete(self, routine_id: str, *, workspace_id: str | None = None) -> bool:
        return bool(self._pool.run(self._async.delete(routine_id, workspace_id)))

    def find_by_agent_id(self, agent_id: str, *, workspace_id: str) -> _list[str]:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.find_by_agent_id(agent_id, workspace_id)
        )

    def mark_started(
        self,
        routine_id: str,
        started_at: str,
        *,
        expected_last_run_at: Any = None,
    ) -> bool:
        """Atomic start-claim CAS (see :meth:`_AsyncRoutinesStore.mark_started`).

        Returns ``True`` iff this replica won the claim. The scheduler tick captures the
        ``last_run_at`` value at due-evaluation and passes it here, gating the claim
        (``IS NOT DISTINCT FROM``, NULL-safe for the first run). The manual ``run_now`` path
        passes :data:`himmy.api.routines.UNCONDITIONAL` to fire regardless (overlap there is
        the host flock's job, not the anchor's).
        """
        from himmy.api.routines import _Unconditional

        conditional = not isinstance(expected_last_run_at, _Unconditional)
        sentinel = None if not conditional else expected_last_run_at
        return bool(
            self._pool.run(
                self._async.mark_started(
                    routine_id, started_at, sentinel, conditional=conditional
                )
            )
        )

    def record_result(
        self,
        routine_id: str,
        *,
        status: str,
        preview: str,
        error: str | None = None,
        delivery: str | None = None,
    ) -> None:
        from himmy.api.routines import PREVIEW_MAX_CHARS

        self._pool.run(
            self._async.record_result(
                routine_id,
                status=status,
                preview=preview[:PREVIEW_MAX_CHARS],
                error=error,
                delivery=delivery,
            )
        )

    def advance_next_fire(self, routine_id: str, next_fire_at: str | None) -> None:
        """Persist the recomputed ``next_fire_at`` (Phase 1; cluster mirror)."""
        self._pool.run(self._async.advance_next_fire(routine_id, next_fire_at))

    def close(self) -> None:
        self._pool.close()


# ============================================================ K4: teams + workflows mirrors


class _AsyncBindingStore:
    """Async body shared by the teams and workflows Postgres mirrors (same shape)."""

    def __init__(self, pool: _AuxPgPool, tenant: str, table: str, model: Any) -> None:
        self._pool = pool
        self._tenant = tenant
        self._table = table
        self._model = model

    async def list(self, workspace_id: str | None) -> list[Any]:
        if workspace_id is None:
            sql = (
                f"SELECT data FROM {self._table} WHERE tenant = $1 "  # noqa: S608
                "ORDER BY created_at DESC"
            )
            params: list[Any] = [self._tenant]
        else:
            sql = (
                f"SELECT data FROM {self._table} "  # noqa: S608
                "WHERE tenant = $1 AND workspace_id = $2 ORDER BY created_at DESC"
            )
            params = [self._tenant, workspace_id]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [self._model.model_validate(dict(r["data"])) for r in rows]

    async def get(self, record_id: str, workspace_id: str | None) -> Any | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT data FROM {self._table} WHERE tenant = $1 AND id = $2",  # noqa: S608
                self._tenant,
                record_id,
            )
        if row is None:
            return None
        record = self._model.model_validate(dict(row["data"]))
        if workspace_id is not None and record.workspace_id != workspace_id:
            return None
        return record

    async def get_by_idempotency_key(
        self, key: str, workspace_id: str
    ) -> Any | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT data FROM {self._table} "  # noqa: S608
                "WHERE tenant = $1 AND idempotency_key = $2 AND workspace_id = $3",
                self._tenant,
                key,
                workspace_id,
            )
        return self._model.model_validate(dict(row["data"])) if row else None

    async def upsert(self, record: Any) -> Any:
        record.updated_at = utc_now_iso()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {self._table} "  # noqa: S608
                "(tenant, id, workspace_id, data, idempotency_key, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6) "
                "ON CONFLICT (tenant, id) DO UPDATE SET "
                "workspace_id = EXCLUDED.workspace_id, data = EXCLUDED.data, "
                "idempotency_key = EXCLUDED.idempotency_key",
                self._tenant,
                record.id,
                record.workspace_id,
                json.loads(record.model_dump_json()),
                record.idempotency_key,
                _norm_ts(record.created_at),
            )
        return record

    async def delete(self, record_id: str, workspace_id: str | None) -> bool:
        async with self._pool.acquire() as conn:
            if workspace_id is None:
                result = await conn.execute(
                    f"DELETE FROM {self._table} WHERE tenant = $1 AND id = $2",  # noqa: S608
                    self._tenant,
                    record_id,
                )
            else:
                result = await conn.execute(
                    f"DELETE FROM {self._table} "  # noqa: S608
                    "WHERE tenant = $1 AND id = $2 AND workspace_id = $3",
                    self._tenant,
                    record_id,
                    workspace_id,
                )
        return _rowcount(result) > 0

    async def find_by_agent_id(
        self, agent_id: str, workspace_id: str
    ) -> _list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, data FROM {self._table} "  # noqa: S608
                "WHERE tenant = $1 AND workspace_id = $2",
                self._tenant,
                workspace_id,
            )
        out: _list[str] = []
        for r in rows:
            members = (dict(r["data"]).get("members")) or []
            if agent_id in members:
                out.append(r["id"])
        return out


class _PostgresBindingStore:
    """Synchronous facade shared by the teams + workflows Postgres mirrors."""

    def __init__(
        self, *, tenant: str, table: str, model: Any, dsn: str | None
    ) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncBindingStore(self._pool, tenant, table, model)

    def list(self, *, workspace_id: str | None = None) -> _list[Any]:
        return self._pool.run(self._async.list(workspace_id))  # type: ignore[no-any-return]

    def get(self, record_id: str, *, workspace_id: str | None = None) -> Any | None:
        return self._pool.run(self._async.get(record_id, workspace_id))  # type: ignore[no-any-return]

    def get_by_idempotency_key(self, key: str, *, workspace_id: str) -> Any | None:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.get_by_idempotency_key(key, workspace_id)
        )

    def upsert(self, record: Any) -> Any:
        return self._pool.run(self._async.upsert(record))

    def delete(self, record_id: str, *, workspace_id: str | None = None) -> bool:
        return bool(self._pool.run(self._async.delete(record_id, workspace_id)))

    def find_by_agent_id(self, agent_id: str, *, workspace_id: str) -> _list[str]:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.find_by_agent_id(agent_id, workspace_id)
        )

    def close(self) -> None:
        self._pool.close()


class PostgresTeamsStore(_PostgresBindingStore):
    """Synchronous Postgres mirror of :class:`TeamsStore` (K4)."""

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        from himmy.api.teams_store import TeamRecord

        super().__init__(tenant=tenant, table="aux_teams", model=TeamRecord, dsn=dsn)


class PostgresWorkflowsStore(_PostgresBindingStore):
    """Synchronous Postgres mirror of :class:`WorkflowsStore` (K4)."""

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        from himmy.api.teams_store import WorkflowRecord

        super().__init__(
            tenant=tenant, table="aux_workflows", model=WorkflowRecord, dsn=dsn
        )


# ============================================================ K5: Studio CRUD sidecars


class _AsyncCalendarStore:
    """Async body of the Postgres Studio-calendar mirror (tenant-scoped)."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def add(self, ev: Any, workspace_id: str | None = None) -> Any:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_calendar_events "
                "(tenant, id, date, time, title, notes, created_at, workspace_id) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (tenant, id) DO UPDATE SET "
                "date = EXCLUDED.date, time = EXCLUDED.time, title = EXCLUDED.title, "
                "notes = EXCLUDED.notes, workspace_id = EXCLUDED.workspace_id",
                self._tenant,
                ev.id,
                ev.date,
                ev.time,
                ev.title,
                ev.notes,
                ev.created_at,
                workspace_id,
            )
        return ev

    async def list(
        self, month: str | None, workspace_id: str | frozenset[str] | None = None
    ) -> list[Any]:
        from himmy.api.studio_calendar import CalendarEvent

        # Mirror the SQLite ``ORDER BY date, time IS NULL, time`` (all-day events last
        # within a day): a NULL time sorts AFTER a present one.
        order = "ORDER BY date, (time IS NULL), time"
        params: list[Any] = [self._tenant]
        sql = "SELECT * FROM aux_calendar_events WHERE tenant = $1"
        if month:
            params.append(f"{month}-%")
            sql += f" AND date LIKE ${len(params)}"
        scope, sparams = _pg_scope(workspace_id, start=len(params) + 1)
        if scope:
            sql += f" AND {scope}"
            params.extend(sparams)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"{sql} {order}", *params)  # noqa: S608
        return [
            CalendarEvent(
                id=r["id"],
                date=r["date"],
                time=r["time"],
                title=r["title"],
                notes=r["notes"],
                created_at=_norm_ts(r["created_at"]),
            )
            for r in rows
        ]

    async def delete(
        self, event_id: str, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM aux_calendar_events "  # noqa: S608
                f"WHERE tenant = $1 AND id = $2{where}",
                self._tenant,
                event_id,
                *sparams,
            )
        return _rowcount(result) > 0


class PostgresCalendarStore:
    """Synchronous, tenant-scoped Postgres mirror of :class:`CalendarStore` (K5)."""

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncCalendarStore(self._pool, tenant)

    def add(self, ev: Any, *, workspace_id: str | None = None) -> Any:
        return self._pool.run(self._async.add(ev, workspace_id))

    def list(
        self,
        *,
        month: str | None = None,
        workspace_id: str | frozenset[str] | None = None,
    ) -> list[Any]:
        return self._pool.run(self._async.list(month, workspace_id))  # type: ignore[no-any-return]

    def delete(
        self, event_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return bool(self._pool.run(self._async.delete(event_id, workspace_id)))

    def close(self) -> None:
        self._pool.close()


class _AsyncCookbookStore:
    """Async body of the Postgres Studio-cookbook mirror (tenant-scoped)."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    def _to_recipe(self, r: Any) -> Any:
        from himmy.api.studio_cookbook import Recipe

        return Recipe(
            id=r["id"],
            name=r["name"],
            agent_path=r["agent_path"],
            prompt=r["prompt"],
            notes=r["notes"],
            created_at=_norm_ts(r["created_at"]),
        )

    async def list(
        self, workspace_id: str | frozenset[str] | None = None
    ) -> list[Any]:
        scope, sparams = _pg_scope(workspace_id, start=2)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM aux_recipes WHERE tenant = $1{where} "  # noqa: S608
                "ORDER BY created_at DESC",
                self._tenant,
                *sparams,
            )
        return [self._to_recipe(r) for r in rows]

    async def get(
        self, recipe_id: str, workspace_id: str | frozenset[str] | None = None
    ) -> Any | None:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM aux_recipes "  # noqa: S608
                f"WHERE tenant = $1 AND id = $2{where}",
                self._tenant,
                recipe_id,
                *sparams,
            )
        return self._to_recipe(row) if row else None

    async def upsert(self, r: Any, workspace_id: str | None = None) -> Any:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_recipes "
                "(tenant, id, name, agent_path, prompt, notes, created_at, workspace_id) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (tenant, id) DO UPDATE SET "
                "name = EXCLUDED.name, agent_path = EXCLUDED.agent_path, "
                "prompt = EXCLUDED.prompt, notes = EXCLUDED.notes, "
                "workspace_id = EXCLUDED.workspace_id",
                self._tenant,
                r.id,
                r.name,
                r.agent_path,
                r.prompt,
                r.notes,
                r.created_at,
                workspace_id,
            )
        return r

    async def delete(
        self, recipe_id: str, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM aux_recipes "  # noqa: S608
                f"WHERE tenant = $1 AND id = $2{where}",
                self._tenant,
                recipe_id,
                *sparams,
            )
        return _rowcount(result) > 0


class PostgresCookbookStore:
    """Synchronous, tenant-scoped Postgres mirror of :class:`CookbookStore` (K5)."""

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncCookbookStore(self._pool, tenant)

    def list(self, *, workspace_id: str | frozenset[str] | None = None) -> list[Any]:
        return self._pool.run(self._async.list(workspace_id))  # type: ignore[no-any-return]

    def get(
        self, recipe_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> Any | None:
        return self._pool.run(self._async.get(recipe_id, workspace_id))

    def upsert(self, r: Any, *, workspace_id: str | None = None) -> Any:
        return self._pool.run(self._async.upsert(r, workspace_id))

    def delete(
        self, recipe_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return bool(self._pool.run(self._async.delete(recipe_id, workspace_id)))

    def close(self) -> None:
        self._pool.close()


class _AsyncNotesStore:
    """Async body of the Postgres Studio-notes mirror (tenant-scoped)."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    def _to_note(self, row: Any) -> Any:
        from himmy.api.studio_notes import Note

        return Note(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            updated_at=_norm_ts(row["updated_at"]),
        )

    async def list(
        self, workspace_id: str | frozenset[str] | None = None
    ) -> list[Any]:
        scope, sparams = _pg_scope(workspace_id, start=2)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM aux_notes WHERE tenant = $1{where} "  # noqa: S608
                "ORDER BY updated_at DESC",
                self._tenant,
                *sparams,
            )
        return [self._to_note(r) for r in rows]

    async def get(
        self, note_id: str, workspace_id: str | frozenset[str] | None = None
    ) -> Any | None:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM aux_notes "  # noqa: S608
                f"WHERE tenant = $1 AND id = $2{where}",
                self._tenant,
                note_id,
                *sparams,
            )
        return self._to_note(row) if row else None

    async def find_by_title(
        self, title: str, workspace_id: str | frozenset[str] | None = None
    ) -> Any | None:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM aux_notes "  # noqa: S608
                f"WHERE tenant = $1 AND title = $2{where} "
                "ORDER BY updated_at DESC LIMIT 1",
                self._tenant,
                title,
                *sparams,
            )
        return self._to_note(row) if row else None

    async def upsert(self, note: Any, workspace_id: str | None = None) -> Any:
        note.updated_at = utc_now_iso()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_notes "
                "(tenant, id, title, body, updated_at, workspace_id) "
                "VALUES ($1, $2, $3, $4, $5, $6) "
                "ON CONFLICT (tenant, id) DO UPDATE SET "
                "title = EXCLUDED.title, body = EXCLUDED.body, "
                "updated_at = EXCLUDED.updated_at, "
                "workspace_id = EXCLUDED.workspace_id",
                self._tenant,
                note.id,
                note.title,
                note.body,
                note.updated_at,
                workspace_id,
            )
        return note

    async def delete(
        self, note_id: str, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM aux_notes "  # noqa: S608
                f"WHERE tenant = $1 AND id = $2{where}",
                self._tenant,
                note_id,
                *sparams,
            )
        return _rowcount(result) > 0


class PostgresNotesStore:
    """Synchronous, tenant-scoped Postgres mirror of :class:`NotesStore` (K5)."""

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncNotesStore(self._pool, tenant)

    def list(self, *, workspace_id: str | frozenset[str] | None = None) -> list[Any]:
        return self._pool.run(self._async.list(workspace_id))  # type: ignore[no-any-return]

    def get(
        self, note_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> Any | None:
        return self._pool.run(self._async.get(note_id, workspace_id))

    def find_by_title(
        self, title: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> Any | None:
        return self._pool.run(self._async.find_by_title(title, workspace_id))

    def upsert(self, note: Any, *, workspace_id: str | None = None) -> Any:
        return self._pool.run(self._async.upsert(note, workspace_id))

    def delete(
        self, note_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return bool(self._pool.run(self._async.delete(note_id, workspace_id)))

    def close(self) -> None:
        self._pool.close()


class _AsyncTasksStore:
    """Async body of the Postgres Studio-tasks mirror (tenant-scoped)."""

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def list(
        self, workspace_id: str | frozenset[str] | None = None
    ) -> list[Any]:
        from himmy.api.studio_tasks import Task

        scope, sparams = _pg_scope(workspace_id, start=2)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM aux_tasks WHERE tenant = $1{where} "  # noqa: S608
                "ORDER BY done, created_at DESC",
                self._tenant,
                *sparams,
            )
        return [
            Task(
                id=r["id"],
                title=r["title"],
                done=bool(r["done"]),
                due=r["due"],
                created_at=_norm_ts(r["created_at"]),
            )
            for r in rows
        ]

    async def add(
        self,
        title: str,
        due: str | None,
        priority: int = 0,
        workspace_id: str | None = None,
    ) -> Any:
        from himmy.api.studio_tasks import Task

        # ``priority`` rides along on the in-memory Task so callers see it back; the PG
        # mirror table isn't migrated for it (priority is a SQLite-path planner feature),
        # so it isn't persisted here — keeps the mirror forward-compatible, never breaks.
        t = Task(title=title, due=due, priority=priority)
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_tasks "
                "(tenant, id, title, done, due, created_at, workspace_id) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                self._tenant,
                t.id,
                t.title,
                t.done,
                t.due,
                t.created_at,
                workspace_id,
            )
        return t

    async def set_done(
        self, task_id: str, done: bool, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        scope, sparams = _pg_scope(workspace_id, start=4)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE aux_tasks SET done = $1 "  # noqa: S608
                f"WHERE tenant = $2 AND id = $3{where}",
                done,
                self._tenant,
                task_id,
                *sparams,
            )
        return _rowcount(result) > 0

    async def update(
        self,
        task_id: str,
        due: str | None = None,
        priority: int | None = None,
        done: bool | None = None,
        workspace_id: str | frozenset[str] | None = None,
    ) -> Any:
        from himmy.api.studio_tasks import Task

        # Patch only the supplied fields; ``priority`` is not a column in the mirror table
        # so it's accepted-and-ignored for persistence (parity with add()).
        sets: list[str] = []
        vals: list[Any] = []
        idx = 1
        if due is not None:
            sets.append(f"due = ${idx}")
            vals.append(due)
            idx += 1
        if done is not None:
            sets.append(f"done = ${idx}")
            vals.append(done)
            idx += 1
        if sets:
            scope, sparams = _pg_scope(workspace_id, start=idx + 2)
            where = f" AND {scope}" if scope else ""
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"UPDATE aux_tasks SET {', '.join(sets)} "  # noqa: S608
                    f"WHERE tenant = ${idx} AND id = ${idx + 1}{where}",
                    *vals,
                    self._tenant,
                    task_id,
                    *sparams,
                )
        rscope, rsparams = _pg_scope(workspace_id, start=3)
        rwhere = f" AND {rscope}" if rscope else ""
        async with self._pool.acquire() as conn:
            r = await conn.fetchrow(
                f"SELECT * FROM aux_tasks "  # noqa: S608
                f"WHERE tenant = $1 AND id = $2{rwhere}",
                self._tenant,
                task_id,
                *rsparams,
            )
        if r is None:
            return None
        t = Task(
            id=r["id"],
            title=r["title"],
            done=bool(r["done"]),
            due=r["due"],
            priority=int(priority) if priority is not None else 0,
            created_at=_norm_ts(r["created_at"]),
        )
        return t

    async def complete_by_title(
        self, title: str, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE aux_tasks SET done = TRUE "  # noqa: S608
                f"WHERE tenant = $1 AND title = $2 AND done = FALSE{where}",
                self._tenant,
                title,
                *sparams,
            )
        return _rowcount(result) > 0

    async def delete(
        self, task_id: str, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        scope, sparams = _pg_scope(workspace_id, start=3)
        where = f" AND {scope}" if scope else ""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM aux_tasks "  # noqa: S608
                f"WHERE tenant = $1 AND id = $2{where}",
                self._tenant,
                task_id,
                *sparams,
            )
        return _rowcount(result) > 0


class PostgresTasksStore:
    """Synchronous, tenant-scoped Postgres mirror of :class:`TasksStore` (K5)."""

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncTasksStore(self._pool, tenant)

    def list(self, *, workspace_id: str | frozenset[str] | None = None) -> list[Any]:
        return self._pool.run(self._async.list(workspace_id))  # type: ignore[no-any-return]

    def add(
        self,
        title: str,
        *,
        due: str | None = None,
        priority: int = 0,
        workspace_id: str | None = None,
    ) -> Any:
        return self._pool.run(self._async.add(title, due, priority, workspace_id))

    def set_done(
        self, task_id: str, done: bool, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return bool(self._pool.run(self._async.set_done(task_id, done, workspace_id)))

    def update(
        self,
        task_id: str,
        *,
        due: str | None = None,
        priority: int | None = None,
        done: bool | None = None,
        workspace_id: str | frozenset[str] | None = None,
    ) -> Any:
        return self._pool.run(
            self._async.update(task_id, due, priority, done, workspace_id)
        )

    def complete_by_title(
        self, title: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return bool(self._pool.run(self._async.complete_by_title(title, workspace_id)))

    def delete(
        self, task_id: str, *, workspace_id: str | frozenset[str] | None = None
    ) -> bool:
        return bool(self._pool.run(self._async.delete(task_id, workspace_id)))

    def close(self) -> None:
        self._pool.close()


# ============================================================ K5: notifications mirror


class _AsyncNotifyStore:
    """Async body of the Postgres notifications mirror (tenant-scoped).

    Backs the durable bell the :mod:`himmy.api.routers.studio_notify` sink keeps. The
    in-process deque stays the live read truth within a process exactly as on the SQLite
    path; this mirror's job is the SAME best-effort durability — seed the deque on the first
    touch (``hydrate``) and persist each item (``persist_item``) so the bell survives a
    restart, with the same ring-bound (``id <= max_id - RING_SIZE`` is pruned).
    """

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def max_id(self) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(MAX(id), 0) AS m FROM aux_notifications "
                "WHERE tenant = $1",
                self._tenant,
            )
        return int(row["m"]) if row else 0

    async def hydrate(self, ring_size: int) -> tuple[list[dict[str, Any]], bool]:
        """Return ``(items oldest-first, forward_telegram)`` for the deque seed."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM aux_notifications WHERE tenant = $1 "
                "ORDER BY id DESC LIMIT $2",
                self._tenant,
                ring_size,
            )
            srow = await conn.fetchrow(
                "SELECT value FROM aux_notify_settings "
                "WHERE tenant = $1 AND key = 'forward_telegram'",
                self._tenant,
            )
        items = [_notify_row_to_item(r) for r in reversed(rows)]
        forward = bool(srow) and srow["value"] == "1"
        return items, forward

    async def persist_item(self, item: dict[str, Any], ring_size: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_notifications "
                "(tenant, id, kind, title, body, link, created_at, read) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (tenant, id) DO UPDATE SET "
                "kind = EXCLUDED.kind, title = EXCLUDED.title, body = EXCLUDED.body, "
                "link = EXCLUDED.link, created_at = EXCLUDED.created_at, "
                "read = EXCLUDED.read",
                self._tenant,
                int(item["id"]),
                item["kind"],
                item["title"],
                item["body"],
                item["link"],
                item["created_at"],
                bool(item["read"]),
            )
            await conn.execute(
                "DELETE FROM aux_notifications WHERE tenant = $1 AND id <= $2",
                self._tenant,
                int(item["id"]) - ring_size,
            )

    async def mark_read(self, ids: list[int] | None) -> None:
        async with self._pool.acquire() as conn:
            if ids is None:
                await conn.execute(
                    "UPDATE aux_notifications SET read = TRUE WHERE tenant = $1",
                    self._tenant,
                )
            else:
                await conn.executemany(
                    "UPDATE aux_notifications SET read = TRUE "
                    "WHERE tenant = $1 AND id = $2",
                    [(self._tenant, i) for i in ids],
                )

    async def set_setting(self, key: str, value: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_notify_settings (tenant, key, value) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (tenant, key) DO UPDATE SET value = EXCLUDED.value",
                self._tenant,
                key,
                value,
            )


class PostgresNotifyStore:
    """Synchronous, tenant-scoped Postgres mirror of the notify sink's durable store (K5).

    Unlike the other CRUD mirrors this is NOT a drop-in for a ``*Store`` class — the notify
    sink (:mod:`himmy.api.routers.studio_notify`) is a module-level deque + best-effort SQL
    mirror, so this exposes the small set of operations that sink performs (hydrate / persist
    / mark-read / set-setting), driven on the shared aux loop.
    """

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncNotifyStore(self._pool, tenant)

    def max_id(self) -> int:
        return int(self._pool.run(self._async.max_id()))

    def hydrate(self, ring_size: int) -> tuple[list[dict[str, Any]], bool]:
        return self._pool.run(self._async.hydrate(ring_size))  # type: ignore[no-any-return]

    def persist_item(self, item: dict[str, Any], ring_size: int) -> None:
        self._pool.run(self._async.persist_item(item, ring_size))

    def mark_read(self, ids: list[int] | None) -> None:
        self._pool.run(self._async.mark_read(ids))

    def set_setting(self, key: str, value: str) -> None:
        self._pool.run(self._async.set_setting(key, value))

    def close(self) -> None:
        self._pool.close()


def _notify_row_to_item(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "kind": row["kind"],
        "title": row["title"],
        "body": row["body"],
        "link": row["link"],
        "created_at": _norm_ts(row["created_at"]),
        "read": bool(row["read"]),
    }


# ============================================================ K5: long-term memory mirror


#: Column order shared by the memory INSERT and SELECT (matches the SQLite store's set).
_MEMORY_COLS = (
    "memory_id, subject_id, kind, text, metadata, created_at, tier, valid_from, "
    "valid_to, superseded_by, confidence, source, stable_key"
)


class _AsyncMemoryStore:
    """Async body of the Postgres long-term-memory mirror (tenant-scoped).

    Mirrors :class:`SqliteMemoryStore` 1:1 — facts in ``aux_memories`` (typed columns so
    ``subject_id`` / ``valid_to`` / ``tier`` stay indexable filters, NOT a JSON blob) and the
    typed graph links in ``aux_memory_links``. Durability routing ONLY: the O(N) recall +
    pgvector ANN index is a separate roadmap line (per the K5 spec).
    """

    def __init__(self, pool: _AuxPgPool, tenant: str) -> None:
        self._pool = pool
        self._tenant = tenant

    async def save(self, record: Any) -> Any:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO aux_memories (tenant, {_MEMORY_COLS}) "  # noqa: S608
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
                "ON CONFLICT (tenant, memory_id) DO UPDATE SET "
                "subject_id = EXCLUDED.subject_id, kind = EXCLUDED.kind, "
                "text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                "created_at = EXCLUDED.created_at, tier = EXCLUDED.tier, "
                "valid_from = EXCLUDED.valid_from, valid_to = EXCLUDED.valid_to, "
                "superseded_by = EXCLUDED.superseded_by, "
                "confidence = EXCLUDED.confidence, source = EXCLUDED.source, "
                "stable_key = EXCLUDED.stable_key",
                self._tenant,
                record.memory_id,
                record.subject_id,
                record.kind,
                record.text,
                dict(record.metadata),
                record.created_at,
                record.tier,
                record.valid_from,
                record.valid_to,
                record.superseded_by,
                record.confidence,
                record.source,
                record.stable_key,
            )
        return record

    async def list(
        self, subject_id: str | None, active_only: bool, tier: str | None
    ) -> list[Any]:
        clauses = ["tenant = $1"]
        params: list[Any] = [self._tenant]
        if subject_id is not None:
            params.append(subject_id)
            clauses.append(f"subject_id = ${len(params)}")
        if active_only:
            clauses.append("valid_to IS NULL")
        if tier is not None:
            params.append(tier)
            clauses.append(f"tier = ${len(params)}")
        where = " AND ".join(clauses)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_MEMORY_COLS} FROM aux_memories "  # noqa: S608
                f"WHERE {where} ORDER BY created_at",
                *params,
            )
        return [_memory_row_to_record(r) for r in rows]

    async def get(self, memory_id: str) -> Any | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_MEMORY_COLS} FROM aux_memories "  # noqa: S608
                "WHERE tenant = $1 AND memory_id = $2",
                self._tenant,
                memory_id,
            )
        return _memory_row_to_record(row) if row else None

    async def delete(self, memory_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM aux_memories WHERE tenant = $1 AND memory_id = $2",
                self._tenant,
                memory_id,
            )
        return _rowcount(result) > 0

    async def delete_by_subject(self, subject_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM aux_memories WHERE tenant = $1 AND subject_id = $2",
                self._tenant,
                subject_id,
            )
        return _rowcount(result)

    async def save_link(self, link: Any) -> Any:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO aux_memory_links "
                "(tenant, link_id, from_memory_id, to_memory_id, relation, metadata, "
                "created_at) VALUES ($1, $2, $3, $4, $5, $6, $7) "
                "ON CONFLICT (tenant, link_id) DO UPDATE SET "
                "from_memory_id = EXCLUDED.from_memory_id, "
                "to_memory_id = EXCLUDED.to_memory_id, relation = EXCLUDED.relation, "
                "metadata = EXCLUDED.metadata, created_at = EXCLUDED.created_at",
                self._tenant,
                link.link_id,
                link.from_memory_id,
                link.to_memory_id,
                link.relation,
                dict(link.metadata),
                link.created_at,
            )
        return link

    async def links_from(self, memory_id: str) -> _list[Any]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM aux_memory_links "
                "WHERE tenant = $1 AND from_memory_id = $2",
                self._tenant,
                memory_id,
            )
        return [_memory_row_to_link(r) for r in rows]

    async def links_to(self, memory_id: str) -> _list[Any]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM aux_memory_links "
                "WHERE tenant = $1 AND to_memory_id = $2",
                self._tenant,
                memory_id,
            )
        return [_memory_row_to_link(r) for r in rows]


class PostgresMemoryStore:
    """Synchronous, tenant-scoped Postgres mirror of :class:`SqliteMemoryStore` (K5).

    Satisfies the :class:`~himmy.services.memory.store.MemoryStore` protocol, so it drops
    into :class:`~himmy.services.memory.service.MemoryService` unchanged.
    """

    def __init__(self, *, tenant: str = "local", dsn: str | None = None) -> None:
        self._tenant = tenant
        self._pool = _new_aux_pool(dsn)
        self._async = _AsyncMemoryStore(self._pool, tenant)

    def save(self, record: Any) -> Any:
        return self._pool.run(self._async.save(record))

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[Any]:
        return self._pool.run(  # type: ignore[no-any-return]
            self._async.list(subject_id, active_only, tier)
        )

    def get(self, memory_id: str) -> Any | None:
        return self._pool.run(self._async.get(memory_id))

    def delete(self, memory_id: str) -> bool:
        return bool(self._pool.run(self._async.delete(memory_id)))

    def delete_by_subject(self, subject_id: str) -> int:
        return int(self._pool.run(self._async.delete_by_subject(subject_id)))

    def save_link(self, link: Any) -> Any:
        return self._pool.run(self._async.save_link(link))

    def links_from(self, memory_id: str) -> _list[Any]:
        return self._pool.run(self._async.links_from(memory_id))  # type: ignore[no-any-return]

    def links_to(self, memory_id: str) -> _list[Any]:
        return self._pool.run(self._async.links_to(memory_id))  # type: ignore[no-any-return]

    def close(self) -> None:
        self._pool.close()


def _memory_row_to_record(row: Any) -> Any:
    from himmy.services.memory.store import MemoryRecord

    metadata = row["metadata"]
    if isinstance(metadata, str):  # defensive: a non-JSONB codec round-trip
        metadata = json.loads(metadata)
    return MemoryRecord(
        memory_id=row["memory_id"],
        subject_id=row["subject_id"],
        kind=row["kind"],
        text=row["text"],
        metadata=dict(metadata or {}),
        created_at=_norm_ts(row["created_at"]),
        tier=row["tier"],
        valid_from=row["valid_from"] or _norm_ts(row["created_at"]),
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
        confidence=row["confidence"],
        source=row["source"],
        stable_key=row["stable_key"],
    )


def _memory_row_to_link(row: Any) -> Any:
    from himmy.services.memory.store import MemoryLink

    metadata = row["metadata"]
    if isinstance(metadata, str):  # defensive
        metadata = json.loads(metadata)
    return MemoryLink(
        link_id=row["link_id"],
        from_memory_id=row["from_memory_id"],
        to_memory_id=row["to_memory_id"],
        relation=row["relation"],
        metadata=dict(metadata or {}),
        created_at=_norm_ts(row["created_at"]),
    )


# --------------------------------------------------------------------------- helpers


def _pg_scope(
    workspace_id: str | frozenset[str] | None,
    *,
    start: int,
    column: str = "workspace_id",
) -> tuple[str, list[str]]:
    """asyncpg ``$N`` analogue of :func:`himmy.api.studio_tenant_scope.scope_clause`.

    The K5 Postgres aux mirrors carry the SAME nullable ``workspace_id`` column as the
    SQLite stores (migration v11), so a tenant-bound Studio principal threaded through the
    drained REST routes is filtered identically on either backend instead of crashing on a
    rejected keyword. The contract matches the SQLite helper byte-for-byte:

    * ``workspace_id is None`` (offline / ``all_tenants`` / the pinned ``tenant='local'``
      single-box mirror) yields an EMPTY fragment + no params — "no filtering", so the
      Postgres read is byte-unchanged from before this column existed;
    * a concrete workspace yields ``(column = $N OR column IS NULL)`` so the caller sees its
      OWN rows AND any legacy ``NULL`` row written before tenant binding;
    * a ``frozenset`` yields ``(column IN ($N, ...) OR column IS NULL)``; an EMPTY set is
      "nothing but legacy NULL rows", never "ALL".

    ``start`` is the next free ``$N`` index (the callers' fixed params — ``tenant``, ids —
    already consume the low numbers), so the fragment composes after the static predicate.
    """
    if workspace_id is None:
        return "", []
    if isinstance(workspace_id, str):
        return f"({column} = ${start} OR {column} IS NULL)", [workspace_id]
    ids = sorted(workspace_id)
    if not ids:
        return f"{column} IS NULL", []
    placeholders = ", ".join(f"${start + i}" for i in range(len(ids)))
    return f"({column} IN ({placeholders}) OR {column} IS NULL)", list(ids)


def _rowcount(result: Any) -> int:
    """Parse asyncpg's command tag (``'UPDATE 1'`` / ``'DELETE 3'``) into a row count."""
    if not isinstance(result, str):  # pragma: no cover - defensive
        return 0
    parts = result.split()
    try:
        return int(parts[-1])
    except (ValueError, IndexError):  # pragma: no cover - defensive
        return 0


def _norm_ts(value: Any) -> str:
    """Normalise a timestamp (str or asyncpg datetime) back to an ISO-8601 string."""
    if value is None:
        return utc_now_iso()
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


__all__ = [
    "PostgresCalendarStore",
    "PostgresCheckpointStore",
    "PostgresConversationStore",
    "PostgresCookbookStore",
    "PostgresGraphCheckpointStore",
    "PostgresMemoryStore",
    "PostgresNotesStore",
    "PostgresNotifyStore",
    "PostgresRoutinesStore",
    "PostgresTasksStore",
    "PostgresTeamsStore",
    "PostgresWorkflowsStore",
]
