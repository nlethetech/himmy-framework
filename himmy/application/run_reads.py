"""Tenant-scoped, NO-MUTATION run reads for :class:`~himmy.application.services.RunAppService`.

Extracted from :mod:`himmy.application.services` as the ``RunReadService`` collaborator in
the staged decomposition of ``RunAppService`` (Blueprint B, step 2 — the *safe leaf*): it
owns the read-only run queries that never mutate state — resolving a run, its provenance
lineage, listing/counting runs, replaying the event stream, and loading the conversation
thread — plus the ``await_run`` terminal-state poll helper.

Behaviour is BYTE-IDENTICAL to the former inline methods:

- every read is tenant-scoped (AAEO-4): a run outside the supplied ``workspace_id`` is
  treated as not-found,
- the reserved ``__local__`` workspace is excluded from the all-workspaces admin views of
  :meth:`list_runs` / :meth:`count_runs` (T2.2), while an explicit ``__local__`` query still
  returns them,
- the collaborator reads its handles LIVE through the shared :class:`_RunContext` (never a
  construction-time snapshot), so a re-pointed ``storage``/``registry`` is observed at once.

``RunAppService``'s public read methods delegate here; its internal ``self.get_run`` callers
continue to work because the facade keeps a delegating shim.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from himmy.application.services import (
    DEFAULT_PAGE_LIMIT,
    _maybe_await,
    _paginate,
)
from himmy.entities.lineage import DEFAULT_TRACE_DEPTH
from himmy.entities.records import stable_id_for
from himmy.services.storage.models import (
    LOCAL_WORKSPACE,
    RunRecord,
    RunStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.application.run_context import _RunContext
    from himmy.entities.lineage import LineageGraph


class RunReadService:
    """Tenant-scoped, no-mutation run reads over the shared :class:`_RunContext`.

    Holds no state of its own beyond the shared context handle; every method reads
    ``storage``/``registry`` live from the context so behaviour matches the former inline
    implementation byte-for-byte.
    """

    def __init__(self, *, context: _RunContext) -> None:
        """Wire the shared run-lifecycle context (source of the live storage/registry handles)."""
        self._ctx = context

    async def get_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> RunRecord | None:
        """Return a run record by id, scoped to a workspace (AAEO-4).

        When ``workspace_id`` is supplied, a run belonging to another workspace is
        treated as not found (returns None).
        """
        run = await self._ctx.storage.get_run(run_id)
        if run is None:
            return None
        if workspace_id is not None and run.workspace_id != workspace_id:
            return None
        return run

    async def get_run_lineage(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        max_depth: int = DEFAULT_TRACE_DEPTH,
        relations: set[str] | list[str] | None = None,
    ) -> LineageGraph | None:
        """Return the provenance subgraph for a run, or None.

        Resolves the run (tenant-scoped), finds its ``chat_thread`` entity — the
        lineage hub — and traces the connected records: the persona it used, the
        prompt, and the context snapshot it was built from. This is the read side
        of the captured lineage that fulfils the documented "trace any run back to
        its persona + evidence" promise.

        Returns None when the run is unknown / out-of-workspace, no entity registry
        is wired, or the thread was never projected into the registry.
        """
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None or self._ctx.registry is None or run.thread_id is None:
            return None
        stable_id = stable_id_for(run.thread_id, namespace="chat_thread")
        root = await _maybe_await(self._ctx.registry.get_latest(stable_id))
        if root is None:
            return None
        return cast(
            "LineageGraph | None",
            await _maybe_await(
                self._ctx.registry.trace(
                    root.record_id, max_depth=max_depth, relations=relations
                )
            ),
        )

    async def list_runs(
        self,
        *,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> list[RunRecord]:
        """List runs filtered by workspace, subject, and/or status, paginated.

        Ordered ``created_at`` desc with ``run_id`` tiebreak, then windowed by
        ``offset``/``limit`` (capped at :data:`MAX_PAGE_LIMIT`) — AAEO-8.

        Cross-tenant safety (T2.2): the all-workspaces view (``workspace_id is None``,
        an authenticated admin listing every tenant) EXCLUDES the reserved
        ``__local__`` workspace so CLI/single-user-local runs never leak into a
        multi-tenant admin list. An explicit query for ``__local__`` still returns
        them (the local Studio/CLI browse their own history).
        """
        runs = await self._ctx.storage.list_runs(
            workspace_id=workspace_id, subject_id=subject_id, status=status
        )
        if workspace_id is None:
            runs = [r for r in runs if r.workspace_id != LOCAL_WORKSPACE]
        return _paginate(
            runs,
            limit=limit,
            offset=offset,
            sort_key=lambda r: (r.created_at, r.run_id),
        )

    async def count_runs(
        self,
        *,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> int:
        """Total run count for the filter (for pagination envelopes).

        Mirrors :meth:`list_runs`' cross-tenant rule so the count matches the page:
        the all-workspaces view excludes the reserved ``__local__`` runs (T2.2).

        T2-runs-pagination: when the backing store exposes a native ``count_runs``
        (the durable SQLite/Postgres services), the total is computed with a single
        ``SELECT COUNT(*)`` — the ``__local__`` exclusion folded into the same WHERE
        clause — instead of loading + JSON-deserializing the entire runs table just
        to ``len()`` it. Stores without a native counter (the in-memory facade) fall
        back to the load+filter path, which is cheap for an in-RAM dict and preserves
        byte-for-byte the same count semantics.
        """
        native_count = getattr(self._ctx.storage, "count_runs", None)
        if native_count is not None:
            return int(
                await native_count(
                    workspace_id=workspace_id,
                    subject_id=subject_id,
                    status=status,
                    exclude_local_workspace=(workspace_id is None),
                )
            )
        runs = await self._ctx.storage.list_runs(
            workspace_id=workspace_id, subject_id=subject_id, status=status
        )
        if workspace_id is None:
            runs = [r for r in runs if r.workspace_id != LOCAL_WORKSPACE]
        return len(runs)

    async def get_run_events(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> list[Any]:
        """Replay the canonical event stream for one run (by its trace id).

        Tenant-scoped (AAEO-4): a run outside ``workspace_id`` yields ``[]``.
        """
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None:
            return []
        if run.trace_id is not None:
            events = await self._ctx.storage.list_events(trace_id=run.trace_id)
            if events:
                return events
        if run.thread_id is not None:
            return await self._ctx.storage.list_events(thread_id=run.thread_id)
        return []

    async def get_run_thread(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> Any:
        """Replay the full conversation thread for one run (tenant-scoped, AAEO-4)."""
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None or run.thread_id is None:
            return None
        return await self._ctx.storage.load_thread(run.thread_id)

    async def get_run_thread_by_thread_id(self, thread_id: str) -> Any:
        """Load the authoritative ChatThread the runtime saved under ``thread_id`` (T2g).

        The thread router pins a run's ``thread_id`` to the conversation id, so this reads
        the latest in-flight thread state from the canonical run store directly (the run
        record may not yet carry the thread_id when a continuation has only just started).
        Returns None when nothing is stored under that id.
        """
        return await self._ctx.storage.load_thread(thread_id)

    async def await_run(self, run_id: str, timeout: float = 5.0) -> RunRecord | None:
        """Poll until the run reaches a terminal state (test/example helper)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = await self._ctx.storage.get_run(run_id)
            if run is not None and run.status in (
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
            ):
                return run
            await asyncio.sleep(0.01)
        return await self._ctx.storage.get_run(run_id)
