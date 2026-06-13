"""CLI application-service container — the ONE wiring the read/write commands share (T2h/T2b).

``himmy recommendations``/``context``/``dashboard``/``runs`` and ``--persist`` all need the
SAME durable application services the ``/v1`` API and Himmy Studio drive — but built for an
*ungoverned, single-user-local* CLI rather than the API container. :func:`build_app_container`
is that single decision point, modelled on the :mod:`himmy.cli.consent` adapter pattern: it
constructs a DURABLE storage backend + the durable shared spine and wires the four
application services (:class:`RunAppService`, :class:`RecommendationAppService`,
:class:`ContextAppService`, :class:`DashboardQueryService`) over them, pointing at exactly
the same ``.himmy/storage.db`` + ``.himmy/spine.db`` the co-located server reads — so a run
``himmy run --persist`` writes is the run ``GET /v1/runs`` and the Studio runs screen list.

DELIBERATELY NOT the API container (``ApiContainer._assemble``): this builder DROPS the
governed consent overlay that ``_assemble`` layers on (``ConsentGatedStorage`` /
``ConsentAwareRegistry`` + the TRAIN consent decider). The CLI's recommendations/context/
dashboard surfaces are an ungoverned local operator view; ``himmy consent`` is the governed
surface and keeps its OWN isolated ledger (:mod:`himmy.cli.consent`). This is NOT implied
parity with ``_assemble`` — it is the ungoverned-local subset by design (reviewer must_fix).

WORKSPACE PINNING (reviewer must_fix): every read AND write goes through ONE pinned
workspace/subject — :data:`LOCAL_WORKSPACE` / :data:`LOCAL_SUBJECT` (``__local__``, the
reserved CLI/single-user-local scope the canonical run store already excludes from
cross-tenant admin list views). It is pinned for BOTH directions because
:meth:`DashboardQueryService.summary` REQUIRES a concrete ``workspace_id: str`` and filters
strictly: a reads-pass-``None`` design would make ``himmy dashboard`` counts disagree with
``himmy runs list`` (which an all-workspaces ``None`` view would *exclude* ``__local__``
from). One pinned scope keeps every CLI surface internally consistent and aligned with what
``--persist`` writes.

The store is durable here (unlike :meth:`StoreFactory.for_cli`, which is intentionally
in-RAM for a zero-setup one-shot ``himmy run``): these commands are an explicit
read/write-history surface, so they open the SAME durable ``.himmy/storage.db`` the server
persists into. ``himmy run``/``chat`` WITHOUT ``--persist`` never construct this container,
so the zero-config in-RAM default for a plain one-shot is preserved byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from himmy.services.storage.models import LOCAL_SUBJECT, LOCAL_WORKSPACE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.application.services import (
        ContextAppService,
        DashboardQueryService,
        RecommendationAppService,
        RunAppService,
    )
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.services.storage.service import StorageService

__all__ = [
    "AppContainer",
    "LOCAL_SUBJECT",
    "LOCAL_WORKSPACE",
    "build_app_container",
]


@dataclass(frozen=True)
class AppContainer:
    """The CLI's durable application-service container (the ungoverned-local wiring).

    Holds the four application services every new read/write command shares plus the
    durable storage + spine they sit on, and the pinned ``__local__`` workspace/subject
    both reads and writes use. Constructed by :func:`build_app_container`; closed by
    :meth:`close` (the store may own a SQLite file handle / Postgres pool).
    """

    storage: StorageService
    spine: EntityRegistryProtocol
    runs: RunAppService
    recommendations: RecommendationAppService
    context: ContextAppService
    dashboard: DashboardQueryService
    workspace_id: str = LOCAL_WORKSPACE
    subject_id: str = LOCAL_SUBJECT

    def close(self) -> None:
        """Release the durable store + spine handles (best-effort, idempotent).

        Handles both close shapes: the SQLite store's ``close`` is a coroutine (awaited
        on a fresh loop here), while the spine registry's is synchronous — so we await
        only when the closer returns an awaitable, never leaving a coroutine un-awaited.
        """
        import asyncio
        import inspect
        from collections.abc import Coroutine
        from typing import Any, cast

        for owned in (self.storage, self.spine):
            closer = getattr(owned, "close", None)
            if not callable(closer):
                continue
            try:
                if inspect.iscoroutinefunction(closer):
                    asyncio.run(closer())
                else:
                    result = closer()
                    if inspect.iscoroutine(result):
                        asyncio.run(cast("Coroutine[Any, Any, Any]", result))
            except Exception:  # pragma: no cover - best-effort teardown
                pass


def build_app_container() -> AppContainer:
    """Construct the CLI's durable application-service container (T2h).

    ONE wiring imported by ``himmy recommendations``/``context``/``dashboard``/``runs`` and
    by ``--persist``. Builds:

    * a DURABLE file-backed :class:`SqliteStorageService` at the canonical
      ``.himmy/storage.db`` (the SAME store the co-located server reads — resolved via
      ``HIMMY_STORE_PATH`` exactly like :meth:`StoreFactory._sqlite_store`), so a CLI write
      is visible to ``/v1`` and Studio and vice-versa;
    * the DURABLE shared spine via :meth:`SpineFactory.for_cli` (``.himmy/spine.db``) so a
      run's provenance projected by the CLI is the same on-disk graph ``himmy seclog`` /
      ``himmy lineage`` read;
    * the four application services over them, pinned to the ``__local__`` workspace/subject.

    No consent overlay is layered on (see the module docstring) — this is the ungoverned
    local subset. ``RunAppService`` here carries no checkpoint store / agent resolver: the
    CLI does not drive HITL or stored-agent runs through this container (those stay on the
    interactive ``himmy run`` path / the ``/v1`` API), so the defaults keep its construction
    minimal and the offline contract intact.
    """
    from himmy.application.services import (
        ContextAppService,
        DashboardQueryService,
        RecommendationAppService,
        RunAppService,
    )
    from himmy.entities.spine_factory import SpineFactory
    from himmy.services.context.service import ContextService

    storage = _durable_store()
    spine = SpineFactory.for_cli()

    context_service = ContextService(
        storage_service=storage, entity_registry=spine
    )
    recommendations = RecommendationAppService(storage=storage, entity_registry=spine)
    context = ContextAppService(context_service=context_service, storage=storage)
    # RunAppService needs a runtime, but the CLI read/write surfaces only call its readers
    # (list_runs/get_run) and never launch a background run through it, so a no-op runtime
    # double keeps the container cheap while satisfying the constructor. ``--persist`` calls
    # ``storage.save_run`` directly (NOT ``create_run``), so it never touches the runtime.
    runs = RunAppService(
        runtime=_NullRuntime(),  # type: ignore[arg-type]
        storage=storage,
        entity_registry=spine,
        recommendation_app=recommendations,
    )
    dashboard = DashboardQueryService(storage=storage)
    return AppContainer(
        storage=storage,
        spine=spine,
        runs=runs,
        recommendations=recommendations,
        context=context,
        dashboard=dashboard,
    )


def _durable_store() -> StorageService:
    """Open the durable SQLite store at the canonical path the server also reads.

    Resolves ``HIMMY_STORE_PATH`` (default ``.himmy/storage.db``) exactly like
    :meth:`StoreFactory._sqlite_store`, so the CLI's durable store IS the server's durable
    store. Deliberately NOT :meth:`StoreFactory.for_cli` (which is in-RAM by contract): the
    persisted-history commands need durability, and ``--persist`` is the opt-in that pays
    for it — a plain one-shot ``himmy run`` never reaches here.
    """
    from typing import cast

    from himmy.config.secrets import get_secret
    from himmy.services.storage.factory import HIMMY_STORE_PATH
    from himmy.services.storage.sqlite import SqliteStorageService

    path = get_secret("HIMMY_STORE_PATH") or HIMMY_STORE_PATH
    # SqliteStorageService mirrors the in-memory StorageService API 1:1 (the StoreFactory
    # contract) but is not a subclass; cast so the container's typed surface stays clean.
    return cast("StorageService", SqliteStorageService(path))


class _NullRuntime:
    """A do-nothing runtime stand-in for the CLI container's :class:`RunAppService`.

    The CLI read/write surfaces only use the run service's READERS and the direct
    ``storage.save_run`` write — they never drive ``create_run`` (background execution).
    A real :class:`SingleAgentRuntime` would pull in the whole inference stack for no
    reason, so this placeholder keeps the container construction free of provider wiring.
    Any accidental call to a runtime method fails loudly rather than silently no-op'ing.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - guard rail
        raise RuntimeError(
            "the CLI app-services container has no live runtime; "
            f"call to runtime.{name} is a bug (use storage.save_run / the readers)"
        )
