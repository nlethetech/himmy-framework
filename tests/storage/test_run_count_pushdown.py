"""T2-runs-pagination: ``count_runs`` is a real ``SELECT COUNT(*)`` push-down.

The pagination envelope's total must be computed without loading + deserializing the
whole runs table, AND it must keep byte-for-byte the same filter semantics as
:meth:`RunAppService.list_runs` — including the reserved ``__local__`` exclusion on the
cross-tenant all-workspaces view — across every backend (SQLite native counter and the
in-memory facade fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.application.services import RunAppService
from himmy.services.storage.inmemory import InMemoryRunStore
from himmy.services.storage.models import LOCAL_WORKSPACE, RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _seed(store: object) -> None:
    """Two tenants (mixed statuses) + one reserved ``__local__`` single-user run."""
    runs = [
        RunRecord(workspace_id="acme", subject_id="u1", status=RunStatus.SUCCEEDED),
        RunRecord(workspace_id="acme", subject_id="u1", status=RunStatus.FAILED),
        RunRecord(workspace_id="acme", subject_id="u2", status=RunStatus.SUCCEEDED),
        RunRecord(workspace_id="beta", subject_id="u3", status=RunStatus.SUCCEEDED),
        RunRecord(
            workspace_id=LOCAL_WORKSPACE,
            subject_id="__local__",
            status=RunStatus.SUCCEEDED,
        ),
    ]
    for r in runs:
        run_async(store.save_run(r))  # type: ignore[attr-defined]


def test_sqlite_count_runs_matches_list_runs(tmp_path: Path) -> None:
    """Every filter combo: SQL COUNT(*) == len(list_runs) on the same predicate."""
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    _seed(store)

    async def check() -> None:
        combos = [
            {},
            {"workspace_id": "acme"},
            {"workspace_id": "beta"},
            {"workspace_id": LOCAL_WORKSPACE},
            {"subject_id": "u1"},
            {"status": RunStatus.SUCCEEDED},
            {"workspace_id": "acme", "status": RunStatus.SUCCEEDED},
            {"workspace_id": "acme", "subject_id": "u1", "status": RunStatus.FAILED},
        ]
        for kw in combos:
            listed = await store.list_runs(**kw)  # type: ignore[arg-type]
            counted = await store.count_runs(**kw)  # type: ignore[arg-type]
            assert counted == len(listed), kw

    run_async(check())


def test_sqlite_count_runs_exclude_local(tmp_path: Path) -> None:
    """``exclude_local_workspace`` drops the reserved ``__local__`` runs (only when
    the workspace is unscoped) and is a no-op for a concrete workspace query."""
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    _seed(store)

    async def check() -> None:
        # All five rows exist; four are non-local.
        assert await store.count_runs() == 5
        assert await store.count_runs(exclude_local_workspace=True) == 4
        # A concrete workspace query ignores the exclude flag (already pinned).
        assert (
            await store.count_runs(
                workspace_id=LOCAL_WORKSPACE, exclude_local_workspace=True
            )
            == 1
        )

    run_async(check())


def test_inmemory_run_store_count_runs() -> None:
    """The in-memory RunStore counter mirrors its own filtered list_runs."""
    store = InMemoryRunStore()
    _seed(store)

    async def check() -> None:
        assert await store.count_runs() == 5
        assert await store.count_runs(exclude_local_workspace=True) == 4
        assert await store.count_runs(workspace_id="acme") == 3
        assert await store.count_runs(subject_id="u1") == 2
        assert await store.count_runs(status=RunStatus.SUCCEEDED) == 4

    run_async(check())


@pytest.mark.parametrize("backend", ["sqlite", "inmemory_facade"])
def test_run_app_count_runs_all_backends_preserve_local_exclusion(
    tmp_path: Path, backend: str
) -> None:
    """``RunAppService.count_runs`` keeps the same __local__ exclusion whether the
    store exposes a native counter (SQLite) or falls back (in-memory facade)."""
    if backend == "sqlite":
        storage: object = SqliteStorageService(str(tmp_path / "storage.db"))
    else:
        # The in-memory StorageService facade has NO native count_runs -> the service
        # takes the load+filter fallback; the count must still match.
        storage = StorageService()
    _seed(storage)

    svc = RunAppService(runtime=None, storage=storage)  # type: ignore[arg-type]

    async def check() -> None:
        # All-workspaces admin view: __local__ hidden from both list and count.
        everything = await svc.list_runs(workspace_id=None)
        assert LOCAL_WORKSPACE not in {r.workspace_id for r in everything}
        assert await svc.count_runs(workspace_id=None) == 4
        # A concrete __local__ query still counts it.
        assert await svc.count_runs(workspace_id=LOCAL_WORKSPACE) == 1
        # Scoped + status counts match the corresponding page length.
        for kw in (
            {"workspace_id": "acme"},
            {"workspace_id": "acme", "status": RunStatus.SUCCEEDED},
            {"subject_id": "u1"},
        ):
            page = await svc.list_runs(**kw)  # type: ignore[arg-type]
            assert await svc.count_runs(**kw) == len(page), kw  # type: ignore[arg-type]

    run_async(check())
