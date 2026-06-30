"""T2.2 — one canonical RunRecord, one writer, read by every surface.

These tests pin the coherence contract:

* a run created by ``POST /v1/runs`` is readable by the Studio runs screen reading the
  SAME canonical store (the headline acceptance);
* a Studio run is mirrored into the canonical store, with ``.himmy/studio.db`` holding
  only a run_id-keyed presentation projection — never an authoritative record;
* the new ``RunStatus.AWAITING_APPROVAL`` round-trips through the (CHECK-constraint-free)
  status string and a pre-AWAITING_APPROVAL row still loads (back-compat is the enum
  string, not a migration file);
* the reserved ``__local__`` workspace is EXCLUDED from cross-tenant all-workspaces list
  views (no leak), yet queryable directly;
* the last-write-wins upsert preserves an authoritative (more-advanced) status.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.studio_canonical import (
    record_to_studio_run,
    save_canonical_run,
    studio_run_to_record,
)
from himmy.api.studio_runs import (
    StudioRun,
    reset_run_store,
)
from himmy.services.storage.models import (
    LOCAL_WORKSPACE,
    RunRecord,
    RunStatus,
)
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    reset_run_store()  # studio.db cache is keyed on cwd; start fresh per test
    return TestClient(create_app())


def _run_studio(client: TestClient, body: dict) -> list[dict]:
    frames: list[dict] = []
    with client.stream("POST", "/api/studio/run", json=body) as r:
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    return frames


def _create_v1_run(client: TestClient) -> str:
    body = {
        "workspace_id": "acme",
        "subject_id": "user-1",
        "persona": {"name": "analyst", "instructions": ["Be brief."]},
        "task": {"title": "greet", "prompt": "say hi"},
    }
    res = client.post("/v1/runs", json=body)
    assert res.status_code == 200, res.text
    run_id = res.json()["run_id"]
    # Fire-and-forget background execution — poll the canonical store to terminal.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        got = client.get(f"/v1/runs/{run_id}", params={"workspace_id": "acme"}).json()
        if got["status"] in ("SUCCEEDED", "FAILED", "AWAITING_APPROVAL"):
            break
        time.sleep(0.02)
    return run_id


# --------------------------------------------------------------------- integration


def test_v1_run_is_readable_by_studio_runs_screen(client: TestClient) -> None:
    """The headline acceptance: a /v1 run is browsable in the Studio runs UI."""
    run_id = _create_v1_run(client)

    listing = client.get("/api/studio/runs").json()
    ids = [item["id"] for item in listing["items"]]
    assert run_id in ids, "a /v1 run must appear in the Studio runs list (one store)"

    detail = client.get(f"/api/studio/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == run_id


def test_studio_run_lands_in_canonical_store(client: TestClient) -> None:
    """A Studio run mirrors into the canonical store the /v1 readers use."""
    frames = _run_studio(
        client, {"agent_path": "agent.yaml", "prompt": "hello", "provider": "stub"}
    )
    run_id = frames[-1]["run_id"]

    # The container's canonical storage is the one /v1 + the CLI read.
    storage = client.app.state.container.storage
    import anyio

    async def _read() -> RunRecord | None:
        return await storage.get_run(run_id)

    canonical = anyio.run(_read)
    assert canonical is not None, "Studio run must be mirrored into the canonical store"
    assert canonical.run_id == run_id
    assert canonical.workspace_id == LOCAL_WORKSPACE
    assert canonical.status == RunStatus.SUCCEEDED
    # The rich Studio projection rides in metadata['studio'] — studio.db is just a cache.
    assert "studio" in canonical.metadata
    assert canonical.metadata["studio"]["agent_name"] == "helper"


# ----------------------------------------------------------------- back-compat enum


@pytest.mark.asyncio
async def test_awaiting_approval_status_round_trips(tmp_path: Path) -> None:
    """The new status string persists + reloads (no CHECK-constraint break)."""
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    rec = RunRecord(
        workspace_id="acme",
        subject_id="s1",
        status=RunStatus.AWAITING_APPROVAL,
        metadata={"checkpoint_id": "cp-123"},
    )
    await store.save_run(rec)

    got = await store.get_run(rec.run_id)
    assert got is not None
    assert got.status is RunStatus.AWAITING_APPROVAL
    # And it is filterable by the new status without erroring.
    awaiting = await store.list_runs(status=RunStatus.AWAITING_APPROVAL)
    assert [r.run_id for r in awaiting] == [rec.run_id]


@pytest.mark.asyncio
async def test_pre_awaiting_approval_row_still_loads(tmp_path: Path) -> None:
    """A runs row written before AWAITING_APPROVAL existed still round-trips.

    Simulates an existing DB by writing a legacy-status run, then re-opening the file:
    the back-compat concern is the enum STRING, not a migration — an older row must load
    untouched and a new-status row must coexist.
    """
    db = str(tmp_path / "storage.db")
    store = SqliteStorageService(db)
    legacy = RunRecord(workspace_id="w", subject_id="s", status=RunStatus.SUCCEEDED)
    await store.save_run(legacy)

    # Re-open the file (a fresh handle, like a process restart) and read both shapes.
    reopened = SqliteStorageService(db)
    new = RunRecord(workspace_id="w", subject_id="s", status=RunStatus.AWAITING_APPROVAL)
    await reopened.save_run(new)

    got_legacy = await reopened.get_run(legacy.run_id)
    got_new = await reopened.get_run(new.run_id)
    assert got_legacy is not None and got_legacy.status is RunStatus.SUCCEEDED
    assert got_new is not None and got_new.status is RunStatus.AWAITING_APPROVAL


# ------------------------------------------------------------- cross-tenant no-leak


@pytest.mark.asyncio
async def test_local_runs_excluded_from_all_tenants_view() -> None:
    """``__local__`` runs are hidden from the all-workspaces list but queryable directly."""
    from himmy.application.services import RunAppService

    storage = StorageService()
    # A tenant run + a single-user-local run, both in the one store.
    await storage.save_run(
        RunRecord(workspace_id="acme", subject_id="u", status=RunStatus.SUCCEEDED)
    )
    local = RunRecord(
        workspace_id=LOCAL_WORKSPACE,
        subject_id="__local__",
        status=RunStatus.SUCCEEDED,
    )
    await storage.save_run(local)

    svc = RunAppService(runtime=None, storage=storage)  # type: ignore[arg-type]

    # All-workspaces view (admin listing everything) must NOT leak __local__.
    everything = await svc.list_runs(workspace_id=None)
    ids = {r.run_id for r in everything}
    assert local.run_id not in ids
    assert await svc.count_runs(workspace_id=None) == 1

    # A concrete query for __local__ still returns it (the local Studio/CLI browse).
    locals_ = await svc.list_runs(workspace_id=LOCAL_WORKSPACE)
    assert [r.run_id for r in locals_] == [local.run_id]


# ----------------------------------------------------------------- last-write-wins


@pytest.mark.asyncio
async def test_save_canonical_run_preserves_authoritative_status() -> None:
    """A stale, LOWER-status projection cannot roll back an advanced/paused status.

    The authoritative ``/v1`` writer advanced a run to ``AWAITING_APPROVAL``; a late
    re-projection of the SAME run carrying a less-advanced status (a re-render that lost
    the pause) must NOT regress the row — the merge keeps the more-advanced status and
    preserves the authoritative metadata.
    """
    storage = StorageService()
    run_id = "run-xyz"
    await storage.save_run(
        RunRecord(
            run_id=run_id,
            workspace_id="acme",
            subject_id="u",
            status=RunStatus.AWAITING_APPROVAL,
            metadata={"checkpoint_id": "cp-1"},
        )
    )
    # A stale projection of the same run that lost the pause (renders as still running).
    stale = RunRecord(
        run_id=run_id,
        workspace_id=LOCAL_WORKSPACE,
        subject_id="__local__",
        status=RunStatus.RUNNING,
        metadata={"studio": {"agent_name": "helper"}},
    )
    merged = await save_canonical_run(storage, stale)
    assert merged.status is RunStatus.AWAITING_APPROVAL  # not rolled back
    # The authoritative metadata is preserved through the merge.
    assert merged.metadata.get("checkpoint_id") == "cp-1"
    # The presentation projection IS refreshed (merged in, not lost).
    assert merged.metadata.get("studio", {}).get("agent_name") == "helper"


@pytest.mark.asyncio
async def test_save_canonical_run_advances_status() -> None:
    """A real completion DOES overwrite an earlier non-terminal status."""
    storage = StorageService()
    run_id = "run-abc"
    await storage.save_run(
        RunRecord(
            run_id=run_id,
            workspace_id="acme",
            subject_id="u",
            status=RunStatus.RUNNING,
        )
    )
    done = StudioRun(id=run_id, created_at="2020-01-01T00:00:00Z", status="ok")
    merged = await save_canonical_run(storage, studio_run_to_record(done))
    assert merged.status is RunStatus.SUCCEEDED


# ------------------------------------------------------------------- projections


def test_record_to_studio_run_renders_v1_only_record() -> None:
    """A canonical RunRecord with no studio.db cache still renders a StudioRun."""
    rec = RunRecord(
        run_id="r1",
        workspace_id="acme",
        subject_id="u",
        status=RunStatus.AWAITING_APPROVAL,
        output_text="partial output",
        metadata={"checkpoint_id": "cp-9"},
    )
    view = record_to_studio_run(rec)
    assert view.id == "r1"
    assert view.status == "awaiting_approval"
    assert view.output == "partial output"
    assert view.checkpoint_id == "cp-9"


@pytest.mark.asyncio
async def test_resolving_run_projects_without_keyerror() -> None:
    """A RESOLVING run renders through the Studio readers without crashing the screen.

    Regression: ``_RUN_STATUS_TO_STUDIO`` is keyed by the exhaustive ``RunStatus`` enum
    and subscripted UNGUARDED from the Studio list + detail readers (which scan every
    canonical run with no tenant scope). The HITL resume CAS makes ``RESOLVING`` a real,
    durable, observable run state (during every resume and for crash-stranded runs), so a
    missing key would KeyError-crash the Studio runs list/detail whenever any run is
    RESOLVING. RESOLVING must project to a non-terminal Studio status ("ok") like RUNNING.
    """
    from himmy.api.studio_canonical import (
        get_studio_run_unified,
        list_studio_runs_unified,
    )

    rec = RunRecord(
        run_id="resolving-1",
        workspace_id="acme",
        subject_id="u",
        status=RunStatus.RESOLVING,
        output_text="mid-resume",
    )

    # 1) Direct projection (the synthesize branch, no studio.db cache row).
    view = record_to_studio_run(rec)
    assert view.id == "resolving-1"
    assert view.status == "ok"  # non-terminal, shown as in-progress like RUNNING

    # 2) The list + detail readers scan EVERY canonical run with no tenant scope.
    storage = StorageService()
    await storage.save_run(rec)
    summaries, total = await list_studio_runs_unified(storage, limit=50, offset=0)
    assert total == 1
    assert summaries[0].id == "resolving-1"
    assert summaries[0].status == "ok"

    detail = await get_studio_run_unified(storage, "resolving-1")
    assert detail is not None and detail.status == "ok"


@pytest.mark.asyncio
async def test_cache_only_run_not_served_to_tenant_bound_reader(tmp_path: Path) -> None:
    """Red-team r3: a cache-only Studio run is NOT served to a tenant-bound reader.

    The studio.db presentation cache carries no tenant stamp. When a canonical record is
    absent (a cache row written before / outliving its canonical mirror) the by-id reader
    must NOT fall back to the cache for a tenant-bound principal — that would bypass the
    workspace allow-list and leak an unattributable run cross-tenant. An unscoped
    (single-box) reader still gets the cache row.
    """
    import os

    from himmy.api.studio_canonical import get_studio_run_unified
    from himmy.api.studio_runs import StudioRun, get_run_store, reset_run_store

    prev_cwd = os.getcwd()
    os.chdir(tmp_path)
    reset_run_store()
    try:
        # Seed a cache-only row (no canonical record exists for it).
        get_run_store().save(
            StudioRun(id="cache-only-1", created_at="2026-01-01T00:00:00Z", output="X")
        )
        storage = StorageService()  # canonical store has NO record for cache-only-1

        # Tenant-bound reader (a non-None allow-list) must read it as not-found.
        scoped = await get_studio_run_unified(
            storage, "cache-only-1", accessible_workspaces=frozenset({"acme"})
        )
        assert scoped is None

        # The unscoped (single-box) reader still gets the cache row (byte-unchanged).
        unscoped = await get_studio_run_unified(
            storage, "cache-only-1", accessible_workspaces=None
        )
        assert unscoped is not None and unscoped.id == "cache-only-1"
    finally:
        reset_run_store()
        os.chdir(prev_cwd)


# --------------------------------------------- subject axis (BOLA, scope-r1#3)


@pytest.mark.asyncio
async def test_studio_run_readers_enforce_subject_axis() -> None:
    """The Studio run list + by-id readers narrow on the SUBJECT axis (not just tenant).

    Vuln scope-r1#3: the Studio ``/api/studio/runs`` family scoped ONLY by tenant
    (``accessible_workspaces``) and never by subject, so a ``subject_scoped`` principal
    blocked from another subject's runs on ``/v1/runs`` and missions could read the SAME runs
    via Studio. The unified readers now also accept ``accessible_subject`` (from
    ``studio_subject_filter``): pinned to a subject, they surface only that subject's runs
    (plus the reserved ``__local__`` single-user runs) and fold a cross-subject by-id fetch to
    not-found. ``None`` (offline / all_tenants / tenant_admin) is byte-unchanged.
    """
    from himmy.api.studio_canonical import (
        get_studio_run_unified,
        list_studio_runs_unified,
    )

    storage = StorageService()
    # Two subjects in the SAME tenant 't', plus one __local__ single-user run.
    await storage.save_run(
        RunRecord(
            run_id="alice-run",
            workspace_id="t",
            subject_id="alice",
            status=RunStatus.SUCCEEDED,
        )
    )
    await storage.save_run(
        RunRecord(
            run_id="bob-run",
            workspace_id="t",
            subject_id="bob",
            status=RunStatus.SUCCEEDED,
        )
    )
    await storage.save_run(
        RunRecord(
            run_id="local-run",
            workspace_id=LOCAL_WORKSPACE,
            subject_id="__local__",
            status=RunStatus.SUCCEEDED,
        )
    )

    # alice (subject_scoped) tenant 't' → sees ONLY its own + the __local__ run, never bob's.
    summaries, total = await list_studio_runs_unified(
        storage,
        limit=50,
        offset=0,
        accessible_workspaces=frozenset({"t", LOCAL_WORKSPACE}),
        accessible_subject="alice",
    )
    ids = {s.id for s in summaries}
    assert ids == {"alice-run", "local-run"}
    assert total == 2  # bob's run never enters the visible slice/count

    # By-id: alice's own + __local__ are visible; bob's folds to not-found.
    assert (
        await get_studio_run_unified(
            storage, "alice-run", accessible_subject="alice"
        )
    ) is not None
    assert (
        await get_studio_run_unified(
            storage, "local-run", accessible_subject="alice"
        )
    ) is not None
    assert (
        await get_studio_run_unified(
            storage, "bob-run", accessible_subject="alice"
        )
    ) is None

    # INVARIANT: no subject filter (offline / all_tenants / tenant_admin) → everything.
    all_summaries, all_total = await list_studio_runs_unified(
        storage, limit=50, offset=0, accessible_subject=None
    )
    assert {s.id for s in all_summaries} == {"alice-run", "bob-run", "local-run"}
    assert all_total == 3
    assert (
        await get_studio_run_unified(storage, "bob-run", accessible_subject=None)
    ) is not None
