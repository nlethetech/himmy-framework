"""Expanded hardening tests for the storage kernel idempotency + defaults.

Complements ``tests/storage/test_storage_service.py`` by pinning the production
behaviour the kernel audit hardened (SE-2/11/12):

- SE-2  ``save_run_if_absent_by_idempotency`` is atomic (no ``await`` between the
        read and the write), so ``asyncio.gather`` of N concurrent same-key
        creates produces EXACTLY ONE run, while distinct keys / distinct
        workspaces each create their own; null-key runs are always created.
- SE-11 storage owns ``updated_at`` — every ``save_run`` write stamps a fresh
        value so it can never drift, and the idempotency index is a stable
        pointer (``setdefault``) that the first writer wins.
- SE-12 mutable model defaults (RunRecord/RecommendationItem/MemoryObject/…) are
        per-instance via ``default_factory``, not shared class literals.

Tests are plain ``def test_*`` functions driving async via ``run_async``
(``asyncio.run``). Entirely offline — no Postgres, no asyncpg, no network.
"""

from __future__ import annotations

import asyncio

import pytest

from opensims.services.storage.models import (
    ActionRecord,
    AgentStateRecord,
    ContextEvidenceRecord,
    EnvironmentStateRecord,
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RunRecord,
)
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


# --------------------------------------------------------------- local fixture
@pytest.fixture()
def storage() -> StorageService:
    """A fresh in-memory StorageService (area fixture, defined locally)."""
    return StorageService()


# ----------------------------------------- SE-2: atomic idempotent creation
def test_gather_many_same_key_creates_exactly_one(storage: StorageService) -> None:
    """asyncio.gather of N same-key creates collapses to exactly one run (SE-2)."""

    async def scenario() -> None:
        runs = [
            RunRecord(workspace_id="w1", subject_id="s", idempotency_key="K")
            for _ in range(25)
        ]
        results = await asyncio.gather(
            *(storage.save_run_if_absent_by_idempotency(r) for r in runs)
        )
        created = sum(1 for _, was_created in results if was_created)
        run_ids = {stored.run_id for stored, _ in results}
        assert created == 1
        assert len(run_ids) == 1
        assert len(await storage.list_runs(workspace_id="w1")) == 1

    run_async(scenario())


def test_gather_distinct_keys_create_distinct_runs(storage: StorageService) -> None:
    """Distinct idempotency keys each create their own run (no over-collapse) (SE-2)."""

    async def scenario() -> None:
        runs = [
            RunRecord(workspace_id="w1", subject_id="s", idempotency_key=f"k{i}")
            for i in range(5)
        ]
        results = await asyncio.gather(
            *(storage.save_run_if_absent_by_idempotency(r) for r in runs)
        )
        assert sum(1 for _, c in results if c) == 5
        assert len(await storage.list_runs(workspace_id="w1")) == 5

    run_async(scenario())


def test_same_key_different_workspace_are_isolated(storage: StorageService) -> None:
    """The idempotency scope is (workspace_id, key); same key in two ws is two runs (SE-2)."""

    async def scenario() -> None:
        a = RunRecord(workspace_id="w1", subject_id="s", idempotency_key="K")
        b = RunRecord(workspace_id="w2", subject_id="s", idempotency_key="K")
        ra, ca = await storage.save_run_if_absent_by_idempotency(a)
        rb, cb = await storage.save_run_if_absent_by_idempotency(b)
        assert ca is True and cb is True
        assert ra.run_id != rb.run_id
        # Each workspace resolves its own run.
        w1 = await storage.load_run_by_idempotency("w1", "K")
        w2 = await storage.load_run_by_idempotency("w2", "K")
        assert w1 is not None and w1.run_id == a.run_id
        assert w2 is not None and w2.run_id == b.run_id

    run_async(scenario())


def test_null_key_runs_always_created(storage: StorageService) -> None:
    """Runs without an idempotency key are never deduplicated (SE-2)."""

    async def scenario() -> None:
        runs = [RunRecord(workspace_id="w1", subject_id="s") for _ in range(4)]
        results = await asyncio.gather(
            *(storage.save_run_if_absent_by_idempotency(r) for r in runs)
        )
        assert all(c for _, c in results)
        assert len(await storage.list_runs(workspace_id="w1")) == 4

    run_async(scenario())


def test_idempotency_index_first_writer_wins(storage: StorageService) -> None:
    """A later same-key write does not stomp the index pointer (setdefault) (SE-2/11)."""

    async def scenario() -> None:
        first = RunRecord(workspace_id="w", subject_id="s", idempotency_key="dup")
        await storage.save_run_if_absent_by_idempotency(first)
        # A plain save_run of a *different* run object with the same key must not
        # repoint the index away from the first writer.
        second = RunRecord(workspace_id="w", subject_id="s", idempotency_key="dup")
        await storage.save_run(second)
        resolved = await storage.load_run_by_idempotency("w", "dup")
        assert resolved is not None and resolved.run_id == first.run_id

    run_async(scenario())


# -------------------------------------------------- SE-11: updated_at stamping
def test_save_run_always_refreshes_updated_at(storage: StorageService) -> None:
    """save_run owns updated_at and rewrites it on every write (SE-11)."""

    async def scenario() -> None:
        run = RunRecord(workspace_id="w", subject_id="s", updated_at="STALE")
        await storage.save_run(run)
        fetched = await storage.get_run(run.run_id)
        assert fetched is not None and fetched.updated_at != "STALE"

    run_async(scenario())


def test_atomic_create_stamps_updated_at(storage: StorageService) -> None:
    """The atomic create path also stamps updated_at on the freshly created run (SE-11)."""

    async def scenario() -> None:
        run = RunRecord(
            workspace_id="w", subject_id="s", idempotency_key="k", updated_at="STALE"
        )
        stored, created = await storage.save_run_if_absent_by_idempotency(run)
        assert created is True
        assert stored.updated_at != "STALE"

    run_async(scenario())


# ---------------------------------------------- SE-12: per-instance defaults
def test_run_record_mutable_default_metadata_is_per_instance() -> None:
    """RunRecord.metadata is a per-instance dict (default_factory), not shared (SE-12)."""
    a = RunRecord(workspace_id="w", subject_id="s")
    b = RunRecord(workspace_id="w", subject_id="s")
    assert a.metadata is not b.metadata
    a.metadata["x"] = 1
    assert b.metadata == {}


def test_recommendation_item_mutable_defaults_are_per_instance() -> None:
    """RecommendationItem.evidence_refs/metadata are per-instance (SE-12)."""
    a = RecommendationItem(
        run_id="r", workspace_id="w", subject_id="s", kind="k", title="t"
    )
    b = RecommendationItem(
        run_id="r", workspace_id="w", subject_id="s", kind="k", title="t"
    )
    assert a.evidence_refs is not b.evidence_refs
    assert a.metadata is not b.metadata
    a.evidence_refs.append("e1")
    assert b.evidence_refs == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MemoryObject(),
        lambda: EpisodicMemoryObject(),
        lambda: AgentStateRecord(),
        lambda: ActionRecord(),
        lambda: EnvironmentStateRecord(),
        lambda: ContextEvidenceRecord(),
    ],
)
def test_record_payload_metadata_defaults_are_per_instance(factory) -> None:  # type: ignore[no-untyped-def]
    """Every record type's payload/metadata defaults are per-instance (SE-12)."""
    a = factory()
    b = factory()
    assert a.payload is not b.payload
    assert a.metadata is not b.metadata
    a.payload["k"] = 1
    assert b.payload == {}
