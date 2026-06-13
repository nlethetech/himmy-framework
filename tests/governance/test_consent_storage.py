"""WS4.6 — ConsentGatedStorage: RETAIN gating at the persistence boundary."""

from __future__ import annotations

import pytest

from himmy.entities.registry import EntityRegistry
from himmy.services.audit.log import SecurityAuditLog
from himmy.services.governance.consent import ConsentPolicy, Purpose
from himmy.services.governance.consent_ledger import ConsentLedger
from himmy.services.governance.consent_storage import (
    GATED_SAVE_METHODS,
    ConsentGatedStorage,
)
from himmy.services.storage.models import MemoryObject, RunRecord
from himmy.services.storage.service import StorageService


def _governed_store() -> tuple[
    ConsentGatedStorage, StorageService, ConsentLedger, SecurityAuditLog
]:
    inner = StorageService()
    registry = EntityRegistry()
    audit = SecurityAuditLog(registry)
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    gated = ConsentGatedStorage(inner, decider=ledger.decision, audit=audit)
    return gated, inner, ledger, audit


def _run(subject_id: str) -> RunRecord:
    return RunRecord(workspace_id="w", subject_id=subject_id, output_text="hello")


@pytest.mark.asyncio
async def test_consented_subject_persists() -> None:
    gated, inner, ledger, _ = _governed_store()
    ledger.grant("alice", Purpose.RETAIN)
    run = _run("alice")
    await gated.save_run(run)
    assert await inner.get_run(run.run_id) is not None


@pytest.mark.asyncio
async def test_unconsented_subject_is_ephemeral() -> None:
    gated, inner, _, audit = _governed_store()
    run = _run("bob")  # no consent record → governed default DENY for RETAIN
    returned = await gated.save_run(run)
    assert returned is run  # caller still gets the record for the live request
    assert await inner.get_run(run.run_id) is None  # but nothing was written
    assert await inner.list_runs(subject_id="bob") == []
    events = audit.recent(event_type="consent_denied_persist")
    assert events and events[0].actor.get("subject") == "bob"


@pytest.mark.asyncio
async def test_idempotent_create_is_gated() -> None:
    gated, inner, _, _ = _governed_store()
    run = _run("carol")
    returned, _created = await gated.save_run_if_absent_by_idempotency(run)
    assert returned is run
    assert await inner.get_run(run.run_id) is None


@pytest.mark.asyncio
async def test_memory_without_subject_fails_closed() -> None:
    gated, inner, _, audit = _governed_store()
    obj = MemoryObject(payload={"text": "a fact"})  # subject_id is None
    await gated.save_memory(obj)
    assert await inner.get_memory(obj.memory_id) is None
    assert audit.recent(event_type="consent_denied_persist")


@pytest.mark.asyncio
async def test_non_subject_bearing_writes_delegate() -> None:
    # save_context_field is not gated (ContextField has no first-class subject_id);
    # it must still round-trip through delegation.
    gated, inner, _, _ = _governed_store()
    from himmy.services.context.models import ContextField

    field = ContextField(key="locale", value="ne-NP")
    saved = await gated.save_context_field(field)  # delegated via __getattr__
    assert saved.key == "locale"


def test_gated_methods_cover_every_subject_bearing_sink() -> None:
    """Sink-drift guard: a new StorageService save_* must be classified, not silent."""
    save_methods = {m for m in dir(StorageService) if m.startswith("save_")}
    # Saves that are intentionally NOT subject-gated (no first-class subject /
    # infrastructure / orchestration world-model state). append_event is gated in A2.
    known_non_subject = {
        "save_thread",
        "save_context_field",
        "save_context_evidence",
        "save_evaluation_run",
        "save_agent_state",
        "save_action",
        "save_environment_state",
        # T2e: a stored AgentSpec is workspace-scoped tenant config with no
        # first-class subject_id — not a subject-bearing PII sink.
        "save_agent_def",
        "save_agent_def_if_absent",
    }
    assert save_methods == GATED_SAVE_METHODS | known_non_subject
