"""WS4.6 — A2 TRAIN enforcement at the runtime + export/replay boundaries.

Covers the four guarantees from the plan's "Train proof":

* a TRAIN-denied subject's run suppresses the raw-I/O capture block AND strips the
  ``rendered_prompt`` from ``INFERENCE_REQUESTED`` — the prompt/completion appear in **no**
  event payload (storage spine *and* entity registry) and **no** cassette;
* a TRAIN-allowed subject keeps both;
* ``consent_decider=None`` ⇒ behaviour byte-identical to a pre-WS4.6 runtime (offline
  regression — the headline invariant);
* :class:`ConsentFilteredExporter` drops non-consented subjects and keeps consented ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType, RunEvent
from himmy.entities.registry import EntityRegistry
from himmy.runtime import SingleAgentRuntime
from himmy.services.governance.consent import ConsentPolicy, Decision, Effect, Purpose
from himmy.services.governance.consent_ledger import ConsentLedger
from himmy.services.governance.training_export import ConsentFilteredExporter
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)
from himmy.services.inference.replay import RecordingClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.models import RunRecord
from himmy.services.storage.service import StorageService

#: A distinctive marker so we can assert it appears / does not appear in any sink.
_SECRET_PROMPT = "TEACHER-B-SECRET-LESSON-PLAN-12345"


def _governed_runtime(
    *, capture_io: bool, with_decider: bool
) -> tuple[SingleAgentRuntime, StorageService, EntityRegistry, ConsentLedger]:
    """A runtime with raw-I/O capture forced on and an optional governed TRAIN decider."""
    storage = StorageService()
    registry = EntityRegistry()
    inference = InferenceService(StubClientManager(), event_sink=storage)
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    rt = SingleAgentRuntime(
        inference_service=inference,
        memory_store=storage,
        entity_registry=registry,
        capture_io=capture_io,
        consent_decider=ledger.decision if with_decider else None,
    )
    return rt, storage, registry, ledger


async def _events(storage: StorageService) -> list[RunEvent]:
    return list(await storage.list_events())


def _payload_blob(events: list[RunEvent]) -> str:
    """Every event payload flattened to one JSON string for substring assertions."""
    return json.dumps([e.payload for e in events], default=str)


def _of_type(events: list[RunEvent], event_type: EventType) -> list[RunEvent]:
    return [e for e in events if e.event_type == event_type]


def _requested(events: list[RunEvent]) -> RunEvent:
    # Two INFERENCE_REQUESTED events exist per turn: the runtime's rich payload and the
    # InferenceService's bare {model_key}. Select the runtime's by its richer payload
    # (order-independent), so the rendered_prompt assertions can't pass for the wrong reason.
    requested = _of_type(events, EventType.INFERENCE_REQUESTED)
    rich = [e for e in requested if set(e.payload) - {"model_key"}]
    return (rich or requested)[0]


def _io_present(events: list[RunEvent]) -> bool:
    """Whether ANY INFERENCE_SUCCEEDED event carries the raw-I/O capture block.

    Two such events exist per turn (one bare event from ``InferenceService`` and one
    from the runtime that carries the optional ``io`` block), so we scan all of them.
    """
    return any(
        "io" in e.payload for e in _of_type(events, EventType.INFERENCE_SUCCEEDED)
    )


# --------------------------------------------------------------------------- runtime gate


@pytest.mark.asyncio
async def test_train_deny_suppresses_io_and_strips_rendered_prompt() -> None:
    """A subject who denied TRAIN: no io block, no rendered_prompt, nothing leaks."""
    rt, storage, registry, ledger = _governed_runtime(
        capture_io=True, with_decider=True
    )
    ledger.deny("teacher_b", Purpose.TRAIN)
    task = Task(
        title="lesson",
        prompt=_SECRET_PROMPT,
        context={"context_subject_id": "teacher_b"},
    )
    await rt.run_task(Persona(name="Tutor"), task)

    events = await _events(storage)
    requested = _requested(events)

    # rendered_prompt is omitted entirely (not blanked) and io block is absent.
    assert "rendered_prompt" not in requested.payload
    assert not _io_present(events)

    # The secret never appears in ANY event payload — on the storage spine OR the
    # entity-registry projection of the same events.
    assert _SECRET_PROMPT not in _payload_blob(events)
    registry_events = [r for r in registry.list_by_kind("run_event")]
    assert _SECRET_PROMPT not in json.dumps(
        [r.payload for r in registry_events], default=str
    )


@pytest.mark.asyncio
async def test_train_allow_keeps_io_and_rendered_prompt() -> None:
    """A subject who granted TRAIN keeps both the io block and the rendered prompt."""
    rt, storage, _registry, ledger = _governed_runtime(
        capture_io=True, with_decider=True
    )
    ledger.grant("teacher_a", Purpose.TRAIN)
    task = Task(
        title="lesson",
        prompt=_SECRET_PROMPT,
        context={"context_subject_id": "teacher_a"},
    )
    await rt.run_task(Persona(name="Tutor"), task)

    events = await _events(storage)
    requested = _requested(events)
    # rendered_prompt is the rendered task template, so the raw prompt is a substring.
    assert _SECRET_PROMPT in requested.payload["rendered_prompt"]
    assert _io_present(events)
    assert _SECRET_PROMPT in _payload_blob(events)


@pytest.mark.asyncio
async def test_unknown_subject_is_denied_by_default_when_governed() -> None:
    """No consent record ⇒ governed default DENY for TRAIN ⇒ suppression fires."""
    rt, storage, _registry, _ledger = _governed_runtime(
        capture_io=True, with_decider=True
    )
    task = Task(
        title="lesson",
        prompt=_SECRET_PROMPT,
        context={"context_subject_id": "teacher_unknown"},
    )
    await rt.run_task(Persona(name="Tutor"), task)
    events = await _events(storage)
    assert "rendered_prompt" not in _requested(events).payload
    assert _SECRET_PROMPT not in _payload_blob(events)


@pytest.mark.asyncio
async def test_agent_id_is_not_a_subject() -> None:
    """persona.agent_id is NOT a data subject: with no context_subject_id, no gate."""
    rt, storage, _registry, _ledger = _governed_runtime(
        capture_io=True, with_decider=True
    )
    # No context_subject_id at all → the human-subject gate cannot fire even though
    # governed; the run captures as normal (the agent itself is not a subject).
    task = Task(title="lesson", prompt=_SECRET_PROMPT)
    await rt.run_task(Persona(name="Tutor"), task)
    events = await _events(storage)
    assert _SECRET_PROMPT in _requested(events).payload["rendered_prompt"]
    assert _io_present(events)


# ----------------------------------------------------------- offline regression (None path)


@pytest.mark.asyncio
async def test_consent_decider_none_is_identical_to_today() -> None:
    """consent_decider=None ⇒ rendered_prompt + io captured verbatim regardless of ctx.

    This is the headline offline-first invariant: a subject with NO opt-in is irrelevant
    when the runtime is ungoverned (decider is None), so the persistence is byte-identical
    to a pre-WS4.6 runtime.
    """
    rt, storage, _registry, _ledger = _governed_runtime(
        capture_io=True, with_decider=False
    )
    task = Task(
        title="lesson",
        prompt=_SECRET_PROMPT,
        context={"context_subject_id": "teacher_b"},  # would deny if governed
    )
    await rt.run_task(Persona(name="Tutor"), task)
    events = await _events(storage)
    assert _SECRET_PROMPT in _requested(events).payload["rendered_prompt"]
    assert _io_present(events)
    assert _SECRET_PROMPT in _payload_blob(events)


@pytest.mark.asyncio
async def test_capture_off_stays_off_with_decider() -> None:
    """The gate never *adds* capture: capture_io=False keeps the io block absent."""
    rt, storage, _registry, ledger = _governed_runtime(
        capture_io=False, with_decider=True
    )
    ledger.grant("teacher_a", Purpose.TRAIN)  # even fully consented
    task = Task(
        title="lesson",
        prompt=_SECRET_PROMPT,
        context={"context_subject_id": "teacher_a"},
    )
    await rt.run_task(Persona(name="Tutor"), task)
    events = await _events(storage)
    assert not _io_present(events)
    # rendered_prompt is independent of capture_io and stays present (consented).
    assert _SECRET_PROMPT in _requested(events).payload["rendered_prompt"]


# ------------------------------------------------------------------- ConsentFilteredExporter


def _exporter(ledger: ConsentLedger) -> ConsentFilteredExporter:
    return ConsentFilteredExporter(decider=ledger.decision)


def _subject_of(rec: object) -> str | None:
    return getattr(rec, "subject_id", None) or None


def test_exporter_drops_non_consented_keeps_consented() -> None:
    """Only TRAIN-granted subjects survive the export funnel; order preserved."""
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    ledger.grant("alice", Purpose.TRAIN)
    ledger.deny("bob", Purpose.TRAIN)
    # carol has no record → governed default DENY.
    records = [
        RunRecord(workspace_id="w", subject_id="alice", output_text="a"),
        RunRecord(workspace_id="w", subject_id="bob", output_text="b"),
        RunRecord(workspace_id="w", subject_id="carol", output_text="c"),
    ]
    kept = _exporter(ledger).filter(records, subject_of=_subject_of)
    assert [r.subject_id for r in kept] == ["alice"]


def test_exporter_fails_closed_on_unresolved_subject() -> None:
    """A record whose subject can't be resolved is dropped (fail-closed)."""
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    records = [RunRecord(workspace_id="w", subject_id="", output_text="x")]
    assert _exporter(ledger).filter(records, subject_of=_subject_of) == []


def test_exporter_ungoverned_keeps_everything() -> None:
    """An ungoverned decider returns ALLOW for all ⇒ nothing is dropped (offline)."""
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=False))
    records = [
        RunRecord(workspace_id="w", subject_id="alice", output_text="a"),
        RunRecord(workspace_id="w", subject_id="bob", output_text="b"),
    ]
    kept = _exporter(ledger).filter(records, subject_of=_subject_of)
    assert [r.subject_id for r in kept] == ["alice", "bob"]


# -------------------------------------------------------- RecordingClientManager TRAIN gate


def _stub_decider(allow: bool) -> Callable[[str, Purpose], Decision]:
    def decide(subject_id: str, purpose: Purpose) -> Decision:
        return Decision(
            subject_id,
            purpose,
            Effect.ALLOW if allow else Effect.DENY,
        )

    return decide


def _request() -> InferenceRequest:
    return InferenceRequest(
        model_key="default",
        messages=[InferenceMessage(role="user", content=_SECRET_PROMPT)],
    )


class _FixedManager:
    provider_name = "fixed"

    def resolve(self, model_key: str) -> str:
        return f"fixed:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            model_path="fixed:echo",
            provider_name="fixed",
            output_text=_SECRET_PROMPT,
            latency_ms=1.0,
        )


@pytest.mark.asyncio
async def test_recording_refuses_non_train_subject() -> None:
    """A non-TRAIN-consented subject's I/O (incl. request_preview) never hits the cassette."""
    rec = RecordingClientManager(
        _FixedManager(),
        consent_decider=_stub_decider(allow=False),
        subject_id="teacher_b",
    )
    resp = await rec.generate(_request())
    assert resp.output_text == _SECRET_PROMPT  # the live answer is still served
    assert rec.cassette.entries == []  # but nothing recorded
    assert _SECRET_PROMPT not in rec.cassette.model_dump_json()


@pytest.mark.asyncio
async def test_recording_keeps_train_consented_subject() -> None:
    """A TRAIN-consented subject is recorded verbatim (including request_preview)."""
    rec = RecordingClientManager(
        _FixedManager(),
        consent_decider=_stub_decider(allow=True),
        subject_id="teacher_a",
    )
    await rec.generate(_request())
    assert len(rec.cassette.entries) == 1
    assert rec.cassette.entries[0].request_preview == _SECRET_PROMPT


@pytest.mark.asyncio
async def test_recording_decider_none_records_everything() -> None:
    """consent_decider=None ⇒ recording unchanged (offline regression)."""
    rec = RecordingClientManager(_FixedManager())
    await rec.generate(_request())
    assert len(rec.cassette.entries) == 1
    assert rec.cassette.entries[0].request_preview == _SECRET_PROMPT
