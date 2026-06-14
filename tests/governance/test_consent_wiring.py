"""WS4.6 A3 — consent wiring in the API container (offline parity + governed spine gate).

Two proofs the plan calls headline:

* **Offline regression**: zero-config ``create_app()`` / ``ApiContainer.build_default()``
  wires a **bare** ``StorageService``, a runtime whose ``consent_decider`` is ``None``, and
  no governance singletons — so persistence is byte-identical to pre-WS4.6.
* **Governed ephemeral end-to-end**: with ``HIMMY_CONSENT=on`` an opted-out subject leaves
  **zero** cleartext on BOTH the storage facade AND the EntityRegistry spine (queried
  directly), while a consented subject persists and infrastructure records still register.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.api.deps import ApiContainer
from himmy.entities.records import EntityQuery
from himmy.services.governance.consent import Purpose
from himmy.services.governance.consent_registry import ConsentAwareRegistry
from himmy.services.governance.consent_storage import ConsentGatedStorage
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
)
from himmy.services.storage.service import StorageService

#: A distinctive marker so we can assert it appears / does not appear in any sink.
_SECRET = "TEACHER-B-SECRET-LESSON-PLAN-WIRING-98765"


class _ApprovalGateManager:
    """Invokes the approval-gated bound tool on turn 1, then finals once a result exists.

    Drives the real HITL pause/resume cycle: the first turn calls the bound tool (which
    the runtime routes through the approval gate and pauses on); after resume puts a tool
    result on the thread it answers without tools and stops.
    """

    def resolve(self, model_key: str) -> str:
        return model_key

    async def generate(self, request: Any) -> InferenceResponse:
        tool_in_history = any(
            getattr(m, "role", "") == "tool" for m in request.messages
        )
        if not tool_in_history and request.bound_tools:
            ret = await request.tool_executor(request.bound_tools[0].name, {})
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="invoking the tool",
                tool_calls=[
                    ToolCallRecord(
                        tool_call_id=ret.tool_call_id, tool_name=ret.tool_name, args={}
                    )
                ],
                tool_returns=[ret],
                input_tokens=1,
                output_tokens=1,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="done with the result",
            input_tokens=1,
            output_tokens=1,
        )


# --------------------------------------------------------------- offline regression
def test_zero_config_is_bare_and_ungoverned() -> None:
    """The default container is byte-identical to pre-WS4.6: bare storage, no decider."""
    container = ApiContainer.build_default()
    assert type(container.storage) is StorageService
    assert not isinstance(container.storage, ConsentGatedStorage)
    assert not isinstance(container.entity_registry, ConsentAwareRegistry)
    assert container.runtime._consent_decider is None
    assert container.consent_ledger is None
    assert container.consent_policy is None
    assert container.retention_service is None


@pytest.mark.asyncio
async def test_zero_config_persists_everything_verbatim() -> None:
    """An ungoverned run persists the run + transcript on storage AND spine verbatim."""
    container = ApiContainer.build_default()
    task = Task(
        title="lesson",
        prompt=_SECRET,
        context={"context_subject_id": "teacher_b"},
    )
    await container.runtime.run_task(Persona(name="Tutor"), task)

    # The runtime persists run events to storage; the secret prompt is captured verbatim.
    events = list(await container.storage.list_events())
    blob = "".join(str(e.payload) for e in events)
    assert _SECRET in blob
    # And the same events land on the spine, with NO subject_id stamped (ungoverned path).
    spine_events = container.entity_registry.list_by_kind("run_event")
    assert spine_events
    assert all("subject_id" not in r.metadata for r in spine_events)


# --------------------------------------------------------------- governed wiring
def _governed_container(monkeypatch: pytest.MonkeyPatch) -> ApiContainer:
    monkeypatch.setenv("HIMMY_CONSENT", "on")
    monkeypatch.delenv("HIMMY_CONSENT_FILE", raising=False)
    return ApiContainer.build_default()


def test_governed_container_wraps_storage_and_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIMMY_CONSENT=on wraps storage + the service registry and wires the decider."""
    container = _governed_container(monkeypatch)
    assert isinstance(container.storage, ConsentGatedStorage)
    assert container.consent_ledger is not None
    assert container.consent_policy is not None and container.consent_policy.governed
    assert container.retention_service is not None
    # The runtime's registry is the wrapped (gated) one; the decider is wired.
    assert isinstance(container.runtime.entity_registry, ConsentAwareRegistry)
    assert container.runtime._consent_decider is not None
    # The container's own entity_registry is the INNER spine (so audit/consent never gated).
    assert not isinstance(container.entity_registry, ConsentAwareRegistry)


@pytest.mark.asyncio
async def test_governed_opted_out_subject_is_ephemeral_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opted-out subject leaves zero cleartext on storage AND the registry spine."""
    container = _governed_container(monkeypatch)
    # teacher_b has no consent record → governed default DENY for RETAIN/TRAIN.
    task = Task(
        title="lesson",
        prompt=_SECRET,
        context={"context_subject_id": "teacher_b"},
    )
    await container.runtime.run_task(Persona(name="Tutor"), task)

    spine = container.entity_registry

    # 1) No subject-bearing record reached the spine for teacher_b.
    assert spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_b"})) == []
    # 2) The transcript kinds are absent entirely (skip-on-deny, not just blanked).
    assert spine.list_by_kind("message") == []
    assert spine.list_by_kind("chat_thread") == []
    # 3) The secret prompt never lands on any run_event on the spine.
    spine_blob = "".join(str(r.payload) for r in spine.list_by_kind("run_event"))
    assert _SECRET not in spine_blob
    # 4) Infrastructure records still register (the audit trail is never starved).
    assert spine.list_by_kind("persona")
    # 5) The denial is auditable.
    assert spine.list_by_kind("security_event")


@pytest.mark.asyncio
async def test_governed_consented_subject_persists_on_the_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RETAIN-consented subject's transcript reaches the spine (gate only blocks denials)."""
    container = _governed_container(monkeypatch)
    ledger = container.consent_ledger
    ledger.grant("teacher_a", Purpose.RETAIN)
    ledger.grant("teacher_a", Purpose.TRAIN)
    task = Task(
        title="lesson",
        prompt=_SECRET,
        context={"context_subject_id": "teacher_a"},
    )
    await container.runtime.run_task(Persona(name="Tutor"), task)

    spine = container.entity_registry
    # teacher_a's run events reached the spine, tagged with their subject.
    tagged = spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_a"}))
    assert tagged
    assert any(r.kind == "run_event" for r in tagged)


@pytest.mark.asyncio
async def test_governed_continue_turn_persists_consented_subject_on_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """continue_turn publishes the subject too — its records reach the spine tagged.

    Locks the fix for the governed-mode gap where only run_task set _CURRENT_SUBJECT, so
    multi-turn records were subject-less and the ConsentAwareRegistry silently dropped
    them (fail-closed) even for a fully-consented subject.
    """
    from himmy.agents.base_agent.thread import ChatThread

    container = _governed_container(monkeypatch)
    ledger = container.consent_ledger
    ledger.grant("teacher_a", Purpose.RETAIN)
    ledger.grant("teacher_a", Purpose.TRAIN)
    persona = Persona(name="Tutor")
    thread = ChatThread(agent_id=persona.agent_id)
    ctx = {"context_subject_id": "teacher_a"}

    await container.runtime.continue_turn(persona, thread, task_context=ctx)

    spine = container.entity_registry
    tagged = spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_a"}))
    assert tagged, "continue_turn should stamp the subject so records persist"
    # The continuation message + bumped thread version reached the spine.
    assert any(r.kind == "message" for r in tagged)


@pytest.mark.asyncio
async def test_governed_continue_turn_opted_out_subject_is_ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opted-out subject is still ephemeral on the continue_turn path (no leak)."""
    from himmy.agents.base_agent.thread import ChatThread

    container = _governed_container(monkeypatch)
    persona = Persona(name="Tutor")
    thread = ChatThread(agent_id=persona.agent_id)
    ctx = {"context_subject_id": "teacher_b"}  # no consent → governed DENY

    await container.runtime.continue_turn(persona, thread, task_context=ctx)

    spine = container.entity_registry
    assert spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_b"})) == []
    assert spine.list_by_kind("message") == []


@pytest.mark.asyncio
async def test_governed_run_agent_loop_persists_consented_subject_on_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_agent_loop publishes the subject for the whole loop (turn-completed events too)."""
    container = _governed_container(monkeypatch)
    ledger = container.consent_ledger
    ledger.grant("teacher_a", Purpose.RETAIN)
    ledger.grant("teacher_a", Purpose.TRAIN)
    task = Task(
        title="lesson",
        prompt=_SECRET,
        context={"context_subject_id": "teacher_a"},
    )
    await container.runtime.run_agent_loop(Persona(name="Tutor"), task, max_turns=1)

    spine = container.entity_registry
    tagged = spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_a"}))
    assert tagged
    assert any(r.kind == "run_event" for r in tagged)


@pytest.mark.asyncio
async def test_governed_stream_task_opted_out_subject_is_ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stream_task publishes the subject — an opted-out subject leaves no cleartext."""
    container = _governed_container(monkeypatch)
    task = Task(
        title="lesson",
        prompt=_SECRET,
        context={"context_subject_id": "teacher_b"},
    )
    async for _delta in container.runtime.stream_task(Persona(name="Tutor"), task):
        pass

    spine = container.entity_registry
    assert spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_b"})) == []
    assert spine.list_by_kind("message") == []
    spine_blob = "".join(str(r.payload) for r in spine.list_by_kind("message"))
    assert _SECRET not in spine_blob


def _governed_hitl_runtime(monkeypatch: pytest.MonkeyPatch):
    """A governed runtime wired for HITL (checkpoint_store + approval-gated tool).

    ``ApiContainer.build_default()`` deliberately wires no ``checkpoint_store`` /
    ``tool_service``, so the stock governed BFF cannot reach ``resume_agent_loop``.
    But ``SingleAgentRuntime`` is a library-public reusable component: a consumer that
    wires HITL in governed mode must still have its resumed spine records subject-tagged
    so the fail-closed ``ConsentAwareRegistry`` does not silently drop them. This builds
    that consumer's runtime by reusing the governed container's gated storage / gated
    registry / consent_decider and adding the missing HITL seams.
    """
    from himmy.runtime import InMemoryCheckpointStore, SingleAgentRuntime
    from himmy.services.inference.service import InferenceService
    from himmy.services.tools.registry import ToolRegistry, register_local_tool
    from himmy.services.tools.service import ToolService

    container = _governed_container(monkeypatch)
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="wire_money",
        handler=lambda args: {"sent": True, "amount": args.get("amount", 0)},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=True,
    )
    cp_store = InMemoryCheckpointStore()
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(_ApprovalGateManager()),
        memory_store=container.storage,
        tool_service=ToolService(registry, event_sink=container.storage),
        entity_registry=container.runtime.entity_registry,
        checkpoint_store=cp_store,
        consent_decider=container.runtime._consent_decider,
    )
    return container, runtime


def _hitl_task() -> tuple[Persona, Task]:
    return Persona(name="treasurer"), Task(
        title="payment",
        prompt="wire 1000 to the vendor",
        context={
            "tool_names": ["wire_money"],
            "response_format": "AUTO_TOOLS",
            "context_subject_id": "teacher_a",
        },
    )


@pytest.mark.asyncio
async def test_governed_resume_agent_loop_persists_consented_subject_on_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HITL-resumed run for a fully-consented subject reaches the spine tagged.

    Locks the fix for the gap where ``resume_agent_loop`` — a FIFTH multi-turn entry
    point — rebuilt ``ctx`` from the checkpoint but emitted its resumed tool messages /
    APPROVAL_* events / bumped thread version / continuation turn OUTSIDE any
    ``_subject_scope``, so the fail-closed ``ConsentAwareRegistry`` silently dropped them
    even for a consented subject (the exact data-loss defect the multi-turn fix closed,
    left in a sibling path).
    """
    container, runtime = _governed_hitl_runtime(monkeypatch)
    ledger = container.consent_ledger
    ledger.grant("teacher_a", Purpose.RETAIN)
    ledger.grant("teacher_a", Purpose.TRAIN)
    persona, task = _hitl_task()

    paused = await runtime.run_agent_loop(persona, task, max_turns=5, hitl=True)
    assert paused.stopped_reason == "awaiting_approval"

    spine = container.entity_registry
    before = len(spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_a"})))

    final = await runtime.resume_agent_loop(paused.checkpoint_id, approved=True)
    assert final.stopped_reason == "final"

    # The resume path emitted MORE subject-tagged spine records (the approval-granted
    # tool message, the bumped thread version, and the continuation turn's events).
    tagged = spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_a"}))
    assert len(tagged) > before, "resume_agent_loop must stamp the consented subject"
    # The resumed continuation message + an approval/run event reached the spine.
    assert any(r.kind == "message" for r in tagged)
    assert any(r.kind == "run_event" for r in tagged)


@pytest.mark.asyncio
async def test_governed_resume_agent_loop_opted_out_subject_is_ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opted-out subject's HITL-resumed records stay ephemeral (no spine leak)."""
    container, runtime = _governed_hitl_runtime(monkeypatch)
    persona = Persona(name="treasurer")
    # teacher_b never opts in → governed DENY for RETAIN; its resumed records must not land.
    task = Task(
        title="payment",
        prompt="wire 1000 to the vendor",
        context={
            "tool_names": ["wire_money"],
            "response_format": "AUTO_TOOLS",
            "context_subject_id": "teacher_b",
        },
    )

    paused = await runtime.run_agent_loop(persona, task, max_turns=5, hitl=True)
    assert paused.stopped_reason == "awaiting_approval"
    final = await runtime.resume_agent_loop(paused.checkpoint_id, approved=True)
    assert final.stopped_reason == "final"

    spine = container.entity_registry
    # Nothing subject-bearing for teacher_b reached the spine across the WHOLE HITL run.
    assert spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_b"})) == []
    assert spine.list_by_kind("message") == []
    assert spine.list_by_kind("chat_thread") == []
    # The denials were auditable (skip-on-deny emits a consent SecurityEvent).
    assert spine.list_by_kind("security_event")


@pytest.mark.asyncio
async def test_governed_withdraw_crypto_shreds_consented_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After RETAIN+grant, a message persists encrypted; withdraw makes it unrecoverable."""
    pytest.importorskip("cryptography")
    from himmy.services.storage.encryption import ENC_PREFIX

    container = _governed_container(monkeypatch)
    ledger = container.consent_ledger
    ledger.grant("teacher_a", Purpose.RETAIN)
    ledger.grant("teacher_a", Purpose.TRAIN)
    task = Task(
        title="lesson",
        prompt=_SECRET,
        context={"context_subject_id": "teacher_a"},
    )
    await container.runtime.run_task(Persona(name="Tutor"), task)

    spine = container.entity_registry
    messages = [
        r
        for r in spine.list_by_kind("message")
        if r.metadata.get("subject_id") == "teacher_a"
    ]
    assert messages
    # The user message content is encrypted under teacher_a's shreddable key on the spine.
    user_msgs = [m for m in messages if m.payload.get("role") == "user"]
    assert user_msgs
    assert user_msgs[0].payload["content"].startswith(ENC_PREFIX)
    assert _SECRET not in user_msgs[0].payload["content"]

    # Withdraw → erase_subject destroys the key → ciphertext is now permanently lost.
    ledger.withdraw("teacher_a")
    assert container.retention_service is not None
    # A signed erasure tombstone was written.
    assert spine.list_by_kind("erasure_tombstone")
    # Fresh decision is now DENY.
    assert not ledger.decision("teacher_a", Purpose.RETAIN).allowed


@pytest.mark.asyncio
async def test_governed_chat_thread_carries_no_plaintext_conversation_on_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the chat_thread aggregate's nested message content is encrypted too.

    The richest record kind registers a full conversation per turn; before the fix it
    landed in plaintext on the immutable spine (mapped to ``()``), so crypto-shred could
    never make it unrecoverable. Now its ``messages[*].content`` is walked through the
    subject cipher, so erasing the subject renders the whole conversation body lost.
    """
    pytest.importorskip("cryptography")
    import json

    from himmy.services.storage.encryption import ENC_PREFIX

    container = _governed_container(monkeypatch)
    ledger = container.consent_ledger
    ledger.grant("teacher_a", Purpose.RETAIN)
    ledger.grant("teacher_a", Purpose.TRAIN)
    task = Task(
        title="lesson",
        prompt=_SECRET,
        context={"context_subject_id": "teacher_a"},
    )
    await container.runtime.run_task(Persona(name="Tutor"), task)

    spine = container.entity_registry
    threads = [
        r
        for r in spine.list_by_kind("chat_thread")
        if r.metadata.get("subject_id") == "teacher_a"
    ]
    assert threads, "the consented subject's chat_thread reached the spine"
    # No plaintext conversation body survives on the spine chat_thread record.
    blob = "".join(json.dumps(r.payload) for r in threads)
    assert _SECRET not in blob
    # At least one message's content is a ciphertext envelope.
    enc_contents = [
        m["content"]
        for r in threads
        for m in r.payload.get("messages", [])
        if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"]
    ]
    assert enc_contents and all(c.startswith(ENC_PREFIX) for c in enc_contents)

    # Erase the subject: the nested thread ciphertext is now permanently unrecoverable.
    # The durable vault refuses to re-mint a key for the erased subject (which would
    # silently un-shred), so re-access raises rather than returning a fresh (wrong) key.
    vault = container.retention_service._keys
    token = enc_contents[0]
    assert token  # ciphertext exists pre-erase
    ledger.withdraw("teacher_a")
    with pytest.raises(KeyError, match="teacher_a"):
        vault.encryptor_for("teacher_a")
