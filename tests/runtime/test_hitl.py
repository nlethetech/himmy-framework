"""Tests for human-in-the-loop pause/resume (approval interrupt + checkpoint)."""

from __future__ import annotations

from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.runtime import (
    InMemoryCheckpointStore,
    SingleAgentRuntime,
    SqliteCheckpointStore,
)
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


class _ApprovalGateManager:
    """Calls the (approval-gated) bound tool until a tool result exists, then finals.

    This exercises the REAL approval gate: turn 1 invokes the bound tool's handler
    (which routes through ToolService.execute and gets denied for lack of
    approval); once a tool result is in the conversation it answers and stops.
    """

    def resolve(self, model_key: str) -> str:
        return model_key

    async def generate(self, request: Any) -> InferenceResponse:
        tool_in_history = any(
            getattr(m, "role", "") == "tool" for m in request.messages
        )
        if not tool_in_history and request.bound_tools:
            ret = await request.bound_tools[0].handler({})
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


def _hitl_runtime(store: Any = None, on_event: Any = None):
    storage = StorageService()
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="wire_money",
        handler=lambda args: {"sent": True, "amount": args.get("amount", 0)},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=True,
    )
    cp_store = store or InMemoryCheckpointStore()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_ApprovalGateManager()),
        memory_store=storage,
        tool_service=ToolService(registry, event_sink=storage),
        checkpoint_store=cp_store,
        on_event=on_event,
    )
    return rt, cp_store


def _task() -> tuple[Persona, Task]:
    return Persona(name="treasurer"), Task(
        title="payment",
        prompt="wire 1000 to the vendor",
        context={"tool_names": ["wire_money"], "response_format": "AUTO_TOOLS"},
    )


def test_loop_pauses_and_checkpoints_on_approval_required() -> None:
    """An approval-required tool call pauses the loop and persists a checkpoint."""
    rt, store = _hitl_runtime()
    persona, task = _task()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))

    assert result.stopped_reason == "awaiting_approval"
    assert result.checkpoint_id is not None
    checkpoint = store.load(result.checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.status == "awaiting_approval"
    assert [p.tool_name for p in checkpoint.pending_tool_calls] == ["wire_money"]


def test_resume_approved_executes_the_tool_and_completes() -> None:
    """Resuming approved runs the real tool and the loop then finishes."""
    rt, store = _hitl_runtime()
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))

    final = run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert final.stopped_reason == "final"
    assert final.final.output_text == "done with the result"
    tool_msgs = [m for m in final.thread.messages if m.role == MessageRole.TOOL]
    assert any('"sent"' in (m.content or "") for m in tool_msgs)  # the tool really ran
    assert store.load(paused.checkpoint_id).status == "approved"


def test_resume_rejected_records_rejection_without_running_the_tool() -> None:
    """Resuming rejected records the rejection and never executes the tool."""
    rt, store = _hitl_runtime()
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))

    final = run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=False))
    assert final.stopped_reason == "final"
    tool_msgs = [m for m in final.thread.messages if m.role == MessageRole.TOOL]
    assert any("rejected" in (m.content or "").lower() for m in tool_msgs)
    assert not any('"sent"' in (m.content or "") for m in tool_msgs)  # tool not run
    assert store.load(paused.checkpoint_id).status == "rejected"


def test_resume_is_idempotent() -> None:
    """A resolved checkpoint cannot be resumed twice."""
    rt, _store = _hitl_runtime()
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, hitl=True))
    run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    with pytest.raises(HimmyError):
        run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))


def test_hitl_requires_a_checkpoint_store() -> None:
    """hitl=True without a checkpoint_store is a configuration error."""
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_ApprovalGateManager()),
        memory_store=StorageService(),
    )
    with pytest.raises(HimmyError):
        run_async(
            rt.run_agent_loop(Persona(name="a"), Task(title="t", prompt="p"), hitl=True)
        )


def test_approval_events_are_emitted() -> None:
    """APPROVAL_REQUIRED on pause and APPROVAL_GRANTED on approve are emitted."""
    events: list[RunEvent] = []

    async def collect(event: RunEvent) -> None:
        events.append(event)

    rt, _store = _hitl_runtime(on_event=collect)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, hitl=True))
    run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    kinds = {e.event_type for e in events}
    assert EventType.APPROVAL_REQUIRED in kinds
    assert EventType.APPROVAL_GRANTED in kinds


def test_checkpoint_survives_a_restart_via_sqlite(tmp_path) -> None:
    """A SQLite checkpoint can be resumed by a fresh runtime (simulated restart)."""
    path = str(tmp_path / "checkpoints.db")
    rt, store = _hitl_runtime(store=SqliteCheckpointStore(path))
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))
    checkpoint_id = paused.checkpoint_id
    store.close()

    # Fresh runtime + reopened store (the original process is "gone").
    rt2, _store2 = _hitl_runtime(store=SqliteCheckpointStore(path))
    final = run_async(rt2.resume_agent_loop(checkpoint_id, approved=True))
    assert final.stopped_reason == "final"
    tool_msgs = [m for m in final.thread.messages if m.role == MessageRole.TOOL]
    assert any('"sent"' in (m.content or "") for m in tool_msgs)
