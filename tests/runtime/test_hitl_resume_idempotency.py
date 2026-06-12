"""Tests for HITL resume idempotency: an approved tool executes exactly once.

The status guard alone (``awaiting_approval`` -> ``approved`` exactly once) does
not cover a crash BETWEEN executing an approved state-mutating tool and resolving
the checkpoint — the checkpoint stays ``awaiting_approval`` and a retried resume
would run the tool a second time. The fix records each executed tool_call_id (and
its result) on the checkpoint the moment it completes, so a repeated resume
replays the recorded result instead of re-executing.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.runtime import InMemoryCheckpointStore, SingleAgentRuntime
from himmy.runtime.checkpoint import (
    APPROVED,
    CHECKPOINT_SCHEMA_VERSION,
    AgentCheckpoint,
    _migrate_checkpoint,
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
    """Calls the (approval-gated) bound tool until a tool result exists, then finals."""

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


class _CrashBeforeResolveStore(InMemoryCheckpointStore):
    """Raises once on the save that flips the checkpoint to ``approved``.

    Simulates a process crash AFTER the approved tool executed but BEFORE the
    checkpoint was resolved — the exact window where a retried resume would
    double-execute a state-mutating tool without the idempotency record.
    """

    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    def save(self, checkpoint: AgentCheckpoint) -> None:
        if checkpoint.status == APPROVED and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated crash before checkpoint resolution")
        super().save(checkpoint)


def _hitl_runtime(store: Any) -> tuple[SingleAgentRuntime, list[dict]]:
    storage = StorageService()
    registry = ToolRegistry()
    executions: list[dict] = []

    def wire_money(args: dict) -> dict:
        executions.append(dict(args))
        return {"sent": True, "execution": len(executions)}

    register_local_tool(
        registry,
        name="wire_money",
        handler=wire_money,
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=True,
    )
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_ApprovalGateManager()),
        memory_store=storage,
        tool_service=ToolService(registry, event_sink=storage),
        checkpoint_store=store,
    )
    return rt, executions


def _task() -> tuple[Persona, Task]:
    return Persona(name="treasurer"), Task(
        title="payment",
        prompt="wire 1000 to the vendor",
        context={"tool_names": ["wire_money"], "response_format": "AUTO_TOOLS"},
    )


def test_resume_retry_after_crash_does_not_re_execute_the_tool() -> None:
    """pause -> resume (crashes before resolve) -> resume again: tool ran ONCE."""
    store = _CrashBeforeResolveStore()
    rt, executions = _hitl_runtime(store)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))
    assert paused.stopped_reason == "awaiting_approval"

    # First resume: the tool executes, then the resolve save "crashes".
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert len(executions) == 1

    # The execution was recorded durably BEFORE the crash window. The atomic claim
    # flipped the checkpoint to ``resolving`` before executing, so a mid-resume crash
    # leaves it there (re-claimable by a retry) — never ``approved`` (it never
    # resolved).
    checkpoint = store.load(paused.checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.status == "resolving"  # claimed, executed, but not resolved
    pending_id = checkpoint.pending_tool_calls[0].tool_call_id
    assert pending_id in checkpoint.executed_tool_results

    # Retried resume: replays the recorded result instead of re-executing.
    final = run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert len(executions) == 1  # the money moved exactly once
    assert final.stopped_reason == "final"
    assert final.final is not None
    assert final.final.output_text == "done with the result"

    # The transcript is coherent: exactly one tool message, carrying the REAL
    # result of the one execution.
    tool_msgs = [
        m
        for m in final.thread.messages
        if m.role == MessageRole.TOOL and '"sent"' in (m.content or "")
    ]
    assert len(tool_msgs) == 1
    assert '"execution": 1' in (tool_msgs[0].content or "")

    # And the checkpoint is now resolved, so a THIRD resume is refused outright.
    loaded = store.load(paused.checkpoint_id)
    assert loaded is not None
    assert loaded.status == "approved"
    with pytest.raises(HimmyError, match="already resolved"):
        run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert len(executions) == 1


def test_clean_resume_records_the_execution_on_the_checkpoint() -> None:
    """A normal approved resume leaves the executed-call record on the checkpoint."""
    store = InMemoryCheckpointStore()
    rt, executions = _hitl_runtime(store)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))

    final = run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert final.stopped_reason == "final"
    assert len(executions) == 1

    checkpoint = store.load(paused.checkpoint_id)
    assert checkpoint is not None
    pending_id = checkpoint.pending_tool_calls[0].tool_call_id
    recorded = checkpoint.executed_tool_results[pending_id]
    assert recorded["outcome"] == "success"
    assert recorded["result"] == {"sent": True, "execution": 1}


def test_v1_checkpoint_migrates_with_empty_execution_record() -> None:
    """A persisted v1 checkpoint (pre-idempotency) loads with an empty record."""
    migrated = _migrate_checkpoint({"schema_version": 1, "persona": {"name": "p"}})
    assert migrated["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert migrated["executed_tool_results"] == {}
    checkpoint = AgentCheckpoint.model_validate(migrated)
    assert checkpoint.executed_tool_results == {}
