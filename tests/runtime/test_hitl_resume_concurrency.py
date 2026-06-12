"""HITL resume is exactly-once under CONCURRENT approvals (atomic claim).

The status check (``awaiting_approval`` -> ``approved``) and the gated tool's
execution were not atomic, so two concurrent resumes of the same checkpoint (a
double-clicked Approve, two browser tabs, an automation retry) both passed the check
and both executed the approval-gated tool — the HITL gate's core promise (a
human-approved action runs ONCE) was broken. The fix makes the resume transition an
atomic claim (compare-and-set ``awaiting_approval``/``resolving`` -> ``resolving`` at
the store, plus a per-checkpoint in-process lock), so only the first resume executes
and the loser is a no-op.

Offline-first: an approval-gated local tool, an InMemory checkpoint store, no provider.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.runtime import InMemoryCheckpointStore, SingleAgentRuntime
from himmy.runtime.checkpoint import (
    AWAITING_APPROVAL,
    RESOLVING,
    AgentCheckpoint,
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
    """Calls the approval-gated tool until a result exists, then finals."""

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


def _hitl_runtime(store: Any) -> tuple[SingleAgentRuntime, list[dict]]:
    storage = StorageService()
    registry = ToolRegistry()
    executions: list[dict] = []

    async def wire_money(args: dict) -> dict:
        # Yield control so two concurrent resumes genuinely interleave at the await.
        await asyncio.sleep(0)
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


def test_concurrent_resumes_execute_the_tool_exactly_once_inmemory() -> None:
    """Two concurrent approvals of the same checkpoint: the money moves ONCE."""
    store = InMemoryCheckpointStore()
    rt, executions = _hitl_runtime(store)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))
    assert paused.stopped_reason == "awaiting_approval"

    async def _race() -> list[Any]:
        return await asyncio.gather(
            rt.resume_agent_loop(paused.checkpoint_id, approved=True),
            rt.resume_agent_loop(paused.checkpoint_id, approved=True),
            return_exceptions=True,
        )

    results = run_async(_race())

    # The gated tool ran exactly once across both concurrent resumes.
    assert len(executions) == 1
    # Exactly one resume succeeded; the loser was refused as already-resolved.
    losers = [r for r in results if isinstance(r, HimmyError)]
    winners = [r for r in results if not isinstance(r, BaseException)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert "already resolved" in str(losers[0])
    # The checkpoint ends resolved.
    final = store.load(paused.checkpoint_id)
    assert final is not None and final.status == "approved"


def test_concurrent_resumes_execute_the_tool_exactly_once_sqlite() -> None:
    """Same guarantee on the durable SQLite store (the shipped Studio store)."""
    store = SqliteCheckpointStore(":memory:")
    rt, executions = _hitl_runtime(store)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))
    assert paused.stopped_reason == "awaiting_approval"

    async def _race() -> list[Any]:
        return await asyncio.gather(
            rt.resume_agent_loop(paused.checkpoint_id, approved=True),
            rt.resume_agent_loop(paused.checkpoint_id, approved=True),
            return_exceptions=True,
        )

    run_async(_race())
    assert len(executions) == 1
    final = store.load(paused.checkpoint_id)
    assert final is not None and final.status == "approved"


def test_claim_is_a_single_winner_compare_and_set() -> None:
    """``claim`` flips awaiting_approval -> resolving for exactly the first caller."""
    for store in (InMemoryCheckpointStore(), SqliteCheckpointStore(":memory:")):
        cp = AgentCheckpoint(status=AWAITING_APPROVAL)
        store.save(cp)
        assert store.claim(cp.checkpoint_id) is True  # first wins
        reloaded = store.load(cp.checkpoint_id)
        assert reloaded is not None and reloaded.status == RESOLVING
        # A second claim from ``resolving`` is allowed (crash-retry re-entry)...
        assert store.claim(cp.checkpoint_id) is True
        # ...but once resolved, the claim is refused.
        store.save(reloaded.model_copy(update={"status": "approved"}))
        assert store.claim(cp.checkpoint_id) is False


def test_claim_unknown_checkpoint_is_false() -> None:
    """Claiming a checkpoint that doesn't exist is a no-op (False), never a crash."""
    for store in (InMemoryCheckpointStore(), SqliteCheckpointStore(":memory:")):
        assert store.claim("nope") is False


def test_second_sequential_resume_after_resolution_is_refused() -> None:
    """A clean resume resolves; a later resume is refused (status no longer claimable)."""
    store = InMemoryCheckpointStore()
    rt, executions = _hitl_runtime(store)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, max_turns=5, hitl=True))

    final = run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert final.stopped_reason == "final"
    assert len(executions) == 1
    with pytest.raises(HimmyError, match="already resolved"):
        run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))
    assert len(executions) == 1
