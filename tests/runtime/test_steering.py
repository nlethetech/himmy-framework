"""Tests for the between-turns steering seam (run_agent_loop steer_queue).

The seam's contract: a thread-safe queue of guidance texts, drained by the
drive loop at the TOP of each continuation turn, each text appended as a USER
message BEFORE the next request is built — so the steer is visible in the very
next turn's request payload. ``steer_queue=None`` (the default) must leave the
loop byte-identical, and turn bounds are unchanged by steering.
"""

from __future__ import annotations

import queue
from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


class _CapturingManager:
    """Scripted manager that records every request's messages.

    Turn 1 calls a tool (so the loop continues); turn 2 (and beyond) answers.
    ``self.requests`` keeps the message list of EVERY inference call so a test
    can assert exactly what the model saw on each turn.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, str]]] = []

    def resolve(self, model_key: str) -> str:
        return f"scripted:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        self.calls += 1
        self.requests.append(
            [
                {"role": str(m.role), "content": m.content or ""}
                for m in request.messages
            ]
        )
        cid = f"c{self.calls}"
        if self.calls == 1:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="acting",
                tool_calls=[
                    ToolCallRecord(tool_call_id=cid, tool_name="probe", args={})
                ],
                tool_returns=[
                    ToolReturnRecord(
                        tool_call_id=cid, tool_name="probe", content={"ok": True}
                    )
                ],
                input_tokens=1,
                output_tokens=1,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="final answer",
            input_tokens=1,
            output_tokens=1,
        )


def _runtime(manager: _CapturingManager) -> SingleAgentRuntime:
    return SingleAgentRuntime(
        inference_service=InferenceService(manager),
        memory_store=StorageService(),
    )


def _go() -> tuple[Persona, Task]:
    return Persona(name="analyst"), Task(title="t", prompt="investigate")


def test_steer_lands_in_next_turns_request_payload() -> None:
    """A queued steer is a USER message in the NEXT continuation turn's request."""
    manager = _CapturingManager()
    rt = _runtime(manager)
    persona, task = _go()
    steers: queue.Queue[str] = queue.Queue()
    steers.put("focus only on Q4 numbers")

    result = run_async(
        rt.run_agent_loop(persona, task, max_turns=4, steer_queue=steers)
    )

    assert result.stopped_reason == "final"
    assert manager.calls == 2
    # Turn 1's request must NOT contain the steer (it was queued for later turns).
    turn1_user = [m for m in manager.requests[0] if m["role"] == "user"]
    assert all("Q4 numbers" not in m["content"] for m in turn1_user)
    # Turn 2's request must carry it as a user message, after the tool exchange.
    turn2 = manager.requests[1]
    steer_msgs = [
        m
        for m in turn2
        if m["role"] == "user" and "focus only on Q4 numbers" in m["content"]
    ]
    assert len(steer_msgs) == 1
    # And the thread itself shows the injected USER message (events/persistence).
    injected = [
        m
        for m in result.thread.messages
        if m.role == MessageRole.USER and m.metadata.get("steer") is True
    ]
    assert [m.content for m in injected] == ["focus only on Q4 numbers"]
    assert steers.empty()


def test_multiple_steers_drain_in_order() -> None:
    """Every queued text is injected, in arrival order, in one drain."""
    manager = _CapturingManager()
    rt = _runtime(manager)
    persona, task = _go()
    steers: queue.Queue[str] = queue.Queue()
    steers.put("first steer")
    steers.put("second steer")
    steers.put("   ")  # blank guidance is dropped, not injected

    result = run_async(
        rt.run_agent_loop(persona, task, max_turns=4, steer_queue=steers)
    )

    turn2 = manager.requests[1]
    contents = [m["content"] for m in turn2 if m["role"] == "user"]
    i_first = next(i for i, c in enumerate(contents) if "first steer" in c)
    i_second = next(i for i, c in enumerate(contents) if "second steer" in c)
    assert i_first < i_second
    injected = [m for m in result.thread.messages if m.metadata.get("steer") is True]
    assert [m.content for m in injected] == ["first steer", "second steer"]


def test_no_steer_queue_is_byte_identical() -> None:
    """steer_queue=None changes nothing: same turns, same request payloads."""
    base = _CapturingManager()
    run_async(_runtime(base).run_agent_loop(*_go(), max_turns=4))

    empty_q = _CapturingManager()
    q: queue.Queue[str] = queue.Queue()
    run_async(_runtime(empty_q).run_agent_loop(*_go(), max_turns=4, steer_queue=q))

    assert base.calls == empty_q.calls == 2
    # Same message roles/contents on every turn (ids/timestamps aside).
    assert base.requests == empty_q.requests


def test_steering_respects_max_turns() -> None:
    """Steering never extends the loop past max_turns (HARD_MAX unchanged)."""

    class _AlwaysActs(_CapturingManager):
        async def generate(self, request: Any) -> InferenceResponse:
            self.calls += 1
            self.requests.append(
                [
                    {"role": str(m.role), "content": m.content or ""}
                    for m in request.messages
                ]
            )
            cid = f"c{self.calls}"
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="acting",
                tool_calls=[
                    ToolCallRecord(tool_call_id=cid, tool_name="probe", args={})
                ],
                tool_returns=[
                    ToolReturnRecord(
                        tool_call_id=cid, tool_name="probe", content={"ok": True}
                    )
                ],
                input_tokens=1,
                output_tokens=1,
            )

    manager = _AlwaysActs()
    rt = _runtime(manager)
    persona, task = _go()
    steers: queue.Queue[str] = queue.Queue()
    for i in range(10):
        steers.put(f"keep going {i}")

    result = run_async(
        rt.run_agent_loop(
            persona, task, max_turns=3, steer_queue=steers, synthesize_empty=False
        )
    )
    assert result.stopped_reason == "max_turns"
    assert manager.calls == 3
