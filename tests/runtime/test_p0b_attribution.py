"""P0-B: implicit signals (guardrail hits, approvals) are attributable + enriched.

Guardrail events now carry trace_id/thread_id (were agent_id-only); approval
events carry decision/agent_name/actor/time_to_decision_ms + the tool_call_id —
the judge-free signals a learning miner reads, made queryable per run/agent/tool.
"""

from __future__ import annotations

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType
from himmy.services.guardrails import (
    BlocklistGuardrail,
    DlpAction,
    DlpGuardrail,
    DlpPolicy,
    GuardrailPipeline,
)
from tests.conftest import run_async
from tests.runtime.test_guardrail_block_enforcement import _CARD, _runtime
from tests.runtime.test_hitl import _hitl_runtime, _task


def _guardrail_events(events: list, *, stage: str) -> list:
    return [
        e
        for e in events
        if e.event_type == EventType.GUARDRAIL_APPLIED
        and e.payload.get("stage") == stage
    ]


# --------------------------------------------------------------- guardrails


def test_output_guardrail_event_attributes_to_run() -> None:
    guard = GuardrailPipeline([BlocklistGuardrail(["forbidden-secret-phrase"])])
    rt, _m, _s, events = _runtime(
        output_text="here is the forbidden-secret-phrase", output_guardrail=guard
    )
    run_async(rt.run_task_detailed(Persona(name="agent"), Task(title="t", prompt="go")))

    out = _guardrail_events(events, stage="output")
    assert out, "expected an output GUARDRAIL_APPLIED event"
    # Was agent_id-only before P0-B — now attributable to the specific run + thread.
    assert out[0].trace_id is not None
    assert out[0].thread_id is not None


def test_input_guardrail_event_attributes_to_thread() -> None:
    guard = GuardrailPipeline(
        [DlpGuardrail(policy=DlpPolicy(actions={"card": DlpAction.BLOCK}))]
    )
    rt, _m, _s, events = _runtime(output_text="ok", input_guardrail=guard)
    run_async(
        rt.run_task_detailed(
            Persona(name="agent"), Task(title="t", prompt=f"card {_CARD}")
        )
    )

    inp = _guardrail_events(events, stage="input")
    assert inp, "expected an input GUARDRAIL_APPLIED event"
    assert inp[0].thread_id is not None


# --------------------------------------------------------------- approvals


def test_approval_granted_event_is_enriched() -> None:
    events: list = []

    async def collect(event) -> None:
        events.append(event)

    rt, _store = _hitl_runtime(on_event=collect)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, hitl=True))
    run_async(
        rt.resume_agent_loop(paused.checkpoint_id, approved=True, actor="alice")
    )

    granted = next(e for e in events if e.event_type == EventType.APPROVAL_GRANTED)
    assert granted.trace_id is not None
    assert granted.thread_id is not None
    assert granted.tool_call_id is not None  # promoted to the event field
    p = granted.payload
    assert p["decision"] == "granted"
    assert p["agent_name"] == "treasurer"
    assert p["actor"] == "alice"
    assert p["tool_name"] == "wire_money"
    assert isinstance(p["time_to_decision_ms"], float)
    assert p["time_to_decision_ms"] >= 0.0


def test_approval_rejected_records_decision_and_actor() -> None:
    events: list = []

    async def collect(event) -> None:
        events.append(event)

    rt, _store = _hitl_runtime(on_event=collect)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, hitl=True))
    run_async(
        rt.resume_agent_loop(paused.checkpoint_id, approved=False, actor="carol")
    )

    rejected = next(e for e in events if e.event_type == EventType.APPROVAL_REJECTED)
    assert rejected.payload["decision"] == "rejected"
    assert rejected.payload["actor"] == "carol"


def test_approval_actor_defaults_to_human() -> None:
    events: list = []

    async def collect(event) -> None:
        events.append(event)

    rt, _store = _hitl_runtime(on_event=collect)
    persona, task = _task()
    paused = run_async(rt.run_agent_loop(persona, task, hitl=True))
    run_async(rt.resume_agent_loop(paused.checkpoint_id, approved=True))

    granted = next(e for e in events if e.event_type == EventType.APPROVAL_GRANTED)
    assert granted.payload["actor"] == "human"
