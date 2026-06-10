"""Tests for the multi-agent orchestrator: handoff + supervisor delegation (offline)."""

from __future__ import annotations

import concurrent.futures
from typing import Any

import pytest

from himmy import build_runtime
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core import HimmyError
from himmy.core.events import EventType
from himmy.entities.registry import EntityRegistry
from himmy.orchestrators import AgentTeam, MultiAgentOrchestrator, TeamMember
from himmy.services.governance.consent import Decision, Effect, Purpose
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from tests.conftest import run_async


def _team_runtime() -> tuple[Any, ToolRegistry]:
    registry = ToolRegistry()
    runtime, _inf, _tools = build_runtime(tool_registry=registry)
    return runtime, registry


def _system_message(thread: Any) -> str:
    """The current leading SYSTEM message content of a thread (or "")."""
    for message in thread.messages:
        if message.role == MessageRole.SYSTEM:
            return message.content
    return ""


def test_handoff_transfers_control_and_terminates() -> None:
    """The entry agent hands off to a tool-less peer, which produces the final answer."""
    runtime, registry = _team_runtime()
    team = AgentTeam(
        members=[
            TeamMember(
                name="triage", persona=Persona(name="triage"), handoffs=["specialist"]
            ),
            TeamMember(name="specialist", persona=Persona(name="specialist")),
        ],
        entry="triage",
    )
    events: list[str] = []

    async def on_event(e: Any) -> None:
        if e.event_type == EventType.AGENT_HANDOFF:
            events.append(f"{e.payload['from']}->{e.payload['to']}")

    orch = MultiAgentOrchestrator(runtime, team, registry, on_event=on_event)
    res = run_async(orch.run("help me"))

    assert res.handoff_chain == ["triage", "specialist"]
    assert res.final_agent == "specialist"
    assert res.stopped_reason == "final"
    assert events == ["triage->specialist"]


def test_handoff_tool_is_registered_for_targets_only() -> None:
    """Only declared handoff targets get a transfer_to_* tool."""
    runtime, registry = _team_runtime()
    team = AgentTeam(
        members=[
            TeamMember(name="a", persona=Persona(name="a"), handoffs=["b"]),
            TeamMember(name="b", persona=Persona(name="b")),
        ],
        entry="a",
    )
    MultiAgentOrchestrator(runtime, team, registry)
    assert registry.get("transfer_to_b") is not None
    assert registry.get("transfer_to_a") is None


def test_delegation_runs_worker_and_returns_answer() -> None:
    """A manager delegating to a worker fires ask_<worker> and gets the worker's answer."""
    runtime, registry = _team_runtime()
    team = AgentTeam(
        members=[
            TeamMember(
                name="manager", persona=Persona(name="manager"), delegates=["worker"]
            ),
            TeamMember(name="worker", persona=Persona(name="worker")),
        ],
        entry="manager",
    )
    delegated: list[str] = []

    async def on_event(e: Any) -> None:
        if e.event_type == EventType.AGENT_DELEGATED:
            delegated.append(e.payload["worker"])

    # Low cap: the offline stub always calls a bound tool under AUTO_TOOLS, so a manager
    # that owns a delegate tool never emits a final turn — a real model would. We assert
    # the delegation HAPPENED, not the stop reason.
    orch = MultiAgentOrchestrator(
        runtime, team, registry, max_turns=3, on_event=on_event
    )
    res = run_async(orch.run("do research"))

    assert registry.get("ask_worker") is not None
    assert "worker" in delegated
    returns = [r for _, turn in res.turns for r in turn.tool_returns]
    assert any(
        r.tool_name == "ask_worker" and "answer" in (r.content or {}) for r in returns
    )


def test_single_agent_no_edges_answers_directly() -> None:
    """A lone tool-less member answers on the first turn (no handoff machinery)."""
    runtime, registry = _team_runtime()
    team = AgentTeam(
        members=[TeamMember(name="solo", persona=Persona(name="solo"))], entry="solo"
    )
    res = run_async(MultiAgentOrchestrator(runtime, team, registry).run("hello"))
    assert res.final_agent == "solo"
    assert res.stopped_reason == "final"
    assert res.turn_count == 1


def test_handoff_reinjects_target_persona_system_prompt() -> None:
    """Regression: after a handoff the TARGET persona's system prompt is in effect.

    The shared thread carries the entry persona's SYSTEM message; without re-injecting
    the target persona's prompt on handoff, the specialist would run under the triage
    persona. We assert (a) the thread's leading SYSTEM message is now the specialist's
    and (b) the specialist's distinctive description reaches the model (the offline
    stub echoes the system prompt head, so the final answer carries it).
    """
    runtime, registry = _team_runtime()
    team = AgentTeam(
        members=[
            TeamMember(
                name="triage",
                persona=Persona(name="triage", description="ZEBRA_TRIAGE_DESK"),
                handoffs=["specialist"],
            ),
            TeamMember(
                name="specialist",
                persona=Persona(
                    name="specialist", description="QUOKKA_SPECIALIST_DESK"
                ),
            ),
        ],
        entry="triage",
    )
    res = run_async(MultiAgentOrchestrator(runtime, team, registry).run("help me"))

    assert res.final_agent == "specialist"
    system = _system_message(res.thread)
    # The specialist's identity is in the SYSTEM message, the triage's is gone.
    assert "specialist agent" in system
    assert "QUOKKA_SPECIALIST_DESK" in system
    assert "triage agent" not in system
    assert "ZEBRA_TRIAGE_DESK" not in system
    # The offline stub echoes the system head into its answer — proof the SWITCHED
    # persona prompt is what actually drove the specialist's turn.
    assert res.output_text is not None
    assert "specialist agent" in res.output_text


def test_handoff_system_message_is_not_duplicated() -> None:
    """A handoff REPLACES the SYSTEM message in place — it does not append a 2nd one."""
    runtime, registry = _team_runtime()
    team = AgentTeam(
        members=[
            TeamMember(
                name="a",
                persona=Persona(name="a", description="AAA"),
                handoffs=["b"],
            ),
            TeamMember(name="b", persona=Persona(name="b", description="BBB")),
        ],
        entry="a",
    )
    res = run_async(MultiAgentOrchestrator(runtime, team, registry).run("go"))
    system_count = sum(1 for m in res.thread.messages if m.role == MessageRole.SYSTEM)
    assert system_count == 1


def test_unknown_handoff_target_raises() -> None:
    """Declaring a handoff to a non-member fails fast at construction."""
    runtime, registry = _team_runtime()
    team = AgentTeam(
        members=[TeamMember(name="a", persona=Persona(name="a"), handoffs=["ghost"])],
        entry="a",
    )
    import pytest

    from himmy.core import HimmyError

    with pytest.raises(HimmyError):
        MultiAgentOrchestrator(runtime, team, registry)


# ---------------------------------------------------------- WS4.6 subject propagation


def _allow_all(subject: str, purpose: Purpose) -> Decision:
    """A permissive consent decider so subject scopes actually publish."""
    return Decision(subject_id=subject, purpose=purpose, effect=Effect.ALLOW)


def _governed_team_runtime() -> tuple[Any, ToolRegistry, EntityRegistry]:
    """A runtime with a consent decider wired plus a directly inspectable spine."""
    spine = EntityRegistry()
    tool_registry = ToolRegistry()
    runtime, _inf, _tools = build_runtime(tool_registry=tool_registry, registry=spine)
    runtime._consent_decider = _allow_all
    return runtime, tool_registry, spine


def _handoff_team() -> AgentTeam:
    return AgentTeam(
        members=[
            TeamMember(
                name="triage", persona=Persona(name="triage"), handoffs=["specialist"]
            ),
            TeamMember(name="specialist", persona=Persona(name="specialist")),
        ],
        entry="triage",
    )


def _delegate_team() -> AgentTeam:
    return AgentTeam(
        members=[
            TeamMember(
                name="manager", persona=Persona(name="manager"), delegates=["worker"]
            ),
            TeamMember(name="worker", persona=Persona(name="worker")),
        ],
        entry="manager",
    )


def test_run_task_context_subject_spans_handoff() -> None:
    """WS4.6: the run's subject is published for the WHOLE team run.

    The orchestrator's AGENT_HANDOFF spine record and every turn's message records
    (including post-handoff turns) carry the subject stamp — without the run-wide
    scope + per-turn propagation they would be subject-less and a fail-closed
    ConsentAwareRegistry would silently drop them on handoff.
    """
    runtime, registry, spine = _governed_team_runtime()
    orch = MultiAgentOrchestrator(runtime, _handoff_team(), registry)
    res = run_async(
        orch.run("help me", task_context={"context_subject_id": "subject_7"})
    )
    assert res.handoff_chain == ["triage", "specialist"]

    events = spine.list_by_kind("run_event")
    handoffs = [
        r for r in events if "AGENT_HANDOFF" in str(r.payload.get("event_type"))
    ]
    assert handoffs
    assert all(r.metadata.get("subject_id") == "subject_7" for r in handoffs)
    # Per-turn propagation: EVERY message record (both sides of the handoff) is
    # stamped, so the nested runtime scopes resolved the same subject.
    messages = spine.list_by_kind("message")
    assert messages
    assert all(r.metadata.get("subject_id") == "subject_7" for r in messages)


def test_run_task_context_subject_reaches_delegate_subrun() -> None:
    """WS4.6: a delegate worker's sub-run inherits the run's subject."""
    runtime, registry, spine = _governed_team_runtime()
    orch = MultiAgentOrchestrator(runtime, _delegate_team(), registry, max_turns=3)
    run_async(orch.run("do research", task_context={"context_subject_id": "subject_9"}))

    events = spine.list_by_kind("run_event")
    delegated = [
        r for r in events if "AGENT_DELEGATED" in str(r.payload.get("event_type"))
    ]
    assert delegated
    assert all(r.metadata.get("subject_id") == "subject_9" for r in delegated)
    # The worker's sub-thread messages are stamped too (no subject-less leak).
    messages = spine.list_by_kind("message")
    assert messages
    assert all(r.metadata.get("subject_id") == "subject_9" for r in messages)


def test_run_without_subject_keeps_records_unstamped() -> None:
    """Zero-config regression: no decider/subject ⇒ no subject_id metadata anywhere."""
    spine = EntityRegistry()
    registry = ToolRegistry()
    runtime, _inf, _tools = build_runtime(tool_registry=registry, registry=spine)
    res = run_async(
        MultiAgentOrchestrator(runtime, _handoff_team(), registry).run("help me")
    )
    assert res.stopped_reason == "final"
    events = spine.list_by_kind("run_event")
    assert events
    assert all("subject_id" not in r.metadata for r in events)


# ------------------------------------------------------------- max_turns synthesis


class _ToolLoopManager:
    """Calls the member's non-final tool with VARYING args every tool turn.

    Varying args defeat the no-progress dedup so the loop genuinely runs into
    ``max_turns`` mid-act; when no tools are bound (the synthesis turn) it answers
    in plain text instead.
    """

    provider_name = "stub"

    def __init__(self) -> None:
        self._calls = 0

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        tools = [t for t in (request.bound_tools or []) if t.name != "final_answer"]
        if tools:
            self._calls += 1
            args = {"q": f"step-{self._calls}"}
            ret = await request.tool_executor(tools[0].name, args)
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="raw tool chatter",
                tool_calls=[
                    ToolCallRecord(
                        tool_call_id=ret.tool_call_id,
                        tool_name=ret.tool_name,
                        args=args,
                    )
                ],
                tool_returns=[ret],
                input_tokens=1,
                output_tokens=1,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="SYNTHESIZED-PLAIN-ANSWER",
            input_tokens=1,
            output_tokens=1,
        )


def _loop_team_runtime() -> tuple[Any, ToolRegistry]:
    registry = ToolRegistry()
    runtime, _inf, _tools = build_runtime(
        tool_registry=registry, inference=InferenceService(_ToolLoopManager())
    )
    register_local_tool(
        registry,
        name="probe",
        handler=lambda args: {"ok": True},
        description="A probe tool the looping member keeps calling.",
    )
    return runtime, registry


def _solo_tool_team() -> AgentTeam:
    return AgentTeam(
        members=[
            TeamMember(name="solo", persona=Persona(name="solo"), tools=["probe"])
        ],
        entry="solo",
    )


def test_max_turns_with_tool_calls_synthesizes_plain_answer() -> None:
    """Hitting max_turns mid-act adds ONE tool-less synthesis turn (no raw output)."""
    runtime, registry = _loop_team_runtime()
    orch = MultiAgentOrchestrator(runtime, _solo_tool_team(), registry, max_turns=2)
    res = run_async(orch.run("dig into this"))

    assert res.stopped_reason == "max_turns_synthesized"
    assert res.output_text == "SYNTHESIZED-PLAIN-ANSWER"
    assert res.turn_count == 3  # the 2 capped turns + 1 synthesis turn
    assert not res.turns[-1][1].tool_calls


def test_max_turns_synthesis_opt_out_keeps_raw_stop() -> None:
    """synthesize_on_max_turns=False preserves the bare max_turns stop."""
    runtime, registry = _loop_team_runtime()
    orch = MultiAgentOrchestrator(
        runtime,
        _solo_tool_team(),
        registry,
        max_turns=2,
        synthesize_on_max_turns=False,
    )
    res = run_async(orch.run("dig into this"))

    assert res.stopped_reason == "max_turns"
    assert res.turn_count == 2
    assert res.turns[-1][1].tool_calls


# --------------------------------------------------- synthetic tool registration


def test_synthetic_tool_collision_with_foreign_tool_raises() -> None:
    """A user tool that happens to be named ask_<member> is not silently clobbered."""
    runtime, registry = _team_runtime()
    register_local_tool(
        registry,
        name="ask_worker",
        handler=lambda args: {"mine": True},
        description="A user's own tool that predates the team.",
    )
    with pytest.raises(HimmyError, match="collision"):
        MultiAgentOrchestrator(runtime, _delegate_team(), registry)
    # The user's tool survives untouched.
    definition = registry.get("ask_worker")
    assert definition is not None
    assert "synthetic" not in definition.metadata


def test_synthetic_tool_reregistration_is_idempotent() -> None:
    """Two orchestrators over one shared registry re-register their tools cleanly."""
    runtime, registry = _team_runtime()
    MultiAgentOrchestrator(runtime, _delegate_team(), registry)
    MultiAgentOrchestrator(runtime, _delegate_team(), registry)  # no collision error
    definition = registry.get("ask_worker")
    assert definition is not None
    assert definition.metadata.get("synthetic") == "delegate"


def test_concurrent_construction_on_shared_registry_is_safe() -> None:
    """Many orchestrators wired in parallel threads never tear the registration."""
    runtime, registry = _team_runtime()

    def _build() -> None:
        MultiAgentOrchestrator(runtime, _delegate_team(), registry)
        MultiAgentOrchestrator(runtime, _handoff_team(), registry)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(_build) for _ in range(32)]:
            future.result()  # surfaces any construction failure

    for name in ("ask_worker", "transfer_to_specialist", "final_answer"):
        assert registry.get(name) is not None
        assert callable(registry.handler_for(name))
