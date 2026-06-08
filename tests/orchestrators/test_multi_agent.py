"""Tests for the multi-agent orchestrator: handoff + supervisor delegation (offline)."""

from __future__ import annotations

from typing import Any

from himmy import build_runtime
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType
from himmy.orchestrators import AgentTeam, MultiAgentOrchestrator, TeamMember
from himmy.services.tools.registry import ToolRegistry
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
