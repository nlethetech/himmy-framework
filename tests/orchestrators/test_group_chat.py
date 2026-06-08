"""Behavioral tests for multi-agent at scale: GroupChat/selector + parallel fan-out.

All offline (the zero-config stub). Covers: round-robin/LLM/callable selection over a
shared audited thread, per-speaker persona re-injection, final-answer/termination/
max-rounds stops, audited GROUP_*/FANOUT_* events, typed subtask/result contracts,
concurrent fan-out with a deterministic submission-order join, and graceful per-worker
failure.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from himmy import build_runtime
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core import HimmyError
from himmy.core.events import EventType
from himmy.orchestrators import (
    AgentTeam,
    TeamMember,
)
from himmy.orchestrators.group_chat import (
    CallableSelector,
    GroupChatOrchestrator,
    LLMSelector,
    RoundRobinSelector,
    SelectionContext,
    SubtaskResult,
    SubtaskSpec,
    fan_out,
)
from himmy.services.tools.registry import ToolRegistry
from tests.conftest import run_async


def _team_runtime() -> tuple[Any, ToolRegistry]:
    registry = ToolRegistry()
    runtime, _inf, _tools = build_runtime(tool_registry=registry)
    return runtime, registry


def _panel() -> AgentTeam:
    return AgentTeam(
        members=[
            TeamMember(
                name="alice", persona=Persona(name="alice", description="ALICE_ANALYST")
            ),
            TeamMember(
                name="bob", persona=Persona(name="bob", description="BOB_BUILDER")
            ),
            TeamMember(
                name="carol", persona=Persona(name="carol", description="CAROL_CRITIC")
            ),
        ],
        entry="alice",
    )


# --------------------------------------------------------------------- group chat


def test_round_robin_cycles_speakers_to_max_rounds() -> None:
    """Round-robin alternates speakers and runs to max_rounds (no final_answer)."""
    runtime, registry = _team_runtime()
    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=RoundRobinSelector(),
        max_rounds=5,
        allow_final_answer=False,
    )
    res = run_async(orch.run("discuss the plan"))
    assert res.speaker_order == ["alice", "bob", "carol", "alice", "bob"]
    assert res.stopped_reason == "max_rounds"
    assert res.round_count == 5


def test_each_speaker_runs_under_its_own_persona() -> None:
    """Per-turn persona re-injection: each speaker's prompt drives ITS turn.

    The offline stub echoes the system-prompt head, so a speaker's answer carries its
    own persona identity — proving the shared thread's SYSTEM message was swapped to
    the current speaker each round (not left as the first speaker's).
    """
    runtime, registry = _team_runtime()
    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=RoundRobinSelector(),
        max_rounds=3,
        allow_final_answer=False,
    )
    res = run_async(orch.run("plan"))
    spoken = {sp: (r.output_text or "") for sp, r in res.turns}
    assert "alice agent" in spoken["alice"]
    assert "bob agent" in spoken["bob"]
    assert "carol agent" in spoken["carol"]
    # And the final thread SYSTEM message belongs to the LAST speaker (carol), once.
    systems = [m for m in res.thread.messages if m.role == MessageRole.SYSTEM]
    assert len(systems) == 1
    assert "carol agent" in systems[0].content


def test_final_answer_ends_the_chat() -> None:
    """A speaker calling final_answer ends the chat immediately (stub binds it)."""
    runtime, registry = _team_runtime()
    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=RoundRobinSelector(),
        max_rounds=6,
        allow_final_answer=True,
    )
    res = run_async(orch.run("decide"))
    # Under the stub, the first speaker calls every bound tool (incl. final_answer).
    assert res.stopped_reason == "final_answer"
    assert res.final_speaker == "alice"
    assert res.round_count == 1
    assert res.output_text == "answer"


def test_terminate_when_predicate_stops_early() -> None:
    """A termination predicate ends the chat after the matching speaker's turn."""
    runtime, registry = _team_runtime()

    def stop_after_bob(speaker: str, _result: Any) -> bool:
        return speaker == "bob"

    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=RoundRobinSelector(),
        max_rounds=10,
        allow_final_answer=False,
        terminate_when=stop_after_bob,
    )
    res = run_async(orch.run("go"))
    assert res.stopped_reason == "terminated"
    assert res.final_speaker == "bob"
    assert res.speaker_order == ["alice", "bob"]


def test_callable_selector_drives_order() -> None:
    """A caller-supplied (manager) selector controls who speaks next."""
    runtime, registry = _team_runtime()
    picks = iter(["carol", "alice", "carol"])

    def manager(_ctx: SelectionContext) -> str:
        return next(picks)

    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=CallableSelector(manager),
        max_rounds=3,
        allow_final_answer=False,
    )
    res = run_async(orch.run("plan"))
    assert res.speaker_order == ["carol", "alice", "carol"]


def test_async_callable_selector_is_awaited() -> None:
    """An async manager selector is awaited correctly."""
    runtime, registry = _team_runtime()

    async def manager(ctx: SelectionContext) -> str:
        await asyncio.sleep(0)
        return "bob" if ctx.round_index == 0 else "alice"

    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=CallableSelector(manager),
        max_rounds=2,
        allow_final_answer=False,
    )
    res = run_async(orch.run("plan"))
    assert res.speaker_order == ["bob", "alice"]


def test_llm_selector_offline_is_deterministic_and_runs() -> None:
    """The LLM selector runs offline (enum schema -> first candidate) without keys."""
    runtime, registry = _team_runtime()
    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=LLMSelector(runtime),
        max_rounds=3,
        allow_final_answer=False,
    )
    res = run_async(orch.run("plan"))
    # Offline structured output picks the first enum value (alice) every round.
    assert res.speaker_order == ["alice", "alice", "alice"]
    assert res.stopped_reason == "max_rounds"


def test_group_chat_emits_audited_events() -> None:
    """STARTED first, one SELECTED+SPOKE per round, FINISHED last."""
    runtime, registry = _team_runtime()
    seen: list[str] = []

    async def on_event(e: Any) -> None:
        seen.append(e.event_type.value)

    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry,
        selector=RoundRobinSelector(),
        max_rounds=2,
        allow_final_answer=False,
        on_event=on_event,
    )
    run_async(orch.run("plan"))
    assert seen[0] == EventType.GROUP_CHAT_STARTED.value
    assert seen[-1] == EventType.GROUP_CHAT_FINISHED.value
    assert seen.count(EventType.GROUP_SPEAKER_SELECTED.value) == 2
    assert seen.count(EventType.GROUP_SPEAKER_SPOKE.value) == 2
    assert seen.count(EventType.GROUP_CHAT_FINISHED.value) == 1


def test_group_chat_events_reach_the_entity_spine() -> None:
    """Group transitions project into the EntityRegistry audit spine (provenance)."""
    registry_tools = ToolRegistry()
    runtime, _inf, _tools = build_runtime(tool_registry=registry_tools)
    orch = GroupChatOrchestrator(
        runtime,
        _panel(),
        registry_tools,
        selector=RoundRobinSelector(),
        max_rounds=2,
        allow_final_answer=False,
    )
    run_async(orch.run("plan"))
    records = runtime.entity_registry.list_by_kind("run_event")
    kinds = {r.payload.get("event_type") for r in records}
    assert EventType.GROUP_CHAT_STARTED.value in kinds
    assert EventType.GROUP_SPEAKER_SELECTED.value in kinds
    assert EventType.GROUP_SPEAKER_SPOKE.value in kinds


def test_unknown_speaker_raises() -> None:
    """Restricting speakers to a non-member fails fast at construction."""
    runtime, registry = _team_runtime()
    with pytest.raises(HimmyError):
        GroupChatOrchestrator(runtime, _panel(), registry, speakers=["alice", "ghost"])


def test_max_rounds_must_be_positive() -> None:
    """A non-positive max_rounds is rejected."""
    runtime, registry = _team_runtime()
    with pytest.raises(HimmyError):
        GroupChatOrchestrator(runtime, _panel(), registry, max_rounds=0)


# ------------------------------------------------------------------------ fan-out


def test_fan_out_runs_all_workers_and_joins_in_order() -> None:
    """Fan-out runs N workers concurrently and joins in SUBMISSION order."""
    runtime, _registry = _team_runtime()
    team = _panel()
    specs = [
        SubtaskSpec(worker="carol", objective="critique"),
        SubtaskSpec(worker="alice", objective="analyze"),
        SubtaskSpec(worker="bob", objective="build"),
    ]
    res = run_async(fan_out(runtime, team, specs))
    assert [r.worker for r in res.results] == ["carol", "alice", "bob"]
    assert all(isinstance(r, SubtaskResult) for r in res.results)
    assert res.ok is True
    assert len(res.threads) == 3
    # Each worker ran in its OWN sub-thread (distinct ids).
    assert len({t.thread_id for t in res.threads}) == 3


def test_fan_out_worker_failure_is_captured_not_raised() -> None:
    """An unknown worker yields a typed ok=False result; the others still succeed."""
    runtime, _registry = _team_runtime()
    team = _panel()
    specs = [
        SubtaskSpec(worker="alice", objective="ok"),
        SubtaskSpec(worker="ghost", objective="missing"),
    ]
    res = run_async(fan_out(runtime, team, specs))
    assert res.results[0].ok is True
    assert res.results[1].ok is False
    assert res.results[1].error is not None
    assert "ghost" in res.results[1].error
    assert res.ok is False  # any failure flips the aggregate


def test_fan_out_emits_audited_events() -> None:
    """FANOUT_STARTED + one FANOUT_WORKER_COMPLETED per worker + FANOUT_JOINED."""
    runtime, _registry = _team_runtime()
    team = _panel()
    seen: list[str] = []

    async def on_event(e: Any) -> None:
        seen.append(e.event_type.value)

    specs = [
        SubtaskSpec(worker="alice", objective="a"),
        SubtaskSpec(worker="bob", objective="b"),
    ]
    run_async(fan_out(runtime, team, specs, on_event=on_event))
    assert seen.count(EventType.FANOUT_STARTED.value) == 1
    assert seen.count(EventType.FANOUT_WORKER_COMPLETED.value) == 2
    assert seen.count(EventType.FANOUT_JOINED.value) == 1


def test_fan_out_empty_specs_join_is_total() -> None:
    """Fan-out over zero specs returns an empty result and still emits a join."""
    runtime, _registry = _team_runtime()
    team = _panel()
    seen: list[str] = []

    async def on_event(e: Any) -> None:
        seen.append(e.event_type.value)

    res = run_async(fan_out(runtime, team, [], on_event=on_event))
    assert res.results == []
    assert res.ok is True
    assert EventType.FANOUT_JOINED.value in seen


def test_fan_out_concurrency_cap_still_runs_all() -> None:
    """A concurrency cap bounds simultaneity but still completes every worker."""
    runtime, _registry = _team_runtime()
    team = _panel()
    specs = [SubtaskSpec(worker="alice", objective=f"task-{i}") for i in range(4)]
    res = run_async(fan_out(runtime, team, specs, concurrency=2))
    assert len(res.results) == 4
    assert all(r.worker == "alice" for r in res.results)
    assert res.ok is True


def test_fan_out_invalid_concurrency_raises() -> None:
    """A concurrency of 0 is rejected."""
    runtime, _registry = _team_runtime()
    team = _panel()
    with pytest.raises(HimmyError):
        run_async(
            fan_out(
                runtime,
                team,
                [SubtaskSpec(worker="alice", objective="x")],
                concurrency=0,
            )
        )


def test_fan_out_result_worker_is_pinned() -> None:
    """The joined result's worker name is the REAL worker, not a synthesized value."""
    runtime, _registry = _team_runtime()
    team = _panel()
    specs = [SubtaskSpec(worker="bob", objective="produce something")]
    res = run_async(fan_out(runtime, team, specs))
    assert res.results[0].worker == "bob"


def test_group_chat_fan_out_method_delegates() -> None:
    """GroupChatOrchestrator.fan_out runs the same typed parallel delegation."""
    runtime, registry = _team_runtime()
    orch = GroupChatOrchestrator(runtime, _panel(), registry)
    specs = [
        SubtaskSpec(worker="alice", objective="a"),
        SubtaskSpec(worker="bob", objective="b"),
    ]
    res = run_async(orch.fan_out(specs))
    assert [r.worker for r in res.results] == ["alice", "bob"]
    assert res.ok is True
