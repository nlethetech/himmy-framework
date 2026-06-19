"""End-to-end self-learning proof (CI-safe, no API key, deterministic).

Two LOCAL tools do the SAME job — ``lookup_reliable`` (returns a valid answer) and
``lookup_flaky`` (always raises -> ``TOOL_FAILED``). An :class:`AgentSpec` binds BOTH
with ``self_learning=True``. We fail the flaky tool until its reputation drops below the
``0.2`` floor and then assert, through the REAL himmy build path (not hand-built stubs):

1. after >=3 failures the flaky tool scores below the floor with ``has_min_samples`` True;
2. the bound-tool list reorders ``lookup_reliable`` ahead of ``lookup_flaky``;
3. a reliability hint NAMING the flaky tool is injected into the prompt;
4. a ``LEARNING_APPLIED`` event fires — and ONLY when the order actually changes.

Plus the three safety guarantees that keep the feature trustworthy:

* off-by-default — ``self_learning=False`` produces NONE of it and an identical order;
* neutral prior — a fresh tool with <3 calls is never demoted (a single early failure
  cannot bury an otherwise-fine tool);
* annotate, don't drop — the flaky tool is flagged but stays bound + callable.

The deterministic stub model (``examples.self_learning_demo.FirstToolManager``) always
reaches for the FIRST advertised tool, so the only lever that can change which tool a real
run executes is himmy's own reorder — which run 4 of the demo exercises live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.self_learning_demo import (
    _FLAKY,
    _RELIABLE,
    TOOLS_MODULE,
    FirstToolManager,
)
from himmy.config.agent_spec import AgentSpec
from himmy.core.events import EventType, RunEvent
from himmy.runtime.from_spec import build_runtime_for_spec
from himmy.services.inference.service import InferenceService
from himmy.services.learning import LearningService, ToolReputationProvider
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


@pytest.fixture(autouse=True)
def _store_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep any durable store under tmp_path and clear any exported DSN."""
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "storage.db"))


def _spec(*, self_learning: bool, tools: list[str]) -> AgentSpec:
    """The two-same-job-tools spec; ``self_learning`` toggles the whole loop."""
    return AgentSpec(
        name="lookup-agent",
        self_learning=self_learning,
        tools_module=TOOLS_MODULE,
        tools=tools,
    )


def _seed_failures(store: StorageService, tool: str, n: int) -> None:
    """Record ``n`` real ``TOOL_FAILED`` events for ``tool`` (what the loop would emit)."""
    for _ in range(n):
        run_async(
            store.append_event(
                RunEvent(event_type=EventType.TOOL_FAILED, payload={"tool_name": tool})
            )
        )


def _inference(store: StorageService) -> InferenceService:
    return InferenceService(FirstToolManager(), event_sink=store)


def _bound_names(runtime: object, names: list[str]) -> list[str]:
    return [bt.name for bt in runtime.tool_service.bound_tools(names)]  # type: ignore[attr-defined]


def _hint(runtime: object) -> str | None:
    adapter = runtime.context_service._adapters["learned_hints"]  # type: ignore[attr-defined]
    field = run_async(adapter.fetch("learned_hints", {}))
    return None if field is None else str(field.value)


# --------------------------------------------------------------------------- (1) score
def test_three_failures_drop_flaky_below_floor() -> None:
    """After 3 failures (0 completions) the flaky tool scores below the 0.2 floor."""
    store = StorageService()
    _seed_failures(store, _FLAKY, 3)

    rep = run_async(
        LearningService(store).get_tool_reputation([_FLAKY, _RELIABLE])
    )
    flaky = rep[_FLAKY]
    assert flaky.failed == 3
    assert flaky.completed == 0
    assert flaky.has_min_samples is True  # 3 == default min_samples
    assert flaky.score == 0.0
    assert flaky.score < 0.2  # below the unreliable floor
    # The never-called reliable tool keeps the neutral prior — it is not punished.
    assert rep[_RELIABLE].score == 1.0
    assert rep[_RELIABLE].has_min_samples is False


# ------------------------------------------------------------------------- (2) reorder
def test_bound_tools_reorder_reliable_ahead_of_flaky() -> None:
    """Through the real from_spec build, the flaky tool sorts AFTER the reliable one."""
    store = StorageService()
    _seed_failures(store, _FLAKY, 3)

    runtime, _registry = build_runtime_for_spec(
        _spec(self_learning=True, tools=[_FLAKY, _RELIABLE]),
        inference=_inference(store),
        storage=store,
    )
    run_async(runtime.reprime_self_learning())  # type: ignore[attr-defined]

    # Declared order is flaky-first; the reputation reorder must flip it.
    assert _bound_names(runtime, [_FLAKY, _RELIABLE]) == [_RELIABLE, _FLAKY]


# ---------------------------------------------------------------------------- (3) hint
def test_reliability_hint_names_the_flaky_tool() -> None:
    """A privacy-safe hint that NAMES the flaky tool is injected into the prompt."""
    store = StorageService()
    _seed_failures(store, _FLAKY, 3)

    runtime, _registry = build_runtime_for_spec(
        _spec(self_learning=True, tools=[_FLAKY, _RELIABLE]),
        inference=_inference(store),
        storage=store,
    )
    run_async(runtime.reprime_self_learning())  # type: ignore[attr-defined]

    hint = _hint(runtime)
    assert hint is not None
    assert f'tool "{_FLAKY}"' in hint  # names the flaky tool
    assert _RELIABLE not in hint  # the healthy tool is NOT flagged
    # Privacy: only tool names + counts, never raw error text from the handler.
    assert "unreachable" not in hint
    assert "failed 3 of its last 3 calls" in hint


# ------------------------------------------------------------- (4) LEARNING_APPLIED fires
def test_learning_applied_emitted_only_when_order_changes() -> None:
    """The reorder emits LEARNING_APPLIED; an already-sorted order emits nothing."""
    store = StorageService()
    _seed_failures(store, _FLAKY, 3)

    # flaky-first candidate order -> the stable sort MOVES it -> one event.
    provider = ToolReputationProvider(LearningService(store), event_sink=store)
    run_async(provider.refresh([_FLAKY, _RELIABLE]))
    applied = run_async(store.list_events(event_type=EventType.LEARNING_APPLIED))
    assert len(applied) == 1
    assert applied[0].payload["deprioritised_tools"] == [_FLAKY]
    assert applied[0].payload["tools_reordered"] == 1

    # reliable-first candidate order -> already sorted -> the stable sort is a no-op,
    # so NO further event fires (best-effort, only on a genuine visible move).
    provider2 = ToolReputationProvider(LearningService(store), event_sink=store)
    run_async(provider2.refresh([_RELIABLE, _FLAKY]))
    applied_after = run_async(store.list_events(event_type=EventType.LEARNING_APPLIED))
    assert len(applied_after) == 1  # unchanged — no new emit


# ----------------------------------------------------------- safety: off by default
def test_off_by_default_no_reorder_no_hint_no_provider() -> None:
    """self_learning=False -> identical order, no hint, no reputation provider at all."""
    store = StorageService()
    _seed_failures(store, _FLAKY, 9)  # plenty of failures on record...

    runtime, _registry = build_runtime_for_spec(
        _spec(self_learning=False, tools=[_FLAKY, _RELIABLE]),
        inference=_inference(store),
        storage=store,
    )
    # ...yet with the flag OFF none of the machinery is wired.
    assert runtime.tool_service._reputation_provider is None  # type: ignore[attr-defined]
    assert "learned_hints" not in getattr(
        runtime.context_service, "_adapters", {}
    )  # type: ignore[attr-defined]
    # Order is byte-identical to the declared (flaky-first) order — no reorder.
    assert _bound_names(runtime, [_FLAKY, _RELIABLE]) == [_FLAKY, _RELIABLE]
    # And no caution is appended to the flaky tool's description.
    bound = {bt.name: bt for bt in runtime.tool_service.bound_tools([_FLAKY, _RELIABLE])}  # type: ignore[attr-defined]
    assert "failed frequently" not in bound[_FLAKY].description


# ----------------------------------------------------------- safety: neutral prior
def test_fresh_tool_below_min_samples_is_never_demoted() -> None:
    """A tool with <3 calls (even an all-failure burst) keeps the neutral prior."""
    store = StorageService()
    _seed_failures(store, _FLAKY, 2)  # 2 < default min_samples of 3

    rep = run_async(LearningService(store).get_tool_reputation([_FLAKY]))
    assert rep[_FLAKY].failed == 2  # the real count is carried...
    assert rep[_FLAKY].has_min_samples is False  # ...but it is sub-threshold...
    assert rep[_FLAKY].score == 1.0  # ...so the score stays neutral (never punished).

    # Through the real build, a sub-threshold flaky tool is NOT reordered or hinted.
    runtime, _registry = build_runtime_for_spec(
        _spec(self_learning=True, tools=[_FLAKY, _RELIABLE]),
        inference=_inference(store),
        storage=store,
    )
    run_async(runtime.reprime_self_learning())  # type: ignore[attr-defined]
    assert _bound_names(runtime, [_FLAKY, _RELIABLE]) == [_FLAKY, _RELIABLE]  # no move
    assert _hint(runtime) is None  # no hint for a not-yet-proven-flaky tool


# ----------------------------------------------------------- safety: annotate, don't drop
def test_flaky_tool_is_flagged_but_still_bound_and_callable() -> None:
    """The flaky tool is annotated + demoted, but stays in the toolset and still runs."""
    store = StorageService()
    _seed_failures(store, _FLAKY, 3)

    runtime, _registry = build_runtime_for_spec(
        _spec(self_learning=True, tools=[_FLAKY, _RELIABLE]),
        inference=_inference(store),
        storage=store,
    )
    run_async(runtime.reprime_self_learning())  # type: ignore[attr-defined]

    bound = {bt.name: bt for bt in runtime.tool_service.bound_tools([_FLAKY, _RELIABLE])}  # type: ignore[attr-defined]
    # Not dropped — both tools are still advertised to the model.
    assert set(bound) == {_FLAKY, _RELIABLE}
    # The flaky tool carries a caution; the reliable one does not.
    assert "failed frequently recently" in bound[_FLAKY].description
    assert "failed frequently recently" not in bound[_RELIABLE].description

    # And it is genuinely still CALLABLE — executing it routes through the real
    # ToolService and raises -> a failed return (a dropped tool could not run at all).
    ret = run_async(
        runtime.tool_service.tool_executor()(_FLAKY, {"query": "x"})  # type: ignore[attr-defined]
    )
    assert ret.tool_name == _FLAKY
    assert ret.outcome == "failed"


# --------------------------------------------------- behavioural flip through the real loop
def _run_turn(store: StorageService, advertised: list[str]) -> str:
    """Run ONE real agent-loop turn with the deterministic stub; return the tool it called."""
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    runtime, _registry = build_runtime_for_spec(
        _spec(self_learning=True, tools=[_FLAKY, _RELIABLE]),
        inference=_inference(store),
        storage=store,
    )
    run_async(runtime.reprime_self_learning())  # type: ignore[attr-defined]
    loop = run_async(
        runtime.run_agent_loop(  # type: ignore[attr-defined]
            Persona(name="lookup", instructions=["Use a lookup tool to answer."]),
            Task(
                title="q",
                prompt="What is the capital of France?",
                context={"tool_names": list(advertised)},
            ),
            max_turns=4,
        )
    )
    called = [c.tool_name for turn in loop.turns for c in turn.tool_calls]
    return called[0] if called else "(none)"


def test_real_loop_tool_choice_flips_after_learning() -> None:
    """The stub model's ACTUAL tool choice flips flaky -> reliable once flaky is demoted.

    The stub always reaches for the FIRST advertised tool, so the flip is driven solely by
    himmy's reorder. With NO history the flaky-first toolset makes it pick (and fail) the
    flaky tool; after enough real failures the same toolset is re-sorted and it picks the
    reliable tool instead — behaviour change, not just internal bookkeeping.
    """
    store = StorageService()

    # Cold start: flaky is first, so the stub picks it and it fails for real.
    assert _run_turn(store, [_FLAKY, _RELIABLE]) == _FLAKY

    # Establish a sub-floor reputation with real failed loop turns (flaky-only toolset).
    for _ in range(3):
        assert _run_turn(store, [_FLAKY]) == _FLAKY

    flaky = run_async(LearningService(store).get_tool_reputation([_FLAKY]))[_FLAKY]
    assert flaky.has_min_samples is True and flaky.score < 0.2

    # Same model, same flaky-first toolset — now the reorder puts reliable first.
    assert _run_turn(store, [_FLAKY, _RELIABLE]) == _RELIABLE
