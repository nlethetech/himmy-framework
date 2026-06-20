"""P2 outcome-based self-learning: the OUTCOME_SCORED event, the blended reputation math,
the trajectory-aware hints, and the off=byte-identical invariant.

Covers (1) the :class:`OutcomeRecorder` emitting a well-formed, tenant-scoped event from a
score / a judge verdict and degrading to a no-op fail-open; (2) the convex blend of the
operational and outcome signals in :class:`LearningService`, including the central invariant
that ``outcome_weight == 0`` (or the feature off) leaves ``score`` byte-identical to the P1
operational score even when outcome events exist; (3) the :class:`TrajectoryAdvisor` looping
/ first-call detection and its hint rendering; (4) the adapter composing P1 + P2 hints while
staying byte-identical to P1 when no advisor is wired.
"""

from __future__ import annotations

from typing import Any

from himmy.core.events import EventType, RunEvent
from himmy.services.learning.adapter import LearnedHintsContextAdapter
from himmy.services.learning.outcome import (
    OutcomeRecorder,
    OutcomeSource,
    clamp_score,
)
from himmy.services.learning.service import (
    NEUTRAL_SCORE,
    LearningService,
)
from himmy.services.learning.trajectory import (
    TrajectoryAdvisor,
    TrajectorySignal,
    render_trajectory_hints,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


# --------------------------------------------------------------------- store helpers
def _store_with(events: list[tuple[EventType, str]]) -> StorageService:
    """Seed an in-memory store with ``(event_type, tool_name)`` tool events."""
    store = StorageService()
    for event_type, tool_name in events:
        run_async(
            store.append_event(
                RunEvent(event_type=event_type, payload={"tool_name": tool_name})
            )
        )
    return store


def _add_outcome(
    store: StorageService,
    tool_name: str,
    score: float,
    *,
    source: OutcomeSource = OutcomeSource.JUDGE,
    workspace_id: str | None = None,
) -> None:
    """Append one OUTCOME_SCORED event attributed to ``tool_name``."""
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.OUTCOME_SCORED,
                workspace_id=workspace_id,
                payload={
                    "tool_name": tool_name,
                    "outcome_score": score,
                    "outcome_source": source.value,
                },
            )
        )
    )


# ------------------------------------------------------------------- clamp_score
def test_clamp_score_bounds_and_rejects_non_numeric() -> None:
    """Scores clamp into [0, 1]; a non-numeric or NaN value raises ValueError."""
    assert clamp_score(0.5) == 0.5
    assert clamp_score(1.5) == 1.0
    assert clamp_score(-0.2) == 0.0
    assert clamp_score("0.7") == 0.7  # numeric string coerces
    for bad in (None, "nope", float("nan")):
        try:
            clamp_score(bad)
        except ValueError:
            pass
        else:  # pragma: no cover - the assert below fails the test loudly
            raise AssertionError(f"clamp_score({bad!r}) should have raised")


# --------------------------------------------------------------- OutcomeRecorder
def test_recorder_emits_wellformed_event() -> None:
    """``record`` emits an OUTCOME_SCORED event with the denormalised tool_name + score."""
    store = StorageService()
    rec = OutcomeRecorder(store, workspace_id="ws1")
    emitted = run_async(
        rec.record(
            0.9,
            source=OutcomeSource.DETERMINISTIC_GRADER,
            tool_name="search",
            run_id="r1",
            rationale="exact match",
        )
    )
    assert emitted is True
    events = run_async(
        store.list_events(event_type=EventType.OUTCOME_SCORED, tool_name="search")
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.payload["outcome_score"] == 0.9
    assert ev.payload["outcome_source"] == "deterministic_grader"
    assert ev.payload["tool_name"] == "search"
    assert ev.payload["run_id"] == "r1"
    assert ev.workspace_id == "ws1"  # tenant scope stamped for the miner


def test_recorder_clamps_and_skips_bad_score() -> None:
    """An out-of-range score clamps; a non-numeric score is skipped (no event, no raise)."""
    store = StorageService()
    rec = OutcomeRecorder(store)
    assert run_async(rec.record(2.0, source=OutcomeSource.USER_FEEDBACK, tool_name="t"))
    assert (
        run_async(rec.record("bad", source=OutcomeSource.USER_FEEDBACK, tool_name="t"))
        is False
    )
    events = run_async(store.list_events(event_type=EventType.OUTCOME_SCORED))
    assert len(events) == 1  # only the clamped (good) one landed
    assert events[0].payload["outcome_score"] == 1.0  # 2.0 clamped to 1.0


def test_recorder_without_sink_is_noop() -> None:
    """A recorder with no sink never emits and returns False — pure fail-open no-op."""
    rec = OutcomeRecorder(None)
    assert rec.enabled is False
    assert run_async(rec.record(0.5, source=OutcomeSource.JUDGE, tool_name="t")) is False


def test_recorder_swallows_sink_failure() -> None:
    """A sink that raises on append degrades to a no-op (returns False), never propagates."""

    class _BrokenSink:
        async def append_event(self, event: RunEvent) -> None:
            raise RuntimeError("storage down")

    rec = OutcomeRecorder(_BrokenSink())
    assert run_async(rec.record(0.5, source=OutcomeSource.JUDGE, tool_name="t")) is False


def test_recorder_judge_verdict_skips_ungraded() -> None:
    """An ungraded judge verdict carries no signal and is NOT recorded; a graded one is."""
    from himmy.benchmark.judge import JudgeVerdict

    store = StorageService()
    rec = OutcomeRecorder(store)
    ungraded = JudgeVerdict(passed=False, score=0.0, rationale="timeout", ungraded=True)
    graded = JudgeVerdict(passed=True, score=0.8, rationale="good answer")
    assert run_async(rec.record_judge_verdict(ungraded, tool_name="t")) is False
    assert run_async(rec.record_judge_verdict(graded, tool_name="t")) is True
    events = run_async(store.list_events(event_type=EventType.OUTCOME_SCORED))
    assert len(events) == 1 and events[0].payload["outcome_score"] == 0.8


def test_recorder_run_level_outcome_has_no_tool_name() -> None:
    """An outcome recorded without a tool is run-level — no tool_name, blends into nothing."""
    store = StorageService()
    rec = OutcomeRecorder(store)
    run_async(rec.record(0.4, source=OutcomeSource.TRAJECTORY_GRADE))
    events = run_async(store.list_events(event_type=EventType.OUTCOME_SCORED))
    assert len(events) == 1 and "tool_name" not in events[0].payload


# ------------------------------------------------------------- blended math
def test_off_invariant_score_equals_operational_even_with_outcomes() -> None:
    """THE invariant: outcome_weight=0 ⇒ score == operational_score, byte-identical to P1.

    Seed a flaky tool (operational 1/5) AND strong outcome scores; with the weight off the
    blended ``score`` must be exactly the operational score and the outcome read must not
    even run (outcome_score stays None).
    """
    store = _store_with(
        [(EventType.TOOL_FAILED, "wire")] * 4 + [(EventType.TOOL_COMPLETED, "wire")]
    )
    for _ in range(5):
        _add_outcome(store, "wire", 1.0)
    svc = LearningService(store)  # default outcome_weight=0.0
    assert svc.outcome_enabled is False
    rep = run_async(svc.get_tool_reputation(["wire"]))["wire"]
    assert rep.operational_score == 1 / 5
    assert rep.score == 1 / 5  # blended == operational, to the bit
    assert rep.score == rep.operational_score
    assert rep.outcome_score is None  # outcome read never ran
    assert rep.outcome_samples == 0


def test_blend_mixes_outcome_when_weight_positive() -> None:
    """A positive weight convex-blends operational and the mean outcome score."""
    # Operational: 4 completed / 1 failed = 0.8. Outcome: mean(0.2, 0.2, 0.2) = 0.2.
    store = _store_with(
        [(EventType.TOOL_COMPLETED, "wire")] * 4 + [(EventType.TOOL_FAILED, "wire")]
    )
    for _ in range(3):
        _add_outcome(store, "wire", 0.2)
    svc = LearningService(store, outcome_weight=0.5)
    rep = run_async(svc.get_tool_reputation(["wire"]))["wire"]
    assert rep.operational_score == 0.8
    assert abs(rep.outcome_score - 0.2) < 1e-9
    assert rep.outcome_samples == 3
    assert abs(rep.score - (0.5 * 0.8 + 0.5 * 0.2)) < 1e-9  # 0.5


def test_blend_falls_back_to_operational_below_min_outcome_samples() -> None:
    """Below min_outcome_samples outcome events, the blend keeps the operational score."""
    store = _store_with([(EventType.TOOL_COMPLETED, "wire")] * 5)  # operational 1.0
    _add_outcome(store, "wire", 0.0)  # a single (sub-threshold) bad outcome
    svc = LearningService(store, outcome_weight=0.9, min_outcome_samples=3)
    rep = run_async(svc.get_tool_reputation(["wire"]))["wire"]
    assert rep.outcome_score is None  # 1 < 3 → no blend
    assert rep.score == 1.0  # stays operational


def test_blend_outcome_can_lower_a_well_running_tool() -> None:
    """The whole point: a tool that always RUNS but produces BAD answers scores down.

    Operational 1.0 (always completes) but outcomes all 0.0; with full outcome weight the
    blended score collapses to the outcome — surfacing the 'runs fine, answers poorly' tool
    that the P1 operational-only signal could never catch.
    """
    store = _store_with([(EventType.TOOL_COMPLETED, "wire")] * 5)
    for _ in range(4):
        _add_outcome(store, "wire", 0.0)
    svc = LearningService(store, outcome_weight=1.0)
    rep = run_async(svc.get_tool_reputation(["wire"]))["wire"]
    assert rep.operational_score == 1.0
    assert rep.outcome_score == 0.0
    assert rep.score == 0.0  # fully outcome-weighted


def test_outcome_scoped_to_workspace() -> None:
    """Outcome events are tenant-scoped exactly like the operational events."""
    store = StorageService()
    for et in [EventType.TOOL_COMPLETED] * 5:
        run_async(
            store.append_event(
                RunEvent(event_type=et, payload={"tool_name": "wire"}, workspace_id="A")
            )
        )
        run_async(
            store.append_event(
                RunEvent(event_type=et, payload={"tool_name": "wire"}, workspace_id="B")
            )
        )
    for _ in range(3):
        _add_outcome(store, "wire", 0.0, workspace_id="A")
        _add_outcome(store, "wire", 1.0, workspace_id="B")
    svc_a = LearningService(store, workspace_id="A", outcome_weight=1.0)
    svc_b = LearningService(store, workspace_id="B", outcome_weight=1.0)
    rep_a = run_async(svc_a.get_tool_reputation(["wire"]))["wire"]
    rep_b = run_async(svc_b.get_tool_reputation(["wire"]))["wire"]
    assert rep_a.outcome_score == 0.0  # only A's bad grades in scope
    assert rep_b.outcome_score == 1.0  # only B's good grades in scope


def test_cold_start_tool_neutral_regardless_of_outcomes() -> None:
    """An under-sampled tool stays neutral even with outcome events and weight on."""
    store = _store_with([(EventType.TOOL_FAILED, "wire")])  # 1 op call < min_samples
    for _ in range(5):
        _add_outcome(store, "wire", 0.0)
    svc = LearningService(store, outcome_weight=1.0)
    rep = run_async(svc.get_tool_reputation(["wire"]))["wire"]
    assert rep.has_min_samples is False
    assert rep.score == NEUTRAL_SCORE  # gated on operational samples, outcome ignored


# ----------------------------------------------------------- trajectory advisor
def _store_with_calls(names: list[str]) -> StorageService:
    """Seed a store with TOOL_CALLED events in the given chronological order."""
    store = StorageService()
    for name in names:
        run_async(
            store.append_event(
                RunEvent(
                    event_type=EventType.TOOL_CALLED, payload={"tool_name": name}
                )
            )
        )
    return store


def test_trajectory_detects_loop() -> None:
    """A back-to-back repeat run at/above the threshold is flagged as looping."""
    store = _store_with_calls(["a", "search", "search", "search", "search"])
    advisor = TrajectoryAdvisor(store, loop_threshold=3)
    signal = run_async(advisor.analyze())
    assert signal.looping_tool == "search"
    assert signal.max_repeat == 4


def test_trajectory_no_loop_below_threshold() -> None:
    """Alternating / short-repeat sequences produce no looping signal."""
    store = _store_with_calls(["a", "b", "a", "b", "a"])
    advisor = TrajectoryAdvisor(store, loop_threshold=3)
    signal = run_async(advisor.analyze())
    assert signal.looping_tool is None
    assert signal.first_call_tools == ("a",)


def test_render_trajectory_hints_loop_and_first_call() -> None:
    """Rendering emits a looping line and a don't-call-first line only for flagged tools."""
    signal = TrajectorySignal(
        looping_tool="search", max_repeat=4, first_call_tools=("flaky",)
    )
    lines = render_trajectory_hints(signal, unreliable_first_tools=("flaky",))
    assert any("looping" in ln and "search" in ln for ln in lines)
    assert any("avoid calling" in ln and "flaky" in ln for ln in lines)


def test_render_trajectory_hints_skips_healthy_first_tool() -> None:
    """A healthy opener (not in unreliable set) gets no don't-call-first hint."""
    signal = TrajectorySignal(
        looping_tool=None, max_repeat=0, first_call_tools=("healthy",)
    )
    assert render_trajectory_hints(signal, unreliable_first_tools=()) == []


def test_trajectory_read_failure_yields_no_signal() -> None:
    """A storage error in the advisor degrades to an empty signal, never raises."""

    class _BrokenLog:
        async def list_events(self, *a: Any, **k: Any) -> list[RunEvent]:
            raise RuntimeError("down")

    advisor = TrajectoryAdvisor(_BrokenLog())
    signal = run_async(advisor.analyze())
    assert signal == TrajectorySignal(
        looping_tool=None, max_repeat=0, first_call_tools=()
    )


# ----------------------------------------------------- adapter composition / off-invariant
def test_adapter_without_advisor_is_byte_identical_to_p1() -> None:
    """No advisor ⇒ the adapter renders exactly the P1 reliability note (no extra section)."""
    store = _store_with([(EventType.TOOL_FAILED, "x")] * 9)  # clearly unreliable
    adapter = LearnedHintsContextAdapter(
        LearningService(store), tool_names=["x"]
    )  # no trajectory_advisor
    field = run_async(adapter.fetch("learned_hints", {}))
    assert field is not None
    value = str(field.value)
    assert value.startswith("Reliability notes from past runs:")
    assert "Run-sequence notes:" not in value  # P2 section absent


def test_adapter_appends_trajectory_section_when_advisor_present() -> None:
    """With an advisor and a real loop, the adapter appends a Run-sequence notes section."""
    store = StorageService()
    # An unreliable tool for the P1 note...
    for _ in range(9):
        run_async(
            store.append_event(
                RunEvent(event_type=EventType.TOOL_FAILED, payload={"tool_name": "x"})
            )
        )
    # ...plus a loop on "x" in the call sequence for the P2 note.
    for _ in range(4):
        run_async(
            store.append_event(
                RunEvent(event_type=EventType.TOOL_CALLED, payload={"tool_name": "x"})
            )
        )
    adapter = LearnedHintsContextAdapter(
        LearningService(store),
        tool_names=["x"],
        trajectory_advisor=TrajectoryAdvisor(store, loop_threshold=3),
        event_sink=store,
    )
    field = run_async(adapter.fetch("learned_hints", {}))
    assert field is not None
    value = str(field.value)
    assert "Reliability notes from past runs:" in value
    assert "Run-sequence notes:" in value
    assert "looping" in value
    # The LEARNING_APPLIED event carries the trajectory hint count (P2 payload extension).
    applied = run_async(store.list_events(event_type=EventType.LEARNING_APPLIED))
    assert applied and applied[-1].payload.get("trajectory_hint_count", 0) >= 1


def test_adapter_hints_on_bad_outcome_with_zero_failures() -> None:
    """P2: a tool that RUNS fine but grades poorly gets an outcome-phrased hint.

    This is the case P1 is structurally blind to (failed == 0), so the legacy hint never
    fired. With the outcome blend on, the blended score drops below the floor and the
    adapter emits a 'ran without error but produced low-quality results' line.
    """
    store = _store_with([(EventType.TOOL_COMPLETED, "x")] * 5)  # 0 failures, op 1.0
    for _ in range(4):
        _add_outcome(store, "x", 0.0)  # consistently bad answers
    adapter = LearnedHintsContextAdapter(
        LearningService(store, outcome_weight=1.0), tool_names=["x"]
    )
    field = run_async(adapter.fetch("learned_hints", {}))
    assert field is not None
    value = str(field.value)
    assert "ran without error but produced low-quality results" in value
    assert 'tool "x"' in value
    assert "has failed" not in value  # not the P1 failure phrasing


def test_adapter_bad_outcome_invisible_when_weight_off() -> None:
    """Same bad-outcome history with the blend OFF ⇒ no hint (byte-identical to P1)."""
    store = _store_with([(EventType.TOOL_COMPLETED, "x")] * 5)
    for _ in range(4):
        _add_outcome(store, "x", 0.0)
    adapter = LearnedHintsContextAdapter(
        LearningService(store), tool_names=["x"]  # outcome_weight defaults 0
    )
    assert run_async(adapter.fetch("learned_hints", {})) is None


def test_adapter_trajectory_only_loop_on_healthy_tool() -> None:
    """A loop on an otherwise-healthy tool still produces a hint (no P1 note needed)."""
    store = StorageService()
    for _ in range(4):
        run_async(
            store.append_event(
                RunEvent(
                    event_type=EventType.TOOL_CALLED, payload={"tool_name": "healthy"}
                )
            )
        )
    adapter = LearnedHintsContextAdapter(
        LearningService(store),
        tool_names=["healthy"],
        trajectory_advisor=TrajectoryAdvisor(store, loop_threshold=3),
    )
    field = run_async(adapter.fetch("learned_hints", {}))
    assert field is not None
    value = str(field.value)
    assert "Reliability notes from past runs:" not in value  # no P1 note
    assert "Run-sequence notes:" in value and "looping" in value
