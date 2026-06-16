"""P1 self-learning: reputation math, the bound-tools reorder, and the hints adapter.

Covers the brain (``LearningService`` reputation math + best-effort swallow), the
behavioural lever (``ToolService.bound_tools`` reorder/annotate via a stubbed provider,
and the unchanged provider=None path), and the prompt-injection adapter
(``LearnedHintsContextAdapter`` returns a privacy-safe hint or None).
"""

from __future__ import annotations

from typing import Any

from himmy.config.agent_spec import AgentSpec
from himmy.core.events import EventType, RunEvent
from himmy.services.context.service import ContextService
from himmy.services.learning.adapter import LearnedHintsContextAdapter
from himmy.services.learning.service import (
    NEUTRAL_SCORE,
    LearningService,
    ToolReputation,
    ToolReputationProvider,
)
from himmy.services.storage.service import StorageService
from himmy.services.tools.models import ToolBackendKind, ToolDefinition
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


def _store_with(events: list[tuple[EventType, str]]) -> StorageService:
    """Build an in-memory store seeded with (event_type, tool_name) tool events."""
    store = StorageService()
    for event_type, tool_name in events:
        run_async(
            store.append_event(
                RunEvent(event_type=event_type, payload={"tool_name": tool_name})
            )
        )
    return store


# --------------------------------------------------------------------- reputation math
def test_unseen_tool_is_neutral_cold_start() -> None:
    """A tool with no recorded events scores the neutral prior (never punished)."""
    svc = LearningService(StorageService())
    rep = run_async(svc.get_tool_reputation(["brand_new"]))
    assert rep["brand_new"].score == NEUTRAL_SCORE
    assert rep["brand_new"].has_min_samples is False
    assert rep["brand_new"].total == 0


def test_min_sample_gate_keeps_neutral_below_threshold() -> None:
    """Below ``min_samples`` scored calls, a tool stays neutral even with a failure."""
    store = _store_with([(EventType.TOOL_FAILED, "flaky")])  # 1 call < default 3
    svc = LearningService(store)
    rep = run_async(svc.get_tool_reputation(["flaky"]))
    assert rep["flaky"].score == NEUTRAL_SCORE
    assert rep["flaky"].has_min_samples is False
    assert rep["flaky"].failed == 1  # the real count is still carried


def test_failing_tool_scores_low_once_sampled() -> None:
    """A mostly-failing, sufficiently-sampled tool gets a low score."""
    events = [(EventType.TOOL_FAILED, "wire")] * 4 + [(EventType.TOOL_COMPLETED, "wire")]
    svc = LearningService(_store_with(events))
    rep = run_async(svc.get_tool_reputation(["wire"]))
    assert rep["wire"].has_min_samples is True
    assert rep["wire"].score == 1 / 5  # completed / (completed + failed)
    assert rep["wire"].failed == 4


def test_all_success_tool_scores_high() -> None:
    """A tool that always completes scores a perfect 1.0."""
    svc = LearningService(_store_with([(EventType.TOOL_COMPLETED, "search")] * 5))
    rep = run_async(svc.get_tool_reputation(["search"]))
    assert rep["search"].score == 1.0
    assert rep["search"].has_min_samples is True


def _store_with_scoped(
    events: list[tuple[EventType, str, str | None]],
) -> StorageService:
    """Seed a store with ``(event_type, tool_name, workspace_id)`` events.

    ``workspace_id`` is stamped on the event so the durable backends denormalise it and a
    ``LearningService(workspace_id=...)`` reads only its own tenant's events back.
    """
    store = StorageService()
    for event_type, tool_name, workspace_id in events:
        run_async(
            store.append_event(
                RunEvent(
                    event_type=event_type,
                    payload={"tool_name": tool_name},
                    workspace_id=workspace_id,
                )
            )
        )
    return store


# ---------------------------------------------------------------- per-tenant scoping
def test_reputation_scoped_to_workspace_isolates_tenants() -> None:
    """Tool failures under tenant A do NOT drag down the SAME tool's score for tenant B.

    Seed 5 failures of ``wire`` under workspace ``A`` and 5 completions under ``B`` on ONE
    shared store. A LearningService scoped to A sees a low (all-failure) score; one scoped
    to B sees a perfect score — proving the per-tenant ``list_events(workspace_id=...)``
    filter, not a global aggregate.
    """
    store = _store_with_scoped(
        [(EventType.TOOL_FAILED, "wire", "A")] * 5
        + [(EventType.TOOL_COMPLETED, "wire", "B")] * 5
    )
    rep_a = run_async(
        LearningService(store, workspace_id="A").get_tool_reputation(["wire"])
    )
    rep_b = run_async(
        LearningService(store, workspace_id="B").get_tool_reputation(["wire"])
    )
    assert rep_a["wire"].has_min_samples is True
    assert rep_a["wire"].score == 0.0  # only A's 5 failures are in scope
    assert rep_a["wire"].failed == 5
    assert rep_b["wire"].score == 1.0  # only B's 5 completions are in scope
    assert rep_b["wire"].failed == 0


def test_tenant_b_run_does_not_see_tenant_a_failing_tool_as_unreliable() -> None:
    """A run under tenant B never inflicts tenant A's failures as a B-side unreliable hint.

    Step 3 of the task: seed TOOL_FAILED for tool X under subject A; prove a run under
    subject B does NOT see X as unreliable, while a run under A does.
    """
    store = _store_with_scoped([(EventType.TOOL_FAILED, "x", "A")] * 9)
    adapter_a = LearnedHintsContextAdapter(
        LearningService(store, workspace_id="A"), tool_names=["x"]
    )
    adapter_b = LearnedHintsContextAdapter(
        LearningService(store, workspace_id="B"), tool_names=["x"]
    )
    field_a = run_async(adapter_a.fetch("learned_hints", {}))
    field_b = run_async(adapter_b.fetch("learned_hints", {}))
    assert field_a is not None and 'tool "x"' in str(field_a.value)  # A sees the warning
    assert field_b is None  # B sees a clean tool — A's failures never crossed the boundary


def test_unscoped_reputation_reads_all_tenants_back_compat() -> None:
    """``workspace_id=None`` reads the whole stream — the historical (pre-tenancy) behaviour.

    Locks in that the scoping is opt-in: an unscoped service still aggregates across the
    workspace-stamped events exactly as it did before the column existed.
    """
    store = _store_with_scoped(
        [(EventType.TOOL_FAILED, "wire", "A")] * 4
        + [(EventType.TOOL_COMPLETED, "wire", "B")] * 1
    )
    rep = run_async(LearningService(store).get_tool_reputation(["wire"]))  # unscoped
    assert rep["wire"].failed == 4  # both tenants' events counted
    assert rep["wire"].completed == 1
    assert rep["wire"].score == 1 / 5


def _store_with_timestamps(
    events: list[tuple[EventType, str, str]],
) -> StorageService:
    """Seed a store with explicit ``(event_type, tool_name, timestamp)`` events.

    Pinning the timestamp makes the recency-window merge/cut in ``_recent_counts``
    deterministic regardless of clock resolution: the merge orders by the recorded
    timestamp, so the test controls exactly which events fall inside the window.
    """
    store = StorageService()
    for event_type, tool_name, timestamp in events:
        run_async(
            store.append_event(
                RunEvent(
                    event_type=event_type,
                    payload={"tool_name": tool_name},
                    timestamp=timestamp,
                )
            )
        )
    return store


def test_recency_window_recent_failures_not_diluted_by_old_completions() -> None:
    """The combined most-recent-``window`` cut means a long completion history can't bury
    a recent burst of failures — the core value of the recency window (round-2 fix).

    Seed >``window`` OLD completions then a small burst of NEWER failures and assert the
    score reflects only the recent window (here all-failures), exercising the merge-and-cut
    slice ``[: window]`` that the <=10-event reputation tests never reach.
    """
    window = 10
    old_completions = [
        (EventType.TOOL_COMPLETED, "wire", f"2026-06-15T00:00:{i:02d}.000000+00:00")
        for i in range(20)  # 20 old completions, well past the window
    ]
    recent_failures = [
        (EventType.TOOL_FAILED, "wire", f"2026-06-15T01:00:{i:02d}.000000+00:00")
        for i in range(window)  # window newer failures, all inside the cut
    ]
    svc = LearningService(
        _store_with_timestamps(old_completions + recent_failures), window=window
    )
    rep = run_async(svc.get_tool_reputation(["wire"]))
    # Only the most-recent ``window`` events survive the cut — all failures — so the old
    # completions are fully cut out and the score collapses to 0.0 (not diluted to ~0.67).
    assert rep["wire"].has_min_samples is True
    assert rep["wire"].failed == window
    assert rep["wire"].completed == 0
    assert rep["wire"].score == 0.0


def test_best_effort_swallow_on_broken_store() -> None:
    """A store that raises on every read degrades to the neutral result, never raising."""

    class _BrokenLog:
        async def list_events(self, *a: Any, **k: Any) -> list[RunEvent]:
            raise RuntimeError("store down")

        async def append_event(self, event: RunEvent) -> None:  # pragma: no cover
            raise RuntimeError("store down")

    svc = LearningService(_BrokenLog())
    rep = run_async(svc.get_tool_reputation(["x"]))
    assert rep["x"].score == NEUTRAL_SCORE
    assert rep["x"].has_min_samples is False


# ------------------------------------------------------------------- bound_tools reorder
def _registry(names: list[str]) -> ToolRegistry:
    reg = ToolRegistry()
    for n in names:
        reg.register(
            ToolDefinition(name=n, description=f"{n} tool", kind=ToolBackendKind.LOCAL)
        )
    return reg


class _StubProvider:
    """A sync reputation snapshot stub for the reorder hook."""

    def __init__(self, scores: dict[str, float], *, floor: float = 0.2) -> None:
        self._scores = scores
        self._floor = floor

    def score_for(self, tool_name: str) -> float:
        return self._scores.get(tool_name, NEUTRAL_SCORE)

    def is_unreliable(self, tool_name: str) -> bool:
        return self._scores.get(tool_name, NEUTRAL_SCORE) < self._floor


def test_bound_tools_reorders_flaky_after_reliable() -> None:
    """A frequently-failing tool sorts AFTER a reliable one (stable, deterministic)."""
    reg = _registry(["flaky", "reliable"])  # insertion order: flaky first
    provider = _StubProvider({"flaky": 0.1, "reliable": 1.0})
    svc = ToolService(reg, reputation_provider=provider)
    names = [bt.name for bt in svc.bound_tools()]
    assert names == ["reliable", "flaky"]


def test_bound_tools_annotates_unreliable_without_dropping() -> None:
    """A tool below the floor is annotated (not dropped); the model can still use it."""
    reg = _registry(["flaky", "reliable"])
    provider = _StubProvider({"flaky": 0.05, "reliable": 1.0})
    svc = ToolService(reg, reputation_provider=provider)
    bound = {bt.name: bt for bt in svc.bound_tools()}
    assert set(bound) == {"flaky", "reliable"}  # nothing dropped
    assert "failed frequently recently" in bound["flaky"].description
    assert "failed frequently recently" not in bound["reliable"].description


def test_bound_tools_stable_on_equal_scores() -> None:
    """Equal scores keep insertion order (stable sort → deterministic)."""
    reg = _registry(["a", "b", "c"])
    provider = _StubProvider({"a": 1.0, "b": 1.0, "c": 1.0})
    svc = ToolService(reg, reputation_provider=provider)
    assert [bt.name for bt in svc.bound_tools()] == ["a", "b", "c"]


def test_bound_tools_unchanged_when_no_provider() -> None:
    """provider=None → byte-for-byte the historical behaviour (no reorder, no annotation)."""
    reg = _registry(["flaky", "reliable"])
    svc = ToolService(reg)  # no reputation_provider
    bound = svc.bound_tools()
    assert [bt.name for bt in bound] == ["flaky", "reliable"]
    assert all("failed frequently" not in bt.description for bt in bound)


def test_reputation_provider_snapshot_roundtrip() -> None:
    """The provider mines history into a sync snapshot the reorder reads with no I/O."""
    # 9 failed + 1 completed → score 0.1, clearly below the default 0.2 floor.
    events = [(EventType.TOOL_FAILED, "wire")] * 9 + [(EventType.TOOL_COMPLETED, "wire")]
    store = _store_with(events)
    provider = ToolReputationProvider(LearningService(store))
    run_async(provider.refresh(["wire", "unseen"]))
    assert provider.score_for("wire") == 0.1
    assert provider.is_unreliable("wire") is True
    assert provider.score_for("unseen") == NEUTRAL_SCORE
    assert provider.is_unreliable("unseen") is False


def test_reputation_provider_emits_learning_applied_on_reorder() -> None:
    """A snapshot that would reorder tools emits a best-effort LEARNING_APPLIED event."""
    events = [(EventType.TOOL_FAILED, "wire")] * 9 + [(EventType.TOOL_COMPLETED, "wire")]
    store = _store_with(events)
    provider = ToolReputationProvider(LearningService(store), event_sink=store)
    run_async(provider.refresh(["wire", "search"]))  # wire 0.1, search neutral 1.0
    applied = run_async(store.list_events(event_type=EventType.LEARNING_APPLIED))
    assert len(applied) == 1
    assert applied[0].payload["tools_reordered"] == 1
    assert applied[0].payload["deprioritised_tools"] == ["wire"]


def test_reputation_provider_no_emit_when_flaky_already_last() -> None:
    """No reorder (the sub-1.0 tool is already last) → no LEARNING_APPLIED event fires."""
    events = [(EventType.TOOL_FAILED, "wire")] * 9 + [(EventType.TOOL_COMPLETED, "wire")]
    store = _store_with(events)
    provider = ToolReputationProvider(LearningService(store), event_sink=store)
    # search (neutral 1.0) is already before wire (0.1) → the stable sort is a no-op.
    run_async(provider.refresh(["search", "wire"]))
    applied = run_async(store.list_events(event_type=EventType.LEARNING_APPLIED))
    assert applied == []


# ----------------------------------------------------------------- learned-hints adapter
def test_adapter_returns_hint_for_unreliable_tool() -> None:
    """The adapter renders a privacy-safe string hint when a tool is unreliable."""
    events = [(EventType.TOOL_FAILED, "wire")] * 9 + [(EventType.TOOL_COMPLETED, "wire")]
    adapter = LearnedHintsContextAdapter(LearningService(_store_with(events)))
    field = run_async(adapter.fetch("learned_hints", {"tool_names": ["wire"]}))
    assert field is not None
    assert isinstance(field.value, str)  # plain string → no JSON-dump leak
    assert 'tool "wire"' in field.value
    assert "failed 9 of its last 10 calls" in field.value


def test_adapter_returns_none_when_all_healthy() -> None:
    """No unreliable tools → no block injected (returns None)."""
    adapter = LearnedHintsContextAdapter(
        LearningService(_store_with([(EventType.TOOL_COMPLETED, "search")] * 5))
    )
    assert run_async(adapter.fetch("learned_hints", {"tool_names": ["search"]})) is None


def test_adapter_returns_none_with_no_candidate_tools() -> None:
    """No candidate tools in scope → None (off-topic / no-tools run injects nothing)."""
    adapter = LearnedHintsContextAdapter(LearningService(StorageService()))
    assert run_async(adapter.fetch("learned_hints", {})) is None


def test_adapter_hint_carries_no_raw_prompt_or_args() -> None:
    """The hint contains only tool names + counts — never the user prompt or tool args."""
    events = [
        (EventType.TOOL_FAILED, "wire"),
        (EventType.TOOL_FAILED, "wire"),
        (EventType.TOOL_FAILED, "wire"),
    ]
    store = StorageService()
    for et, name in events:
        run_async(
            store.append_event(
                RunEvent(
                    event_type=et,
                    payload={
                        "tool_name": name,
                        "tool_args": {"secret_account": "1234-PRIVATE"},
                        "result": "raw error: SECRET-TOKEN leaked here",
                    },
                )
            )
        )
    adapter = LearnedHintsContextAdapter(LearningService(store))
    field = run_async(
        adapter.fetch("learned_hints", {"tool_names": ["wire"], "query": "PRIVATE-PROMPT"})
    )
    assert field is not None
    text = str(field.value)
    assert "1234-PRIVATE" not in text
    assert "SECRET-TOKEN" not in text
    assert "PRIVATE-PROMPT" not in text


def test_reputation_dataclass_total() -> None:
    """ToolReputation.total is completed + failed."""
    rep = ToolReputation(
        tool_name="t", completed=2, failed=3, score=0.4, has_min_samples=True
    )
    assert rep.total == 5


# ----------------------------------------------- end-to-end: real build_snapshot path
def test_bound_adapter_emits_hint_through_real_build_snapshot() -> None:
    """A bound adapter renders a hint when driven via make_task's real build spec.

    Regression for the dead hint half of the loop: the snapshot ``scope`` never carries
    the run's tool list (build_snapshot's scope = subject_id/task_id/metadata only), so the
    adapter must be PINNED with the run's bound tools (as the from_spec wiring now does via
    ``bind_tools``). We drive the exact build spec ``make_task`` emits — not a hand-built
    ``{"tool_names": [...]}`` scope — through a real ContextService.build_snapshot.
    """
    events = [(EventType.TOOL_FAILED, "wire")] * 9 + [(EventType.TOOL_COMPLETED, "wire")]
    store = _store_with(events)
    adapter = LearnedHintsContextAdapter(LearningService(store))
    adapter.bind_tools(["wire"])  # mirrors from_spec priming once the registry exists
    service = ContextService(storage_service=store, adapters=[adapter])

    spec = AgentSpec(name="x", self_learning=True, tools=["wire"])
    task = spec.make_task("do the thing")
    snapshot = run_async(
        service.build_snapshot(
            subject_id="x",
            build_spec=task.context["context_build_spec"],
            metadata=None,  # the real runtime passes ctx['context_metadata'] (no tool list)
        )
    )
    field = snapshot.fields.get("learned_hints")
    assert field is not None
    assert isinstance(field.value, str)
    assert 'tool "wire"' in field.value


def test_unbound_adapter_injects_nothing_through_real_build_snapshot() -> None:
    """An UN-bound adapter produces no field via the real path (the bug being fixed).

    Locks in WHY ``bind_tools`` is required: with no pinned tools and a scope that lacks
    ``tool_names`` (the real build_snapshot scope), the hint half resolves to nothing.
    """
    events = [(EventType.TOOL_FAILED, "wire")] * 9 + [(EventType.TOOL_COMPLETED, "wire")]
    store = _store_with(events)
    adapter = LearnedHintsContextAdapter(LearningService(store))  # NOT bound
    service = ContextService(storage_service=store, adapters=[adapter])

    spec = AgentSpec(name="x", self_learning=True, tools=["wire"])
    task = spec.make_task("do the thing")
    snapshot = run_async(
        service.build_snapshot(
            subject_id="x",
            build_spec=task.context["context_build_spec"],
            metadata=None,
        )
    )
    assert snapshot.fields.get("learned_hints") is None
