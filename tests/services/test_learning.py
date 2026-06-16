"""P1 self-learning: reputation math, the bound-tools reorder, and the hints adapter.

Covers the brain (``LearningService`` reputation math + best-effort swallow), the
behavioural lever (``ToolService.bound_tools`` reorder/annotate via a stubbed provider,
and the unchanged provider=None path), and the prompt-injection adapter
(``LearnedHintsContextAdapter`` returns a privacy-safe hint or None).
"""

from __future__ import annotations

from typing import Any

from himmy.core.events import EventType, RunEvent
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


def test_best_effort_swallow_on_broken_store() -> None:
    """A store that raises on every read degrades to the neutral result, never raising."""

    class _BrokenLog:
        async def list_events(self, *a: Any, **k: Any) -> list[RunEvent]:
            raise RuntimeError("store down")

        async def count_events(self, *a: Any, **k: Any) -> int:
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
