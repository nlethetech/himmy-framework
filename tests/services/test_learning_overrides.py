"""Tests for Sprint-2 operator overrides (pin / mute / reset / clear + trust gate).

Covers the L1 core engine: the shared :func:`apply_override`, the durable
workspace-scoped :class:`OverrideStore` on the entity spine, the runtime behaviour
change through :class:`LearningService` / :class:`ToolReputationProvider`, the RESET
event cutoff, the trust-feedback weight gate, and the FEATURE-OFF byte-identical
invariant against Sprint 1.
"""

from __future__ import annotations

import pytest

from himmy.core.events import EventType, RunEvent
from himmy.core.ids import utc_now_iso
from himmy.entities.sqlite_registry import SqliteEntityRegistry
from himmy.services.learning.overrides import (
    TRUST_FEEDBACK_DEFAULT_WEIGHT,
    OverrideKind,
    OverrideSet,
    OverrideStore,
    ToolOverride,
    apply_override,
)
from himmy.services.learning.report import build_learning_report
from himmy.services.learning.service import (
    NEUTRAL_SCORE,
    LearningService,
    ToolReputation,
    ToolReputationProvider,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _emit(
    store: StorageService, tool: str, ok: bool, *, workspace: str | None = None
) -> None:
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.TOOL_COMPLETED if ok else EventType.TOOL_FAILED,
                workspace_id=workspace,
                payload={"tool_name": tool},
            )
        )
    )


def _flaky_rep(tool: str = "flaky") -> ToolReputation:
    """A sufficiently-sampled, genuinely-flaky reputation (score below the floor)."""
    return ToolReputation(
        tool_name=tool,
        completed=1,
        failed=9,
        score=0.1,
        has_min_samples=True,
        operational_score=0.1,
    )


# --------------------------------------------------------------------------- (1)
def test_apply_override_pin_and_mute_force_reliable() -> None:
    rep = _flaky_rep()
    for kind in (OverrideKind.PIN, OverrideKind.MUTE):
        ov = ToolOverride(tool_name="flaky", kind=kind)
        out = apply_override(rep, ov)
        assert out.score == NEUTRAL_SCORE
        assert out.has_min_samples is False
        # Underlying counts preserved so the report can still show the numbers.
        assert out.completed == 1 and out.failed == 9


def test_apply_override_none_and_reset_are_identity() -> None:
    rep = _flaky_rep()
    assert apply_override(rep, None) is rep
    reset = ToolOverride(
        tool_name="flaky", kind=OverrideKind.RESET, reset_cutoff=utc_now_iso()
    )
    # RESET is handled upstream (event cutoff), not a post-read reputation change.
    assert apply_override(rep, reset) is rep
    clear = ToolOverride(tool_name="flaky", kind=OverrideKind.CLEAR)
    assert apply_override(rep, clear) is rep


# --------------------------------------------------------------------------- (2)
def test_store_roundtrip_set_get_clear() -> None:
    store = OverrideStore(SqliteEntityRegistry())
    assert store.load().by_tool == {}

    store.set_override("flaky", OverrideKind.PIN, actor="cli", source="cli")
    loaded = store.load()
    assert loaded.for_tool("flaky") is not None
    assert loaded.for_tool("flaky").kind is OverrideKind.PIN

    # Re-pin then mute -> latest version wins.
    store.set_override("flaky", OverrideKind.MUTE)
    assert store.load().for_tool("flaky").kind is OverrideKind.MUTE

    # Clear -> back to no override (tombstone, not visible).
    store.clear_override("flaky")
    assert store.load().for_tool("flaky") is None


def test_store_version_chain_history_is_audited() -> None:
    registry = SqliteEntityRegistry()
    store = OverrideStore(registry)
    store.set_override("flaky", OverrideKind.PIN)
    store.set_override("flaky", OverrideKind.MUTE)
    store.clear_override("flaky")

    stable = OverrideStore._override_stable_id(None, "flaky")
    history = registry.get_history(stable)
    # Three immutable versions: pin, mute, clear — the who-changed-what-when trail.
    assert [r.version for r in history] == [1, 2, 3]
    assert [r.payload["kind"] for r in history] == ["pin", "mute", "clear"]


def test_store_tenant_isolation() -> None:
    store = OverrideStore(SqliteEntityRegistry())
    store.set_override("flaky", OverrideKind.PIN, workspace_id="A")

    assert store.load("A").for_tool("flaky") is not None
    # Workspace B cannot see A's override.
    assert store.load("B").for_tool("flaky") is None
    # The unscoped default subject is likewise isolated from A.
    assert store.load().for_tool("flaky") is None


def test_store_reset_stamps_cutoff() -> None:
    store = OverrideStore(SqliteEntityRegistry())
    ov = store.set_override("flaky", OverrideKind.RESET)
    assert ov is not None and ov.reset_cutoff is not None
    loaded = store.load()
    assert loaded.reset_cutoff_for("flaky") == ov.reset_cutoff


# --------------------------------------------------------------------------- (3)
def test_pinned_flaky_tool_is_reliable_through_provider() -> None:
    store = StorageService()
    _emit(store, "flaky", True)
    for _ in range(9):
        _emit(store, "flaky", False)  # 10% -> genuinely flaky

    overrides = OverrideSet(by_tool={"flaky": ToolOverride("flaky", OverrideKind.PIN)})
    learning = LearningService(store, override_set=overrides)
    provider = ToolReputationProvider(learning)
    run_async(provider.refresh(["flaky"]))

    assert provider.score_for("flaky") == NEUTRAL_SCORE
    assert provider.is_unreliable("flaky") is False


def test_muted_flaky_tool_is_reliable_through_provider() -> None:
    store = StorageService()
    for _ in range(10):
        _emit(store, "flaky", False)

    overrides = OverrideSet(by_tool={"flaky": ToolOverride("flaky", OverrideKind.MUTE)})
    learning = LearningService(store, override_set=overrides)
    provider = ToolReputationProvider(learning)
    run_async(provider.refresh(["flaky"]))

    assert provider.score_for("flaky") == NEUTRAL_SCORE
    assert provider.is_unreliable("flaky") is False


def test_pin_shows_as_pinned_and_not_flaky_in_report() -> None:
    store = StorageService()
    _emit(store, "flaky", True)
    for _ in range(9):
        _emit(store, "flaky", False)

    overrides = OverrideSet(by_tool={"flaky": ToolOverride("flaky", OverrideKind.PIN)})
    report = run_async(build_learning_report(store, override_set=overrides))
    row = {t.tool_name: t for t in report.tools}["flaky"]
    assert row.override == "pinned"
    assert row.flaky is False
    assert report.flaky_count == 0


# --------------------------------------------------------------------------- (4)
def test_reset_cutoff_drops_pre_cutoff_events() -> None:
    store = StorageService()
    # A burst of pre-cutoff failures that would make the tool flaky.
    for _ in range(10):
        _emit(store, "flaky", False)

    cutoff = utc_now_iso()  # everything at/before now is forgotten

    overrides = OverrideSet(
        by_tool={
            "flaky": ToolOverride("flaky", OverrideKind.RESET, reset_cutoff=cutoff)
        }
    )
    learning = LearningService(store, override_set=overrides)
    reps = run_async(learning.get_tool_reputation(["flaky"]))
    rep = reps["flaky"]
    # Pre-cutoff events dropped -> below min_samples -> neutral (forgotten, not punished).
    assert rep.has_min_samples is False
    assert rep.score == NEUTRAL_SCORE
    assert rep.completed == 0 and rep.failed == 0

    # New post-cutoff failures rebuild the reputation from now.
    for _ in range(10):
        _emit(store, "flaky", False)
    reps2 = run_async(learning.get_tool_reputation(["flaky"]))
    assert reps2["flaky"].failed == 10
    assert reps2["flaky"].score == pytest.approx(0.0)


def test_reset_does_not_filter_other_tools() -> None:
    store = StorageService()
    for _ in range(10):
        _emit(store, "other", False)
    cutoff = utc_now_iso()
    overrides = OverrideSet(
        by_tool={
            "flaky": ToolOverride("flaky", OverrideKind.RESET, reset_cutoff=cutoff)
        }
    )
    learning = LearningService(store, override_set=overrides)
    reps = run_async(learning.get_tool_reputation(["other"]))
    # ``other`` has no cutoff -> its events are NOT dropped.
    assert reps["other"].failed == 10


# --------------------------------------------------------------------------- (5)
def test_trust_gate_lifts_weight_only_when_on_and_spec_zero() -> None:
    store = StorageService()
    _emit(store, "t", True)

    # Trust ON, spec weight 0 -> lifted to the default.
    on = OverrideSet(trust_feedback=True)
    report_on = run_async(
        build_learning_report(store, outcome_weight=0.0, override_set=on)
    )
    assert report_on.trust_feedback is True
    assert report_on.outcome_weight == TRUST_FEEDBACK_DEFAULT_WEIGHT

    # Trust OFF -> stays at spec weight (0.0).
    off = OverrideSet(trust_feedback=False)
    report_off = run_async(
        build_learning_report(store, outcome_weight=0.0, override_set=off)
    )
    assert report_off.trust_feedback is False
    assert report_off.outcome_weight == 0.0

    # Trust ON but an explicit positive spec weight -> NOT overridden.
    report_explicit = run_async(
        build_learning_report(store, outcome_weight=0.7, override_set=on)
    )
    assert report_explicit.outcome_weight == pytest.approx(0.7)


# --------------------------------------------------------------------------- (6)
def test_feature_off_report_byte_identical_to_sprint1() -> None:
    store = StorageService()
    _emit(store, "flaky", True)
    for _ in range(9):
        _emit(store, "flaky", False)
    for _ in range(5):
        _emit(store, "clean", True)

    baseline = run_async(build_learning_report(store))
    empty = run_async(
        build_learning_report(store, override_set=OverrideSet())
    )
    none_set = run_async(build_learning_report(store, override_set=None))

    # Same rows, scores, flaky verdicts, weight — overrides add only an inert label.
    def _shape(r: object) -> list[tuple]:
        return [
            (t.tool_name, t.score, t.flaky, t.has_min_samples, t.override)  # type: ignore[attr-defined]
            for t in r.tools  # type: ignore[attr-defined]
        ]

    assert _shape(baseline) == [
        (t.tool_name, t.score, t.flaky, t.has_min_samples, None) for t in baseline.tools
    ]
    assert _shape(empty) == _shape(baseline)
    assert _shape(none_set) == _shape(baseline)
    assert empty.outcome_weight == baseline.outcome_weight
    assert empty.trust_feedback is False
    assert empty.flaky_count == baseline.flaky_count


def test_feature_off_provider_byte_identical_to_sprint1() -> None:
    store = StorageService()
    for _ in range(10):
        _emit(store, "flaky", False)

    sprint1 = LearningService(store)
    p1 = ToolReputationProvider(sprint1)
    run_async(p1.refresh(["flaky"]))

    sprint2_empty = LearningService(store, override_set=OverrideSet())
    p2 = ToolReputationProvider(sprint2_empty)
    run_async(p2.refresh(["flaky"]))

    assert p1.score_for("flaky") == p2.score_for("flaky")
    assert p1.is_unreliable("flaky") == p2.is_unreliable("flaky")
    assert p2.is_unreliable("flaky") is True  # still genuinely flaky with no override
