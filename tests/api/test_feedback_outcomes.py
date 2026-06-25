"""👍/👎 feedback → per-tool OUTCOME_SCORED (the positive-signal learning producer).

The verdict is attributed to the tools the run actually used. The events are recorded now
and visible in the Learning panel, but only blend into reputation when an agent opts into
``outcome_weight > 0`` (off by default).
"""

from __future__ import annotations

import pytest

from himmy.api.studio_canonical import (
    resolve_canonical_storage,
    set_canonical_storage_provider,
)
from himmy.api.studio_feedback import attribute_feedback_outcomes
from himmy.core.events import EventType, RunEvent
from himmy.services.learning.outcome import OutcomeRecorder, OutcomeSource
from himmy.services.learning.report import build_learning_report
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


@pytest.fixture()
def store() -> StorageService:
    """An in-memory store installed as the canonical storage for the producer to find."""
    s = StorageService()
    set_canonical_storage_provider(lambda: s)
    try:
        yield s
    finally:
        set_canonical_storage_provider(None)


def _tool_completed(store: StorageService, tool: str, *, trace_id: str) -> None:
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.TOOL_COMPLETED,
                trace_id=trace_id,
                payload={"tool_name": tool},
            )
        )
    )


def _outcome_events(store: StorageService) -> list[RunEvent]:
    return run_async(store.list_events(event_type=EventType.OUTCOME_SCORED))


def test_record_user_feedback_emits_one_per_distinct_tool() -> None:
    s = StorageService()
    rec = OutcomeRecorder(s)
    emitted = run_async(
        rec.record_user_feedback(1.0, tool_names=["a", "b", "a"])  # dedupe -> 2
    )
    assert emitted == 2
    events = run_async(s.list_events(event_type=EventType.OUTCOME_SCORED))
    assert {e.payload["tool_name"] for e in events} == {"a", "b"}
    assert all(e.payload["outcome_source"] == OutcomeSource.USER_FEEDBACK.value for e in events)
    assert all(e.payload["outcome_score"] == 1.0 for e in events)


def test_feedback_attributed_to_the_tools_the_run_used(store: StorageService) -> None:
    _tool_completed(store, "search", trace_id="t1")
    _tool_completed(store, "fetch", trace_id="t1")
    _tool_completed(store, "other", trace_id="t2")  # a different run — must NOT be scored

    emitted = run_async(
        attribute_feedback_outcomes(verdict="up", run_id="t1", trace_id="t1")
    )
    assert emitted == 2
    scored = {e.payload["tool_name"] for e in _outcome_events(store)}
    assert scored == {"search", "fetch"}


def test_thumbs_down_records_zero_score(store: StorageService) -> None:
    _tool_completed(store, "flaky", trace_id="t9")
    run_async(attribute_feedback_outcomes(verdict="down", run_id="t9", trace_id="t9"))
    events = _outcome_events(store)
    assert events and all(e.payload["outcome_score"] == 0.0 for e in events)


def test_unknown_verdict_is_a_noop(store: StorageService) -> None:
    _tool_completed(store, "x", trace_id="t1")
    assert run_async(attribute_feedback_outcomes(verdict="sideways", run_id="t1")) == 0
    assert _outcome_events(store) == []


def test_no_tools_no_outcomes(store: StorageService) -> None:
    # A run that called no tools yields nothing to attribute.
    assert run_async(attribute_feedback_outcomes(verdict="up", run_id="empty")) == 0
    assert _outcome_events(store) == []


def test_recorded_but_inert_until_opted_in(store: StorageService) -> None:
    """The feedback is recorded but does NOT change reputation at the default weight=0;
    it only blends when an agent opts into outcome_weight > 0."""
    # 5 successful calls -> operational score 1.0; then 3 thumbs-down feedbacks on it.
    for _ in range(5):
        _tool_completed(store, "tool", trace_id="run")
    for i in range(3):
        run_async(
            attribute_feedback_outcomes(verdict="down", run_id=f"r{i}", trace_id="run")
        )

    # Default weight 0: score stays operational (1.0) — feedback is inert.
    off = run_async(build_learning_report(store))
    assert off.tools[0].score == pytest.approx(1.0)

    # Opted in: the thumbs-down outcomes drag the blended score below 1.0.
    on = run_async(build_learning_report(store, outcome_weight=0.5))
    assert on.tools[0].score < 1.0
