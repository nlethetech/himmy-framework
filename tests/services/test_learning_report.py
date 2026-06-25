"""Tests for the read-only learning report (``himmy learned`` / Studio panel data)."""

from __future__ import annotations

import pytest

from himmy.core.events import EventType, RunEvent
from himmy.services.learning.report import build_learning_report
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _emit(store: StorageService, tool: str, ok: bool, *, workspace: str | None = None) -> None:
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.TOOL_COMPLETED if ok else EventType.TOOL_FAILED,
                workspace_id=workspace,
                payload={"tool_name": tool},
            )
        )
    )


def test_empty_store_has_no_data() -> None:
    report = run_async(build_learning_report(StorageService()))
    assert report.has_data is False
    assert report.tools == []
    assert report.flaky_count == 0


def test_classifies_flaky_watch_reliable_and_undersampled() -> None:
    store = StorageService()
    _emit(store, "flaky", True)
    for _ in range(9):
        _emit(store, "flaky", False)  # 10% -> flaky (< 0.2 floor)
    for _ in range(8):
        _emit(store, "watch", True)
    for _ in range(2):
        _emit(store, "watch", False)  # 80% -> has failures but above floor
    for _ in range(5):
        _emit(store, "clean", True)  # 100% -> reliable
    for _ in range(2):
        _emit(store, "fresh", True)  # under min_samples (2 < 3)

    report = run_async(build_learning_report(store))
    by_name = {t.tool_name: t for t in report.tools}

    assert by_name["flaky"].flaky is True
    assert by_name["flaky"].score == pytest.approx(0.1)
    assert by_name["watch"].flaky is False
    assert by_name["watch"].failed == 2
    assert by_name["clean"].flaky is False and by_name["clean"].failed == 0
    # Under min_samples: neutral prior, never punished.
    assert by_name["fresh"].has_min_samples is False
    assert by_name["fresh"].flaky is False
    assert report.flaky_count == 1
    # Worst-first ordering: the flaky tool sorts ahead of the reliable ones.
    assert report.tools[0].tool_name == "flaky"


def test_recent_reorders_surfaced() -> None:
    store = StorageService()
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.LEARNING_APPLIED,
                payload={"tools_reordered": 2, "deprioritised_tools": ["a", "b"]},
            )
        )
    )
    report = run_async(build_learning_report(store))
    assert len(report.recent_reorders) == 1
    assert report.recent_reorders[0].deprioritised_tools == ["a", "b"]
    assert report.recent_reorders[0].tools_reordered == 2


def test_workspace_scoping_isolates_reputation() -> None:
    store = StorageService()
    # tenant A: tool fails a lot; tenant B: same-named tool is clean.
    for _ in range(9):
        _emit(store, "shared", False, workspace="A")
    _emit(store, "shared", True, workspace="A")
    for _ in range(10):
        _emit(store, "shared", True, workspace="B")

    report_a = run_async(build_learning_report(store, workspace_id="A"))
    report_b = run_async(build_learning_report(store, workspace_id="B"))
    assert report_a.tools[0].flaky is True  # A sees the failures
    assert report_b.tools[0].flaky is False  # B's reputation is clean — no cross-pollination
