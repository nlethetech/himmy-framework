"""Structural guards for the standing `core` benchmark suite (no model needed).

These catch the failure modes that would silently weaken the benchmark itself: a
vacuous combinator grade (always-True), a network-requiring pack (flaky in CI), or an
approval-gated tool (can never execute non-interactively).
"""

from __future__ import annotations

from pathlib import Path

from himmy.benchmark import default_suite
from himmy.benchmark.models import BenchmarkSuite

# Packs that work fully offline and need no approval — the only ones a CI-safe,
# reproducible suite may use.
_OFFLINE_PACKS = {"utils", "files", "data", "nepal", "agentic", "memory", "knowledge"}


def _suite() -> BenchmarkSuite:
    return default_suite()


def test_core_suite_loads_with_real_breadth() -> None:
    suite = _suite()
    assert suite.name == "core"
    assert len(suite.tasks) >= 12  # a real suite, not a smoke test
    cats = {t.category for t in suite.tasks}
    # The capability categories the framework's value depends on.
    assert {"sql", "files", "multistep", "rag", "memory", "regression"} <= cats


def test_every_combinator_grade_is_non_vacuous() -> None:
    # all_of/any_of with no `of` would pass vacuously — a silent false signal.
    def _check(spec: dict) -> None:
        if spec.get("type") in ("all_of", "any_of"):
            sub = spec.get("of")
            assert sub, f"combinator grade missing 'of': {spec}"
            for s in sub:
                _check(s)

    for task in _suite().tasks:
        _check(task.grade)


def test_suite_is_offline_and_non_gated() -> None:
    for task in _suite().tasks:
        assert set(task.packs) <= _OFFLINE_PACKS, (
            f"{task.id} uses a non-offline pack: {set(task.packs) - _OFFLINE_PACKS}"
        )


def test_regression_cases_are_present() -> None:
    # The documented small-model failure modes must stay codified.
    ids = {t.id for t in _suite().tasks}
    assert {
        "rx_sql_literal_recovery",
        "rx_todo_flat_array",
        "rx_synthesis_after_tool",
    } <= ids


def test_core_yaml_is_the_packaged_default() -> None:
    # default_suite() must read the shipped file, not drift from it.
    packaged = BenchmarkSuite.from_yaml(Path("himmy/benchmark/suites/core.yaml"))
    assert {t.id for t in packaged.tasks} == {t.id for t in _suite().tasks}
