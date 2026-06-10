"""Structural guards for the standing `core` benchmark suite (no model needed).

These catch the failure modes that would silently weaken the benchmark itself: a
vacuous combinator grade (always-True), a network-requiring pack (flaky in CI), or an
approval-gated tool (can never execute non-interactively). They also pin the suite's
breadth: every capability category carries at least five tasks so per-category pass
rates are signal, not noise.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from himmy.benchmark import BenchmarkRunner, ModelSpec, default_suite
from himmy.benchmark.models import BenchmarkSuite
from tests.conftest import run_async

# Packs that work fully offline and need no approval — the only ones a CI-safe,
# reproducible suite may use.
_OFFLINE_PACKS = {"utils", "files", "data", "nepal", "agentic", "memory", "knowledge"}

# The six tasks the checked-in baseline gate pins; their definitions are load-bearing for
# `benchmarks/baseline.json` and must not change shape (no trajectory/team/judge added).
_GATE_TASK_IDS = {
    "arithmetic",
    "file_read",
    "sql_count",
    "multistep_file_math",
    "rag_grounded",
    "rx_synthesis_after_tool",
}

# Every capability category must reach this many tasks so its pass rate is meaningful.
_MIN_TASKS_PER_CATEGORY = 5


def _suite() -> BenchmarkSuite:
    return default_suite()


def test_core_suite_loads_with_real_breadth() -> None:
    suite = _suite()
    assert suite.name == "core"
    assert len(suite.tasks) >= 12  # a real suite, not a smoke test
    cats = {t.category for t in suite.tasks}
    # The capability categories the framework's value depends on.
    assert {"sql", "files", "multistep", "rag", "memory", "regression"} <= cats


def test_every_category_has_at_least_five_tasks() -> None:
    # Per-category pass rates are only meaningful with enough tasks; a thin category is
    # statistical noise. Guard that every category stays at >=5.
    counts = Counter(t.category for t in _suite().tasks)
    thin = {cat: n for cat, n in counts.items() if n < _MIN_TASKS_PER_CATEGORY}
    assert not thin, f"categories under {_MIN_TASKS_PER_CATEGORY} tasks: {thin}"


def test_task_ids_are_unique() -> None:
    ids = [t.id for t in _suite().tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids in the core suite"


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


def test_nepal_task_grades_match_toolkit_ground_truth() -> None:
    # The hard-coded nepal grade values must agree with the nepal toolkit's OWN functions,
    # so the suite can't silently drift from the tools it grades against.
    import datetime

    from himmy.benchmark.graders import grade
    from himmy.nepal.calendar import ad_to_bs, nepali_fiscal_year
    from himmy.nepal.language import normalize_nepali

    by_id = {t.id: t for t in _suite().tasks}

    bs_ny = ad_to_bs(datetime.date(2026, 4, 14))  # BS new year
    assert bs_ny.month_name("en").lower() == "baishakh"
    # The all_of grade needs both the BS year and the month name in the answer.
    answer = f"{bs_ny.year}-01-01 ({bs_ny.month_name('en')})"
    assert grade(by_id["nepal_date_new_year"].grade, answer)

    fy = nepali_fiscal_year(datetime.date(2026, 6, 5))
    assert grade(by_id["nepal_fiscal_year"].grade, f"The fiscal year is {fy}.")

    # 'Nepal' normalizes to 'nepal' (script-fold) — the transliterate grade value.
    assert grade(by_id["nepal_transliterate"].grade, normalize_nepali("Nepal"))


def test_baseline_gate_tasks_stay_single_agent_answer_graded() -> None:
    # The six pinned gate tasks must keep their simple shape — no trajectory/team/judge
    # block — or the baseline floor + compare_to_baseline semantics could shift.
    by_id = {t.id: t for t in _suite().tasks}
    for tid in _GATE_TASK_IDS:
        task = by_id[tid]
        assert not task.trajectory, f"gate task {tid} grew a trajectory block"
        assert not task.team, f"gate task {tid} grew a team block"
        assert not task.judge, f"gate task {tid} grew a judge block"


def test_every_task_runs_against_the_stub_provider() -> None:
    # Mechanical runnability of the WHOLE suite: against the offline stub provider every
    # task must build its runtime, materialize its files/sqlite fixtures, execute its
    # tools, and have its grader run — without any trial raising. (Stub answers need not
    # PASS grading; we assert the run completes and grading executed, i.e. zero errors.)
    from himmy.cli.provider import build_inference_for

    factory: Any = lambda spec: build_inference_for("stub", spec.model)  # noqa: E731
    runner = BenchmarkRunner(
        trials=1,
        inference_factory=factory,
        clock=lambda: 0.0,
        append_history=False,
    )
    suite = _suite()
    cards = run_async(runner.run(suite, [ModelSpec("stub", "m")]))
    card = cards[0]
    assert card.total_trials == len(suite.tasks)
    # No trial recorded an error → every fixture materialized and every grader ran.
    offenders = {ts.task_id: ts.errors for ts in card.task_scores if ts.errors}
    assert not offenders, f"tasks errored against the stub: {offenders}"
    # Grading actually executed for each task (a bool was produced per trial).
    for ts in card.task_scores:
        assert all(isinstance(tr.correct, bool) for tr in ts.trials)


def test_task_with_no_scoring_block_is_rejected_at_load() -> None:
    # A task with no grade, no trajectory, and no judge would pass every trial unchecked
    # (correct=True with nothing to check), silently inflating accuracy. from_dict must
    # reject it at load time rather than score it as a vacuous 100%. Finding 4.
    import pytest

    with pytest.raises(ValueError, match="no grade, trajectory, or judge"):
        BenchmarkSuite.from_dict(
            {"name": "bad", "tasks": [{"id": "blank", "prompt": "do a thing"}]}
        )
    # A trajectory-only task (no grade, no judge) is still valid — it has a scoring block.
    ok = BenchmarkSuite.from_dict(
        {
            "name": "ok",
            "tasks": [
                {
                    "id": "traj_only",
                    "prompt": "use a tool",
                    "trajectory": {"type": "tool_called", "value": "calculator"},
                }
            ],
        }
    )
    assert ok.tasks[0].id == "traj_only" and not ok.tasks[0].grade
