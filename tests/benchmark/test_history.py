"""Tests for benchmark run history: append, load, trend/regression, disable, corrupt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from himmy.benchmark.history import (
    HISTORY_PATH,
    PATH_ENV,
    append_run,
    compute_trends,
    history_enabled,
    load_history,
    resolve_history_path,
)
from himmy.benchmark.models import ModelScorecard, ModelSpec, TaskScore, TrialResult


def _card(name: str, correct: list[bool]) -> ModelScorecard:
    trials = [
        TrialResult("t1", "x", ["calculator"], ok, True, 1.0 + i, 1, 1, 1, 0.0)
        for i, ok in enumerate(correct)
    ]
    return ModelScorecard(
        spec=ModelSpec("ollama", name),
        task_scores=[TaskScore("t1", "arithmetic", trials)],
    )


def test_append_writes_one_line_per_card(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    written = append_run(
        [_card("m1", [True, True]), _card("m2", [True, False])],
        suite_name="core",
        path=path,
        when="2026-06-10T00:00:00+00:00",
        sha="abc123",
        enabled=True,
    )
    assert written == 2
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["model"] == "ollama:m1"
    assert rec["suite"] == "core"
    assert rec["sha"] == "abc123"
    assert rec["trials"] == 2
    assert rec["task_outcomes"] == {"t1": [True, True]}
    assert rec["metrics"]["accuracy"] == 1.0


def test_append_is_appending_not_truncating(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    append_run([_card("m1", [True])], suite_name="core", path=path, enabled=True)
    append_run([_card("m1", [False])], suite_name="core", path=path, enabled=True)
    assert len(load_history(path)) == 2


def test_disable_switch_via_param(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    written = append_run(
        [_card("m1", [True])], suite_name="core", path=path, enabled=False
    )
    assert written == 0
    assert not path.exists()


def test_disable_switch_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.jsonl"
    monkeypatch.setenv("HIMMY_BENCH_NO_HISTORY", "1")
    # enabled=None defers to the env switch → disabled.
    written = append_run([_card("m1", [True])], suite_name="core", path=path)
    assert written == 0
    assert history_enabled(enabled=None) is False
    # An explicit enabled=True overrides the env switch.
    assert append_run([_card("m1", [True])], suite_name="c", path=path, enabled=True)


def test_loader_skips_corrupt_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.jsonl"
    good = json.dumps(
        {"model": "ollama:m1", "suite": "core", "metrics": {"accuracy": 0.9}}
    )
    path.write_text(good + "\n" + "{not valid json" + "\n" + good + "\n")
    with caplog.at_level("WARNING"):
        records = load_history(path)
    assert len(records) == 2  # the garbled middle line is skipped
    assert any("corrupt history line" in r.message for r in caplog.records)


def test_loader_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_history(tmp_path / "nope.jsonl") == []


def test_trend_flags_accuracy_drop() -> None:
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "metrics": {"accuracy": 0.9, "error_rate": 0.0},
        },
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t2",
            "metrics": {"accuracy": 0.7, "error_rate": 0.0},
        },
    ]
    trends = compute_trends(records, threshold=0.1)
    assert len(trends) == 1
    trend = trends[0]
    assert trend.previous_when == "t1" and trend.latest_when == "t2"
    assert trend.regressed
    acc = next(t for t in trend.trends if t.metric == "accuracy")
    assert acc.delta == pytest.approx(-0.2)
    assert acc.regressed


def test_trend_flags_error_rate_rise() -> None:
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "metrics": {"accuracy": 0.9, "error_rate": 0.0},
        },
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t2",
            "metrics": {"accuracy": 0.9, "error_rate": 0.3},
        },
    ]
    trend = compute_trends(records, threshold=0.1)[0]
    assert trend.regressed
    err = next(t for t in trend.trends if t.metric == "error_rate")
    assert err.regressed and err.delta == pytest.approx(0.3)


def test_trend_no_regression_within_threshold() -> None:
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "metrics": {"accuracy": 0.9},
        },
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t2",
            "metrics": {"accuracy": 0.85},
        },
    ]
    trend = compute_trends(records, threshold=0.1)[0]
    assert not trend.regressed  # 0.05 drop is within the 0.1 threshold


def test_trend_single_run_has_no_deltas() -> None:
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "metrics": {"accuracy": 0.9},
        },
    ]
    trend = compute_trends(records)[0]
    assert trend.previous_when is None
    assert not trend.regressed
    assert all(t.delta is None for t in trend.trends)


def test_trend_separates_models_and_suites() -> None:
    records = [
        {
            "model": "ollama:a",
            "suite": "core",
            "when": "t1",
            "metrics": {"accuracy": 0.9},
        },
        {
            "model": "ollama:a",
            "suite": "core",
            "when": "t2",
            "metrics": {"accuracy": 0.9},
        },
        {
            "model": "ollama:b",
            "suite": "core",
            "when": "t1",
            "metrics": {"accuracy": 0.5},
        },
    ]
    trends = compute_trends(records)
    keys = {(t.model, t.suite) for t in trends}
    assert keys == {("ollama:a", "core"), ("ollama:b", "core")}


def test_trend_noise_floor_suppresses_small_suite_swing() -> None:
    # 30 trials (the gate run: 6 tasks x 5 trials). A 0.10 accuracy drop is ~1 sigma of
    # pure binomial noise at this n, so it must NOT flag a regression on the absolute
    # threshold alone — the sample-size-aware floor suppresses it.
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "trials": 30,
            "metrics": {"accuracy": 0.70},
        },
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t2",
            "trials": 30,
            "metrics": {"accuracy": 0.58},
        },
    ]
    trend = compute_trends(records, threshold=0.10)[0]
    acc = next(t for t in trend.trends if t.metric == "accuracy")
    assert acc.delta == pytest.approx(-0.12)
    # delta -0.12 clears the 0.10 absolute threshold but NOT the ~0.21 noise floor at n=30.
    assert not acc.regressed
    assert not trend.regressed


def test_trend_noise_floor_allows_large_real_drop() -> None:
    # A 0.40 drop at 30 trials clears both the absolute threshold and the noise floor.
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "trials": 30,
            "metrics": {"accuracy": 0.85},
        },
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t2",
            "trials": 30,
            "metrics": {"accuracy": 0.45},
        },
    ]
    trend = compute_trends(records, threshold=0.10)[0]
    acc = next(t for t in trend.trends if t.metric == "accuracy")
    assert acc.regressed and trend.regressed


def test_trend_noise_floor_scales_with_trial_count() -> None:
    # The same 0.12 accuracy drop that is noise at n=30 IS a regression at a large n,
    # where the binomial noise floor shrinks below the absolute threshold.
    def _recs(n: int) -> list[dict[str, object]]:
        return [
            {
                "model": "ollama:m",
                "suite": "core",
                "when": "t1",
                "trials": n,
                "metrics": {"accuracy": 0.70},
            },
            {
                "model": "ollama:m",
                "suite": "core",
                "when": "t2",
                "trials": n,
                "metrics": {"accuracy": 0.58},
            },
        ]

    assert not compute_trends(_recs(30), threshold=0.10)[0].regressed
    assert compute_trends(_recs(2000), threshold=0.10)[0].regressed


def test_trend_error_rate_rise_gated_by_noise_floor() -> None:
    # error_rate (lower-is-better) is gated by the same noise floor on a rise.
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "trials": 12,
            "metrics": {"accuracy": 0.5, "error_rate": 0.0},
        },
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t2",
            "trials": 12,
            "metrics": {"accuracy": 0.5, "error_rate": 0.12},
        },
    ]
    trend = compute_trends(records, threshold=0.10)[0]
    err = next(t for t in trend.trends if t.metric == "error_rate")
    # 12-trial multiagent run: a 0.12 rise is within noise at this n.
    assert not err.regressed


def test_trend_without_trials_falls_back_to_absolute_threshold() -> None:
    # Records with no `trials` (legacy) keep the pure absolute-threshold behaviour.
    records = [
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t1",
            "metrics": {"accuracy": 0.70},
        },
        {
            "model": "ollama:m",
            "suite": "core",
            "when": "t2",
            "metrics": {"accuracy": 0.58},
        },
    ]
    trend = compute_trends(records, threshold=0.10)[0]
    acc = next(t for t in trend.trends if t.metric == "accuracy")
    # No trial count → noise floor 0.0 → the 0.12 drop trips the 0.10 absolute threshold.
    assert acc.regressed


def test_resolve_history_path_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HIMMY_BENCH_HISTORY_PATH wins over everything, so the CLI writer and the Studio
    # reader resolve to the same file regardless of cwd / install layout (findings 8, 12).
    target = tmp_path / "custom" / "history.jsonl"
    monkeypatch.setenv(PATH_ENV, str(target))
    assert resolve_history_path() == target
    assert resolve_history_path(tmp_path / "elsewhere") == target


def test_resolve_history_path_reader_is_scoped_to_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Studio reader passes its launch cwd as project_root; with no env override the
    # resolver returns that cwd's benchmarks/history.jsonl (whether or not it exists), so
    # a fresh project reads as "missing" instead of latching onto the installed package
    # dir. This keeps writer (repo root) and reader (repo root) on one file, and isolates
    # a project launched elsewhere. Findings 8, 12.
    monkeypatch.delenv(PATH_ENV, raising=False)
    proj = tmp_path / "proj"
    assert resolve_history_path(proj) == proj / "benchmarks" / "history.jsonl"


def test_resolve_history_path_writer_default_is_package_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The CLI writer passes no project_root, so the default is the package-anchored
    # HISTORY_PATH (the repo's benchmarks/ on a source checkout).
    monkeypatch.delenv(PATH_ENV, raising=False)
    assert resolve_history_path() == HISTORY_PATH


def test_writer_and_reader_share_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end: append via the writer's default resolution, read via the same resolver
    # from a DIFFERENT cwd, and still see the record — the divergence the findings flag.
    target = tmp_path / "benchmarks" / "history.jsonl"
    monkeypatch.setenv(PATH_ENV, str(target))
    # Writer uses resolve_history_path() when no explicit path is given.
    written = append_run([_card("m1", [True, True])], suite_name="core", enabled=True)
    assert written == 1
    # Reader (default path) resolves the same env-overridden file from anywhere.
    monkeypatch.chdir(tmp_path)
    records = load_history()
    assert len(records) == 1
    assert records[0]["model"] == "ollama:m1"


def test_round_trip_append_then_trends(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    append_run(
        [_card("m", [True, True, True, True])],
        suite_name="core",
        path=path,
        when="t1",
        enabled=True,
    )
    append_run(
        [_card("m", [True, False, False, False])],
        suite_name="core",
        path=path,
        when="t2",
        enabled=True,
    )
    trend = compute_trends(load_history(path), threshold=0.1)[0]
    assert trend.regressed  # accuracy 1.0 -> 0.25
