"""Tests for the benchmark baseline gate (floors, re-baselining, gate config)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from himmy.benchmark import (
    build_baseline,
    compare_to_baseline,
    default_suite,
    load_baseline,
    subset_suite,
)

_REPO_BASELINE = Path(__file__).parents[2] / "benchmarks" / "baseline.json"


def _results(
    *,
    accuracy: float = 0.8,
    tool: float | None = 1.0,
    error_rate: float = 0.0,
    trials: int = 5,
    by_category: dict[str, float] | None = None,
    name: str = "ollama:m",
) -> dict[str, Any]:
    """A minimal ``to_json``-shaped results dict."""
    return {
        "suite": "core-gate",
        "trials": trials,
        "models": [
            {
                "name": name,
                "accuracy": accuracy,
                "tool_call_accuracy": tool,
                "error_rate": error_rate,
                "by_category": by_category or {},
            }
        ],
    }


def _baseline(
    *,
    acc_floor: float = 0.6,
    tool_floor: float | None = 0.8,
    err_ceiling: float = 0.2,
    min_trials: int = 5,
    cat_floors: dict[str, float] | None = None,
) -> dict[str, Any]:
    floors: dict[str, Any] = {"accuracy": acc_floor}
    if tool_floor is not None:
        floors["tool_call_accuracy"] = tool_floor
    if cat_floors:
        floors["by_category"] = cat_floors
    return {
        "schema": 1,
        "suite": "core-gate",
        "gate": {
            "tasks": ["arithmetic"],
            "trials": min_trials,
            "min_trials": min_trials,
        },
        "models": {
            "ollama:m": {"floors": floors, "ceilings": {"error_rate": err_ceiling}}
        },
    }


# --- compare_to_baseline ------------------------------------------------------------


def test_compare_passes_when_above_floors() -> None:
    assert compare_to_baseline(_results(), _baseline()) == []


def test_compare_fails_on_accuracy_below_floor() -> None:
    failures = compare_to_baseline(_results(accuracy=0.5), _baseline(acc_floor=0.6))
    assert any("accuracy 50%" in f and "60%" in f for f in failures)


def test_compare_fails_on_tool_call_regression_and_on_missing_metric() -> None:
    below = compare_to_baseline(_results(tool=0.5), _baseline(tool_floor=0.8))
    assert any("tool-call" in f for f in below)
    # A floor with no measured value (None) must fail, not silently pass.
    absent = compare_to_baseline(_results(tool=None), _baseline(tool_floor=0.8))
    assert any("tool-call" in f for f in absent)


def test_compare_fails_on_error_rate_above_ceiling() -> None:
    failures = compare_to_baseline(_results(error_rate=0.5), _baseline(err_ceiling=0.2))
    assert any("error rate" in f for f in failures)


def test_compare_fails_on_category_floor_and_missing_category() -> None:
    base = _baseline(cat_floors={"sql": 0.5})
    below = compare_to_baseline(_results(by_category={"sql": 0.2}), base)
    assert any("'sql'" in f for f in below)
    missing = compare_to_baseline(_results(by_category={}), base)
    assert any("'sql'" in f for f in missing)


def test_compare_fails_on_missing_model() -> None:
    failures = compare_to_baseline(_results(name="ollama:other"), _baseline())
    assert any("missing from benchmark results" in f for f in failures)


def test_compare_fails_on_too_few_trials() -> None:
    failures = compare_to_baseline(_results(trials=1), _baseline(min_trials=5))
    assert any("1 trial(s)" in f for f in failures)


# --- build_baseline -----------------------------------------------------------------


def test_build_baseline_derives_floors_with_margin() -> None:
    gate = {"tasks": ["arithmetic"], "trials": 5, "min_trials": 5}
    base = build_baseline(
        _results(accuracy=0.8, tool=1.0, error_rate=0.1),
        sha="abc123",
        margin=0.1,
        gate=gate,
    )
    entry = base["models"]["ollama:m"]
    assert entry["floors"]["accuracy"] == pytest.approx(0.7)
    assert entry["floors"]["tool_call_accuracy"] == pytest.approx(0.9)
    assert entry["ceilings"]["error_rate"] == pytest.approx(0.2)
    assert entry["measured"]["accuracy"] == pytest.approx(0.8)
    assert base["baselined_at"]["sha"] == "abc123"
    assert base["gate"] == gate
    # A freshly built baseline must gate its own run (round-trip sanity).
    assert (
        compare_to_baseline(_results(accuracy=0.8, tool=1.0, error_rate=0.1), base)
        == []
    )


def test_build_baseline_clamps_floors_and_validates_margin() -> None:
    base = build_baseline(_results(accuracy=0.05, error_rate=0.95), margin=0.2)
    entry = base["models"]["ollama:m"]
    assert entry["floors"]["accuracy"] == 0.0
    assert entry["ceilings"]["error_rate"] == 1.0
    with pytest.raises(ValueError):
        build_baseline(_results(), margin=1.5)


# --- subset_suite + the checked-in baseline file ------------------------------------


def test_subset_suite_keeps_order_and_rejects_unknown_ids() -> None:
    suite = subset_suite(default_suite(), ["file_read", "arithmetic"])
    assert [t.id for t in suite.tasks] == ["file_read", "arithmetic"]
    assert suite.name == "core-gate"
    with pytest.raises(ValueError, match="no_such_task"):
        subset_suite(default_suite(), ["no_such_task"])


def test_load_baseline_rejects_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline(bad)


def test_checked_in_baseline_is_valid_and_matches_the_core_suite() -> None:
    """Drift guard: the repo baseline must stay runnable against the core suite."""
    base = load_baseline(_REPO_BASELINE)
    assert base["schema"] == 1
    gate = base["gate"]
    assert int(gate["trials"]) >= 5 and int(gate["min_trials"]) >= 5
    # Every gate task must still exist in the packaged core suite.
    subset = subset_suite(default_suite(), [str(t) for t in gate["tasks"]])
    assert len(subset.tasks) == len(gate["tasks"]) >= 5
    # Floors are sane probabilities and the recorded provenance is present.
    for entry in base["models"].values():
        for value in entry["floors"].values():
            if isinstance(value, dict):  # by_category
                value = max(value.values(), default=0.0)
            assert 0.0 <= float(value) <= 1.0
        assert 0.0 <= float(entry["ceilings"]["error_rate"]) <= 1.0
    assert base["baselined_at"]["sha"]
    assert "ollama:qwen2.5:3b-instruct" in base["models"]
