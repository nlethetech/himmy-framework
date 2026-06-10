"""Baseline gate: compare a benchmark run against checked-in metric floors.

``benchmarks/baseline.json`` records, per model, the floors a change must clear
(accuracy, tool-call accuracy, optional per-category floors, an error-rate ceiling)
together with the git SHA / date / trial count of the run that produced them, plus the
``gate`` config (which task subset and how many trials the PR gate runs).

:func:`compare_to_baseline` takes the machine record from
:func:`~himmy.benchmark.report.to_json` and returns human-readable failures (empty
list == pass), so CI can fail a PR on a real regression instead of eyeballing nightly
numbers. :func:`build_baseline` re-derives a baseline from a fresh run (floors =
measured − margin); ``make bench-rebaseline`` wires it up. The thin CLI around both
lives in ``scripts/bench_gate.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from himmy.benchmark.models import BenchmarkSuite

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Bump when the baseline file layout changes incompatibly.
SCHEMA_VERSION = 1

#: Default safety margin between a measured value and the floor derived from it.
DEFAULT_MARGIN = 0.15


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a baseline file (raises on a malformed one)."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError(f"{path}: not a baseline file (needs a 'models' mapping)")
    return data


def subset_suite(suite: BenchmarkSuite, task_ids: Sequence[str]) -> BenchmarkSuite:
    """The sub-suite holding exactly ``task_ids`` (order kept; unknown ids raise)."""
    by_id = {t.id: t for t in suite.tasks}
    missing = [tid for tid in task_ids if tid not in by_id]
    if missing:
        raise ValueError(f"unknown task id(s) in gate config: {', '.join(missing)}")
    return BenchmarkSuite(
        name=f"{suite.name}-gate", tasks=[by_id[tid] for tid in task_ids]
    )


def compare_to_baseline(results: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Failure messages for every metric below its baseline floor (empty == pass).

    ``results`` is the dict :func:`~himmy.benchmark.report.to_json` produces. A model
    listed in the baseline but absent from the results is a failure (a gate that
    silently skips a model isn't a gate), and so is a run with fewer trials per task
    than the baseline's ``gate.min_trials`` (a thin run can clear floors by luck).
    """
    failures: list[str] = []
    min_trials = int((baseline.get("gate") or {}).get("min_trials", 1) or 1)
    got_trials = int(results.get("trials", 0) or 0)
    if got_trials < min_trials:
        failures.append(
            f"run used {got_trials} trial(s) per task; baseline requires "
            f">= {min_trials}"
        )
    by_name: dict[str, dict[str, Any]] = {
        str(m.get("name")): m for m in results.get("models", [])
    }
    for name, entry in baseline.get("models", {}).items():
        card = by_name.get(name)
        if card is None:
            failures.append(f"{name}: model missing from benchmark results")
            continue
        floors: dict[str, Any] = entry.get("floors") or {}
        ceilings: dict[str, Any] = entry.get("ceilings") or {}

        acc_floor = floors.get("accuracy")
        accuracy = float(card.get("accuracy") or 0.0)
        if acc_floor is not None and accuracy < float(acc_floor):
            failures.append(
                f"{name}: accuracy {accuracy:.0%} < baseline floor "
                f"{float(acc_floor):.0%}"
            )

        tool_floor = floors.get("tool_call_accuracy")
        if tool_floor is not None:
            tool = card.get("tool_call_accuracy")
            if tool is None or float(tool) < float(tool_floor):
                got = "—" if tool is None else f"{float(tool):.0%}"
                failures.append(
                    f"{name}: tool-call accuracy {got} < baseline floor "
                    f"{float(tool_floor):.0%}"
                )

        cats: dict[str, Any] = card.get("by_category") or {}
        for cat, cat_floor in (floors.get("by_category") or {}).items():
            val = cats.get(cat)
            if val is None or float(val) < float(cat_floor):
                got = "—" if val is None else f"{float(val):.0%}"
                failures.append(
                    f"{name}: category '{cat}' accuracy {got} < baseline floor "
                    f"{float(cat_floor):.0%}"
                )

        err_ceiling = ceilings.get("error_rate")
        error_rate = float(card.get("error_rate") or 0.0)
        if err_ceiling is not None and error_rate > float(err_ceiling):
            failures.append(
                f"{name}: error rate {error_rate:.0%} > baseline ceiling "
                f"{float(err_ceiling):.0%}"
            )
    return failures


def build_baseline(
    results: dict[str, Any],
    *,
    sha: str = "unknown",
    margin: float = DEFAULT_MARGIN,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a baseline from one run: floors = measured − ``margin`` (clamped).

    ``margin`` absorbs trial-to-trial noise so the gate only trips on a real
    regression; tighten it as nightly trial counts grow. ``gate`` (task subset +
    trials) is carried through verbatim so a re-baseline keeps the gate config.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    models: dict[str, Any] = {}
    for card in results.get("models", []):
        accuracy = float(card.get("accuracy") or 0.0)
        tool = card.get("tool_call_accuracy")
        error_rate = float(card.get("error_rate") or 0.0)
        floors: dict[str, Any] = {"accuracy": _floor(accuracy, margin)}
        if tool is not None:
            floors["tool_call_accuracy"] = _floor(float(tool), margin)
        models[str(card.get("name"))] = {
            "measured": {
                "accuracy": accuracy,
                "tool_call_accuracy": tool,
                "error_rate": error_rate,
                "by_category": dict(card.get("by_category") or {}),
            },
            "floors": floors,
            "ceilings": {"error_rate": _ceiling(error_rate, margin)},
        }
    return {
        "schema": SCHEMA_VERSION,
        "suite": str(results.get("suite", "")),
        "baselined_at": {
            "sha": sha,
            "date": datetime.now(UTC).date().isoformat(),
            "trials": int(results.get("trials", 0) or 0),
        },
        "gate": dict(gate or {}),
        "models": models,
    }


def _floor(measured: float, margin: float) -> float:
    return max(0.0, round(measured - margin, 3))


def _ceiling(measured: float, margin: float) -> float:
    return min(1.0, round(measured + margin, 3))


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_MARGIN",
    "load_baseline",
    "subset_suite",
    "compare_to_baseline",
    "build_baseline",
]
