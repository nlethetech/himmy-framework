"""Append-only benchmark run history + per-model trend analysis.

Every scored benchmark run appends one JSON line per ``(model, suite)`` to
``benchmarks/history.jsonl`` (in the repo, committed): the git SHA it ran at, an ISO
timestamp, the model id, suite name, trials per task, the raw per-task pass/fail
outcomes, and the aggregate metrics. This is the *time series* behind the single-shot
``benchmarks/baseline.json`` gate — it answers "is the 7B drifting down over the last
ten runs?", not just "did this PR clear the floor?".

The file is JSONL on purpose: appends are atomic-ish (a single ``write`` of one line
under ``open(..., "a")``), so concurrent runs don't corrupt each other's records, and a
truncated/garbled tail line is skipped by the loader rather than crashing it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from himmy.core.ids import utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from himmy.benchmark.models import ModelScorecard

_LOG = logging.getLogger(__name__)

#: Default history file, alongside the checked-in baseline. Anchored to the package so a
#: source checkout writes into the repo's ``benchmarks/`` dir. See :func:`resolve_history_path`
#: for the override / cwd-fallback logic both the writer and the Studio reader share.
HISTORY_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "history.jsonl"

#: Env switch to disable history appends globally (CI lanes that shouldn't write back).
DISABLE_ENV = "HIMMY_BENCH_NO_HISTORY"

#: Env override for the history file location (absolute path). Set this when running the
#: CLI and Studio from different directories, or on a wheel install where the package dir
#: is not writable — both the writer and the Studio reader honour it, so they never split.
PATH_ENV = "HIMMY_BENCH_HISTORY_PATH"


def resolve_history_path(project_root: Path | None = None) -> Path:
    """The single history-file resolver shared by the writer and the Studio reader.

    Resolution order:

    1. the ``HIMMY_BENCH_HISTORY_PATH`` env override (absolute path) — the cross-cwd
       unifier: set it and BOTH the CLI writer and the Studio reader pin to the same file
       regardless of launch dir or install layout (the fix for a wheel install, where the
       package dir is read-only, or for CLI and Studio launched from different dirs);
    2. when a ``project_root`` is given (the Studio reader passes its launch cwd):
       ``project_root/benchmarks/history.jsonl`` — the reader is scoped to its launch
       directory, so a project that has never been benchmarked reads as "missing" rather
       than latching onto whatever sits in the installed package dir;
    3. otherwise :data:`HISTORY_PATH`, the package-anchored default the CLI writer uses on
       a source checkout.

    On a source checkout the CLI writer (no ``project_root`` ⇒ :data:`HISTORY_PATH` =
    ``<repo>/benchmarks/history.jsonl``) and a Studio launched from the repo root
    (``project_root`` = the same repo) resolve to the *same* file, so the History panel
    shows what the CLI wrote. When the two are launched from different directories (or
    himmy is wheel-installed), pin both with the env override.
    """
    override = os.environ.get(PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if project_root is not None:
        return project_root / "benchmarks" / "history.jsonl"
    return HISTORY_PATH


#: A metric is flagged as a regression when its latest-vs-previous delta crosses this
#: many *absolute* percentage points (e.g. accuracy 0.80 -> 0.69 is a 0.11 drop, past
#: 0.10) AND that delta also clears the run's noise floor (see below). It is an absolute
#: percentage-point delta, not a fraction of the previous value. error_rate is flagged
#: when it *rises* past the threshold; accuracy-type metrics when they *drop*.
DEFAULT_REGRESSION_THRESHOLD = 0.10

#: z for the noise-floor bound (one-sided 95%): a between-run accuracy delta of two
#: identical-quality runs has SD ~= sqrt(2*p*(1-p)/n), so a drop only clears noise when
#: it exceeds ``z * SD``. We require BOTH the absolute threshold above AND this
#: sample-size-aware bound, so small suites (6 tasks x 5 trials = 30 trials, where the
#: 0.10 threshold is ~1 sigma of pure noise) stop firing spurious regressions.
_NOISE_Z = 1.6448536269514722

#: Aggregate metric keys carried per run record.
_METRIC_KEYS = (
    "accuracy",
    "tool_call_accuracy",
    "irrelevance_accuracy",
    "error_rate",
    "p50_latency_s",
    "p95_latency_s",
    "mean_cost",
)
#: Metrics where a *lower* value is worse (a drop is a regression).
_HIGHER_IS_BETTER = ("accuracy", "tool_call_accuracy", "irrelevance_accuracy")
#: Metrics where a *higher* value is worse (a rise is a regression).
_LOWER_IS_BETTER = ("error_rate",)


def git_sha(*, short: bool = True) -> str | None:
    """Current git SHA via ``git rev-parse``, or ``None`` outside a repo / on error."""
    args = ["git", "rev-parse"]
    if short:
        args.append("--short=12")
    args.append("HEAD")
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            args,
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):  # git missing / timed out
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def history_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve whether to write history: explicit ``enabled`` wins, else the env switch.

    Appending is the default; set ``HIMMY_BENCH_NO_HISTORY=1`` (or pass
    ``enabled=False``) to turn it off.
    """
    if enabled is not None:
        return enabled
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in {"1", "true", "yes"}


def _record(
    card: ModelScorecard, *, suite_name: str, when: str, sha: str | None
) -> dict[str, Any]:
    """One JSON-able history line for a single model's scorecard."""
    return {
        "sha": sha,
        "when": when,
        "model": card.spec.name,
        "provider": card.spec.provider,
        "model_id": card.spec.model,
        "suite": suite_name,
        "trials": card.task_scores[0].n if card.task_scores else 0,
        "task_outcomes": {
            s.task_id: [t.correct for t in s.trials] for s in card.task_scores
        },
        "metrics": {
            "accuracy": card.accuracy,
            "tool_call_accuracy": card.tool_call_accuracy,
            # Irrelevance/abstention headline (None when the suite has no irrelevance
            # task). Tracked for trend/regression like the other accuracy-type metrics.
            "irrelevance_accuracy": card.irrelevance_accuracy,
            "error_rate": card.error_rate,
            "p50_latency_s": card.p50_latency,
            "p95_latency_s": card.p95_latency,
            "mean_cost": card.mean_cost,
        },
    }


def append_run(
    cards: Sequence[ModelScorecard],
    *,
    suite_name: str,
    path: Path | None = None,
    when: str | None = None,
    sha: str | None = None,
    enabled: bool | None = None,
) -> int:
    """Append one history line per scorecard; return the number of lines written.

    No-op (returns 0) when history is disabled. The git SHA is captured automatically
    (``None`` outside a repo) unless supplied. Each line is written with a single
    ``write`` call to an ``"a"``-mode handle so concurrent appends interleave cleanly.
    """
    if not history_enabled(enabled=enabled):
        return 0
    if not cards:
        return 0
    target = path or resolve_history_path()
    stamp = when or utc_now_iso()
    resolved_sha = sha if sha is not None else git_sha()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(_record(c, suite_name=suite_name, when=stamp, sha=resolved_sha))
        + "\n"
        for c in cards
    )
    with target.open("a", encoding="utf-8") as fh:
        fh.write(payload)
    return len(cards)


def load_history(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every well-formed history record (oldest first); skip corrupt lines.

    A truncated or garbled line (e.g. a half-written record from a crashed run) is
    logged at WARNING and skipped — a bad tail line never crashes the loader.
    """
    target = path or resolve_history_path()
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for lineno, raw in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            _LOG.warning("skipping corrupt history line %d in %s", lineno, target)
            continue
        if isinstance(obj, dict):
            records.append(obj)
        else:
            _LOG.warning("skipping non-object history line %d in %s", lineno, target)
    return records


@dataclass(frozen=True)
class MetricTrend:
    """Latest-vs-previous delta for one metric, with a regression flag."""

    metric: str
    latest: float | None
    previous: float | None
    delta: float | None
    regressed: bool


@dataclass(frozen=True)
class ModelTrend:
    """Trend for one ``(model, suite)``: latest vs the immediately prior run."""

    model: str
    suite: str
    latest_when: str
    previous_when: str | None
    trends: list[MetricTrend] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        """True if any tracked metric crossed the regression threshold."""
        return any(t.regressed for t in self.trends)

    @property
    def regressions(self) -> list[MetricTrend]:
        """The metrics that regressed (for a focused report)."""
        return [t for t in self.trends if t.regressed]


def _metric(record: dict[str, Any], key: str) -> float | None:
    value = (record.get("metrics") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _trials(record: dict[str, Any]) -> int:
    value = record.get("trials")
    return int(value) if isinstance(value, (int, float)) else 0


def _noise_floor(latest: float, previous: float, n: int) -> float:
    """Sample-size-aware bound a between-run proportion delta must clear to signal.

    Two runs of identical quality at rate ``p`` over ``n`` trials each produce a delta
    with SD ``sqrt(2*p*(1-p)/n)``; a real change must beat ``z`` of those SDs. We use the
    midpoint of the two rates for ``p`` (the H0 rate is unknown; the midpoint is the
    natural pooled estimate) and return ``z * SD``. With no trial count (``n <= 0``) the
    floor is ``0.0`` — the absolute threshold alone gates, preserving old behaviour for
    records written before ``trials`` was recorded.
    """
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, (latest + previous) / 2.0))
    return _NOISE_Z * math.sqrt(2.0 * p * (1.0 - p) / n)


def _trend_for(
    latest: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    threshold: float,
) -> list[MetricTrend]:
    # Trials per task in each run; the between-run noise of a pooled proportion shrinks
    # with the trial count, so we gate every flag on a sample-size-aware floor too.
    n = min(_trials(latest), _trials(previous)) if previous is not None else 0
    trends: list[MetricTrend] = []
    for key in _METRIC_KEYS:
        cur = _metric(latest, key)
        prev = _metric(previous, key) if previous is not None else None
        delta = (cur - prev) if (cur is not None and prev is not None) else None
        regressed = False
        if delta is not None and cur is not None and prev is not None:
            # A regression must clear BOTH the absolute percentage-point threshold AND
            # the run's binomial noise floor — so a 0.10 swing on a 30-trial gate run
            # (where 0.10 is ~1 sigma of pure noise) no longer fires a spurious flag.
            floor = max(threshold, _noise_floor(cur, prev, n))
            if key in _HIGHER_IS_BETTER and delta < -floor:
                regressed = True
            elif key in _LOWER_IS_BETTER and delta > floor:
                regressed = True
        trends.append(
            MetricTrend(
                metric=key,
                latest=cur,
                previous=prev,
                delta=delta,
                regressed=regressed,
            )
        )
    return trends


def compute_trends(
    records: Sequence[dict[str, Any]],
    *,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> list[ModelTrend]:
    """Per ``(model, suite)`` trend: latest run vs the previous one of the same pair.

    ``records`` is :func:`load_history` output (assumed oldest-first as appended). A
    metric regresses when an accuracy-type metric falls by more than ``threshold`` or
    ``error_rate`` rises by more than ``threshold``. Pairs with a single run still
    appear (no previous → no deltas, nothing flagged). Result is sorted by
    ``(model, suite)`` for stable output.
    """
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        model = str(rec.get("model", ""))
        suite = str(rec.get("suite", ""))
        if not model:
            continue
        by_key.setdefault((model, suite), []).append(rec)

    out: list[ModelTrend] = []
    for (model, suite), runs in sorted(by_key.items()):
        latest = runs[-1]
        previous = runs[-2] if len(runs) >= 2 else None
        out.append(
            ModelTrend(
                model=model,
                suite=suite,
                latest_when=str(latest.get("when", "")),
                previous_when=str(previous.get("when", "")) if previous else None,
                trends=_trend_for(latest, previous, threshold=threshold),
            )
        )
    return out


__all__ = [
    "HISTORY_PATH",
    "DISABLE_ENV",
    "PATH_ENV",
    "DEFAULT_REGRESSION_THRESHOLD",
    "resolve_history_path",
    "git_sha",
    "history_enabled",
    "append_run",
    "load_history",
    "compute_trends",
    "MetricTrend",
    "ModelTrend",
]
