"""Model benchmark: mathematically evaluate models/configs on the himmy framework.

Run a fixed :class:`~himmy.benchmark.models.BenchmarkSuite` against any number of
:class:`~himmy.benchmark.models.ModelSpec`s over N trials, and get a
:class:`~himmy.benchmark.models.ModelScorecard` per model — correctness with a Wilson
95% confidence interval, tool-call accuracy, and p50/p95 latency — so "is the 7B worth
it?" and "did the tool router help?" become measured, not guessed.

    from himmy.benchmark import BenchmarkRunner, ModelSpec, default_suite, render_markdown

    cards = await BenchmarkRunner(trials=5).run(
        default_suite(),
        [ModelSpec("ollama", "qwen2.5:0.5b-instruct"),
         ModelSpec("ollama", "qwen2.5:3b-instruct")],
    )
    print(render_markdown(cards))
"""

from __future__ import annotations

from pathlib import Path

from himmy.benchmark.baseline import (
    build_baseline,
    compare_to_baseline,
    load_baseline,
    subset_suite,
)
from himmy.benchmark.history import (
    append_run,
    compute_trends,
    load_history,
)
from himmy.benchmark.judge import JudgeVerdict, SameModelJudgeError
from himmy.benchmark.models import (
    BenchmarkSuite,
    BenchmarkTask,
    ModelScorecard,
    ModelSpec,
    TaskScore,
    TrialResult,
    compare_scorecards,
)
from himmy.benchmark.report import render_leaderboard, render_markdown, to_json
from himmy.benchmark.runner import BenchmarkRunner
from himmy.benchmark.stats import McNemarResult, mcnemar_exact, mcnemar_from_outcomes


def default_suite() -> BenchmarkSuite:
    """Load the packaged ``core`` benchmark suite."""
    return BenchmarkSuite.from_yaml(Path(__file__).parent / "suites" / "core.yaml")


def nepali_suite() -> BenchmarkSuite:
    """Load the packaged ``nepali`` language-understanding suite.

    The product's key model-selection question is "which model understands Nepali?".
    Most tasks are graded deterministically (transliteration round-trip and BS-calendar
    reasoning use the nepal toolkit's own functions for ground truth); the open-ended
    summarization task is judge-graded (reported, never gated).
    """
    return BenchmarkSuite.from_yaml(Path(__file__).parent / "suites" / "nepali.yaml")


def irrelevance_suite() -> BenchmarkSuite:
    """Load the packaged ``irrelevance`` abstention suite.

    Every task binds tools the prompt does NOT need and grades ``max_tool_calls: 0`` —
    the correct behaviour is to abstain (call no tool). Feeds the ``irrelevance_accuracy``
    headline metric, reported separately from tool-call accuracy (over-calling is a top
    small/open-model failure mode, scored first-class by BFCL). Reported, not gated.
    """
    return BenchmarkSuite.from_yaml(
        Path(__file__).parent / "suites" / "irrelevance.yaml"
    )


def bfcl_suite() -> BenchmarkSuite:
    """Load the packaged ``bfcl`` multi-turn + nested-argument tool-use suite.

    Two BFCL axes the core suite does not isolate: MULTI-TURN tasks (category
    ``multi_turn``) where a tool's result must feed a later call's argument (graded with
    the ``result_feeds_arg`` data-flow predicate), and NESTED / COMPLEX-ARG tasks (object/
    array arguments) graded with the argument-level matchers. Offline + self-contained like
    the rest of the benchmark. Reported, not gated.
    """
    return BenchmarkSuite.from_yaml(Path(__file__).parent / "suites" / "bfcl.yaml")


def multiagent_suite() -> BenchmarkSuite:
    """Load the packaged ``multiagent`` collaboration suite (handoff/delegation/chat).

    Reported, not gated: these tasks exercise the multi-agent orchestrators (the
    framework's differentiator) but are not part of the baseline gate subset.
    """
    return BenchmarkSuite.from_yaml(
        Path(__file__).parent / "suites" / "multiagent.yaml"
    )


__all__ = [
    "BenchmarkSuite",
    "BenchmarkTask",
    "ModelSpec",
    "TrialResult",
    "TaskScore",
    "ModelScorecard",
    "BenchmarkRunner",
    "render_markdown",
    "render_leaderboard",
    "to_json",
    "default_suite",
    "nepali_suite",
    "multiagent_suite",
    "irrelevance_suite",
    "bfcl_suite",
    "JudgeVerdict",
    "SameModelJudgeError",
    "load_baseline",
    "subset_suite",
    "compare_to_baseline",
    "build_baseline",
    "compare_scorecards",
    "McNemarResult",
    "mcnemar_exact",
    "mcnemar_from_outcomes",
    "append_run",
    "load_history",
    "compute_trends",
]
