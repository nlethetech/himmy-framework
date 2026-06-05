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

from himmy.benchmark.models import (
    BenchmarkSuite,
    BenchmarkTask,
    ModelScorecard,
    ModelSpec,
    TaskScore,
    TrialResult,
)
from himmy.benchmark.report import render_markdown, to_json
from himmy.benchmark.runner import BenchmarkRunner


def default_suite() -> BenchmarkSuite:
    """Load the packaged ``core`` benchmark suite."""
    return BenchmarkSuite.from_yaml(Path(__file__).parent / "suites" / "core.yaml")


__all__ = [
    "BenchmarkSuite",
    "BenchmarkTask",
    "ModelSpec",
    "TrialResult",
    "TaskScore",
    "ModelScorecard",
    "BenchmarkRunner",
    "render_markdown",
    "to_json",
    "default_suite",
]
