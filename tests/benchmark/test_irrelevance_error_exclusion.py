"""Regression test: errored irrelevance trials must NOT count as abstentions.

A provider/schema error (e.g. a 400) yields ``tools_called == []`` with ``error``
set — but the model never got to DECIDE to abstain, so counting it as a correct
abstention silently inflates the irrelevance score. The metric must exclude errored
trials from both numerator and denominator (error visibility is ``error_rate``'s job).
"""

from __future__ import annotations

from himmy.benchmark.models import TaskScore, TrialResult


def _trial(tools_called: list[str], *, error: str | None = None) -> TrialResult:
    return TrialResult(
        task_id="t",
        answer="",
        tools_called=tools_called,
        correct=not tools_called and error is None,
        tool_ok=None,
        latency_s=0.0,
        turns=1,
        input_tokens=1,
        output_tokens=1,
        cost=0.0,
        error=error,
    )


def test_abstained_excludes_errored_trials() -> None:
    score = TaskScore(
        task_id="irrel",
        category="irrelevance",
        trials=[
            _trial([]),  # clean abstention -> counts
            _trial([], error="400 invalid schema"),  # errored -> must NOT count
            _trial(["calc"]),  # clean over-call -> not an abstention
        ],
    )
    # Only the single CLEAN no-tool trial counts as a real abstention.
    assert score.abstained == 1
    # Denominator excludes the errored trial: 1 abstention / 2 clean trials = 0.5.
    assert score.abstention_rate == 0.5


def test_all_errored_irrelevance_has_no_abstention_rate() -> None:
    score = TaskScore(
        task_id="irrel",
        category="irrelevance",
        trials=[_trial([], error="400"), _trial([], error="500")],
    )
    assert score.abstained == 0
    assert score.abstention_rate is None  # no clean trial to measure
