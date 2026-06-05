"""Tests for benchmark scoring aggregation (TaskScore + ModelScorecard)."""

from __future__ import annotations

from himmy.benchmark.models import (
    BenchmarkSuite,
    ModelScorecard,
    ModelSpec,
    TaskScore,
    TrialResult,
)


def _trial(
    correct: bool, tool_ok: bool | None, lat: float, err: str | None = None
) -> TrialResult:
    return TrialResult(
        task_id="t",
        answer="a",
        tools_called=[],
        correct=correct,
        tool_ok=tool_ok,
        latency_s=lat,
        turns=1,
        input_tokens=1,
        output_tokens=1,
        cost=0.01,
        error=err,
    )


def test_task_score_pass_rate_and_tool_rate() -> None:
    score = TaskScore(
        task_id="t",
        category="calc",
        trials=[
            _trial(True, True, 1.0),
            _trial(True, False, 2.0),
            _trial(False, True, 3.0),
        ],
    )
    assert score.n == 3
    assert score.successes == 2
    assert abs(score.pass_rate - 2 / 3) < 1e-9
    lo, hi = score.pass_ci
    assert lo < score.pass_rate < hi
    assert abs(score.tool_call_rate - 2 / 3) < 1e-9  # 2 of 3 tool trials called it
    assert score.p50_latency == 2.0


def test_task_score_no_tool_expectation() -> None:
    score = TaskScore("t", "qa", [_trial(True, None, 1.0), _trial(False, None, 2.0)])
    assert score.tool_call_rate is None  # task expected no tool


def test_model_scorecard_aggregates() -> None:
    a = TaskScore("a", "calc", [_trial(True, True, 1.0), _trial(True, True, 2.0)])
    b = TaskScore("b", "qa", [_trial(False, None, 5.0), _trial(True, None, 9.0)])
    card = ModelScorecard(spec=ModelSpec("ollama", "m"), task_scores=[a, b])
    assert card.total_trials == 4
    assert card.accuracy == 0.75  # 3 of 4 correct
    lo, hi = card.accuracy_ci
    assert lo < 0.75 < hi
    assert card.tool_call_accuracy == 1.0  # both tool trials hit
    assert card.by_category() == {"calc": 1.0, "qa": 0.5}
    assert card.p95_latency >= card.p50_latency
    assert card.spec.name == "ollama:m"


def test_model_scorecard_error_rate() -> None:
    s = TaskScore(
        "a", "x", [_trial(False, None, 1.0, err="boom"), _trial(True, None, 1.0)]
    )
    card = ModelScorecard(ModelSpec("ollama", "m"), [s])
    assert card.error_rate == 0.5


def test_suite_from_dict_and_pack_union() -> None:
    suite = BenchmarkSuite.from_dict(
        {
            "name": "core",
            "tasks": [
                {
                    "id": "c",
                    "prompt": "calc",
                    "packs": ["utils"],
                    "grade": {"type": "contains", "value": "x"},
                },
                {
                    "id": "f",
                    "prompt": "file",
                    "packs": ["files", "utils"],
                    "expect_tools": ["read_file"],
                    "grade": {"type": "contains", "value": "y"},
                },
            ],
        }
    )
    assert suite.name == "core"
    assert len(suite.tasks) == 2
    assert suite.tasks[1].expect_tools == ["read_file"]
    assert suite.packs == ["utils", "files"]  # union, order-preserved
