"""Runner + report integration for trajectory grading (deterministic, injected stub)."""

from __future__ import annotations

from typing import Any

from himmy.benchmark import BenchmarkRunner, BenchmarkSuite, ModelSpec, to_json
from himmy.benchmark.models import ModelScorecard, TaskScore, TrialResult
from himmy.benchmark.report import render_markdown
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


class _SequencedTools:
    """Calls each tool in ``tools`` (one per turn, in order), then answers ``answer``."""

    def __init__(self, tools: list[str], answer: str) -> None:
        self._tools = tools
        self._answer = answer
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return "scripted"

    async def generate(self, request: Any) -> InferenceResponse:
        idx = self.calls
        self.calls += 1
        if idx < len(self._tools):
            name = self._tools[idx]
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="",
                tool_calls=[
                    ToolCallRecord(tool_call_id=f"c{idx}", tool_name=name, args={})
                ],
                tool_returns=[
                    ToolReturnRecord(
                        tool_call_id=f"c{idx}", tool_name=name, content={"ok": True}
                    )
                ],
                input_tokens=1,
                output_tokens=1,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=self._answer,
            input_tokens=1,
            output_tokens=2,
        )


def _suite(trajectory: dict[str, Any]) -> BenchmarkSuite:
    return BenchmarkSuite.from_dict(
        {
            "name": "t",
            "tasks": [
                {
                    "id": "seq",
                    "category": "multistep",
                    "prompt": "discover then query",
                    "packs": ["data"],
                    "grade": {"type": "contains", "value": "done"},
                    "trajectory": trajectory,
                }
            ],
        }
    )


def _run(tools: list[str], trajectory: dict[str, Any]) -> ModelScorecard:
    factory = lambda spec: InferenceService(  # noqa: E731
        _SequencedTools(tools, "done")
    )
    runner = BenchmarkRunner(trials=2, inference_factory=factory, clock=lambda: 0.0)
    cards = run_async(runner.run(_suite(trajectory), [ModelSpec("ollama", "m")]))
    return cards[0]


def test_yaml_round_trip_carries_trajectory() -> None:
    traj = {"type": "tool_sequence", "value": ["sql_schema", "sql_query"]}
    suite = _suite(traj)
    assert suite.tasks[0].trajectory == traj


def test_runner_records_answer_and_trajectory_ok() -> None:
    # Correct path: sql_schema before sql_query → trajectory passes.
    card = _run(
        ["sql_schema", "sql_query"],
        {"type": "tool_sequence", "value": ["sql_schema", "sql_query"]},
    )
    trial = card.task_scores[0].trials[0]
    assert trial.answer_ok is True
    assert trial.trajectory_ok is True
    assert trial.correct is True
    assert trial.tools_called == ["sql_schema", "sql_query"]  # ordered, with repeats
    assert card.trajectory_failures == 0


def test_bad_trajectory_fails_trial_despite_right_answer() -> None:
    # Wrong order: answer still "done" but the trajectory grader fails the trial.
    card = _run(
        ["sql_query", "sql_schema"],
        {"type": "tool_sequence", "value": ["sql_schema", "sql_query"]},
    )
    trial = card.task_scores[0].trials[0]
    assert trial.answer_ok is True  # answer was right
    assert trial.trajectory_ok is False  # but the path was wrong
    assert trial.correct is False  # so the trial fails
    assert card.trajectory_failures == 2  # both trials


def test_no_trajectory_leaves_trajectory_ok_none() -> None:
    card = _run(["calculator"], {})
    trial = card.task_scores[0].trials[0]
    assert trial.trajectory_ok is None
    assert trial.correct is True  # answer-only grading
    assert card.trajectory_failures == 0


def test_max_tool_calls_bounds_runaway_loop() -> None:
    card = _run(
        ["calculator", "calculator", "calculator"],
        {"type": "max_tool_calls", "value": 2},
    )
    trial = card.task_scores[0].trials[0]
    assert trial.trajectory_ok is False  # 3 calls > 2


def test_report_shows_trajectory_failures_distinctly() -> None:
    # Hand-build a card with a clear trajectory failure (result produced, bad path).
    good = TrialResult(
        "seq",
        "done",
        ["sql_schema", "sql_query"],
        True,
        None,
        1.0,
        2,
        1,
        1,
        0.0,
        answer_ok=True,
        trajectory_ok=True,
    )
    bad = TrialResult(
        "seq",
        "done",
        ["sql_query"],
        False,
        None,
        1.0,
        1,
        1,
        1,
        0.0,
        answer_ok=True,
        trajectory_ok=False,
    )
    card = ModelScorecard(
        spec=ModelSpec("ollama", "m"),
        task_scores=[TaskScore("seq", "multistep", [good, bad])],
    )
    md = render_markdown([card], suite_name="core")
    assert "Traj✗" in md  # the column appears
    assert "Trajectory failures" in md  # the diagnostic section
    assert "seq 1/2" in md  # one of two trials had a bad path

    data = to_json([card], suite_name="core")
    model = data["models"][0]
    assert model["trajectory_failures"] == 1
    assert model["tasks"][0]["trajectory_failures"] == 1


def test_report_omits_trajectory_section_when_none() -> None:
    clean = TrialResult(
        "seq",
        "done",
        [],
        True,
        None,
        1.0,
        0,
        1,
        1,
        0.0,
        answer_ok=True,
        trajectory_ok=None,
    )
    card = ModelScorecard(
        spec=ModelSpec("ollama", "m"),
        task_scores=[TaskScore("seq", "multistep", [clean])],
    )
    md = render_markdown([card], suite_name="core")
    assert "Traj✗" not in md
    assert "Trajectory failures" not in md
