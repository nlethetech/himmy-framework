"""Irrelevance/abstention suite + headline metric (QUICK WIN 5).

Over-calling — using a tool when none is needed — is a top small/open-model failure that
BFCL scores first-class. The irrelevance suite binds tools the task does NOT need and
grades ``max_tool_calls: 0``; ``irrelevance_accuracy`` is reported SEPARATELY from
``tool_call_accuracy``. A model that abstains scores 1.0; one that over-calls scores 0.0.
"""

from __future__ import annotations

from typing import Any

from himmy.benchmark import (
    BenchmarkRunner,
    BenchmarkSuite,
    ModelSpec,
    irrelevance_suite,
    render_markdown,
    to_json,
)
from himmy.benchmark.models import ModelScorecard, TaskScore, TrialResult
from himmy.benchmark.trajectory import RecordedCall, Trajectory
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


def test_irrelevance_suite_loads_and_is_well_formed() -> None:
    suite = irrelevance_suite()
    assert suite.name == "irrelevance"
    assert suite.tasks
    for task in suite.tasks:
        # Every task is an irrelevance task that binds tools but grades max_tool_calls: 0.
        assert task.category == "irrelevance"
        assert task.packs, f"{task.id} must bind tools the prompt doesn't need"
        assert task.trajectory == {"type": "max_tool_calls", "value": 0}


class _AbstainModel:
    """Never calls a tool — answers directly (the correct irrelevance behaviour)."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def resolve(self, model_key: str) -> str:
        return "scripted"

    async def generate(self, request: Any) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=self._answer,
            input_tokens=1,
            output_tokens=1,
        )


class _OverCaller:
    """Always reaches for a tool first (over-calling), then answers."""

    def __init__(self, tool: str, answer: str) -> None:
        self._tool = tool
        self._answer = answer
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return "scripted"

    async def generate(self, request: Any) -> InferenceResponse:
        idx = self.calls
        self.calls += 1
        if idx == 0:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="",
                tool_calls=[
                    ToolCallRecord(tool_call_id="c0", tool_name=self._tool, args={})
                ],
                tool_returns=[
                    ToolReturnRecord(
                        tool_call_id="c0", tool_name=self._tool, content={"ok": True}
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


def _suite() -> BenchmarkSuite:
    return BenchmarkSuite.from_dict(
        {
            "name": "irr",
            "tasks": [
                {
                    "id": "irr_knows",
                    "category": "irrelevance",
                    "prompt": "What is 2+2? Don't use tools.",
                    "packs": ["utils"],
                    "grade": {"type": "numeric", "value": 4},
                    "trajectory": {"type": "max_tool_calls", "value": 0},
                }
            ],
        }
    )


def _run(make_model: Any) -> ModelScorecard:
    # Build a FRESH model per trial — the stubs carry a per-run turn counter, so reusing
    # one instance across trials would let its counter advance past the over-call turn.
    factory = lambda spec: InferenceService(make_model())  # noqa: E731
    runner = BenchmarkRunner(trials=2, inference_factory=factory, clock=lambda: 0.0)
    cards = run_async(runner.run(_suite(), [ModelSpec("ollama", "m")]))
    return cards[0]


def test_abstaining_model_scores_irrelevance_accuracy_one() -> None:
    card = _run(lambda: _AbstainModel("4"))
    assert card.irrelevance_accuracy == 1.0
    assert card.has_irrelevance_tier is True
    assert card.irrelevance_total_trials == 2
    # The trial is also correct (right answer, abstained, max_tool_calls: 0 satisfied).
    assert card.accuracy == 1.0


def test_over_calling_model_scores_irrelevance_accuracy_zero() -> None:
    card = _run(lambda: _OverCaller("calculator", "4"))
    assert card.irrelevance_accuracy == 0.0
    # Over-calling also fails the trajectory grader (max_tool_calls: 0), so the trial fails.
    trial = card.task_scores[0].trials[0]
    assert trial.trajectory_ok is False
    assert trial.correct is False


def test_irrelevance_accuracy_is_separate_from_tool_call_accuracy() -> None:
    # An irrelevance task declares no `expect_tools`, so tool_call_accuracy is None there;
    # irrelevance_accuracy is its own number derived from abstention.
    card = _run(lambda: _AbstainModel("4"))
    assert card.tool_call_accuracy is None
    assert card.irrelevance_accuracy == 1.0


def test_no_irrelevance_tier_gives_none() -> None:
    # A non-irrelevance suite leaves irrelevance_accuracy None (column hidden).
    card = ModelScorecard(
        spec=ModelSpec("ollama", "m"),
        task_scores=[
            TaskScore(
                "t",
                "arithmetic",
                [TrialResult("t", "x", [], True, None, 1.0, 1, 1, 1, 0.0)],
            )
        ],
    )
    assert card.irrelevance_accuracy is None
    assert card.has_irrelevance_tier is False


def _irrelevance_card(abstained: bool) -> ModelScorecard:
    traj = (
        Trajectory()
        if abstained
        else Trajectory(turns=[[RecordedCall("calculator")]])
    )
    trial = TrialResult(
        "irr",
        "4",
        [] if abstained else ["calculator"],
        abstained,
        None,
        1.0,
        1,
        1,
        1,
        0.0,
        trajectory=traj,
        answer_ok=True,
        trajectory_ok=abstained,
    )
    return ModelScorecard(
        spec=ModelSpec("ollama", "m", label="m"),
        task_scores=[TaskScore("irr", "irrelevance", [trial])],
    )


def test_report_surfaces_abstain_column_and_json() -> None:
    good = _irrelevance_card(abstained=True)
    md = render_markdown([good], suite_name="irrelevance")
    assert "Abstain" in md  # the column appears only when the suite has irrelevance tasks
    assert "100%" in md

    data = to_json([good], suite_name="irrelevance")
    model = data["models"][0]
    assert model["irrelevance_accuracy"] == 1.0
    assert model["irrelevance_total_trials"] == 1


def test_report_omits_abstain_column_without_irrelevance_tasks() -> None:
    card = ModelScorecard(
        spec=ModelSpec("ollama", "m"),
        task_scores=[
            TaskScore(
                "t",
                "arithmetic",
                [TrialResult("t", "x", [], True, None, 1.0, 1, 1, 1, 0.0)],
            )
        ],
    )
    md = render_markdown([card], suite_name="core")
    assert "Abstain" not in md


def test_report_abstain_zero_for_over_caller() -> None:
    bad = _irrelevance_card(abstained=False)
    data = to_json([bad], suite_name="irrelevance")
    assert data["models"][0]["irrelevance_accuracy"] == 0.0
