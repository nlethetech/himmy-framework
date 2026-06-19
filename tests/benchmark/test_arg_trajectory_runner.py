"""Runner integration for arg-level + parallel trajectory capture (FOUNDATION B).

A deterministic injected stub emits tool calls WITH args (and, in one case, two calls in
a single turn) so we can assert the runner records the full structured trajectory and the
arg/parallel graders score it.
"""

from __future__ import annotations

from typing import Any

from himmy.benchmark import BenchmarkRunner, BenchmarkSuite, ModelSpec
from himmy.benchmark.models import ModelScorecard
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


class _ArgTool:
    """First turn: call ``calculator`` with the given args; then answer."""

    def __init__(self, args: dict[str, Any], result: Any, answer: str) -> None:
        self._args = args
        self._result = result
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
                    ToolCallRecord(
                        tool_call_id="c0", tool_name="calculator", args=self._args
                    )
                ],
                tool_returns=[
                    ToolReturnRecord(
                        tool_call_id="c0",
                        tool_name="calculator",
                        content=self._result,
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
                    "id": "argtask",
                    "category": "arithmetic",
                    "prompt": "compute it",
                    "packs": ["utils"],
                    "grade": {"type": "contains", "value": "done"},
                    "trajectory": trajectory,
                }
            ],
        }
    )


def _run(tool: _ArgTool, trajectory: dict[str, Any]) -> ModelScorecard:
    factory = lambda spec: InferenceService(tool)  # noqa: E731
    runner = BenchmarkRunner(trials=1, inference_factory=factory, clock=lambda: 0.0)
    cards = run_async(runner.run(_suite(trajectory), [ModelSpec("ollama", "m")]))
    return cards[0]


def test_runner_captures_call_args() -> None:
    tool = _ArgTool({"expression": "17*23"}, 391, "done")
    card = _run(
        tool,
        {"type": "arg_equals", "tool": "calculator", "arg": "expression", "value": "17*23"},
    )
    trial = card.task_scores[0].trials[0]
    # The flat name-only view is preserved (back-compat).
    assert trial.tools_called == ["calculator"]
    # The structured trajectory carries the args the model produced.
    calls = trial.trajectory.calls
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert calls[0].args == {"expression": "17*23"}
    assert trial.trajectory_ok is True
    assert trial.correct is True


def test_runner_arg_grader_fails_on_wrong_args() -> None:
    # Right tool, WRONG args — the dominant small-model failure, now measurable.
    tool = _ArgTool({"expression": "1+1"}, 2, "done")
    card = _run(
        tool,
        {"type": "arg_equals", "tool": "calculator", "arg": "expression", "value": "17*23"},
    )
    trial = card.task_scores[0].trials[0]
    assert trial.answer_ok is True  # the answer text was "done"
    assert trial.trajectory_ok is False  # but the args were wrong
    assert trial.correct is False
    assert card.trajectory_failures == 1


def test_runner_captures_tool_result_for_result_contains() -> None:
    tool = _ArgTool({"expression": "x"}, "the answer is MARIGOLD", "done")
    card = _run(
        tool, {"type": "result_contains", "tool": "calculator", "value": "marigold"}
    )
    trial = card.task_scores[0].trials[0]
    assert trial.trajectory.calls[0].has_result is True
    assert trial.trajectory_ok is True


class _ParallelTool:
    """Emit two tool calls in a SINGLE assistant turn, then answer."""

    def __init__(self) -> None:
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
                    ToolCallRecord(tool_call_id="a", tool_name="calculator", args={"expression": "1+1"}),
                    ToolCallRecord(tool_call_id="b", tool_name="current_time", args={}),
                ],
                tool_returns=[
                    ToolReturnRecord(tool_call_id="a", tool_name="calculator", content=2),
                    ToolReturnRecord(tool_call_id="b", tool_name="current_time", content="now"),
                ],
                input_tokens=1,
                output_tokens=1,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="done",
            input_tokens=1,
            output_tokens=2,
        )


def test_runner_preserves_within_turn_parallelism() -> None:
    factory = lambda spec: InferenceService(_ParallelTool())  # noqa: E731
    runner = BenchmarkRunner(trials=1, inference_factory=factory, clock=lambda: 0.0)
    suite = _suite({"type": "parallel_in_turn", "value": ["calculator", "current_time"]})
    cards = run_async(runner.run(suite, [ModelSpec("ollama", "m")]))
    trial = cards[0].task_scores[0].trials[0]
    # Both calls collapsed into ONE turn — parallelism preserved (not flattened to 2 turns).
    assert trial.trajectory.max_parallel == 2
    assert len(trial.trajectory.turns[0]) == 2
    assert trial.trajectory_ok is True
    # Flat name-only view still lists both, in order, for the legacy graders.
    assert trial.tools_called == ["calculator", "current_time"]
