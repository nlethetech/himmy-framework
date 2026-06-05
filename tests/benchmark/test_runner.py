"""Tests for the benchmark runner (deterministic, injected inference)."""

from __future__ import annotations

from typing import Any

from himmy.benchmark import BenchmarkRunner, BenchmarkSuite, ModelSpec
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


class _TwoTurn:
    """Turn 1 calls ``tool_name``; turn 2 returns ``answer`` (no tools)."""

    def __init__(self, tool_name: str, answer: str) -> None:
        self._tool = tool_name
        self._answer = answer
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return "scripted"

    async def generate(self, request: Any) -> InferenceResponse:
        self.calls += 1
        if self.calls == 1:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="",
                tool_calls=[
                    ToolCallRecord(tool_call_id="c1", tool_name=self._tool, args={})
                ],
                tool_returns=[
                    ToolReturnRecord(
                        tool_call_id="c1", tool_name=self._tool, content={"result": 391}
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


def _suite(grade_value: str, expect_tool: str | None) -> BenchmarkSuite:
    task: dict[str, Any] = {
        "id": "calc",
        "category": "arithmetic",
        "prompt": "compute it",
        "packs": ["utils"],
        "grade": {"type": "contains", "value": grade_value},
    }
    if expect_tool:
        task["expect_tools"] = [expect_tool]
    return BenchmarkSuite.from_dict({"name": "t", "tasks": [task]})


def _runner(factory: Any, trials: int = 3) -> BenchmarkRunner:
    return BenchmarkRunner(
        trials=trials,
        inference_factory=factory,
        clock=lambda: 0.0,
    )


def test_scores_correct_and_tool_ok() -> None:
    factory = lambda spec: InferenceService(  # noqa: E731
        _TwoTurn("calculator", "the answer is 391")
    )
    cards = run_async(
        _runner(factory).run(_suite("391", "calculator"), [ModelSpec("ollama", "m")])
    )
    card = cards[0]
    assert card.total_trials == 3
    assert card.accuracy == 1.0  # every trial graded correct
    assert card.tool_call_accuracy == 1.0  # the expected tool was called each time
    assert card.error_rate == 0.0


def test_wrong_answer_scores_zero() -> None:
    factory = lambda spec: InferenceService(  # noqa: E731
        _TwoTurn("calculator", "I have no idea")
    )
    cards = run_async(
        _runner(factory).run(_suite("391", "calculator"), [ModelSpec("ollama", "m")])
    )
    assert cards[0].accuracy == 0.0


def test_wrong_tool_scores_tool_miss() -> None:
    factory = lambda spec: InferenceService(  # noqa: E731
        _TwoTurn("web_search", "the answer is 391")
    )
    cards = run_async(
        _runner(factory).run(_suite("391", "calculator"), [ModelSpec("ollama", "m")])
    )
    assert cards[0].accuracy == 1.0  # answer correct
    assert cards[0].tool_call_accuracy == 0.0  # but it called the wrong tool


def test_exception_is_recorded_not_raised() -> None:
    def _boom(spec: Any) -> Any:
        raise RuntimeError("provider exploded")

    cards = run_async(
        _runner(_boom).run(_suite("391", "calculator"), [ModelSpec("ollama", "m")])
    )
    assert cards[0].accuracy == 0.0
    assert cards[0].error_rate == 1.0


def test_no_tool_expectation_excluded_from_tool_accuracy() -> None:
    factory = lambda spec: InferenceService(  # noqa: E731
        _TwoTurn("calculator", "the answer is 391")
    )
    cards = run_async(
        _runner(factory).run(_suite("391", None), [ModelSpec("ollama", "m")])
    )
    assert cards[0].tool_call_accuracy is None  # task expected no specific tool
