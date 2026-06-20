"""BFCL multi-turn + nested-argument suite, runner wiring, and leaderboard (FEATURE #4).

The ``bfcl`` suite isolates two BFCL axes: MULTI-TURN tasks (a tool result feeds the next
call's argument, graded by ``result_feeds_arg``) and NESTED/COMPLEX-ARG tasks (object/
array arguments, graded with the argument-level matchers). The runner must mark
argument-graded tasks (`uses_arg_grader`) so the leaderboard pools its arg-accuracy and
multi-turn columns. All offline + deterministic via injected stubs.
"""

from __future__ import annotations

from typing import Any

from himmy.benchmark import (
    BenchmarkRunner,
    BenchmarkSuite,
    ModelSpec,
    bfcl_suite,
)
from himmy.benchmark.models import (
    MULTI_TURN_CATEGORY,
    ModelScorecard,
)
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async

# --- suite shape --------------------------------------------------------------------


def test_bfcl_suite_loads_and_is_well_formed() -> None:
    suite = bfcl_suite()
    assert suite.name == "bfcl"
    assert suite.tasks
    categories = {t.category for t in suite.tasks}
    assert MULTI_TURN_CATEGORY in categories
    assert "nested_args" in categories
    # Every task declares a scoring block and binds the tools it exercises.
    for task in suite.tasks:
        assert task.grade or task.trajectory
        assert task.packs, f"{task.id} must bind tools"


def test_bfcl_suite_packs_resolve() -> None:
    # The union of packs the suite needs must all be real built-in packs (offline run).
    from himmy.toolkit.pack import BUILTIN_PACKS

    for pack in bfcl_suite().packs:
        assert pack in BUILTIN_PACKS, pack


def test_multi_turn_tasks_use_result_feeds_arg_or_sequence() -> None:
    suite = bfcl_suite()
    mt = [t for t in suite.tasks if t.category == MULTI_TURN_CATEGORY]
    assert len(mt) >= 3
    for task in mt:
        # A multi-turn task must assert a chained tool path (sequence) and most assert the
        # data-flow itself (result_feeds_arg, possibly nested under all_of).
        body = str(task.trajectory)
        assert "result_feeds_arg" in body or "tool_sequence" in body


# --- runner: a chained stub that carries a result into the next call's arg ----------


class _ChainModel:
    """Turn 0: call ``sql_query`` (returns 30). Turn 1: call ``calculator`` with 30*2.

    Models the multi-turn data-flow the suite measures — the second call's argument is
    derived from the first call's RESULT.
    """

    def __init__(self, *, carry: bool) -> None:
        # carry=False simulates the failure: the model re-derives/hallucinates the bridge.
        self._carry = carry
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
                tool_calls=[ToolCallRecord(tool_call_id="c0", tool_name="sql_query", args={"sql": "select quantity from animals where name='chickens'"})],
                tool_returns=[ToolReturnRecord(tool_call_id="c0", tool_name="sql_query", content={"rows": [{"quantity": 30}]})],
                input_tokens=1,
                output_tokens=1,
            )
        if idx == 1:
            expr = "30*2" if self._carry else "99*2"
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="",
                tool_calls=[ToolCallRecord(tool_call_id="c1", tool_name="calculator", args={"expression": expr})],
                tool_returns=[ToolReturnRecord(tool_call_id="c1", tool_name="calculator", content=60 if self._carry else 198)],
                input_tokens=1,
                output_tokens=1,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="60",
            input_tokens=1,
            output_tokens=2,
        )


def _multi_turn_suite() -> BenchmarkSuite:
    return BenchmarkSuite.from_dict(
        {
            "name": "mt",
            "tasks": [
                {
                    "id": "mt",
                    "category": MULTI_TURN_CATEGORY,
                    "prompt": "query then double",
                    "packs": ["data", "utils"],
                    "grade": {"type": "numeric", "value": 60},
                    "trajectory": {
                        "type": "all_of",
                        "of": [
                            {"type": "tool_sequence", "value": ["sql_query", "calculator"]},
                            {"type": "result_feeds_arg", "from_tool": "sql_query", "to_tool": "calculator", "arg": "expression", "value": "30"},
                        ],
                    },
                }
            ],
        }
    )


def _run(make_model: Any) -> ModelScorecard:
    factory = lambda spec: InferenceService(make_model())  # noqa: E731
    runner = BenchmarkRunner(trials=2, inference_factory=factory, clock=lambda: 0.0)
    cards = run_async(runner.run(_multi_turn_suite(), [ModelSpec("ollama", "m")]))
    return cards[0]


def test_runner_passes_multi_turn_when_result_carried() -> None:
    card = _run(lambda: _ChainModel(carry=True))
    assert card.multi_turn_accuracy == 1.0
    assert card.has_multi_turn_tier is True
    assert card.multi_turn_total_trials == 2
    trial = card.task_scores[0].trials[0]
    assert trial.trajectory_ok is True
    assert trial.correct is True
    # The data-flow is visible in the structured trajectory.
    names = trial.tools_called
    assert names == ["sql_query", "calculator"]


def test_runner_fails_multi_turn_when_bridge_value_hallucinated() -> None:
    # Right tools, right order, but the calculator used 99 instead of the result's 30.
    card = _run(lambda: _ChainModel(carry=False))
    trial = card.task_scores[0].trials[0]
    # The answer text "60" still passes the answer grader (numeric 60)...
    assert trial.answer_ok is True
    # ...but the data-flow grader fails: the result did not feed the arg.
    assert trial.trajectory_ok is False
    assert trial.correct is False
    assert card.multi_turn_accuracy == 0.0


def test_runner_marks_arg_graded_tasks() -> None:
    # The multi-turn task uses result_feeds_arg (an arg-level grader), so its TaskScore is
    # flagged and folds into the arg-accuracy headline.
    card = _run(lambda: _ChainModel(carry=True))
    assert card.task_scores[0].uses_arg_grader is True
    assert card.has_arg_tier is True
    assert card.arg_accuracy == 1.0
    assert card.arg_total_trials == 2


def test_name_only_trajectory_task_is_not_arg_graded() -> None:
    suite = BenchmarkSuite.from_dict(
        {
            "name": "n",
            "tasks": [
                {
                    "id": "seq",
                    "category": MULTI_TURN_CATEGORY,
                    "prompt": "x",
                    "packs": ["utils"],
                    "grade": {"type": "contains", "value": "60"},
                    "trajectory": {"type": "tool_sequence", "value": ["sql_query", "calculator"]},
                }
            ],
        }
    )
    factory = lambda spec: InferenceService(_ChainModel(carry=True))  # noqa: E731
    runner = BenchmarkRunner(trials=1, inference_factory=factory, clock=lambda: 0.0)
    card = run_async(runner.run(suite, [ModelSpec("ollama", "m")]))[0]
    # A name-only trajectory grader is NOT an arg-level grader.
    assert card.task_scores[0].uses_arg_grader is False
    assert card.has_arg_tier is False
    assert card.arg_accuracy is None
    # But it is still a multi-turn task.
    assert card.has_multi_turn_tier is True
