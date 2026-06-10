"""Nepali language-understanding suite: YAML parsing + deterministic tasks end-to-end.

Offline and deterministic. The deterministic tasks run the agent against a SCRIPTED
inference stub that returns the right tool result + a final answer, and the test asserts
the suite's hard-coded grade values agree with the nepal toolkit's OWN ground-truth
functions (so the suite can never silently drift from the tools it grades against).
"""

from __future__ import annotations

import datetime
from typing import Any

from himmy.benchmark import BenchmarkRunner, ModelSpec, nepali_suite
from himmy.benchmark.graders import grade
from himmy.nepal.calendar import ad_to_bs
from himmy.nepal.language import transliterate
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


class _ToolThenAnswer:
    """Turn 1 calls ``tool`` (returning ``content``); turn 2 answers ``answer``."""

    def __init__(self, tool: str, content: dict[str, Any], answer: str) -> None:
        self._tool = tool
        self._content = content
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
                        tool_call_id="c1", tool_name=self._tool, content=self._content
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


class _PlainAnswer:
    """Answers ``answer`` immediately with no tool calls."""

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
            output_tokens=2,
        )


def _task(suite: Any, task_id: str) -> Any:
    return next(t for t in suite.tasks if t.id == task_id)


# --- YAML parsing ---------------------------------------------------------------------


def test_suite_parses_and_has_five_tasks() -> None:
    suite = nepali_suite()
    assert suite.name == "nepali"
    assert len(suite.tasks) >= 5
    ids = {t.id for t in suite.tasks}
    assert {
        "devanagari_instruction",
        "code_switched_query",
        "transliteration_roundtrip",
        "bs_calendar_reasoning",
        "nepali_summarization",
    } <= ids


def test_only_summarization_is_judge_tier() -> None:
    suite = nepali_suite()
    judge_tasks = [t.id for t in suite.tasks if t.judge]
    assert judge_tasks == ["nepali_summarization"]
    # The judge task carries a rubric + threshold; deterministic tasks carry a grade.
    summ = _task(suite, "nepali_summarization")
    assert summ.judge["rubric"].strip()
    assert summ.judge["threshold"] == 0.6
    assert not summ.grade  # no deterministic grader for the open-ended task


# --- ground truth comes from the nepal toolkit, not hand-written constants ------------


def test_transliteration_grade_matches_toolkit_ground_truth() -> None:
    suite = nepali_suite()
    task = _task(suite, "transliteration_roundtrip")
    # The grade value must be the toolkit's actual transliteration of नेपाल.
    truth = transliterate("नेपाल")
    assert grade(task.grade, truth) is True
    assert task.grade["value"] == truth


def test_bs_calendar_grade_matches_toolkit_ground_truth() -> None:
    suite = nepali_suite()
    task = _task(suite, "bs_calendar_reasoning")
    truth_year = str(ad_to_bs(datetime.date(2026, 4, 14)).year)
    assert task.grade["value"] == truth_year
    assert grade(task.grade, f"The year is {truth_year} BS.") is True


# --- deterministic tasks end-to-end against the stub provider -------------------------


def _run_task(task_id: str, make_manager: Any, trials: int = 2) -> Any:
    suite = nepali_suite()
    one = next(t for t in suite.tasks if t.id == task_id)
    from himmy.benchmark.models import BenchmarkSuite

    sub = BenchmarkSuite(name="nepali", tasks=[one])
    # A FRESH manager per trial (the stub counts turns; reusing one across trials would
    # leak its call counter and break the scripted tool-then-answer sequence).
    runner = BenchmarkRunner(
        trials=trials,
        inference_factory=lambda s: InferenceService(make_manager()),
        clock=lambda: 0.0,
    )
    cards = run_async(runner.run(sub, [ModelSpec("ollama", "m")]))
    return cards[0]


def test_devanagari_instruction_following() -> None:
    card = _run_task("devanagari_instruction", lambda: _PlainAnswer("red"))
    assert card.accuracy == 1.0


def test_code_switched_query() -> None:
    card = _run_task("code_switched_query", lambda: _PlainAnswer("Kathmandu"))
    assert card.accuracy == 1.0


def test_transliteration_roundtrip_end_to_end() -> None:
    truth = transliterate("नेपाल")
    make = lambda: _ToolThenAnswer(  # noqa: E731
        "nepali_transliterate",
        {"text": "नेपाल", "roman": truth, "normalized": "nepal"},
        f"The Roman form is {truth}.",
    )
    card = _run_task("transliteration_roundtrip", make)
    assert card.accuracy == 1.0
    score = card.task_scores[0]
    # The trajectory grader (tool_called nepali_transliterate) must pass too.
    assert all(t.trajectory_ok is True for t in score.trials)
    assert all("nepali_transliterate" in t.tools_called for t in score.trials)


def test_bs_calendar_reasoning_end_to_end() -> None:
    year = str(ad_to_bs(datetime.date(2026, 4, 14)).year)
    make = lambda: _ToolThenAnswer(  # noqa: E731
        "nepali_date",
        {"ad": "2026-04-14", "bs": f"{year}-01-01", "bs_month_en": "Baishakh"},
        f"त्यो मिति {year} सालमा पर्छ।",
    )
    card = _run_task("bs_calendar_reasoning", make)
    assert card.accuracy == 1.0
    assert all(t.trajectory_ok is True for t in card.task_scores[0].trials)


def test_wrong_answer_fails_deterministic_task() -> None:
    # A model that answers the wrong color fails the Devanagari instruction task.
    card = _run_task("devanagari_instruction", lambda: _PlainAnswer("blue"))
    assert card.accuracy == 0.0


def test_full_suite_runs_with_no_judge_configured() -> None:
    # The whole nepali suite (incl. the judge-graded summarization task) must RUN with a
    # plain ModelSpec that has no judge_model — the judge-tier trial is recorded ungraded
    # (the documented degrade path), NOT an uncaught ValueError that aborts the run.
    # Regression test for findings 3 & 14.
    suite = nepali_suite()
    runner = BenchmarkRunner(
        trials=1,
        inference_factory=lambda s: InferenceService(_PlainAnswer("कुनै उत्तर")),
        clock=lambda: 0.0,
    )
    cards = run_async(runner.run(suite, [ModelSpec("ollama", "m")]))  # no judge_model
    card = cards[0]
    # The judge-tier task ran but produced no graded verdict (no judge configured).
    assert card.has_judge_tier is True
    assert card.judge_total_trials == 1
    assert card.judge_ungraded == 1
    assert card.judge_accuracy is None  # nothing graded ⇒ no pass rate
    # The four deterministic tasks still ran (they are not in the judge tier).
    assert len(card.deterministic_scores) == 4
    assert card.total_trials == 4
