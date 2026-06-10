"""LLM-judge grader tier: adapter, runner wiring, gate exclusion, agreement harness.

All offline: the candidate runs against the deterministic stub manager, and the judge
runs against a SCRIPTED inference stub that returns a controllable structured verdict
(pass / fail / malformed) so the three judge states (pass / fail / ungraded) are exercised
without a live model.
"""

from __future__ import annotations

from typing import Any

from himmy.benchmark import (
    BenchmarkRunner,
    BenchmarkSuite,
    ModelSpec,
    compare_to_baseline,
    to_json,
)
from himmy.benchmark.judge import (
    JudgeVerdict,
    SameModelJudgeError,
    build_judge,
    build_judge_spec,
    judge_answer,
    resolve_judge_model,
    run_judge_agreement,
)
from himmy.benchmark.report import render_markdown
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import InferenceResponse, InferenceStatus
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async


class _ScriptedJudge:
    """A judge inference stub that returns a fixed structured verdict.

    ``mode`` ∈ {"pass", "fail", "malformed", "none", "error"} drives the judge states:
    a high score (pass), a low score (graded fail), an unparseable score (SUCCESS but
    non-numeric → UNGRADED), structured output missing entirely (SUCCESS, no score →
    UNGRADED), or an explicit provider failure (UNGRADED).
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode

    def resolve(self, model_key: str) -> str:
        return "judge"

    async def generate(self, request: Any) -> InferenceResponse:
        if self._mode == "error":
            from himmy.services.inference.models import (
                InferenceError,
                InferenceErrorCode,
            )

            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                output_text="",
                error=InferenceError(
                    code=InferenceErrorCode.TIMEOUT, message="judge timed out"
                ),
            )
        if self._mode == "none":
            # SUCCESS but the model emitted non-JSON text → no structured output at all.
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="not json at all",
                output_structured=None,
                input_tokens=1,
                output_tokens=1,
            )
        if self._mode == "pass":
            structured: dict[str, Any] = {"score": 0.95, "rationale": "great answer"}
        elif self._mode == "fail":
            structured = {"score": 0.1, "rationale": "poor answer"}
        else:  # malformed: a non-numeric score the metric cannot parse
            structured = {"score": "not-a-number", "rationale": "??"}
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="",
            output_structured=structured,
            input_tokens=1,
            output_tokens=1,
        )


def _judge_metric(mode: str) -> Any:
    spec = ModelSpec(
        provider="ollama",
        model="candidate",
        judge_provider="ollama",
        judge_model="judge",
    )
    metric, judge_id = build_judge(
        spec, inference_factory=lambda s: InferenceService(_ScriptedJudge(mode))
    )
    return metric, judge_id


# --- the adapter: pass / fail / malformed / error -------------------------------------


def test_judge_pass_verdict() -> None:
    metric, jid = _judge_metric("pass")
    v = run_async(
        judge_answer(
            metric,
            rubric="be good",
            threshold=0.6,
            prompt="q",
            answer="a",
            judge_model=jid,
        )
    )
    assert isinstance(v, JudgeVerdict)
    assert v.passed is True
    assert v.ungraded is False
    assert v.score >= 0.6


def test_judge_fail_verdict() -> None:
    metric, jid = _judge_metric("fail")
    v = run_async(
        judge_answer(
            metric,
            rubric="be good",
            threshold=0.6,
            prompt="q",
            answer="a",
            judge_model=jid,
        )
    )
    assert v.passed is False
    assert v.ungraded is False


def test_judge_malformed_score_is_ungraded() -> None:
    # SUCCESS but a non-numeric score = an unparseable verdict: the judge never produced
    # a real verdict, so the trial is UNGRADED (not a graded 0.0 fail that would deflate
    # the candidate's judge pass rate). Regression test for the three-state contract.
    metric, jid = _judge_metric("malformed")
    v = run_async(
        judge_answer(
            metric,
            rubric="be good",
            threshold=0.6,
            prompt="q",
            answer="a",
            judge_model=jid,
        )
    )
    assert v.ungraded is True
    assert v.passed is False
    assert v.score == 0.0


def test_judge_missing_structured_output_is_ungraded() -> None:
    # SUCCESS with no structured output at all (model emitted plain text) is also an
    # unparseable verdict → UNGRADED, not a graded fail.
    metric, jid = _judge_metric("none")
    v = run_async(
        judge_answer(
            metric,
            rubric="be good",
            threshold=0.6,
            prompt="q",
            answer="a",
            judge_model=jid,
        )
    )
    assert v.ungraded is True
    assert v.passed is False


def test_judge_provider_failure_is_ungraded() -> None:
    # A judge timeout / provider failure marks the trial UNGRADED (distinct from fail).
    metric, jid = _judge_metric("error")
    v = run_async(
        judge_answer(
            metric,
            rubric="be good",
            threshold=0.6,
            prompt="q",
            answer="a",
            judge_model=jid,
        )
    )
    assert v.ungraded is True
    assert v.passed is False


# --- judge != candidate enforcement ---------------------------------------------------


def test_resolve_judge_rejects_same_model() -> None:
    spec = ModelSpec(
        provider="ollama", model="m", judge_provider="ollama", judge_model="m"
    )
    try:
        resolve_judge_model(spec, spec.provider)
    except SameModelJudgeError as exc:
        assert "may not grade its own output" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected SameModelJudgeError")


def test_resolve_judge_requires_a_judge_model() -> None:
    spec = ModelSpec(provider="ollama", model="m")  # no judge_model
    try:
        resolve_judge_model(spec, spec.provider)
    except ValueError as exc:
        assert "judge model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_resolve_judge_allows_distinct_model() -> None:
    spec = ModelSpec(
        provider="ollama", model="cand", judge_provider="anthropic", judge_model="judge"
    )
    assert resolve_judge_model(spec, spec.provider) == ("anthropic", "judge")


def test_resolve_judge_rejects_ollama_latest_tag_alias() -> None:
    # Ollama treats "llama3.2" and "llama3.2:latest" as the SAME model — the judge!=
    # candidate check must normalize the :latest tag, not just compare raw strings.
    spec = ModelSpec(
        provider="ollama",
        model="llama3.2",
        judge_provider="ollama",
        judge_model="llama3.2:latest",
    )
    try:
        resolve_judge_model(spec, spec.provider)
    except SameModelJudgeError as exc:
        assert "same model" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected SameModelJudgeError for :latest alias")


def test_resolve_judge_rejects_claude_cli_short_alias() -> None:
    # claude-cli accepts both "haiku" and the full id — they resolve to one model, so a
    # candidate "haiku" judged by "claude-haiku-4-5" is self-grading and must be refused.
    spec = ModelSpec(
        provider="claude-cli",
        model="haiku",
        judge_provider="claude-cli",
        judge_model="claude-haiku-4-5",
    )
    try:
        resolve_judge_model(spec, spec.provider)
    except SameModelJudgeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected SameModelJudgeError for claude-cli alias")


def test_resolve_judge_allows_distinct_ollama_tags() -> None:
    # Genuinely different Ollama models are still allowed (no false collision).
    spec = ModelSpec(
        provider="ollama",
        model="llama3.2",
        judge_provider="ollama",
        judge_model="qwen2.5:3b",
    )
    assert resolve_judge_model(spec, spec.provider) == ("ollama", "qwen2.5:3b")


def test_build_judge_spec_validation() -> None:
    assert build_judge_spec({"rubric": "be good", "threshold": 0.7}) == ("be good", 0.7)
    for bad in ({}, {"rubric": "   "}, {"rubric": "x", "threshold": 1.5}):
        try:
            build_judge_spec(bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {bad!r}")


# --- runner wiring: judge tier end-to-end ---------------------------------------------


def _judge_suite() -> BenchmarkSuite:
    return BenchmarkSuite.from_dict(
        {
            "name": "jt",
            "tasks": [
                {
                    "id": "summarize",
                    "category": "nepali",
                    "prompt": "summarize the paragraph",
                    "judge": {
                        "rubric": "a faithful one-sentence summary",
                        "threshold": 0.6,
                    },
                }
            ],
        }
    )


def _candidate_spec() -> ModelSpec:
    return ModelSpec(
        provider="ollama",
        model="candidate",
        judge_provider="ollama",
        judge_model="judge",
    )


def _run_judge_suite(mode: str, trials: int = 2) -> Any:
    runner = BenchmarkRunner(
        trials=trials,
        clock=lambda: 0.0,
        # Candidate uses the offline stub manager (echoes text); judge uses the scripted
        # stub returning a controllable verdict.
        inference_factory=lambda s: InferenceService(StubClientManager()),
        judge_inference_factory=lambda s: InferenceService(_ScriptedJudge(mode)),
    )
    cards = run_async(runner.run(_judge_suite(), [_candidate_spec()]))
    return cards[0]


def test_runner_judge_pass_marks_correct_and_judged() -> None:
    card = _run_judge_suite("pass")
    assert card.has_judge_tier is True
    assert card.judge_accuracy == 1.0
    assert card.judge_ungraded == 0
    assert card.judge_total_trials == 2
    score = card.judge_scores[0]
    assert all(t.judged for t in score.trials)
    assert all(t.judge_ok is True for t in score.trials)


def test_runner_judge_fail_marks_graded_fail() -> None:
    card = _run_judge_suite("fail")
    assert card.judge_accuracy == 0.0
    assert card.judge_ungraded == 0  # graded, just failed


def test_runner_judge_timeout_marks_ungraded() -> None:
    card = _run_judge_suite("error")
    # An ungraded trial is NOT a graded fail: judge_accuracy is None (nothing graded).
    assert card.judge_ungraded == 2
    assert card.judge_accuracy is None


def test_runner_judge_with_grade_and_trajectory_anded() -> None:
    # A judge-tier task that ALSO declares a `grade` and a `trajectory` block: `correct`
    # must AND the judge verdict with the deterministic gates (BenchmarkTask.judge says
    # "in addition to" the grade; TrialResult invariant: correct = answer_ok and traj_ok).
    # The stub echoes the prompt (so a `contains: NONEXISTENT` grade fails) and calls no
    # tools (so a `tool_called` trajectory fails), while the judge PASSES — yet the trial
    # must be `correct=False`. Regression test for finding 6.
    suite = BenchmarkSuite.from_dict(
        {
            "name": "jt",
            "tasks": [
                {
                    "id": "combo",
                    "category": "nepali",
                    "prompt": "summarize the paragraph",
                    "grade": {"type": "contains", "value": "NONEXISTENT_TOKEN_ZZZ"},
                    "trajectory": {"type": "tool_called", "value": "never_called_tool"},
                    "judge": {"rubric": "good", "threshold": 0.6},
                }
            ],
        }
    )
    runner = BenchmarkRunner(
        trials=1,
        clock=lambda: 0.0,
        inference_factory=lambda s: InferenceService(StubClientManager()),
        judge_inference_factory=lambda s: InferenceService(_ScriptedJudge("pass")),
    )
    card = run_async(runner.run(suite, [_candidate_spec()]))[0]
    trial = card.judge_scores[0].trials[0]
    # The judge passed, but the deterministic gates failed → trial is NOT correct.
    assert trial.judge_ok is True
    assert trial.answer_ok is False
    assert trial.trajectory_ok is False
    assert trial.correct is False


def test_runner_preflight_rejects_same_model_judge() -> None:
    spec = ModelSpec(
        provider="ollama", model="m", judge_provider="ollama", judge_model="m"
    )
    runner = BenchmarkRunner(
        trials=1,
        clock=lambda: 0.0,
        judge_inference_factory=lambda s: InferenceService(_ScriptedJudge("pass")),
    )
    try:
        run_async(runner.run(_judge_suite(), [spec]))
    except SameModelJudgeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected SameModelJudgeError before any trial ran")


# --- judge tier is reported, NEVER gating ---------------------------------------------


def test_judge_tier_excluded_from_deterministic_accuracy_and_gate() -> None:
    card = _run_judge_suite("fail")  # judge fails every trial
    # Deterministic accuracy ignores the judge-tier task entirely (no deterministic tasks
    # in this suite ⇒ accuracy over an empty deterministic tier is 0.0 / total_trials 0).
    assert card.total_trials == 0
    assert card.deterministic_scores == []
    results = to_json([card], suite_name="jt")
    model = results["models"][0]
    # The judge sub-record carries the judge numbers; deterministic accuracy stays clean.
    assert model["accuracy"] == 0.0
    assert model["judge"]["pass_rate"] == 0.0
    assert model["judge"]["total_trials"] == 2
    # A baseline that floors accuracy must IGNORE judge tasks: the failing judge tier must
    # not trip the gate (there is no deterministic signal to fail on).
    baseline = {
        "gate": {"min_trials": 1},
        "models": {
            "ollama:candidate": {
                "floors": {"accuracy": 0.0},
                "ceilings": {"error_rate": 1.0},
            }
        },
    }
    assert compare_to_baseline(results, baseline) == []


def test_report_separates_judge_tier_and_names_model() -> None:
    card = _run_judge_suite("pass")
    md = render_markdown([card], suite_name="jt")
    assert "Judge tier (LLM-graded — reported, NOT gated)" in md
    assert "judged by ollama:judge" in md
    assert "Ungraded" in md


def test_report_shows_ungraded_count() -> None:
    card = _run_judge_suite("error")
    md = render_markdown([card], suite_name="jt")
    assert "Judge tier" in md
    # 2 ungraded trials should surface in the per-model judge row.
    assert "| 2 |" in md


# --- judge-agreement harness ----------------------------------------------------------


class _LabelAwareJudge:
    """A judge stub that scores based on whether the answer text contains 'GOOD'.

    Lets the agreement harness be tested deterministically: rows whose answer says GOOD
    pass, the rest fail — so a validation set labelled to match yields 100% agreement.
    """

    def resolve(self, model_key: str) -> str:
        return "judge"

    async def generate(self, request: Any) -> InferenceResponse:
        text = " ".join(m.content for m in request.messages)
        score = 0.95 if "GOOD" in text else 0.05
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="",
            output_structured={"score": score, "rationale": "ok"},
            input_tokens=1,
            output_tokens=1,
        )


def test_judge_agreement_perfect() -> None:
    spec = ModelSpec(
        provider="ollama", model="cand", judge_provider="ollama", judge_model="judge"
    )
    metric, jid = build_judge(
        spec, inference_factory=lambda s: InferenceService(_LabelAwareJudge())
    )
    rows = [
        {"task": "t", "rubric": "r", "answer": "this is GOOD", "human_label": "good"},
        {"task": "t", "rubric": "r", "answer": "this is bad", "human_label": "bad"},
        {"task": "t", "rubric": "r", "answer": "also GOOD", "human_label": "good"},
    ]
    res = run_async(run_judge_agreement(rows, metric=metric, judge_model=jid))
    assert res.n == 3
    assert res.graded == 3
    assert res.matches == 3
    assert res.agreement == 1.0


def test_judge_agreement_counts_ungraded_separately() -> None:
    spec = ModelSpec(
        provider="ollama", model="cand", judge_provider="ollama", judge_model="judge"
    )
    metric, jid = build_judge(
        spec, inference_factory=lambda s: InferenceService(_ScriptedJudge("error"))
    )
    rows = [{"task": "t", "rubric": "r", "answer": "x", "human_label": "good"}]
    res = run_async(run_judge_agreement(rows, metric=metric, judge_model=jid))
    assert res.ungraded == 1
    assert res.graded == 0
    assert res.agreement == 0.0


def test_validation_path_is_cwd_independent(tmp_path: Any, monkeypatch: Any) -> None:
    # JUDGE_VALIDATION_PATH must be package-anchored (absolute), so load_judge_validation()
    # works from any cwd — not only the repo root. Regression for finding 19.
    import os

    from himmy.benchmark.judge import JUDGE_VALIDATION_PATH, load_judge_validation

    assert os.path.isabs(JUDGE_VALIDATION_PATH)
    monkeypatch.chdir(tmp_path)  # a directory with no benchmarks/ subtree
    rows = load_judge_validation()  # must NOT raise FileNotFoundError
    assert len(rows) >= 10


def test_shipped_validation_set_loads_and_runs() -> None:
    from himmy.benchmark.judge import load_judge_validation

    rows = load_judge_validation()
    assert len(rows) >= 10
    assert {r["human_label"] for r in rows} == {"good", "bad"}
    spec = ModelSpec(
        provider="ollama", model="cand", judge_provider="ollama", judge_model="judge"
    )
    metric, jid = build_judge(
        spec, inference_factory=lambda s: InferenceService(_LabelAwareJudge())
    )
    res = run_async(run_judge_agreement(rows, metric=metric, judge_model=jid))
    assert res.n == len(rows)
    assert res.graded == len(rows)  # the label-aware stub grades every row
