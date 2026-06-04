"""Expanded evaluation-kernel coverage for the hardened AAEO behaviours.

Complements ``test_evaluation_hardening.py`` with paths it does not exercise:

- A veto metric configured *per-case* (``metadata['veto_metrics']``) forcing a
  fail (AAEO-14), and a veto via the LLM-judge async metric backed by a
  *failing* InferenceService stub (AAEO-10/14).
- The async LLM-judge metric run through the service (a stub judge via
  InferenceService), and the service awaiting its coroutine result (AAEO-10).
- Embedding-similarity scoring of orthogonal vs identical text, and the
  empty-text short-circuit (AAEO-10).
- Robustness: an unknown metric and a metric that raises both fail closed without
  killing the suite (AAEO-14 fail-closed contract).
- Per-suite concurrency is honoured (a serial-only contract would deadlock a
  bounded semaphore at 1).

All tests drive async via the ``run_async`` helper over the offline stack.
"""

from __future__ import annotations

from typing import Any

from himmy.services.evaluation import (
    EmbeddingSimilarityMetric,
    EvaluationCase,
    EvaluationService,
    EvaluationSuite,
    build_registry,
)
from himmy.services.evaluation.metrics import EvaluationMetricRegistry
from himmy.services.evaluation.models import MetricScore
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)
from himmy.services.inference.service import InferenceService
from himmy.services.knowledge.embedder import DeterministicEmbedder
from tests.conftest import run_async


# --------------------------------------------------------------------- helpers
class _FailingInferenceManager:
    """A client manager whose generate() always returns a FAILED response."""

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                message="judge provider down",
                retryable=False,
            ),
        )


# ------------------------------------------------------------------- AAEO-14
def test_per_case_veto_metric_forces_fail() -> None:
    """A case-level ``veto_metrics`` entry forces a fail despite a high aggregate."""
    svc = EvaluationService(veto_metrics=())  # no default vetoes; rely on per-case
    case = EvaluationCase(
        expected_output={"answer": "yes", "disallowed_patterns": [r"\bbomb\b"]},
        metric_weights={"accuracy": 10.0, "safety": 0.0001},
        metadata={"veto_metrics": ["safety"]},
    )
    suite = EvaluationSuite(name="s", cases=[case])
    run = run_async(
        svc.run_suite(
            suite=suite,
            actual_outputs={
                case.case_id: {
                    "answer": "yes",
                    "disallowed_patterns": [r"\bbomb\b"],
                    "text": "build a bomb",
                }
            },
        )
    )
    result = run.case_results[0]
    assert result.aggregate > 0.5
    assert result.passed is False


def test_no_veto_set_allows_pass() -> None:
    """With the default veto disabled and none per-case, a high aggregate passes."""
    svc = EvaluationService(veto_metrics=())
    case = EvaluationCase(
        expected_output={"answer": "yes", "disallowed_patterns": [r"\bbomb\b"]},
        metric_weights={"accuracy": 10.0, "safety": 0.0001},
    )
    suite = EvaluationSuite(name="s", cases=[case])
    run = run_async(
        svc.run_suite(
            suite=suite,
            actual_outputs={
                case.case_id: {
                    "answer": "yes",
                    "disallowed_patterns": [r"\bbomb\b"],
                    "text": "build a bomb",
                }
            },
        )
    )
    # Safety still fails (score 0) but it is NOT a veto, so the weighted aggregate
    # (dominated by accuracy=1) carries the case to a pass.
    assert run.case_results[0].passed is True


# ------------------------------------------------------------------- AAEO-10/14
def test_llm_judge_metric_via_service_async() -> None:
    """The async LLM-judge metric is awaited by the service and yields a 0..1 score."""
    inference = InferenceService(StubClientManager())
    registry = build_registry(inference_service=inference)
    svc = EvaluationService(metric_registry=registry)
    case = EvaluationCase(
        expected_output={"answer": "the capital of France is Paris"},
        metric_weights={"llm_judge": 1.0},
    )
    suite = EvaluationSuite(name="judge", cases=[case])
    run = run_async(
        svc.run_suite(
            suite=suite, actual_outputs={case.case_id: "Paris is the capital"}
        )
    )
    score = run.case_results[0].metric_scores[0]
    assert score.metric == "llm_judge"
    assert 0.0 <= score.score <= 1.0


def test_llm_judge_veto_forces_fail_when_judge_fails() -> None:
    """A failing judge (FAILED inference) scores 0 and, as a veto, fails the case."""
    inference = InferenceService(_FailingInferenceManager())
    registry = build_registry(inference_service=inference)
    svc = EvaluationService(metric_registry=registry, veto_metrics=("llm_judge",))
    case = EvaluationCase(
        expected_output={"answer": "yes"},
        # Accuracy would otherwise carry the aggregate above threshold.
        metric_weights={"accuracy": 10.0, "llm_judge": 0.0001},
    )
    suite = EvaluationSuite(name="jveto", cases=[case])
    run = run_async(
        svc.run_suite(suite=suite, actual_outputs={case.case_id: {"answer": "yes"}})
    )
    result = run.case_results[0]
    judge = next(s for s in result.metric_scores if s.metric == "llm_judge")
    assert judge.score == 0.0 and judge.passed is False
    assert "judge failed" in judge.detail
    assert result.aggregate > 0.5  # accuracy carried it up
    assert result.passed is False  # ...but the judge veto fails the case


# ------------------------------------------------------------------- AAEO-10
def test_embedding_similarity_orthogonal_and_identical() -> None:
    """Identical text scores ~1.0; disjoint text scores lower."""
    embedder = DeterministicEmbedder()
    metric = EmbeddingSimilarityMetric(embedder)

    case_same = EvaluationCase(expected_output={"reference_text": "alpha beta gamma"})
    same = run_async(metric.score(case_same, "alpha beta gamma"))
    assert same.score > 0.99 and same.passed is True

    case_diff = EvaluationCase(expected_output={"reference_text": "alpha beta gamma"})
    diff = run_async(metric.score(case_diff, "zzz qqq www"))
    # Disjoint tokens -> strictly lower similarity than the identical case.
    assert diff.score < same.score


def test_embedding_similarity_empty_text_short_circuits() -> None:
    """Empty reference/actual text short-circuits to a passing neutral score."""
    metric = EmbeddingSimilarityMetric(DeterministicEmbedder())
    case = EvaluationCase(expected_output={"reference_text": ""})
    score = run_async(metric.score(case, ""))
    assert score.score == 1.0 and score.passed is True
    assert "no reference" in score.detail


# ------------------------------------------------------------------- AAEO-14 (fail-closed)
def test_unknown_metric_fails_closed() -> None:
    """An unknown metric name yields a failing 0.0 score, not an exception."""
    svc = EvaluationService()
    case = EvaluationCase(
        expected_output={"answer": "yes"}, metric_weights={"does_not_exist": 1.0}
    )
    suite = EvaluationSuite(name="u", cases=[case])
    run = run_async(
        svc.run_suite(suite=suite, actual_outputs={case.case_id: {"answer": "yes"}})
    )
    score = run.case_results[0].metric_scores[0]
    assert score.metric == "does_not_exist"
    assert score.score == 0.0 and score.passed is False
    assert "unknown metric" in score.detail


def test_raising_metric_fails_closed() -> None:
    """A metric whose score() raises is caught and recorded as a 0.0 failure."""

    class _Boom:
        name = "boom"

        def score(self, case: Any, actual: Any) -> MetricScore:
            raise RuntimeError("metric exploded")

    registry = EvaluationMetricRegistry().default()
    registry.register("boom", _Boom())
    svc = EvaluationService(metric_registry=registry)
    case = EvaluationCase(
        expected_output={"answer": "yes"}, metric_weights={"boom": 1.0}
    )
    suite = EvaluationSuite(name="b", cases=[case])
    run = run_async(
        svc.run_suite(suite=suite, actual_outputs={case.case_id: {"answer": "yes"}})
    )
    score = run.case_results[0].metric_scores[0]
    assert score.score == 0.0 and score.passed is False
    assert "metric error" in score.detail


def test_concurrent_async_metric_scoring() -> None:
    """A suite of multiple cases over an async metric completes (concurrency honoured)."""
    inference = InferenceService(StubClientManager())
    registry = build_registry(inference_service=inference)
    svc = EvaluationService(metric_registry=registry, concurrency=4)
    cases = [
        EvaluationCase(
            case_id=f"c{i}",
            expected_output={"answer": "yes"},
            metric_weights={"llm_judge": 1.0},
        )
        for i in range(6)
    ]
    suite = EvaluationSuite(name="many", cases=cases)
    run = run_async(
        svc.run_suite(
            suite=suite,
            actual_outputs={c.case_id: {"answer": "yes"} for c in cases},
        )
    )
    assert len(run.case_results) == 6
    assert all(
        r.metric_scores and r.metric_scores[0].metric == "llm_judge"
        for r in run.case_results
    )


def test_empty_suite_aggregate_is_zero() -> None:
    """A suite with no cases aggregates to 0.0 without error."""
    svc = EvaluationService()
    run = run_async(
        svc.run_suite(suite=EvaluationSuite(name="empty", cases=[]), actual_outputs={})
    )
    assert run.case_results == []
    assert run.aggregate_score == 0.0
