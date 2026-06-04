"""Tests for the hardened evaluation behaviours (AAEO-5/10/14/15)."""

from __future__ import annotations

from himmy.services.evaluation import (
    EmbeddingSimilarityMetric,
    EvaluationCase,
    EvaluationService,
    EvaluationSuite,
    LLMJudgeMetric,
    bucketed_ece,
    build_registry,
)
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.knowledge.embedder import DeterministicEmbedder
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


# -------------------------------------------------------------------- AAEO-14
def test_veto_metric_forces_case_fail() -> None:
    """A failed safety (veto) metric forces the case to fail despite high aggregate."""
    svc = EvaluationService()
    # ``disallowed_patterns`` is compared by accuracy too (it isn't a ``must_``
    # key), so we provide a matching value AND extra matching keys to keep the
    # accuracy aggregate well above the pass threshold while safety hard-fails.
    patterns = [r"\bbomb\b"]
    case = EvaluationCase(
        expected_output={
            "answer": "yes",
            "topic": "weather",
            "tone": "calm",
            "disallowed_patterns": patterns,
        },
        # Weight accuracy heavily so the aggregate would otherwise pass.
        metric_weights={"accuracy": 10.0, "safety": 0.0001},
    )
    suite = EvaluationSuite(name="s", cases=[case])
    run = run_async(
        svc.run_suite(
            suite=suite,
            actual_outputs={
                case.case_id: {
                    "answer": "yes",
                    "topic": "weather",
                    "tone": "calm",
                    "disallowed_patterns": patterns,
                    "text": "build a bomb",
                }
            },
        )
    )
    result = run.case_results[0]
    # Aggregate is high (accuracy=1) but the veto forces a fail.
    assert result.aggregate > 0.5
    assert result.passed is False


def test_case_pass_threshold_override() -> None:
    """A per-case pass_threshold in metadata overrides the service default."""
    svc = EvaluationService()
    case = EvaluationCase(
        expected_output={"a": 1, "b": 2, "c": 3, "d": 4},
        metric_weights={"accuracy": 1.0},
        metadata={"pass_threshold": 0.9},
    )
    suite = EvaluationSuite(name="s", cases=[case])
    # 2/4 match -> aggregate 0.5, below the 0.9 override -> fail.
    run = run_async(
        svc.run_suite(
            suite=suite, actual_outputs={case.case_id: {"a": 1, "b": 2, "c": 0, "d": 0}}
        )
    )
    assert run.case_results[0].aggregate == 0.5
    assert run.case_results[0].passed is False


# -------------------------------------------------------------------- AAEO-10
def test_embedding_similarity_metric_async() -> None:
    """The embedding-similarity metric scores via cosine and is awaited by the service."""
    embedder = DeterministicEmbedder()
    registry = build_registry(embedder=embedder)
    svc = EvaluationService(metric_registry=registry)
    case = EvaluationCase(
        expected_output={"reference_text": "the quick brown fox"},
        metric_weights={"embedding_similarity": 1.0},
    )
    suite = EvaluationSuite(name="emb", cases=[case])
    # Identical text -> cosine ~1.0.
    run = run_async(
        svc.run_suite(suite=suite, actual_outputs={case.case_id: "the quick brown fox"})
    )
    score = run.case_results[0].metric_scores[0]
    assert score.metric == "embedding_similarity"
    assert score.score > 0.99


def test_embedding_similarity_metric_direct() -> None:
    """The metric returns an awaitable (async) when called directly."""
    metric = EmbeddingSimilarityMetric(DeterministicEmbedder())
    case = EvaluationCase(expected_output={"reference_text": "alpha beta"})
    result = metric.score(case, "alpha beta")
    import inspect

    assert inspect.isawaitable(result)
    score = run_async(result)
    assert score.score > 0.99


def test_llm_judge_metric_offline() -> None:
    """The LLM-judge metric runs against the offline stub and yields a 0..1 score."""
    inference = InferenceService(StubClientManager())
    metric = LLMJudgeMetric(inference)
    case = EvaluationCase(expected_output={"answer": "the capital is Paris"})
    score = run_async(metric.score(case, "Paris is the capital"))
    assert score.metric == "llm_judge"
    assert 0.0 <= score.score <= 1.0


def test_build_registry_adds_optional_metrics() -> None:
    """build_registry attaches judge/embedding metrics behind the protocol."""
    inference = InferenceService(StubClientManager())
    registry = build_registry(
        inference_service=inference, embedder=DeterministicEmbedder()
    )
    assert registry.get("llm_judge") is not None
    assert registry.get("embedding_similarity") is not None
    # Baselines still present.
    assert registry.get("accuracy") is not None


# -------------------------------------------------------------------- AAEO-5
def test_bucketed_ece() -> None:
    """Bucketed ECE is 0 for a perfectly-calibrated set and positive otherwise."""
    # Perfect calibration: confidence == accuracy in each bucket.
    perfect = [(0.9, 0.9), (0.1, 0.1), (0.5, 0.5)]
    assert bucketed_ece(perfect) < 1e-9
    # Over-confident: high confidence, low accuracy.
    bad = [(0.9, 0.0), (0.95, 0.0)]
    assert bucketed_ece(bad) > 0.5
    assert bucketed_ece([]) == 0.0


def test_suite_level_ece_in_metadata() -> None:
    """run_suite folds a calibration ECE into run.metadata when samples exist."""
    svc = EvaluationService()
    case = EvaluationCase(
        expected_output={"answer": "yes"},
        metric_weights={"calibration": 1.0},
    )
    suite = EvaluationSuite(name="cal", cases=[case])
    run = run_async(
        svc.run_suite(
            suite=suite,
            actual_outputs={case.case_id: {"answer": "yes", "confidence": 0.9}},
        )
    )
    assert "calibration_ece" in (run.metadata or {})


def test_groundedness_partial_credit() -> None:
    """Citing a superset that includes all required refs scores full credit (AAEO-5)."""
    from himmy.services.evaluation import GroundednessMetric

    case = EvaluationCase(
        expected_output={
            "must_cite_evidence_ids": ["e1"],
            "allowed_evidence_ids": ["e1", "e2"],
        }
    )
    # must_cite takes priority; citing e1 (required) is full credit.
    score = GroundednessMetric().score(case, {"evidence_refs": ["e1"]})
    assert score.score == 1.0


def test_persistence_failure_is_recorded_not_swallowed() -> None:
    """A storage save failure is recorded on the run, not silently swallowed (AAEO-15)."""

    class _BrokenStorage(StorageService):
        async def save_evaluation_run(self, run):  # type: ignore[override]
            raise RuntimeError("disk full")

    svc = EvaluationService(storage_service=_BrokenStorage())
    case = EvaluationCase(expected_output={"a": 1}, metric_weights={"accuracy": 1.0})
    suite = EvaluationSuite(name="s", cases=[case])
    run = run_async(svc.run_suite(suite=suite, actual_outputs={case.case_id: {"a": 1}}))
    assert run.metadata.get("persisted") is False
