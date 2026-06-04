"""Tests for the evaluation kernel: built-in metrics, weighting, suite runs."""

from __future__ import annotations

from himmy.services.evaluation import (
    AccuracyMetric,
    EvaluationCase,
    EvaluationService,
    EvaluationSuite,
    GroundednessMetric,
    SafetyMetric,
    default_metric_registry,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def test_accuracy_metric_exact_match_fraction() -> None:
    """Accuracy scores the fraction of expected keys that match."""
    case = EvaluationCase(expected_output={"a": 1, "b": 2})
    score = AccuracyMetric().score(case, {"a": 1, "b": 99})
    assert score.metric == "accuracy"
    assert score.score == 0.5


def test_groundedness_penalizes_hallucinated_refs() -> None:
    """Citing a ref not in the allowed set scores 0."""
    case = EvaluationCase(expected_output={"must_cite_evidence_ids": ["e1", "e2"]})
    good = GroundednessMetric().score(case, {"evidence_refs": ["e1", "e2"]})
    assert good.score == 1.0
    bad = GroundednessMetric().score(case, {"evidence_refs": ["e1", "rogue"]})
    assert bad.score == 0.0
    assert not bad.passed


def test_safety_metric_flags_disallowed_pattern() -> None:
    """Safety scores 0 when a disallowed pattern appears."""
    case = EvaluationCase(expected_output={"disallowed_patterns": [r"\bsecret\b"]})
    flagged = SafetyMetric().score(case, {"text": "this is a secret"})
    assert flagged.score == 0.0
    clean = SafetyMetric().score(case, {"text": "all clear"})
    assert clean.score == 1.0


def test_default_registry_has_five_metrics() -> None:
    """The shared default registry resolves all five built-ins."""
    for name in ("accuracy", "relevance", "groundedness", "safety", "calibration"):
        assert default_metric_registry.get(name) is not None
    assert default_metric_registry.get("nonsense") is None


def test_run_suite_aggregates_and_persists() -> None:
    """run_suite scores each case, aggregates, and saves when storage is wired."""
    storage = StorageService()
    svc = EvaluationService(storage_service=storage)
    case = EvaluationCase(
        expected_output={"answer": "yes"},
        metric_weights={"accuracy": 1.0},
    )
    suite = EvaluationSuite(name="qa", cases=[case])
    run = run_async(
        svc.run_suite(suite=suite, actual_outputs={case.case_id: {"answer": "yes"}})
    )
    assert run.suite_name == "qa"
    assert len(run.case_results) == 1
    assert run.case_results[0].aggregate == 1.0
    assert run.aggregate_score == 1.0
    # Persisted for the dashboard.
    assert run_async(storage.get_evaluation_run(run.run_id)) is not None


def test_run_suite_weighted_aggregate() -> None:
    """Per-metric weights drive the case aggregate."""
    svc = EvaluationService()
    # accuracy will be 0 (mismatch); safety will be 1 (clean) -> weighted toward safety.
    case = EvaluationCase(
        expected_output={"answer": "yes"},
        metric_weights={"accuracy": 1.0, "safety": 3.0},
    )
    suite = EvaluationSuite(name="w", cases=[case])
    run = run_async(
        svc.run_suite(suite=suite, actual_outputs={case.case_id: {"answer": "no"}})
    )
    # (0*1 + 1*3) / (1+3) = 0.75
    assert abs(run.case_results[0].aggregate - 0.75) < 1e-9
