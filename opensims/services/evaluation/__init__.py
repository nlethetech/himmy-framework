"""Evaluation kernel: typed scorecards over agent outputs (aggregate + per-case)."""

from __future__ import annotations

from opensims.services.evaluation.metrics import (
    AccuracyMetric,
    CalibrationMetric,
    EmbeddingSimilarityMetric,
    EvaluationMetricRegistry,
    GroundednessMetric,
    LLMJudgeMetric,
    MetricEvaluator,
    RelevanceMetric,
    SafetyMetric,
    bucketed_ece,
    build_registry,
    default_metric_registry,
)
from opensims.services.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRun,
    EvaluationSuite,
    MetricScore,
)
from opensims.services.evaluation.service import EvaluationService

__all__ = [
    "EvaluationCase",
    "EvaluationSuite",
    "EvaluationRun",
    "EvaluationCaseResult",
    "MetricScore",
    "MetricEvaluator",
    "EvaluationMetricRegistry",
    "EvaluationService",
    "AccuracyMetric",
    "RelevanceMetric",
    "GroundednessMetric",
    "SafetyMetric",
    "CalibrationMetric",
    "EmbeddingSimilarityMetric",
    "LLMJudgeMetric",
    "build_registry",
    "bucketed_ece",
    "default_metric_registry",
]
