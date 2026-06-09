"""Evaluation kernel: typed scorecards over agent outputs (aggregate + per-case)."""

from __future__ import annotations

from himmy.services.evaluation.metrics import (
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
from himmy.services.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRun,
    EvaluationSuite,
    MetricScore,
)
from himmy.services.evaluation.privacy import (
    PRIVACY_AUDIT_REPORT_KIND,
    PrivacyAuditConfig,
    PrivacyAuditReport,
    PrivacyAuditService,
    PrivacyFinding,
    PrivacyMetricResult,
    build_privacy_probe_suite,
)
from himmy.services.evaluation.service import EvaluationService

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
    # WS4.7 privacy & compliance audit harness (Part B) — re-exported public API.
    "PRIVACY_AUDIT_REPORT_KIND",
    "PrivacyAuditService",
    "PrivacyAuditConfig",
    "PrivacyAuditReport",
    "PrivacyMetricResult",
    "PrivacyFinding",
    "build_privacy_probe_suite",
]
