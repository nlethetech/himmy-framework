"""WS4.7 — Ragas-inspired privacy & compliance audit harness (Part B).

Offline-first: importing this package wires nothing and runs no I/O. The harness is
opt-in (a CLI/HTTP surface the operator invokes) and its consent-derived metrics stay
``skipped`` until Part A is wired, so the zero-config path is unaffected.
"""

from __future__ import annotations

from himmy.services.evaluation.privacy.catalog import (
    ALL_SCENARIOS,
    DEFAULT_PERSONAS,
    ProbePersona,
    ProbeScenario,
    ProbeTemplate,
    ScenarioPlan,
    default_scenario_plans,
    fake_pii_for,
    leak_patterns_for,
    planted_pii_blob,
    probe_pii_labels,
    template_for,
)
from himmy.services.evaluation.privacy.metrics import (
    AuditIntegrityMetric,
    ErasureCompletenessMetric,
    PiiLeakageMetric,
    RecordedDataMetric,
    RetentionComplianceMetric,
    build_recorded_registry,
    consent_coverage_metric,
    denial_enforcement_metric,
    purpose_limitation_metric,
)
from himmy.services.evaluation.privacy.models import (
    PRIVACY_AUDIT_REPORT_KIND,
    MetricKind,
    PrivacyAuditConfig,
    PrivacyAuditReport,
    PrivacyFinding,
    PrivacyMetricResult,
    RecordedDataContext,
    Severity,
)
from himmy.services.evaluation.privacy.probes import (
    DeterministicProbeSynthesizer,
    LLMProbeSynthesizer,
    PrivacyProbeGenerator,
    ProbeSynthesizer,
    build_privacy_probe_suite,
)
from himmy.services.evaluation.privacy.service import PrivacyAuditService

__all__ = [
    # typed shapes (B0)
    "PRIVACY_AUDIT_REPORT_KIND",
    "MetricKind",
    "Severity",
    "PrivacyFinding",
    "PrivacyMetricResult",
    "PrivacyAuditConfig",
    "RecordedDataContext",
    "PrivacyAuditReport",
    # recorded-data metrics (B1)
    "RecordedDataMetric",
    "PiiLeakageMetric",
    "AuditIntegrityMetric",
    "RetentionComplianceMetric",
    "ErasureCompletenessMetric",
    "consent_coverage_metric",
    "purpose_limitation_metric",
    "denial_enforcement_metric",
    "build_recorded_registry",
    # probe catalog (B2)
    "ProbeScenario",
    "ALL_SCENARIOS",
    "ProbePersona",
    "DEFAULT_PERSONAS",
    "ProbeTemplate",
    "ScenarioPlan",
    "default_scenario_plans",
    "fake_pii_for",
    "probe_pii_labels",
    "leak_patterns_for",
    "planted_pii_blob",
    "template_for",
    # probe generator (B2)
    "ProbeSynthesizer",
    "DeterministicProbeSynthesizer",
    "LLMProbeSynthesizer",
    "PrivacyProbeGenerator",
    "build_privacy_probe_suite",
    # audit service (B3)
    "PrivacyAuditService",
]
