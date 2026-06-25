"""WS4.7 — privacy & compliance audit: the typed shapes the harness rolls up.

This module is the data-only foundation of the Ragas-inspired privacy audit (Part B).
It defines what a single privacy metric verdict looks like
(:class:`PrivacyMetricResult`), the structurally-PII-safe finding it may attach
(:class:`PrivacyFinding`), the rolled-up scorecard the service registers
(:class:`PrivacyAuditReport`), the per-suite scoring policy
(:class:`PrivacyAuditConfig`), and the pre-materialized read-side bundle the
deterministic recorded-data metrics are fed (:class:`RecordedDataContext`).

Two invariants are enforced *structurally* here, not by convention:

* **A finding can never carry a raw PII value.** :class:`PrivacyFinding` forbids
  extra fields (``extra="forbid"``) and exposes only ``code``/``severity``/``count``
  plus *reference ids* (record/subject) — there is deliberately no
  ``value``/``text``/``match`` field, so the signed report cannot become a PII sink
  even by accident (must_fix #8).
* **Re-running an audit never collides on the content address.**
  :meth:`PrivacyAuditReport.to_record` mints a fresh ``stable_id`` per run
  (:func:`himmy.core.ids.new_uuid`), so two reports over an identical store project
  to two distinct, separately-verifiable :class:`~himmy.entities.records.EntityRecord`
  versions rather than idempotently de-duplicating.

The module has no storage/runtime coupling: it is pure data plus one projection, so
it imports green and tests with zero I/O before the rest of Part B (or Part A) exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from himmy.core.ids import new_uuid, utc_now_iso

# Concrete record/service types are imported at runtime (not under TYPE_CHECKING)
# because :class:`RecordedDataContext` is a pydantic model whose fields reference them,
# so the annotations must resolve when the model is built. None of these modules import
# the privacy package, so there is no import cycle (the field types are leaf data/model
# classes). ``EntityRegistryProtocol`` is a ``runtime_checkable`` Protocol, allowed via
# ``arbitrary_types_allowed`` — and crucially it accepts ANY registry backend (in-memory
# OR the durable :class:`SqliteEntityRegistry`) structurally, instead of the old nominal
# ``EntityRegistry`` check that rejected a durable spine at validation time.
from himmy.entities.protocol import EntityRegistryProtocol
from himmy.services.audit.models import SecurityEvent
from himmy.services.governance.retention import RetentionService, SubjectKeyVault
from himmy.services.storage.models import (
    EpisodicMemoryObject,
    MemoryObject,
    RunRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.records import EntityRecord

#: EntityRecord kind for a registered privacy audit scorecard.
PRIVACY_AUDIT_REPORT_KIND = "privacy_audit_report"

#: Which half of the audit a metric belongs to: a live adversarial probe run, or a
#: deterministic scan over already-recorded data.
MetricKind = Literal["probe", "recorded"]

#: Severity ladder for a finding (lexical, not numeric, to stay refactor-stable).
Severity = Literal["info", "low", "medium", "high", "critical"]


class PrivacyFinding(BaseModel):
    """A privacy issue a metric surfaced — counts + reference ids ONLY.

    By construction this model can never carry a raw PII value: it forbids extra
    fields and exposes no ``value``/``text``/``match`` member. A finding points at the
    offending records/subjects by *id* (``record_refs``/``subject_refs``) so an
    investigator can pull the underlying record through the normal access-controlled
    path, while the signed report itself stays free of cleartext PII (must_fix #8).

    Attributes:
        code: A stable machine code for the issue (e.g. ``"pii_in_output"``).
        severity: The :data:`Severity` ladder rung.
        count: How many occurrences this finding aggregates.
        record_refs: ``record_id``/``run_id`` references to the offending records.
        subject_refs: ``subject_id`` references to the affected subjects.
        detail: A short, PII-free human description (a category, never a value).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Severity = "medium"
    count: int = 0
    record_refs: list[str] = Field(default_factory=list)
    subject_refs: list[str] = Field(default_factory=list)
    detail: str = ""


class PrivacyMetricResult(BaseModel):
    """One metric's verdict for the audit (score in ``0.0..1.0``).

    A metric is either a live adversarial ``probe`` (scored against the agent) or a
    deterministic ``recorded``-data scan. A metric that cannot run because its seam
    is absent (e.g. a consent metric before Part A lands) is reported as
    ``skipped=True`` rather than a misleading ``score=0`` failure, and is excluded
    from the posture roll-up (must_fix #6).

    Attributes:
        metric: The metric name (matches the registered metric / suite weight key).
        kind: Whether this is a ``probe`` or a ``recorded``-data metric.
        score: The metric verdict in ``0.0..1.0`` (1.0 is fully compliant).
        passed: Whether ``score`` cleared ``threshold``.
        threshold: The pass bar applied to ``score``.
        weight: This metric's weight in the weighted posture aggregate.
        detail: A short, PII-free human summary.
        skipped: True when the metric could not run (excluded from posture).
        findings: PII-safe :class:`PrivacyFinding`s this metric raised.
    """

    metric: str
    kind: MetricKind = "recorded"
    score: float = 0.0
    passed: bool = False
    threshold: float = 0.0
    weight: float = 1.0
    detail: str = ""
    skipped: bool = False
    findings: list[PrivacyFinding] = Field(default_factory=list)


class PrivacyAuditConfig(BaseModel):
    """The scoring policy for one audit run: thresholds, veto metrics, lookback.

    Mirrors how a suite carries its own knobs: ``thresholds`` overrides the
    per-metric pass bar (falling back to ``default_threshold``); ``veto_metrics`` are
    the metrics that, if failed, fail the whole posture regardless of the weighted
    aggregate (default ``('safety',)`` — kept opt-in per must_fix #8); and
    ``lookback_seconds`` bounds how far back the recorded-data half scans
    (``None`` ⇒ unbounded).

    ``provider`` is the wired model-provider name; the LLM probe metrics (the
    aspect-critic / rubric scorers) are gated on it being a *real* provider
    (``provider not in (None, 'stub')`` — must_fix #5), mirroring the probe synthesizer
    gate, so the offline / stub path never lights up a model-backed metric.

    Attributes:
        thresholds: Per-metric pass bar overrides.
        default_threshold: Pass bar for any metric not in ``thresholds``.
        veto_metrics: Metrics whose failure vetoes the whole posture.
        lookback_seconds: Recorded-data scan window (``None`` ⇒ unbounded).
        provider: The wired provider name; ``None``/``'stub'`` ⇒ LLM metrics skipped.
        retention_max_age_seconds: The retention window for ``retention_compliance``
            (``None`` ⇒ the metric is not registered, mirroring ``build_recorded_registry``).
    """

    thresholds: dict[str, float] = Field(default_factory=dict)
    default_threshold: float = 0.7
    veto_metrics: tuple[str, ...] = ("safety",)
    lookback_seconds: float | None = None
    provider: str | None = None
    retention_max_age_seconds: float | None = None

    def threshold_for(self, metric: str) -> float:
        """The pass bar for ``metric`` (its override, else :attr:`default_threshold`)."""
        return self.thresholds.get(metric, self.default_threshold)


class RecordedDataContext(BaseModel):
    """The pre-materialized read-side bundle the recorded-data metrics scan.

    The service ``await``s the async storage surface *first* and hands the metrics
    concrete, already-resolved lists — so a deterministic ``recorded`` metric is a
    pure, synchronous function over data and never touches an async store (must_fix
    #1, the no-sync-awaits-async correction). ``entity_registry`` exposes the
    ``security_event``/``erasure_tombstone``/``consent``/``privacy_audit_report``
    kinds via ``list_by_kind`` (the only kinds that are first-class registry records,
    must_fix #2); runs/memory/episodic are pre-fetched here, not projected as kinds.

    The optional services are the Part A / WS4.2 seams: when absent (``None``) the
    consent-derived metrics report ``skipped`` rather than failing, so Part B is
    independently shippable and lights up automatically once Part A is wired.

    Attributes:
        run_records: Pre-fetched runs in the lookback window.
        memory: Pre-fetched cognitive memory objects.
        episodic: Pre-fetched episodic memory objects.
        security_events: Pre-fetched ``security_event`` records (denials etc.).
        entity_registry: The registry, for ``list_by_kind`` on first-class kinds.
        consent_service: Optional consent ledger (Part A) — ``None`` ⇒ skip its metrics.
        retention_service: Optional retention/erasure service (WS4.2).
        key_vault: Optional per-subject key vault, for erasure-completeness checks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_records: list[RunRecord] = Field(default_factory=list)
    memory: list[MemoryObject] = Field(default_factory=list)
    episodic: list[EpisodicMemoryObject] = Field(default_factory=list)
    security_events: list[SecurityEvent] = Field(default_factory=list)
    entity_registry: EntityRegistryProtocol | None = None
    consent_service: Any = None
    retention_service: RetentionService | None = None
    key_vault: SubjectKeyVault | None = None


class PrivacyAuditReport(BaseModel):
    """The rolled-up scorecard for one audit run — the WS4.7 signed evidence artefact.

    Both halves (the live ``probe`` run and the ``recorded``-data scan) roll into one
    report. ``posture_score`` is the weighted aggregate over the *non-skipped*
    metrics; ``passed`` is the overall verdict (every veto metric passed AND the
    aggregate cleared its bar — the service computes this). The report is registered
    as its own :class:`~himmy.entities.records.EntityRecord`
    (:data:`PRIVACY_AUDIT_REPORT_KIND`) so it inherits the tamper-evident signed
    bundle (must_fix #3 — an ``EvaluationRun`` is not in the registry).

    Attributes:
        report_id: This report's id.
        suite_name: The probe suite that was run (``""`` for a recorded-only audit).
        posture_score: Weighted aggregate over non-skipped metrics (``0.0..1.0``).
        passed: The overall posture verdict.
        metrics: Every :class:`PrivacyMetricResult` (probe + recorded).
        lookback_seconds: The recorded-data scan window used (``None`` ⇒ unbounded).
        workspace_id: Tenant scope this report was run for (``None`` ⇒ global/ops). It
            is projected into the record metadata so the tenant-scoped audit bundle
            filter can exclude another tenant's reports.
        metadata: Derived diagnostics (e.g. counts of records scanned).
        created_at: When the audit ran.
    """

    report_id: str = Field(default_factory=new_uuid)
    suite_name: str = ""
    posture_score: float = 0.0
    passed: bool = False
    metrics: list[PrivacyMetricResult] = Field(default_factory=list)
    lookback_seconds: float | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    @property
    def findings(self) -> list[PrivacyFinding]:
        """All findings across every metric (PII-safe by construction)."""
        return [f for m in self.metrics for f in m.findings]

    def to_record(self) -> EntityRecord:
        """Project this report into a registrable, tamper-evident ``EntityRecord``.

        Only counts and reference ids are projected into the payload — every
        :class:`PrivacyFinding` is already structurally free of raw values, so the
        registered record can never become a PII sink. A **fresh** ``stable_id``
        (:func:`~himmy.core.ids.new_uuid`) is minted per call so re-running the audit
        over an identical store yields a *new* content-addressed version instead of
        colliding with the prior run's ``record_id`` (each report is independently
        signed and verifiable).
        """
        from himmy.entities.records import EntityRecord

        return EntityRecord.create(
            stable_id=new_uuid(),
            version=1,
            kind=PRIVACY_AUDIT_REPORT_KIND,
            payload={
                "report_id": self.report_id,
                "suite_name": self.suite_name,
                "posture_score": self.posture_score,
                "passed": self.passed,
                "lookback_seconds": self.lookback_seconds,
                "metrics": [
                    {
                        "metric": m.metric,
                        "kind": m.kind,
                        "score": m.score,
                        "passed": m.passed,
                        "threshold": m.threshold,
                        "weight": m.weight,
                        "skipped": m.skipped,
                        "detail": m.detail,
                        "findings": [
                            {
                                "code": f.code,
                                "severity": f.severity,
                                "count": f.count,
                                "record_refs": list(f.record_refs),
                                "subject_refs": list(f.subject_refs),
                                "detail": f.detail,
                            }
                            for f in m.findings
                        ],
                    }
                    for m in self.metrics
                ],
            },
            metadata={
                "report_id": self.report_id,
                "suite_name": self.suite_name,
                "passed": self.passed,
                "created_at": self.created_at,
                "workspace_id": self.workspace_id,
            },
        )


__all__ = [
    "PRIVACY_AUDIT_REPORT_KIND",
    "MetricKind",
    "Severity",
    "PrivacyFinding",
    "PrivacyMetricResult",
    "PrivacyAuditConfig",
    "RecordedDataContext",
    "PrivacyAuditReport",
]
