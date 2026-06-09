"""Tests for the WS4.7 deterministic recorded-data privacy metrics (B1).

Every metric is exercised on BOTH the happy path (a clean store scores ``1.0`` or
``skipped``) AND a planted-failure path:

* ``pii_leakage`` drops below 1.0 when an unredacted email / SSN / Luhn-valid card / a
  Nepal mobile is planted in a run output — and the raised finding carries only LABELS +
  counts, never the matched string (the explicit "no raw value anywhere in the report"
  assertion lives in :func:`test_report_never_contains_raw_pii`).
* ``retention_compliance`` drops when a record is over the age window.
* ``erasure_completeness`` drops when a subject's key survives or a post-erasure
  reference survives, and is ``skipped`` when there are no tombstones.
* ``audit_integrity`` is ``skipped`` on the first run (no baseline) and drops when a
  baselined record's content is mutated.
* the consent-derived metrics are ``skipped=True`` (never ``score=0``) until Part A.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from himmy.entities.integrity import content_hash, export_audit_bundle
from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.services.evaluation.privacy import (
    AuditIntegrityMetric,
    ErasureCompletenessMetric,
    PiiLeakageMetric,
    PrivacyAuditReport,
    PrivacyMetricResult,
    RecordedDataContext,
    RetentionComplianceMetric,
    build_recorded_registry,
)
from himmy.services.governance.retention import (
    ERASURE_KIND,
    RetentionService,
    SubjectKeyVault,
)
from himmy.services.storage.models import MemoryObject, RunRecord, RunStatus

# A fake-but-format-valid card (passes Luhn) and a fake SSN — no real PII anywhere.
_FAKE_CARD = "4242424242424242"
_FAKE_SSN = "123-45-6789"
_FAKE_EMAIL = "teacher.b@example.com"
_FAKE_NP_MOBILE = "9812345678"


def _run(
    *,
    subject_id: str = "teacher_a",
    output_text: str | None = "the lesson summary is ready",
    created_at: str | None = None,
) -> RunRecord:
    """A minimal SUCCEEDED run with a clean output by default."""
    kwargs: dict[str, object] = {
        "workspace_id": "ws-1",
        "subject_id": subject_id,
        "status": RunStatus.SUCCEEDED,
        "output_text": output_text,
    }
    if created_at is not None:
        kwargs["created_at"] = created_at
    return RunRecord(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------- pii_leakage


def test_pii_leakage_clean_store_scores_one() -> None:
    """A store with no PII in any output scores a perfect 1.0 with no findings."""
    ctx = RecordedDataContext(run_records=[_run(), _run(subject_id="teacher_b")])
    result = PiiLeakageMetric().score(ctx)

    assert result.metric == "pii_leakage"
    assert result.kind == "recorded"
    assert result.score == 1.0
    assert result.passed is True
    assert result.skipped is False
    assert result.findings == []


def test_pii_leakage_empty_store_scores_one() -> None:
    """Nothing to scan is vacuously compliant (1.0), not a zero failure."""
    result = PiiLeakageMetric().score(RecordedDataContext())
    assert result.score == 1.0
    assert result.passed is True


def test_pii_leakage_detects_planted_pii_with_labels_only() -> None:
    """A planted email/SSN/card + a Nepal mobile drop the score with LABEL-only finding."""
    leaky = _run(
        subject_id="teacher_b",
        output_text=(
            f"contact {_FAKE_EMAIL}, ssn {_FAKE_SSN}, card {_FAKE_CARD}, "
            f"mobile {_FAKE_NP_MOBILE}"
        ),
    )
    ctx = RecordedDataContext(run_records=[_run(), leaky])
    result = PiiLeakageMetric().score(ctx)

    # 1 of 2 scanned outputs leaked.
    assert result.score == 0.5
    assert result.passed is False
    assert len(result.findings) == 1

    finding = result.findings[0]
    assert finding.code == "pii_in_output"
    assert finding.severity == "high"
    assert finding.record_refs == [leaky.run_id]
    assert finding.subject_refs == ["teacher_b"]

    # The detail is LABELS only — the categories that matched, never the values.
    labels = set(finding.detail.split(","))
    assert {"email", "ssn", "card"} <= labels
    assert "np_mobile" in labels  # the Nepal rule set is included (must_fix #6)
    # 4 distinct matches counted.
    assert finding.count >= 4

    # CRITICAL: no raw matched value leaked into the finding anywhere.
    blob = finding.model_dump_json()
    for raw in (_FAKE_EMAIL, _FAKE_SSN, _FAKE_CARD, _FAKE_NP_MOBILE):
        assert raw not in blob


def test_pii_leakage_scans_structured_and_error_surfaces() -> None:
    """Structured output and the error field are scanned, not just output_text."""
    run = RunRecord(
        workspace_id="ws-1",
        subject_id="teacher_b",
        status=RunStatus.FAILED,
        output_text=None,
        output_structured={"note": f"reach me at {_FAKE_EMAIL}"},
        error=f"failed sending to {_FAKE_NP_MOBILE}",
    )
    result = PiiLeakageMetric().score(RecordedDataContext(run_records=[run]))
    # Both surfaces leaked ⇒ 2/2 scanned outputs leaked ⇒ score 0.0.
    assert result.score == 0.0
    assert len(result.findings) == 2


# ------------------------------------------------------------------ retention


def _iso(delta_seconds: float) -> str:
    """An ISO timestamp ``delta_seconds`` in the past."""
    return (datetime.now(UTC) - timedelta(seconds=delta_seconds)).isoformat()


def test_retention_compliance_all_fresh_scores_one() -> None:
    """When every record is inside the window the metric is fully compliant."""
    registry = EntityRegistry()
    metric = RetentionComplianceMetric(
        retention_service=RetentionService(registry),
        max_age_seconds=3600,
    )
    ctx = RecordedDataContext(
        run_records=[_run(created_at=_iso(10)), _run(created_at=_iso(60))],
    )
    result = metric.score(ctx)
    assert result.score == 1.0
    assert result.passed is True
    assert result.findings == []


def test_retention_compliance_flags_over_age_record() -> None:
    """An over-age record drops the score and is referenced by id only."""
    registry = EntityRegistry()
    metric = RetentionComplianceMetric(
        retention_service=RetentionService(registry),
        max_age_seconds=3600,
    )
    fresh = _run(created_at=_iso(10))
    stale = _run(subject_id="teacher_b", created_at=_iso(7200))
    stale_mem = MemoryObject(subject_id="teacher_b", created_at=_iso(99999))
    ctx = RecordedDataContext(run_records=[fresh, stale], memory=[stale_mem])

    result = metric.score(ctx)
    # 2 of 3 scanned records expired.
    assert abs(result.score - (1.0 - 2 / 3)) < 1e-9
    assert result.passed is False
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.count == 2
    assert stale.run_id in finding.record_refs
    assert stale_mem.memory_id in finding.record_refs
    assert "teacher_b" in finding.subject_refs


# --------------------------------------------------------- erasure_completeness


def _erase(registry: EntityRegistry, vault: SubjectKeyVault, subject: str) -> None:
    """Crypto-shred ``subject`` and register a tombstone via the real service."""
    RetentionService(registry, key_vault=vault).erase_subject(subject, reason="test")


def test_erasure_completeness_skipped_without_tombstones() -> None:
    """No tombstones ⇒ skipped (not a misleading zero)."""
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    metric = ErasureCompletenessMetric(key_vault=vault)
    result = metric.score(RecordedDataContext(entity_registry=registry))
    assert result.skipped is True
    assert result.passed is False  # default; excluded from posture by the service


def test_erasure_completeness_clean_erasure_scores_one() -> None:
    """A shredded subject with no surviving references is a complete erasure."""
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    vault.key_for("teacher_a")  # the subject once had a key
    _erase(registry, vault, "teacher_a")  # crypto-shred + tombstone

    metric = ErasureCompletenessMetric(key_vault=vault)
    # No run/memory references the erased subject post-erasure (only an unrelated run).
    ctx = RecordedDataContext(
        run_records=[_run(subject_id="teacher_a_other")],
        entity_registry=registry,
    )
    result = metric.score(ctx)
    assert result.skipped is False
    assert result.score == 1.0
    assert result.passed is True
    assert result.findings == []


def test_erasure_completeness_flags_surviving_key() -> None:
    """A tombstone present but the key NOT shredded is an incomplete erasure."""
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    # Register a tombstone WITHOUT shredding the key (simulate a half-done erasure).
    registry.register(
        EntityRecord.create(
            stable_id="ts-1",
            version=1,
            kind=ERASURE_KIND,
            payload={"subject_id": "teacher_a", "crypto_shredded": False},
        )
    )
    vault.key_for("teacher_a")  # key still present ⇒ not shredded

    result = ErasureCompletenessMetric(key_vault=vault).score(
        RecordedDataContext(entity_registry=registry)
    )
    assert result.score == 0.0
    assert result.passed is False
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.subject_refs == ["teacher_a"]
    assert "key_not_shredded" in finding.detail


def test_erasure_completeness_flags_surviving_reference() -> None:
    """A post-erasure run still referencing the subject is an incomplete erasure."""
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    vault.key_for("teacher_a")
    _erase(registry, vault, "teacher_a")  # key shredded + tombstone

    surviving = _run(subject_id="teacher_a")  # a cleartext reference that survived
    result = ErasureCompletenessMetric(key_vault=vault).score(
        RecordedDataContext(run_records=[surviving], entity_registry=registry)
    )
    assert result.score == 0.0
    assert result.passed is False
    finding = result.findings[0]
    assert "post_erasure_reference" in finding.detail
    assert surviving.run_id in finding.record_refs


def test_erasure_completeness_flags_bundle_verify_failure() -> None:
    """A failing bundle re-verification is surfaced as an incomplete erasure."""
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    vault.key_for("teacher_a")
    _erase(registry, vault, "teacher_a")

    result = ErasureCompletenessMetric(
        key_vault=vault, verify_bundle=lambda: False
    ).score(RecordedDataContext(entity_registry=registry))
    assert result.score == 0.0
    assert "bundle_verify_failed" in result.findings[0].detail


# ------------------------------------------------------------------ audit_integrity


def test_audit_integrity_first_run_is_skipped() -> None:
    """With no baseline (a fresh deployment) the metric skips, never falsely fails."""
    registry = EntityRegistry()
    result = AuditIntegrityMetric(baseline=None).score(
        RecordedDataContext(entity_registry=registry)
    )
    assert result.skipped is True
    assert result.passed is False


def test_audit_integrity_clean_against_baseline_scores_one() -> None:
    """An untouched store matches its signed baseline exactly."""
    registry = EntityRegistry()
    stored = registry.register(
        EntityRecord.create(
            stable_id="sec-1",
            version=1,
            kind="security_event",
            payload={"event_type": "access", "subject_id": "teacher_a"},
        )
    )
    baseline = export_audit_bundle([stored], [], secret="s3cret")

    result = AuditIntegrityMetric(baseline=baseline).score(
        RecordedDataContext(entity_registry=registry)
    )
    assert result.skipped is False
    assert result.score == 1.0
    assert result.passed is True


def test_audit_integrity_detects_tampered_record() -> None:
    """A record mutated after the baseline drags the score and raises a finding."""
    registry = EntityRegistry()
    original = EntityRecord.create(
        stable_id="sec-1",
        version=1,
        kind="security_event",
        payload={"event_type": "access", "subject_id": "teacher_a"},
    )
    registry.register(original)
    # Baseline captures the ORIGINAL content hash.
    baseline = {original.record_id: content_hash(original)}

    # Now mutate the stored row in place (same id, different payload).
    tampered = original.model_copy(update={"payload": {"event_type": "TAMPERED"}})
    registry._records[original.record_id] = tampered  # noqa: SLF001 - test fixture

    result = AuditIntegrityMetric(baseline=baseline).score(
        RecordedDataContext(entity_registry=registry)
    )
    assert result.score == 0.0
    assert result.passed is False
    assert result.findings[0].code == "audit_record_tampered"
    assert original.record_id in result.findings[0].record_refs


# ----------------------------------------------- consent-derived (None-guarded)


def test_consent_metrics_are_skipped_not_zero() -> None:
    """Until Part A is wired the consent metrics report skipped, never score=0."""
    registry = EntityRegistry()
    metrics = build_recorded_registry(retention_service=RetentionService(registry))
    consent_names = {
        "consent_coverage",
        "purpose_limitation",
        "denial_enforcement",
    }
    by_name = {m.name: m for m in metrics if m.name in consent_names}
    assert set(by_name) == consent_names

    ctx = RecordedDataContext(consent_service=None, entity_registry=registry)
    for name in consent_names:
        result = by_name[name].score(ctx)
        assert result.skipped is True, name
        assert result.score == 0.0  # default, but excluded from posture because skipped


# ----------------------------------------------------------- build_recorded_registry


def test_build_registry_includes_only_present_sources() -> None:
    """A metric is registered only when its data source is wired."""
    # Bare: only the always-on metrics + the (skipped) consent metrics.
    bare = {m.name for m in build_recorded_registry()}
    assert "pii_leakage" in bare
    assert "audit_integrity" in bare
    assert "retention_compliance" not in bare  # no retention_service/max_age
    assert "erasure_completeness" not in bare  # no key_vault

    # Fully wired.
    registry = EntityRegistry()
    full = {
        m.name
        for m in build_recorded_registry(
            retention_service=RetentionService(registry),
            max_age_seconds=3600,
            key_vault=SubjectKeyVault(),
        )
    }
    assert "retention_compliance" in full
    assert "erasure_completeness" in full


def test_build_registry_retention_needs_both_service_and_age() -> None:
    """retention_compliance is gated on BOTH the service AND a max-age window."""
    registry = EntityRegistry()
    # Service but no max_age ⇒ not registered.
    names = {
        m.name
        for m in build_recorded_registry(retention_service=RetentionService(registry))
    }
    assert "retention_compliance" not in names


# --------------------------------------------- explicit no-raw-PII-in-report guard


def test_report_never_contains_raw_pii() -> None:
    """The serialized full report must contain NO raw matched PII string anywhere.

    This is the headline structural guarantee: a planted leak surfaces a finding, but
    the rolled-up :class:`PrivacyAuditReport` (the signed evidence artefact) is scanned
    end-to-end and proven free of every raw value.
    """
    leaky = _run(
        subject_id="teacher_b",
        output_text=(
            f"email {_FAKE_EMAIL} ssn {_FAKE_SSN} card {_FAKE_CARD} "
            f"mobile {_FAKE_NP_MOBILE}"
        ),
    )
    ctx = RecordedDataContext(run_records=[leaky])
    leakage = PiiLeakageMetric().score(ctx)
    assert not leakage.passed  # the leak was caught

    report = PrivacyAuditReport(
        suite_name="privacy_baseline",
        metrics=[leakage, PrivacyMetricResult(metric="audit_integrity", skipped=True)],
    )

    # Scan EVERY serialization surface of the report + its projected EntityRecord.
    surfaces = [
        report.model_dump_json(),
        json.dumps(report.to_record().payload),
        json.dumps(report.to_record().metadata),
    ]
    for raw in (_FAKE_EMAIL, _FAKE_SSN, _FAKE_CARD, _FAKE_NP_MOBILE):
        for surface in surfaces:
            assert raw not in surface
