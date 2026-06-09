"""Tests for the WS4.7 privacy-audit typed shapes (B0).

The two structural invariants under test:

* a registered :class:`PrivacyAuditReport` projects ONLY counts/refs into its
  ``EntityRecord`` payload, with a fresh ``stable_id`` per run (no content-address
  collision on re-run); and
* a :class:`PrivacyFinding` can NEVER carry a raw PII value — it forbids extra fields
  and has no ``value``/``text``/``match`` member.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from himmy.entities.integrity import (
    content_hash,
    export_audit_bundle,
    verify_audit_bundle,
)
from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.services.evaluation.privacy import (
    PRIVACY_AUDIT_REPORT_KIND,
    PrivacyAuditConfig,
    PrivacyAuditReport,
    PrivacyFinding,
    PrivacyMetricResult,
    RecordedDataContext,
)


def _report() -> PrivacyAuditReport:
    """A small report with one recorded metric that raised a PII-safe finding."""
    return PrivacyAuditReport(
        suite_name="privacy_baseline",
        posture_score=0.5,
        passed=False,
        metrics=[
            PrivacyMetricResult(
                metric="pii_leakage",
                kind="recorded",
                score=0.5,
                passed=False,
                threshold=0.9,
                weight=2.0,
                detail="1 leak over 2 outputs",
                findings=[
                    PrivacyFinding(
                        code="pii_in_output",
                        severity="high",
                        count=1,
                        record_refs=["run-1"],
                        subject_refs=["teacher_b"],
                        detail="email_address",
                    )
                ],
            )
        ],
    )


# --------------------------------------------------------------- PrivacyFinding


def test_finding_forbids_extra_fields() -> None:
    """A finding rejects any extra field — it can never smuggle a raw value."""
    with pytest.raises(ValidationError):
        PrivacyFinding(code="x", value="alice@example.com")  # type: ignore[call-arg]


def test_finding_has_no_value_like_field() -> None:
    """The model schema defines no field that could hold a raw PII value."""
    field_names = set(PrivacyFinding.model_fields)
    assert "value" not in field_names
    assert "text" not in field_names
    assert "match" not in field_names
    # Only counts + reference ids + a category detail are permitted.
    assert field_names == {
        "code",
        "severity",
        "count",
        "record_refs",
        "subject_refs",
        "detail",
    }


def test_finding_is_frozen() -> None:
    """A finding is immutable once built (no after-the-fact value injection)."""
    finding = PrivacyFinding(code="pii_in_output", count=1)
    with pytest.raises(ValidationError):
        finding.code = "mutated"  # type: ignore[misc]


# ----------------------------------------------------- PrivacyAuditReport.to_record


def test_to_record_builds_registrable_entity_record() -> None:
    """``to_record`` yields a record that registers cleanly under the report kind."""
    registry = EntityRegistry()
    record = _report().to_record()

    assert isinstance(record, EntityRecord)
    assert record.kind == PRIVACY_AUDIT_REPORT_KIND
    stored = registry.register(record)
    assert registry.get(stored.record_id) is not None


def test_to_record_payload_is_counts_and_refs_only() -> None:
    """The projected payload carries only counts/ids — no raw values, no PII sink."""
    report = _report()
    record = report.to_record()

    metric = record.payload["metrics"][0]
    finding = metric["findings"][0]
    assert set(finding) == {
        "code",
        "severity",
        "count",
        "record_refs",
        "subject_refs",
        "detail",
    }
    assert finding["count"] == 1
    assert finding["record_refs"] == ["run-1"]
    assert finding["subject_refs"] == ["teacher_b"]
    # The finding category is preserved; no value field is even projectable.
    assert "value" not in finding


def test_to_record_fresh_stable_id_per_run() -> None:
    """Re-projecting identical content mints a new stable_id (no collision)."""
    report = _report()
    first = report.to_record()
    second = report.to_record()

    assert first.stable_id != second.stable_id
    assert first.record_id != second.record_id
    # Distinct records, so both can coexist + be independently registered.
    registry = EntityRegistry()
    registry.register(first)
    registry.register(second)
    assert len(registry.list_by_kind(PRIVACY_AUDIT_REPORT_KIND)) == 2


def test_to_record_round_trips_through_signed_bundle() -> None:
    """A registered report is covered by the tamper-evident signed bundle."""
    registry = EntityRegistry()
    stored = registry.register(_report().to_record())

    bundle = export_audit_bundle([stored], [], secret="s3cret")
    clean = verify_audit_bundle(bundle, [stored], [], secret="s3cret")
    assert clean.ok

    tampered = stored.model_copy(update={"payload": {"posture_score": 1.0}})
    dirty = verify_audit_bundle(bundle, [tampered], [], secret="s3cret")
    assert not dirty.ok
    assert content_hash(tampered) != content_hash(stored)


# ----------------------------------------------------------- PrivacyMetricResult


def test_metric_result_defaults_and_skip_flag() -> None:
    """A metric defaults to recorded/not-skipped; a skipped metric is explicit."""
    recorded = PrivacyMetricResult(metric="retention_compliance")
    assert recorded.kind == "recorded"
    assert recorded.skipped is False

    skipped = PrivacyMetricResult(metric="consent_coverage", skipped=True)
    assert skipped.skipped is True


def test_report_findings_aggregates_across_metrics() -> None:
    """``report.findings`` flattens findings from every metric."""
    report = PrivacyAuditReport(
        metrics=[
            PrivacyMetricResult(
                metric="pii_leakage",
                findings=[PrivacyFinding(code="a"), PrivacyFinding(code="b")],
            ),
            PrivacyMetricResult(
                metric="erasure_completeness",
                findings=[PrivacyFinding(code="c")],
            ),
        ]
    )
    assert {f.code for f in report.findings} == {"a", "b", "c"}


# ------------------------------------------------------------- PrivacyAuditConfig


def test_config_threshold_resolution() -> None:
    """``threshold_for`` uses the override, else the default."""
    config = PrivacyAuditConfig(thresholds={"pii_leakage": 0.95}, default_threshold=0.7)
    assert config.threshold_for("pii_leakage") == 0.95
    assert config.threshold_for("retention_compliance") == 0.7


def test_config_veto_default_is_safety_only() -> None:
    """Veto stays opt-in: the service default is just ('safety',)."""
    assert PrivacyAuditConfig().veto_metrics == ("safety",)


# ----------------------------------------------------------- RecordedDataContext


def test_recorded_context_optional_seams_default_none() -> None:
    """The Part A / WS4.2 seams default to None so Part B ships standalone."""
    ctx = RecordedDataContext()
    assert ctx.run_records == []
    assert ctx.consent_service is None
    assert ctx.retention_service is None
    assert ctx.key_vault is None
    assert ctx.entity_registry is None


def test_recorded_context_holds_pre_materialized_data() -> None:
    """The context carries concrete lists + the registry (no async store ref)."""
    registry = EntityRegistry()
    ctx = RecordedDataContext(entity_registry=registry)
    assert ctx.entity_registry is registry
    assert isinstance(ctx.security_events, list)
