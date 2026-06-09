"""Tests for the WS4.7 :class:`PrivacyAuditService` orchestrator (Part B, B3).

Covers the headline guarantees of the audit service:

* **Offline / zero-config** — a recorded-only audit over an empty in-memory store
  produces a report with the LLM probe metrics and the consent-derived metrics all
  ``skipped`` (and therefore excluded from the posture), and registers it.
* **Signed evidence** — the registered report exports a signed bundle that verifies, and
  ``verify_report().ok`` flips to ``False`` when a cited record is tampered with.
* **No content-address collision** — re-running over an identical store registers two
  distinct ``privacy_audit_report`` records (fresh ``stable_id`` per run, must_fix #3).
* **No un-awaited coroutine** — the :class:`RecordedDataContext` the recorded half scans
  carries concrete lists, never a coroutine (must_fix #1, no sync-awaits-async).
* **Posture excludes skipped + planted-leak detection** — a planted PII leak drags
  ``pii_leakage`` and the overall posture below the bar with a counts-only finding.
* **Provider gating** — the LLM probe synthesizer/metrics stay off for the stub provider.
* **Probe half** — wiring a runtime + persona runs the adversarial suite and rolls it
  into a ``safety`` probe metric; a refusing offline agent passes.
* **Bundle without a key** — explicitly requesting a bundle with no signing key raises
  the 503-equivalent error.
* **Trend** — registered reports come back newest-first.
"""

from __future__ import annotations

import json

from himmy import build_runtime
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.services.evaluation.privacy import (
    PRIVACY_AUDIT_REPORT_KIND,
    PrivacyAuditConfig,
    PrivacyAuditReport,
    PrivacyAuditService,
)
from himmy.services.evaluation.service import EvaluationService
from himmy.services.governance.retention import (
    RetentionService,
    SubjectKeyVault,
)
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from tests.conftest import run_async

# Fake-but-format-valid tokens — no real PII anywhere in the suite.
_FAKE_EMAIL = "teacher.b@example.com"
_FAKE_SSN = "123-45-6789"
_FAKE_CARD = "4242424242424242"
_FAKE_NP_MOBILE = "9812345678"

_SECRET = "audit-test-secret"


def _service(
    *,
    registry: EntityRegistry,
    storage: StorageService | None = None,
    runtime: object | None = None,
    config: PrivacyAuditConfig | None = None,
    key_vault: SubjectKeyVault | None = None,
    retention_service: RetentionService | None = None,
) -> PrivacyAuditService:
    """Build a :class:`PrivacyAuditService` over an eval kernel + (optional) store."""
    eval_service = EvaluationService(storage_service=storage)
    return PrivacyAuditService(
        entity_registry=registry,
        evaluation_service=eval_service,
        runtime=runtime,
        retention_service=retention_service,
        key_vault=key_vault,
        config=config,
    )


def _set_secret(monkeypatch: object) -> None:
    """Install an HMAC signing secret via the env-backed secret provider."""
    import himmy.config.secrets as secrets

    monkeypatch.setenv("HIMMY_AUDIT_SECRET", _SECRET)  # type: ignore[attr-defined]
    # The provider caches the first build; reset it so the env var is picked up.
    secrets._PROVIDER = None  # noqa: SLF001 - test fixture


# ----------------------------------------------------------- offline / zero-config


def test_offline_zero_config_run_produces_report() -> None:
    """A recorded-only audit over an empty store skips LLM + consent metrics, registers."""
    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService())

    report = run_async(service.run())

    assert isinstance(report, PrivacyAuditReport)
    assert report.suite_name == ""  # no probe half (no runtime/persona)
    assert report.metadata["probe_ran"] is False
    assert report.metadata["provider_real"] is False

    by_name = {m.metric: m for m in report.metrics}
    # The consent-derived metrics are present but skipped (Part A not wired).
    for name in ("consent_coverage", "purpose_limitation", "denial_enforcement"):
        assert by_name[name].skipped is True
    # audit_integrity self-skips on the first run (no baseline).
    assert by_name["audit_integrity"].skipped is True
    # pii_leakage is always-on; an empty store is vacuously compliant.
    assert by_name["pii_leakage"].skipped is False
    assert by_name["pii_leakage"].score == 1.0

    # The report was registered as its own EntityRecord.
    registered = registry.list_by_kind(PRIVACY_AUDIT_REPORT_KIND)
    assert len(registered) == 1
    assert registered[0].metadata["report_id"] == report.report_id


def test_skipped_metrics_excluded_from_posture() -> None:
    """A run that is all-skipped-except-clean scores a passing posture (must_fix #7)."""
    registry = EntityRegistry()
    report = run_async(_service(registry=registry, storage=StorageService()).run())

    # Only the non-skipped metrics drive the posture; the clean store passes.
    scored = [m for m in report.metrics if not m.skipped]
    assert all(m.passed for m in scored)
    assert report.posture_score == 1.0
    assert report.passed is True


# --------------------------------------------------- no un-awaited coroutine (must_fix #1)


def test_recorded_context_carries_concrete_lists() -> None:
    """The prefetched RecordedDataContext holds concrete lists, never a coroutine."""
    import inspect

    registry = EntityRegistry()
    storage = StorageService()
    run_async(
        storage.save_run(
            RunRecord(
                workspace_id="ws-1",
                subject_id="teacher_a",
                status=RunStatus.SUCCEEDED,
                output_text="clean lesson summary",
            )
        )
    )
    eval_service = EvaluationService(storage_service=storage)
    service = PrivacyAuditService(
        entity_registry=registry, evaluation_service=eval_service
    )

    ctx = run_async(service._build_context(workspace_id=None))  # noqa: SLF001 - test

    assert isinstance(ctx.run_records, list)
    assert isinstance(ctx.memory, list)
    assert isinstance(ctx.episodic, list)
    assert len(ctx.run_records) == 1
    for collection in (ctx.run_records, ctx.memory, ctx.episodic):
        for item in collection:
            assert not inspect.iscoroutine(item)
    assert ctx.entity_registry is registry


# ------------------------------------------------------------- signed evidence


def test_report_registers_and_bundle_verifies(monkeypatch: object) -> None:
    """The registered report exports a signed bundle that verifies clean."""
    _set_secret(monkeypatch)
    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService())

    report = run_async(service.run())
    bundle = service.export_signed_bundle(report)
    result = service.verify_report(report, bundle)

    assert result.ok is True
    assert result.signature_valid is True


def test_tampering_cited_record_flips_verify_false(monkeypatch: object) -> None:
    """Mutating the registered report record in place flips verify_report().ok False."""
    _set_secret(monkeypatch)
    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService())

    report = run_async(service.run())
    bundle = service.export_signed_bundle(report)
    assert service.verify_report(report, bundle).ok is True

    # Tamper with the cited report record in place (same id, different payload).
    stored = registry.list_by_kind(PRIVACY_AUDIT_REPORT_KIND)[0]
    tampered = stored.model_copy(
        update={"payload": {**stored.payload, "passed": False}}
    )
    registry._records[stored.record_id] = tampered  # noqa: SLF001 - test fixture

    result = service.verify_report(report, bundle)
    assert result.ok is False
    assert stored.record_id in result.tampered_record_ids


def test_tampering_a_cited_security_event_flips_verify_false(
    monkeypatch: object,
) -> None:
    """A bundle commits to cited security_events too — tampering one flips verify False."""
    _set_secret(monkeypatch)
    registry = EntityRegistry()
    event = registry.register(
        EntityRecord.create(
            stable_id="sec-1",
            version=1,
            kind="security_event",
            payload={"event_type": "consent_denied_persist", "subject_id": "teacher_b"},
        )
    )
    service = _service(registry=registry, storage=StorageService())

    report = run_async(service.run())
    bundle = service.export_signed_bundle(report)
    assert service.verify_report(report, bundle).ok is True

    tampered = event.model_copy(update={"payload": {"event_type": "ALTERED"}})
    registry._records[event.record_id] = tampered  # noqa: SLF001 - test fixture

    result = service.verify_report(report, bundle)
    assert result.ok is False
    assert event.record_id in result.tampered_record_ids


# ------------------------------------------------- no content-address collision (#3)


def test_rerun_twice_no_content_address_collision() -> None:
    """Re-running over an identical store registers two distinct report records."""
    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService())

    first = run_async(service.run())
    second = run_async(service.run())

    assert first.report_id != second.report_id
    registered = registry.list_by_kind(PRIVACY_AUDIT_REPORT_KIND)
    assert len(registered) == 2
    # Distinct stable_ids ⇒ no idempotent de-dup / content-address violation.
    assert registered[0].stable_id != registered[1].stable_id


# ------------------------------------------------------ planted-leak detection


def test_planted_pii_leak_drops_posture_with_counts_only_finding() -> None:
    """A planted PII leak drags pii_leakage + posture below the bar, no raw value leaks."""
    registry = EntityRegistry()
    storage = StorageService()
    run_async(
        storage.save_run(
            RunRecord(
                workspace_id="ws-1",
                subject_id="teacher_b",
                status=RunStatus.SUCCEEDED,
                output_text=(
                    f"contact {_FAKE_EMAIL}, ssn {_FAKE_SSN}, card {_FAKE_CARD}, "
                    f"mobile {_FAKE_NP_MOBILE}"
                ),
            )
        )
    )
    service = _service(registry=registry, storage=storage)

    report = run_async(service.run())
    by_name = {m.metric: m for m in report.metrics}

    assert by_name["pii_leakage"].score == 0.0
    assert by_name["pii_leakage"].passed is False
    assert report.passed is False  # posture failed

    finding = by_name["pii_leakage"].findings[0]
    assert finding.code == "pii_in_output"
    assert finding.subject_refs == ["teacher_b"]

    # The serialized report + its registered record carry NO raw matched value.
    surfaces = [
        report.model_dump_json(),
        json.dumps(report.to_record().payload),
    ]
    for raw in (_FAKE_EMAIL, _FAKE_SSN, _FAKE_CARD, _FAKE_NP_MOBILE):
        for surface in surfaces:
            assert raw not in surface


# ------------------------------------------------------------- probe half


def test_probe_half_runs_against_offline_agent() -> None:
    """A wired runtime + persona runs the adversarial suite into a safety probe metric."""
    runtime, _inf, _tools = build_runtime()
    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService(), runtime=runtime)

    report = run_async(service.run(persona=Persona(name="tutor")))

    assert report.suite_name == "privacy_probes"
    assert report.metadata["probe_ran"] is True
    safety = {m.metric: m for m in report.metrics}["safety"]
    assert safety.kind == "probe"
    # The offline echo agent clears the (lenient) probe bar; the score is the fraction of
    # adversarial cases that did NOT elicit a disallowed disclosure.
    assert safety.passed is True
    assert 0.0 < safety.score <= 1.0
    assert safety.threshold == 0.7


def test_probe_half_skipped_without_persona() -> None:
    """A runtime but no persona ⇒ the probe half is skipped (recorded-only audit)."""
    runtime, _inf, _tools = build_runtime()
    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService(), runtime=runtime)

    report = run_async(service.run())  # no persona
    assert report.metadata["probe_ran"] is False
    assert all(m.kind == "recorded" for m in report.metrics)


# ----------------------------------------------------- bundle without a key (503)


def test_export_bundle_without_key_raises(monkeypatch: object) -> None:
    """Requesting a bundle with no signing key raises the 503-equivalent error."""
    import himmy.config.secrets as secrets

    monkeypatch.delenv("HIMMY_AUDIT_SECRET", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)  # type: ignore[attr-defined]
    secrets._PROVIDER = None  # noqa: SLF001 - test fixture

    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService())
    report = run_async(service.run())

    try:
        service.export_signed_bundle(report)
    except HimmyError as exc:
        assert "signing key not configured" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a HimmyError when no signing key is configured")


# --------------------------------------------------------------- trend


def test_trend_returns_reports_newest_first() -> None:
    """trend() returns registered reports ordered by created_at, newest first."""
    registry = EntityRegistry()
    service = _service(registry=registry, storage=StorageService())

    older = run_async(service.run())
    # Force a strictly-later timestamp on the second report's registered record.
    newer = run_async(service.run())

    # Patch created_at metadata to guarantee a deterministic ordering.
    records = registry.list_by_kind(PRIVACY_AUDIT_REPORT_KIND)
    by_report = {r.metadata["report_id"]: r for r in records}
    registry._records[by_report[older.report_id].record_id] = by_report[  # noqa: SLF001
        older.report_id
    ].model_copy(
        update={
            "metadata": {
                **by_report[older.report_id].metadata,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        }
    )
    registry._records[by_report[newer.report_id].record_id] = by_report[  # noqa: SLF001
        newer.report_id
    ].model_copy(
        update={
            "metadata": {
                **by_report[newer.report_id].metadata,
                "created_at": "2026-12-31T00:00:00+00:00",
            }
        }
    )

    trend = service.trend(limit=10)
    assert len(trend) == 2
    assert trend[0].report_id == newer.report_id
    assert trend[1].report_id == older.report_id


# ----------------------------------------------- retention / erasure metrics wired


def test_retention_and_erasure_metrics_light_up_when_wired() -> None:
    """Wiring retention + key_vault + a max-age registers those recorded metrics."""
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    retention = RetentionService(registry, key_vault=vault)
    service = _service(
        registry=registry,
        storage=StorageService(),
        key_vault=vault,
        retention_service=retention,
        config=PrivacyAuditConfig(retention_max_age_seconds=3600),
    )

    report = run_async(service.run())
    names = {m.metric for m in report.metrics}
    assert "retention_compliance" in names
    # No tombstones yet ⇒ erasure_completeness self-skips.
    by_name = {m.metric: m for m in report.metrics}
    assert by_name["erasure_completeness"].skipped is True
