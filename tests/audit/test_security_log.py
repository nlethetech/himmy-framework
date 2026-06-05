"""WS1.4 — SecurityAuditLog: record/query + tamper-evidence of the trail."""

from __future__ import annotations

from himmy.entities.integrity import export_audit_bundle, verify_audit_bundle
from himmy.entities.registry import EntityRegistry
from himmy.services.audit import SECURITY_EVENT_KIND, SecurityAuditLog, SecurityEvent


def _log() -> tuple[SecurityAuditLog, EntityRegistry]:
    reg = EntityRegistry()
    return SecurityAuditLog(reg), reg


def test_record_and_recent_filters() -> None:
    log, _reg = _log()
    log.record(
        SecurityEvent(
            event_type="access",
            outcome="allow",
            workspace_id="t",
            actor={"subject": "a"},
        )
    )
    log.record(
        SecurityEvent(
            event_type="authz_denied",
            outcome="deny",
            workspace_id="t",
            actor={"subject": "b"},
        )
    )
    log.record(
        SecurityEvent(
            event_type="access",
            outcome="allow",
            workspace_id="other",
            actor={"subject": "c"},
        )
    )
    assert len(log.recent(workspace_id="t")) == 2
    denied = log.recent(event_type="authz_denied")
    assert len(denied) == 1 and denied[0].actor["subject"] == "b"
    assert len(log.recent()) == 3
    assert len(log.recent(limit=1)) == 1


def test_events_are_stored_as_security_event_entities() -> None:
    log, reg = _log()
    log.record(SecurityEvent(event_type="access", actor={"subject": "a"}))
    records = reg.list_by_kind(SECURITY_EVENT_KIND)
    assert len(records) == 1
    assert records[0].payload["event_type"] == "access"


def test_audit_trail_is_tamper_evident() -> None:
    """A signed bundle over the audit entities detects any after-the-fact edit."""
    log, reg = _log()
    log.record(
        SecurityEvent(event_type="auth_failure", outcome="deny", actor={"subject": "x"})
    )
    log.record(
        SecurityEvent(event_type="access", outcome="allow", actor={"subject": "y"})
    )
    records = reg.list_by_kind(SECURITY_EVENT_KIND)

    bundle = export_audit_bundle(records, [], secret="audit-key")
    assert verify_audit_bundle(bundle, records, [], secret="audit-key").ok

    # Tamper with one event's payload (e.g. flip a deny to an allow) → detected.
    tampered = list(records)
    tampered[0] = tampered[0].model_copy(
        update={"payload": {**tampered[0].payload, "outcome": "allow"}}
    )
    result = verify_audit_bundle(bundle, tampered, [], secret="audit-key")
    assert not result.ok
    assert tampered[0].record_id in result.tampered_record_ids
