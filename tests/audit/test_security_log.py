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


def test_recent_windows_metadata_filtered_reads(tmp_path: object) -> None:
    """ee sec-r2: metadata-backed filters take the bounded DB-side reader, never a full scan.

    ``workspace_id`` / ``actor_subject`` live in each record's ``metadata``, so a
    metadata-only ``recent()`` must delegate to ``list_recent_by_kind`` (windowed +
    ``LIMIT`` pushed down) and must NOT call the unbounded ``list_by_kind`` that
    materializes+validates the whole audit history. Exercised on BOTH backends.
    """
    import os

    from himmy.entities.sqlite_registry import SqliteEntityRegistry

    mem = EntityRegistry()
    sql = SqliteEntityRegistry(os.path.join(str(tmp_path), "spine.db"))
    try:
        for reg in (mem, sql):
            log = SecurityAuditLog(reg)
            for i in range(50):
                log.record(
                    SecurityEvent(
                        event_type="authz_denied",
                        outcome="deny",
                        workspace_id="t" if i % 2 == 0 else "other",
                        actor={"subject": "mallory" if i % 2 == 0 else "eve"},
                    )
                )

            full_calls = {"n": 0}
            orig_full = reg.list_by_kind

            def _spy(kind: str, _orig: object = orig_full, _c: object = full_calls) -> object:
                _c["n"] += 1  # type: ignore[index]
                return _orig(kind)  # type: ignore[operator]

            reg.list_by_kind = _spy  # type: ignore[method-assign,assignment]

            # Metadata-backed filter → bounded reader only, no full scan, correct rows.
            got = log.recent(limit=5, workspace_id="t", actor_subject="mallory")
            assert full_calls["n"] == 0
            assert len(got) == 5
            assert all(e.workspace_id == "t" for e in got)
            assert all(e.actor["subject"] == "mallory" for e in got)

            # The non-metadata filters (event_type / workspace_ids) still fall back —
            # they must remain correct even though they take the full-history path.
            reg.list_by_kind = orig_full  # type: ignore[method-assign,assignment]
            assert len(log.recent(event_type="authz_denied", limit=100)) == 50
    finally:
        sql.close()


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
