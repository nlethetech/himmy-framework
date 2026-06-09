"""WS4.7 B4 — the ``himmy audit privacy`` CLI (a CI gate over the privacy posture).

Asserts:

* a clean store exits ``0`` and prints a PASS scorecard;
* a planted PII leak fails the posture and exits ``1`` with a counts-only finding (and the
  raw planted value never appears in the scorecard / JSON);
* ``--export-bundle`` writes a verifiable signed bundle over the registered report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import himmy.cli.audit as cli_audit
from himmy.cli.__main__ import main
from himmy.entities.integrity import AuditBundle, verify_audit_bundle
from himmy.entities.registry import EntityRegistry
from himmy.services.audit.log import SecurityAuditLog
from himmy.services.evaluation.privacy import PrivacyAuditService
from himmy.services.evaluation.service import EvaluationService
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from tests.conftest import run_async

# Fake-but-format-valid token — no real PII.
_FAKE_EMAIL = "leaked.teacher@example.com"


def _service_over(storage: StorageService) -> PrivacyAuditService:
    """A privacy service over a fresh in-memory registry + the given storage."""
    registry = EntityRegistry()
    return PrivacyAuditService(
        entity_registry=registry,
        evaluation_service=EvaluationService(storage_service=storage),
        security_audit=SecurityAuditLog(registry),
    )


# ----------------------------------------------------------------- clean store
def test_clean_store_exits_zero_with_scorecard(monkeypatch: Any, capsys: Any) -> None:
    """An empty store ⇒ passing posture ⇒ exit 0 + a PASS scorecard."""
    monkeypatch.setattr(
        cli_audit, "_build_service", lambda: _service_over(StorageService())
    )
    rc = main(["audit", "privacy"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "privacy audit: PASS" in out
    assert "pii_leakage" in out


def test_clean_store_json(monkeypatch: Any, capsys: Any) -> None:
    """``--json`` prints a machine-readable report and still exits 0 on a clean store."""
    import json

    monkeypatch.setattr(
        cli_audit, "_build_service", lambda: _service_over(StorageService())
    )
    rc = main(["audit", "privacy", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert any(m["metric"] == "pii_leakage" for m in payload["metrics"])


# --------------------------------------------------------------- failing posture
def _leaky_storage() -> StorageService:
    """A storage holding one recorded run whose output leaks a PII value."""
    storage = StorageService()
    run_async(
        storage.save_run(
            RunRecord(
                workspace_id="ws-1",
                subject_id="teacher_b",
                status=RunStatus.SUCCEEDED,
                output_text=f"reach me at {_FAKE_EMAIL}",
            )
        )
    )
    return storage


def test_failing_posture_exits_one(monkeypatch: Any, capsys: Any) -> None:
    """A planted PII leak fails the posture ⇒ exit 1 with a counts-only finding."""
    monkeypatch.setattr(
        cli_audit, "_build_service", lambda: _service_over(_leaky_storage())
    )
    rc = main(["audit", "privacy"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "privacy audit: FAIL" in out
    assert "pii_in_output" in out
    # The raw planted value never appears in the scorecard (PII-safe by construction).
    assert _FAKE_EMAIL not in out


def test_failing_posture_json_has_no_raw_value(monkeypatch: Any, capsys: Any) -> None:
    """The JSON scorecard for a failing posture carries no raw matched value."""
    monkeypatch.setattr(
        cli_audit, "_build_service", lambda: _service_over(_leaky_storage())
    )
    rc = main(["audit", "privacy", "--json"])
    assert rc == 1
    assert _FAKE_EMAIL not in capsys.readouterr().out


# --------------------------------------------------------------- export bundle
def test_export_bundle_writes_verifiable_bundle(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """``--export-bundle`` writes a signed bundle that verifies and covers the report."""
    import himmy.config.secrets as secrets

    monkeypatch.setenv("HIMMY_AUDIT_SECRET", "cli-audit-secret")
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    secrets._PROVIDER = None  # noqa: SLF001 - reset the cached provider

    service = _service_over(StorageService())
    monkeypatch.setattr(cli_audit, "_build_service", lambda: service)

    out_path = tmp_path / "bundle.json"
    rc = main(["audit", "privacy", "--export-bundle", str(out_path)])
    assert rc == 0
    assert out_path.is_file()

    bundle = AuditBundle.model_validate_json(out_path.read_text())
    assert bundle.algorithm == "HMAC-SHA256"
    # The bundle covers the privacy_audit_report the run registered (record_id -> hash map).
    report_ids = {
        r.record_id
        for r in service._registry.list_by_kind("privacy_audit_report")  # noqa: SLF001
    }
    assert report_ids
    assert report_ids <= set(bundle.records)
    assert verify_audit_bundle(
        bundle, [], [], secret="cli-audit-secret"
    ).signature_valid


def test_export_bundle_without_key_exits_one(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    """Requesting a bundle with no signing key configured ⇒ a non-zero exit + stderr error."""
    import himmy.config.secrets as secrets

    monkeypatch.delenv("HIMMY_AUDIT_SECRET", raising=False)
    monkeypatch.delenv("HIMMY_AUDIT_PRIVATE_KEY", raising=False)
    secrets._PROVIDER = None  # noqa: SLF001 - reset the cached provider

    monkeypatch.setattr(
        cli_audit, "_build_service", lambda: _service_over(StorageService())
    )
    rc = main(["audit", "privacy", "--export-bundle", str(tmp_path / "b.json")])
    assert rc == 1
    assert "signing key not configured" in capsys.readouterr().err
