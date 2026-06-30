"""red-team scope-r5: by-construction tenant scoping for the lineage / seclog / infra reads.

Four CONFIRMED cross-tenant / posture-disclosure holes a TENANT-BOUND ``admin`` (or any
tenant browse role) could reach once auth is configured — all closed by routing the read
through the centralized scoping the rest of the app enforces, and each asserted byte-unchanged
on the offline / ``all_tenants`` path:

* ``GET /api/studio/lineage/entity/{id}`` + ``/graph`` — a tenant-bound admin read every
  tenant's record payloads (prompts/persona/context) and run-lineage subgraphs because the
  readers did NO tenant scoping. Now the run path threads ``resolve_workspace`` into
  ``get_run_lineage`` and the entity path refuses a subgraph whose owning ``chat_thread``
  workspace is outside the caller's tenant allow-list (404, fail-closed on unanchored).
* ``GET /api/studio/seclog`` — passed ``workspace_id=None`` so a tenant-bound admin read every
  tenant's security-audit events. Now scoped via ``studio_tenant_filter``.
* ``GET /v1/diagnostics`` — leaked operator deployment posture (binary paths, key-presence
  map, DSN host, DB paths) to any tenant browse role. Now posture-free for a tenant-bound
  caller via ``reveal_host_posture``.
* ``GET /v1/models`` — leaked host LLM-binary install state + live local Ollama inventory to
  tenant browse roles. Now ``reveal_host_posture=False`` for a tenant-bound caller.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy
from himmy.entities.records import EntityRecord
from himmy.services.audit.models import SecurityEvent

# A tenant-bound admin: the mintable misconfiguration the verdicts hinge on
# (roles=["admin"] but all_tenants=False, scoped to ONE tenant).
_ADMIN_POLICY = AccessPolicy.from_mapping({"admin": ["*:*"]})


def _tenant_admin_app(subject: str, tenant: str) -> TestClient:
    """A built app authenticated as a TENANT-BOUND admin (``*:*`` but scoped to ``tenant``)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                subject, tenant_ids=[tenant], roles=["admin"], auth_method="apikey"
            )
        }
    )
    app.state.access_policy = _ADMIN_POLICY
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    return client


# ============================ lineage (high) ============================


def _registry(client: TestClient) -> Any:
    return client.app.state.container.entity_registry  # type: ignore[attr-defined]


def _seed_thread_graph(registry: Any, thread_id: str, workspace_id: str) -> dict[str, Any]:
    """A realistic lineage whose ``chat_thread`` hub is stamped with ``workspace_id``.

    The thread's own ``metadata['workspace_id']`` rides into its projected record payload
    (``payload['metadata']``) exactly as ``RunAppService._persist`` projects a real thread.
    """
    thread = EntityRecord.create(
        stable_id=thread_id,
        version=1,
        kind="chat_thread",
        payload={"name": "secret thread", "metadata": {"workspace_id": workspace_id}},
        metadata={"workspace_id": workspace_id},
    )
    persona = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="persona",
        payload={"name": "Analyst", "prompt": "you are tenant-secret"},
    )
    snapshot = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="context_snapshot",
        payload={"summary": "tenant-private evidence"},
    )
    for rec in (thread, persona, snapshot):
        registry.register(rec)
    registry.link(
        from_record_id=thread.record_id,
        to_record_id=persona.record_id,
        relation="uses_persona",
    )
    registry.link(
        from_record_id=thread.record_id,
        to_record_id=snapshot.record_id,
        relation="built_from",
    )
    return {"thread": thread, "persona": persona, "snapshot": snapshot}


def test_tenant_admin_cannot_read_other_tenants_lineage_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-A admin gets 404 (not the payload) for tenant-B's lineage records."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A")
    recs = _seed_thread_graph(_registry(client), str(uuid.uuid4()), workspace_id="B")

    # The thread hub itself (B-stamped) and a B-owned node hanging off it both 404.
    for rec in (recs["thread"], recs["persona"], recs["snapshot"]):
        r = client.get(f"/api/studio/lineage/entity/{rec.record_id}")
        assert r.status_code == 404, (rec.payload, r.text)
        assert "prompt" not in r.text and "evidence" not in r.text


def test_tenant_admin_cannot_read_other_tenants_lineage_graph_by_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/graph?entity_id=`` over a tenant-B subgraph is a clean 404 for tenant-A's admin."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A")
    recs = _seed_thread_graph(_registry(client), str(uuid.uuid4()), workspace_id="B")
    r = client.get(
        "/api/studio/lineage/graph", params={"entity_id": recs["thread"].record_id}
    )
    assert r.status_code == 404
    assert "Analyst" not in r.text


def test_tenant_admin_reads_own_tenant_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same tenant-A admin DOES read its OWN tenant-A subgraph (scope is real, not a wall)."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A")
    recs = _seed_thread_graph(_registry(client), str(uuid.uuid4()), workspace_id="A")
    detail = client.get(f"/api/studio/lineage/entity/{recs['thread'].record_id}")
    assert detail.status_code == 200, detail.text
    graph = client.get(
        "/api/studio/lineage/graph", params={"entity_id": recs["thread"].record_id}
    )
    assert graph.status_code == 200
    assert {n["kind"] for n in graph.json()["nodes"]} >= {"chat_thread", "persona"}


def test_tenant_admin_unanchored_record_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record with NO workspace anchor is 404 to a tenant-bound caller (fail closed)."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A")
    rec = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="context_snapshot",
        payload={"summary": "orphan secret"},
    )
    _registry(client).register(rec)
    r = client.get(f"/api/studio/lineage/entity/{rec.record_id}")
    assert r.status_code == 404
    assert "orphan secret" not in r.text


def test_offline_lineage_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline (no authenticator): an unstamped record reads fine — byte-unchanged."""
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(ApiContainer.build_default()))
    rec = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="context_snapshot",
        payload={"summary": "single-box evidence"},
    )
    client.app.state.container.entity_registry.register(rec)  # type: ignore[attr-defined]
    r = client.get(f"/api/studio/lineage/entity/{rec.record_id}")
    assert r.status_code == 200
    assert r.json()["payload"]["summary"] == "single-box evidence"


# ============================ seclog (medium) ============================


def _seed_seclog(client: TestClient) -> None:
    audit = client.app.state.security_audit  # type: ignore[attr-defined]
    audit.record(
        SecurityEvent(
            event_type="authz_denied",
            outcome="deny",
            actor={"subject": "mallory-A"},
            resource="tool:wire_money",
            action="execute",
            workspace_id="A",
            created_at="2026-06-13T10:00:00Z",
            detail="A-private deny",
        )
    )
    audit.record(
        SecurityEvent(
            event_type="access",
            outcome="allow",
            actor={"subject": "eve-B"},
            resource="runs",
            action="list",
            workspace_id="B",
            created_at="2026-06-13T11:00:00Z",
            detail="B-private access",
        )
    )


def test_tenant_admin_seclog_is_tenant_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-A admin sees only A's security events — never tenant-B's recon data."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A")
    _seed_seclog(client)
    body = client.get("/api/studio/seclog").json()
    workspaces_seen = {e["actor"] for e in body["items"]}
    assert workspaces_seen == {"mallory-A"}
    assert "eve-B" not in client.get("/api/studio/seclog").text
    # The filter-options list is likewise scoped (no B-only types).
    assert body["types"] == ["authz_denied"]


def test_offline_seclog_sees_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline: the seclog shows EVERY tenant's events — byte-unchanged single-box view."""
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(ApiContainer.build_default()))
    _seed_seclog(client)
    body = client.get("/api/studio/seclog").json()
    assert {e["actor"] for e in body["items"]} == {"mallory-A", "eve-B"}


# ============================ /v1/diagnostics (low) ============================


def _v1_browse_app(role: str, tenant: str = "A") -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=[tenant], roles=[role], auth_method="apikey"
            )
        }
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    return client


def test_tenant_diagnostics_strips_host_posture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant browse role gets a POSTURE-FREE diagnostics report (no paths/keys/DSN)."""
    monkeypatch.setenv(
        "HIMMY_DATABASE_URL", "postgresql://user:supersecret@db.internal:5432/himmy"
    )
    body = _v1_browse_app("viewer").get("/v1/diagnostics").json()
    # The provider-key presence inventory is gone.
    assert "keys" not in body["doctor"]
    # Provider rows keep availability but lose the absolute binary path.
    for provider in body["doctor"]["providers"]:
        assert "path" not in provider
        assert "available" in provider or "ok" in provider
    # The storage DSN host is gone (backend NAME stays).
    assert body["storage"].get("dsn") is None
    assert body["storage"]["backend"] == "postgresql"
    assert "db.internal" not in _v1_browse_app("viewer").get("/v1/diagnostics").text
    # The absolute .himmy/*.db paths are gone.
    for db_file in body["db_files"]:
        assert "path" not in db_file


def test_offline_diagnostics_keeps_full_posture() -> None:
    """Offline / all_tenants: the full report (keys, paths, DSN) is byte-unchanged."""
    body = TestClient(create_app(ApiContainer.build_default())).get(
        "/v1/diagnostics"
    ).json()
    assert "keys" in body["doctor"]
    for db_file in body["db_files"]:
        assert "path" in db_file


# ============================ /v1/models (low) ============================


def test_tenant_models_catalog_is_posture_free() -> None:
    """A tenant browse role gets the posture-free picker (no live host probe / inventory)."""
    body = _v1_browse_app("viewer").get("/v1/models").json()
    by_provider = {p["provider"]: p for p in body}
    # Posture-free: providers advertised as SUPPORTED (static True), no live Ollama inventory.
    assert by_provider["ollama"]["available"] is True
    assert by_provider["ollama"]["models"] == []


def test_offline_models_catalog_probes_host() -> None:
    """Offline / all_tenants: the catalog still host-probes (byte-unchanged)."""
    import shutil

    body = TestClient(create_app(ApiContainer.build_default())).get("/v1/models").json()
    by_provider = {p["provider"]: p for p in body}
    # claude-cli availability reflects a real host probe (shutil.which), not a static True.
    assert by_provider["claude-cli"]["available"] == (shutil.which("claude") is not None)
