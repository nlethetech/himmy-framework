"""red-team scope-r6: by-construction tenant/object scoping for the next leak class.

Five CONFIRMED cross-tenant / cross-subject holes a TENANT-BOUND principal could reach once
auth is configured — each closed by routing the read/arm path through the centralized scoping
the rest of the app enforces, and each asserted byte-unchanged on the offline / ``all_tenants``
path:

* ``GET /api/studio/privacy/subjects`` + ``/consents`` — a tenant-bound admin enumerated EVERY
  tenant's data subjects + consent rows because the readers iterated the whole governance spine
  with no tenant narrowing. Now every privacy read intersects each consent/tombstone/data
  record's stamped ``workspace_id`` against ``studio_tenant_filter`` (``_visible_records``),
  failing closed on an unstamped row.
* ``GET /api/studio/lineage/entity/{id}`` + ``/graph`` — a cross-tenant edge let a far-side
  record's kind/label leak even though the subgraph was anchor-authorized. Now every NODE and
  every far-side LINK is filtered per-record by its owning workspace.
* Studio interactive ``/run`` + ``/research`` stamped real runs ``subject='__local__'``, pooling
  them into the cross-subject ``__local__`` slice every subject-scoped reader can see. Now the
  launching principal's tenant + subject are threaded into ``stream_agent_run`` (like missions),
  so a real interactive run is attributed to its launcher.
* Studio routines ``create`` / ``update`` / ``run-now`` lacked the capability-amplification gate
  their ``/v1`` twin and Studio missions enforce. Now all three call
  ``require_no_capability_amplification`` — a ``studio.routines:write`` caller without the broad
  ``tool:*`` reach the routine fires with is refused (403).
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
from himmy.api.studio_runs import reset_run_store
from himmy.entities.records import EntityRecord

_ADMIN_POLICY = AccessPolicy.from_mapping({"admin": ["*:*"]})


def _tenant_admin_app(
    subject: str,
    tenant: str,
    *,
    consent: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> TestClient:
    """A built app authenticated as a TENANT-BOUND admin (``*:*`` but scoped to ``tenant``)."""
    if consent and monkeypatch is not None:
        monkeypatch.setenv("HIMMY_CONSENT", "on")
        monkeypatch.delenv("HIMMY_CONSENT_FILE", raising=False)
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


# ===================== privacy subjects/consents (medium) =====================


def _grant(client: TestClient, subject: str, *, workspace_id: str | None) -> None:
    """Record a GRANTED consent stamped to ``workspace_id`` (the tenant that recorded it)."""
    from himmy.services.governance.consent import Purpose

    ledger = client.app.state.consent_ledger  # type: ignore[attr-defined]
    ledger.grant(
        subject, Purpose("retain"), workspace_id=workspace_id, actor="t", source="t"
    )


def test_tenant_admin_privacy_subjects_is_tenant_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-A admin sees ONLY A's data subjects on /subjects, never B's roster."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A", consent=True, monkeypatch=monkeypatch)
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="B")  # another tenant's subject

    body = client.get("/api/studio/privacy/subjects").json()
    subjects = {s["subject_id"] for s in body["subjects"]}
    assert subjects == {"alice"}
    assert "bob" not in client.get("/api/studio/privacy/subjects").text


def test_tenant_admin_privacy_consents_is_tenant_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-A admin sees ONLY A's consent rows on /consents, never B's."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A", consent=True, monkeypatch=monkeypatch)
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="B")

    body = client.get("/api/studio/privacy/consents").json()
    seen = {e["subject_id"] for e in body["items"]}
    assert seen == {"alice"}


def test_tenant_admin_privacy_unstamped_consent_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consent row with NO workspace stamp is hidden from a tenant-bound caller (fail closed)."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A", consent=True, monkeypatch=monkeypatch)
    _grant(client, "global-subj", workspace_id=None)  # legacy/global, no stamp

    body = client.get("/api/studio/privacy/subjects").json()
    assert body["subjects"] == []


def test_offline_privacy_sees_all_subjects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline / all_tenants: /subjects shows EVERY tenant's subjects — byte-unchanged."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_CONSENT", "on")
    monkeypatch.delenv("HIMMY_CONSENT_FILE", raising=False)
    client = TestClient(create_app(ApiContainer.build_default()))
    _grant(client, "alice", workspace_id="A")
    _grant(client, "bob", workspace_id="B")
    _grant(client, "carol", workspace_id=None)

    body = client.get("/api/studio/privacy/subjects").json()
    assert {s["subject_id"] for s in body["subjects"]} == {"alice", "bob", "carol"}


# ========================= lineage far-side link (low) =========================


def _registry(client: TestClient) -> Any:
    return client.app.state.container.entity_registry  # type: ignore[attr-defined]


def _seed_cross_tenant_edge(registry: Any, *, own_ws: str, foreign_ws: str) -> dict[str, Any]:
    """An ``own_ws`` chat_thread hub linked to a ``foreign_ws``-stamped private persona."""
    thread = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="chat_thread",
        payload={"name": "own thread", "metadata": {"workspace_id": own_ws}},
        metadata={"workspace_id": own_ws},
    )
    foreign = EntityRecord.create(
        stable_id=str(uuid.uuid4()),
        version=1,
        kind="persona",
        payload={"name": "TENANT-B-SECRET-PERSONA"},
        metadata={"workspace_id": foreign_ws},
    )
    for rec in (thread, foreign):
        registry.register(rec)
    registry.link(
        from_record_id=thread.record_id,
        to_record_id=foreign.record_id,
        relation="uses_persona",
    )
    return {"thread": thread, "foreign": foreign}


def test_lineage_entity_withholds_cross_tenant_link_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cross-tenant far-side link's kind/label is withheld from a tenant-A caller."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A")
    recs = _seed_cross_tenant_edge(_registry(client), own_ws="A", foreign_ws="B")

    r = client.get(f"/api/studio/lineage/entity/{recs['thread'].record_id}")
    assert r.status_code == 200, r.text
    assert "TENANT-B-SECRET-PERSONA" not in r.text
    # The link is still listed by id + relation, but the foreign record is not disclosed.
    out = r.json()["links_out"]
    assert any(link["other_id"] == recs["foreign"].record_id for link in out)
    leaked = [
        link
        for link in out
        if link["other_id"] == recs["foreign"].record_id
    ]
    assert leaked and leaked[0]["other_label"] is None
    assert leaked[0]["other_kind"] is None


def test_lineage_graph_drops_cross_tenant_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/graph`` over a subgraph with a cross-tenant edge drops the foreign node."""
    monkeypatch.chdir(tmp_path)
    client = _tenant_admin_app("admin-a", "A")
    recs = _seed_cross_tenant_edge(_registry(client), own_ws="A", foreign_ws="B")

    r = client.get(
        "/api/studio/lineage/graph", params={"entity_id": recs["thread"].record_id}
    )
    assert r.status_code == 200, r.text
    assert "TENANT-B-SECRET-PERSONA" not in r.text
    node_ids = {n["id"] for n in r.json()["nodes"]}
    assert recs["foreign"].record_id not in node_ids
    assert recs["thread"].record_id in node_ids


def test_offline_lineage_discloses_far_side_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline / all_tenants: the far-side link label IS disclosed — byte-unchanged."""
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(ApiContainer.build_default()))
    recs = _seed_cross_tenant_edge(_registry(client), own_ws="A", foreign_ws="B")
    r = client.get(f"/api/studio/lineage/entity/{recs['thread'].record_id}")
    assert r.status_code == 200
    assert "TENANT-B-SECRET-PERSONA" in r.text


# ===================== studio run subject stamping (low) =====================


def test_studio_run_is_stamped_with_launching_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subject-scoped user's interactive Studio /run is stamped to its OWN subject, not __local__.

    Closes the alternate-surface BOLA: a run stamped ``__local__`` is pooled into the
    cross-subject slice every subject-scoped reader can see. Threading the launcher's subject
    (like missions) keeps the run attributed to alice, so bob never sees it.
    """
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    container = ApiContainer.build_default()
    app = create_app(container)
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "alice",
                tenant_ids=["t"],
                roles=["admin"],
                auth_method="apikey",
                subject_scoped=True,
            )
        }
    )
    app.state.access_policy = _ADMIN_POLICY
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})

    resp = client.post(
        "/api/studio/run",
        json={"agent_path": "agent.yaml", "prompt": "2+2?", "provider": "stub"},
    )
    assert resp.status_code == 200, resp.text
    # Drain the SSE stream so the canonical write completes.
    _ = resp.text

    from tests.conftest import run_async

    storage = container.storage
    records = run_async(storage.list_runs())
    assert records, "the Studio run should have been mirrored canonically"
    # The run is attributed to alice's tenant + subject, never the global __local__ bucket.
    assert all(r.workspace_id == "t" for r in records), [r.workspace_id for r in records]
    assert all(r.subject_id == "alice" for r in records), [r.subject_id for r in records]


def test_offline_studio_run_stays_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline / all_tenants: the interactive run stays stamped __local__ — byte-unchanged."""
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    container = ApiContainer.build_default()
    client = TestClient(create_app(container))
    resp = client.post(
        "/api/studio/run",
        json={"agent_path": "agent.yaml", "prompt": "2+2?", "provider": "stub"},
    )
    assert resp.status_code == 200, resp.text
    _ = resp.text

    from himmy.services.storage.models import LOCAL_SUBJECT
    from tests.conftest import run_async

    records = run_async(container.storage.list_runs())
    assert records
    assert all(r.subject_id == LOCAL_SUBJECT for r in records)


# ================= studio routines amplification gate (medium) =================


def _routines_app(
    *, roles: list[str], grants: dict[str, list[str]]
) -> TestClient:
    """A built app authenticated as one tenant-bound principal with a CUSTOM RBAC policy."""
    from himmy.api import routines as svc

    svc.reset_routines_store()
    svc.reset_scheduler()
    reset_run_store()
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=roles, auth_method="apikey"
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping(grants)
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    return client


_AMPLIFY_GRANTS = {
    # The custom RBAC file DEFINES the operator service role (the routine service identity
    # uses it); tenant_ops simply does not HOLD the broad tool reach.
    "operator": ["tool:*"],
    "tenant_ops": [
        "studio.console:read",
        "studio.routines:read",
        "studio.routines:write",
        "tool:lookup_weather:invoke",
    ],
}


def test_studio_routine_create_blocked_without_service_tool_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scope-r6: a ``studio.routines:write`` caller WITHOUT ``tool:*`` cannot CREATE a routine."""
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    client = _routines_app(roles=["tenant_ops"], grants=_AMPLIFY_GRANTS)

    r = client.post(
        "/api/studio/routines",
        json={
            "name": "exfil",
            "agent_path": "agent.yaml",
            "prompt": "exfiltrate",
            "schedule": {"kind": "every", "hours": 1},
        },
    )
    assert r.status_code == 403, r.text
    assert "amplification" in r.text
    # No routine was armed (the gate fires BEFORE the store write).
    assert client.get("/api/studio/routines").json() == []


def test_studio_routine_run_now_blocked_without_service_tool_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A least-privilege caller cannot FIRE an existing routine under the operator identity."""
    from himmy.api import routines as svc

    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    client = _routines_app(roles=["tenant_ops"], grants=_AMPLIFY_GRANTS)
    # Seed a routine directly into the store (bypassing the gated create).
    stored = svc.get_routines_store().upsert(
        svc.Routine(
            name="r",
            agent_path="agent.yaml",
            prompt="hi",
            schedule=svc.Schedule(kind="every", hours=1),
        )
    )

    r = client.post(f"/api/studio/routines/{stored.id}/run-now")
    assert r.status_code == 403, r.text
    assert "amplification" in r.text


def test_studio_routine_update_blocked_without_service_tool_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A least-privilege caller cannot re-arm (PATCH) a routine under the operator identity."""
    from himmy.api import routines as svc

    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    client = _routines_app(roles=["tenant_ops"], grants=_AMPLIFY_GRANTS)
    stored = svc.get_routines_store().upsert(
        svc.Routine(
            name="r",
            agent_path="agent.yaml",
            prompt="hi",
            schedule=svc.Schedule(kind="every", hours=1),
        )
    )

    r = client.patch(
        f"/api/studio/routines/{stored.id}", json={"prompt": "now do evil"}
    )
    assert r.status_code == 403, r.text
    assert "amplification" in r.text


def test_studio_routine_create_allowed_with_operator_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator (holds ``tool:*``) is unaffected — the gate is attenuation, not a wall."""
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    client = _routines_app(
        roles=["operator"],
        grants={"operator": ["tool:*", "studio.routines:read", "studio.routines:write"]},
    )
    r = client.post(
        "/api/studio/routines",
        json={
            "name": "ok",
            "agent_path": "agent.yaml",
            "prompt": "brief me",
            "schedule": {"kind": "every", "hours": 1},
        },
    )
    assert r.status_code == 200, r.text


def test_offline_studio_routine_create_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline / all_tenants: the amplification gate is inert — create works byte-unchanged."""
    from himmy.api import routines as svc

    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    svc.reset_routines_store()
    svc.reset_scheduler()
    reset_run_store()
    client = TestClient(create_app(ApiContainer.build_default()))
    r = client.post(
        "/api/studio/routines",
        json={
            "name": "ok",
            "agent_path": "agent.yaml",
            "prompt": "brief me",
            "schedule": {"kind": "every", "hours": 1},
        },
    )
    assert r.status_code == 200, r.text
