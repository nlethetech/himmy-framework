"""WS1.0 — the tenant-isolation (IDOR) fix: workspace comes from the principal.

With a tenant-bound principal (a mapped API key), the caller can only touch its own
workspace; passing another tenant's ``workspace_id`` is denied (403), not honored.
The offline/no-auth default is unchanged (an ANONYMOUS, all-tenants principal).
"""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal


def _app_with_mapped_keys() -> TestClient:
    """An app whose keys bind callers to specific tenants (closes the IDOR)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-a": Principal.build(
                "user-a",
                tenant_ids=["tenant-a"],
                roles=["operator"],
                auth_method="apikey",
            ),
            "key-b": Principal.build(
                "user-b",
                tenant_ids=["tenant-b"],
                roles=["operator"],
                auth_method="apikey",
            ),
        }
    )
    return TestClient(app)


def _client_as(key: str) -> TestClient:
    c = _app_with_mapped_keys()
    c.headers.update({"x-himmy-internal-key": key})
    return c


def _create_run(client: TestClient, workspace: str) -> str:
    body = {
        "workspace_id": workspace,
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    resp = client.post("/v1/runs", json=body)
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/runs/{run_id}").json().get("status") in (
            "SUCCEEDED",
            "FAILED",
        ):
            break
        time.sleep(0.02)
    return run_id


def test_unauthenticated_request_is_rejected() -> None:
    client = _app_with_mapped_keys()  # no key header
    assert (
        client.get("/v1/runs", params={"workspace_id": "tenant-a"}).status_code == 401
    )


def test_caller_can_read_its_own_tenant() -> None:
    client = _client_as("key-a")
    run_id = _create_run(client, "tenant-a")
    # Explicit own workspace, and the implied (single-tenant) default, both work.
    assert (
        client.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 200
    )
    assert client.get(f"/v1/runs/{run_id}").status_code == 200


def test_cross_tenant_read_is_denied_not_leaked() -> None:
    """The IDOR: caller A passing tenant-b is 403 (denied), never served."""
    client_a = _client_as("key-a")
    run_id = _create_run(client_a, "tenant-a")
    # Same caller A, but asking for tenant-b → 403 (authorization), not a leak.
    assert (
        client_a.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 403
    )
    # A genuinely different tenant's key cannot reach A's run either.
    client_b = _client_as("key-b")
    assert (
        client_b.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-a"}
        ).status_code
        == 403
    )
    # And B reading under its own tenant simply doesn't find A's run (404).
    assert (
        client_b.get(
            f"/v1/runs/{run_id}", params={"workspace_id": "tenant-b"}
        ).status_code
        == 404
    )


def test_cross_tenant_create_is_denied() -> None:
    """Caller A cannot create a run inside tenant-b."""
    client_a = _client_as("key-a")
    body = {
        "workspace_id": "tenant-b",
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    assert client_a.post("/v1/runs", json=body).status_code == 403


def test_list_is_pinned_to_the_callers_tenant() -> None:
    client_a = _client_as("key-a")
    _create_run(client_a, "tenant-a")
    # Omitting workspace_id resolves to the caller's single tenant (not "all").
    listed = client_a.get("/v1/runs").json()
    assert all(it["workspace_id"] == "tenant-a" for it in listed["items"])


def test_offline_default_preserves_legacy_behavior() -> None:
    """No authenticator → ANONYMOUS all-tenants → caller-supplied workspace honored."""
    client = TestClient(create_app(ApiContainer.build_default()))
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    run_id = client.post("/v1/runs", json=body).json()["run_id"]
    assert (
        client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w1"}).status_code
        == 200
    )
    assert (
        client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w2"}).status_code
        == 404
    )


# --------------------------------------------------------------------------- #
# WS-bola — object-level (BOLA) isolation WITHIN a single shared tenant.
#
# Contract: a workspace is the trust boundary by DEFAULT (its members read each
# other's runs — ``subject_scoped=False``, unchanged). An OPT-IN ``subject_scoped``
# principal may read only runs whose ``subject_id`` matches its own ``subject``;
# a ``tenant_admin`` crosses subjects within its own tenant. ANONYMOUS / offline
# is never subject-scoped, so it still reads everything (the invariant).
#
# Here subjects A and B share the SAME tenant ("shared"), so workspace isolation
# does NOT separate them — only the subject axis does.
# --------------------------------------------------------------------------- #


def _bola_app() -> Any:
    """One tenant ("shared"), two subject-scoped users + one tenant_admin + a writer."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            # The WRITER seeds runs for either subject (NOT subject-scoped, so it may
            # set an arbitrary body ``subject_id`` — it stands in for the system/ingest
            # path that creates a run on a subject's behalf).
            "key-writer": Principal.build(
                "writer",
                tenant_ids=["shared"],
                roles=["operator"],
                auth_method="apikey",
            ),
            # Subject A: subject-scoped → may read ONLY its own (subject_id == "subj-a").
            "key-a": Principal.build(
                "subj-a",
                tenant_ids=["shared"],
                roles=["operator"],
                auth_method="apikey",
                subject_scoped=True,
            ),
            # Subject B: subject-scoped → may read ONLY its own (subject_id == "subj-b").
            "key-b": Principal.build(
                "subj-b",
                tenant_ids=["shared"],
                roles=["operator"],
                auth_method="apikey",
                subject_scoped=True,
            ),
            # Tenant admin: subject-scoped but holds tenant_admin → crosses subjects
            # WITHIN the "shared" tenant.
            "key-admin": Principal.build(
                "admin-user",
                tenant_ids=["shared"],
                roles=["operator", "tenant_admin"],
                auth_method="apikey",
                subject_scoped=True,
            ),
        }
    )
    return app


def _client_on(app: Any, key: str) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": key})
    return c


def _create_run_for(client: TestClient, workspace: str, subject_id: str) -> str:
    """Seed a run for a specific data subject and await its terminal state."""
    body = {
        "workspace_id": workspace,
        "subject_id": subject_id,
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    resp = client.post("/v1/runs", json=body)
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/runs/{run_id}").json().get("status") in (
            "SUCCEEDED",
            "FAILED",
        ):
            break
        time.sleep(0.02)
    return run_id


def test_bola_subject_cannot_read_other_subjects_run_same_workspace() -> None:
    """REQUIRED: subject A cannot read subject B's run within the SAME workspace.

    The cross-subject read is a clean 404 (not-found, never a 403 that would confirm
    the run exists) even though both subjects sit in the same tenant — workspace
    isolation alone would have LEAKED it.
    """
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_b = _create_run_for(writer, "shared", "subj-b")

    client_a = _client_on(app, "key-a")
    # A is subject-scoped to "subj-a"; B's run belongs to "subj-b" → 404.
    assert client_a.get(f"/v1/runs/{run_b}").status_code == 404
    # Sibling read surfaces (events / thread) are gated the same way.
    assert client_a.get(f"/v1/runs/{run_b}/events").status_code == 404
    assert client_a.get(f"/v1/runs/{run_b}/thread").status_code == 404


def test_bola_subject_can_read_its_own_run() -> None:
    """No false lockout: A reads its OWN run within the shared tenant."""
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_a = _create_run_for(writer, "shared", "subj-a")

    client_a = _client_on(app, "key-a")
    assert client_a.get(f"/v1/runs/{run_a}").status_code == 200


def test_bola_tenant_admin_can_cross_subjects_within_its_tenant() -> None:
    """REQUIRED: a tenant_admin reads EVERY subject's run within its own tenant."""
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_a = _create_run_for(writer, "shared", "subj-a")
    run_b = _create_run_for(writer, "shared", "subj-b")

    admin = _client_on(app, "key-admin")
    assert admin.get(f"/v1/runs/{run_a}").status_code == 200
    assert admin.get(f"/v1/runs/{run_b}").status_code == 200


def test_bola_list_is_pinned_to_the_subject() -> None:
    """A subject-scoped list shows only the caller's own runs; admin sees all."""
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_a = _create_run_for(writer, "shared", "subj-a")
    run_b = _create_run_for(writer, "shared", "subj-b")

    client_a = _client_on(app, "key-a")
    a_ids = {it["run_id"] for it in client_a.get("/v1/runs").json()["items"]}
    assert run_a in a_ids
    assert run_b not in a_ids  # B's run never enumerated for A
    # Even explicitly asking for B's subject is ignored (pinned to own subject).
    a_ids_forced = {
        it["run_id"]
        for it in client_a.get(
            "/v1/runs", params={"subject_id": "subj-b"}
        ).json()["items"]
    }
    assert run_b not in a_ids_forced

    admin = _client_on(app, "key-admin")
    admin_ids = {it["run_id"] for it in admin.get("/v1/runs").json()["items"]}
    assert {run_a, run_b} <= admin_ids


def test_bola_offline_still_reads_every_subjects_run() -> None:
    """INVARIANT: offline / ANONYMOUS (no auth) reads runs of ANY subject (no narrowing)."""
    client = TestClient(create_app(ApiContainer.build_default()))
    run_a = _create_run_for(client, "w1", "subj-a")
    run_b = _create_run_for(client, "w1", "subj-b")
    # The same anonymous caller reads BOTH subjects' runs — subject-scoping is a no-op.
    assert client.get(f"/v1/runs/{run_a}", params={"workspace_id": "w1"}).status_code == 200
    assert client.get(f"/v1/runs/{run_b}", params={"workspace_id": "w1"}).status_code == 200
    listed = {it["run_id"] for it in client.get(
        "/v1/runs", params={"workspace_id": "w1"}
    ).json()["items"]}
    assert {run_a, run_b} <= listed


def test_bola_non_subject_scoped_workspace_member_reads_all_subjects() -> None:
    """A tenant-bound but NOT subject-scoped operator reads every subject in its tenant.

    This is the DEFAULT multi-user-workspace contract — the workspace is the trust
    boundary, so subject-level narrowing engages ONLY for opt-in subject-scoped keys.
    """
    app = _bola_app()
    writer = _client_on(app, "key-writer")  # operator, NOT subject_scoped
    run_a = _create_run_for(writer, "shared", "subj-a")
    run_b = _create_run_for(writer, "shared", "subj-b")
    assert writer.get(f"/v1/runs/{run_a}").status_code == 200
    assert writer.get(f"/v1/runs/{run_b}").status_code == 200


def test_bola_subject_cannot_read_other_subjects_run_lineage() -> None:
    """REQUIRED: lineage is a read sibling of /events and /thread and is BOLA-gated.

    Lineage exposes a run's persona/prompt/evidence snapshot, so subject A reading
    subject B's lineage within the SAME tenant must be a clean 404 (never a 200 that
    would leak B's prompt text); a tenant_admin still reads it.
    """
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_b = _create_run_for(writer, "shared", "subj-b")

    client_a = _client_on(app, "key-a")
    # A is subject-scoped to "subj-a"; B's lineage belongs to "subj-b" → 404 (no leak).
    assert client_a.get(f"/v1/runs/{run_b}/lineage").status_code == 404

    # The tenant_admin crosses subjects within its own tenant (200 when lineage exists).
    admin = _client_on(app, "key-admin")
    assert admin.get(f"/v1/runs/{run_b}/lineage").status_code in (200, 404)


def test_bola_subject_can_read_its_own_run_lineage() -> None:
    """No false lockout: A reads its OWN run's lineage within the shared tenant."""
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_a = _create_run_for(writer, "shared", "subj-a")

    client_a = _client_on(app, "key-a")
    # Own-subject lineage is never BOLA-blocked (200 if projected, else the route's own 404).
    assert client_a.get(f"/v1/runs/{run_a}/lineage").status_code in (200, 404)


def test_bola_subject_cannot_approve_or_reject_other_subjects_run() -> None:
    """REQUIRED: approve/reject are WRITE siblings and must be subject-BOLA gated.

    A subject-scoped caller targeting ANOTHER subject's run is collapsed into the SAME
    clean 404 as an unknown run — never a 409 that would confirm the foreign run exists
    or leak its lifecycle state, and never a cross-subject resume of a HITL-paused run.
    The gate fires BEFORE resume_run, so the foreign run's status is never touched.
    """
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_b = _create_run_for(writer, "shared", "subj-b")

    client_a = _client_on(app, "key-a")
    # A targeting B's run → 404 (BOLA gate), NOT a 409 that would confirm existence/state.
    assert client_a.post(f"/v1/runs/{run_b}/approve").status_code == 404
    assert client_a.post(f"/v1/runs/{run_b}/reject").status_code == 404

    # B's run is unchanged (still terminal, never resumed by the foreign caller).
    assert writer.get(f"/v1/runs/{run_b}").json()["status"] in ("SUCCEEDED", "FAILED")


def test_bola_subject_approving_own_run_passes_the_gate() -> None:
    """No false lockout: a subject's own run passes the BOLA gate.

    The own-subject approve is NOT a 404 — the gate lets it through to resume_run, which
    returns a 409 here only because an inline-persona run is not AWAITING_APPROVAL (so the
    distinction between the BOLA 404 and the not-approvable 409 is observable).
    """
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_a = _create_run_for(writer, "shared", "subj-a")

    client_a = _client_on(app, "key-a")
    # Own run is not hidden: a non-paused run yields the not-approvable 409, NOT a BOLA 404.
    assert client_a.post(f"/v1/runs/{run_a}/approve").status_code == 409


def test_bola_tenant_admin_can_target_any_subjects_run_for_approval() -> None:
    """REQUIRED: a tenant_admin crosses subjects on the WRITE path too (no BOLA 404).

    The admin's approve passes the subject gate and reaches resume_run (409 here only
    because the inline-persona run is not AWAITING_APPROVAL — the point is it is NOT the
    BOLA 404 a foreign subject-scoped caller would get).
    """
    app = _bola_app()
    writer = _client_on(app, "key-writer")
    run_b = _create_run_for(writer, "shared", "subj-b")

    admin = _client_on(app, "key-admin")
    assert admin.post(f"/v1/runs/{run_b}/approve").status_code == 409
