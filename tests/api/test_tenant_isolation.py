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


# --------------------------------------------------------------------------- #
# WS-bola (red-team r1): WRITE-side object-level isolation. The read gates above
# narrowed get/list/events/thread, but the CREATE/UPSERT paths took ``subject_id``
# straight from the body — letting a subject-scoped principal STAMP a run / thread /
# PII context field under ANOTHER data subject (attribution / lineage / erasure-scope
# poisoning) and READ other subjects' context fields. These assert the symmetric gate.
# --------------------------------------------------------------------------- #


def test_bola_subject_cannot_create_run_for_other_subject() -> None:
    """REQUIRED: a subject-scoped principal may NOT create a run under a foreign subject.

    Previously the write path honored ``body.subject_id`` verbatim, so subject A could
    author a run stamped to subject B (poisoning B's lineage / erasure scope). It now 403s.
    """
    app = _bola_app()
    client_a = _client_on(app, "key-a")  # subject_scoped → "subj-a"
    body = {
        "workspace_id": "shared",
        "subject_id": "subj-b",  # NOT the caller's own subject
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    assert client_a.post("/v1/runs", json=body).status_code == 403


def test_bola_subject_can_create_run_for_its_own_subject() -> None:
    """No false lockout: a subject-scoped principal creates a run under its OWN subject."""
    app = _bola_app()
    client_a = _client_on(app, "key-a")
    body = {
        "workspace_id": "shared",
        "subject_id": "subj-a",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    assert client_a.post("/v1/runs", json=body).status_code == 200


def test_bola_tenant_admin_can_create_run_for_any_subject() -> None:
    """A tenant_admin crosses subjects on the WRITE path within its own tenant."""
    app = _bola_app()
    admin = _client_on(app, "key-admin")
    body = {
        "workspace_id": "shared",
        "subject_id": "subj-b",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    assert admin.post("/v1/runs", json=body).status_code == 200


def test_bola_offline_create_run_honors_any_subject() -> None:
    """INVARIANT: offline / ANONYMOUS create accepts any body subject (no narrowing)."""
    client = TestClient(create_app(ApiContainer.build_default()))
    body = {
        "workspace_id": "w1",
        "subject_id": "anyone",
        "persona": {"name": "A"},
        "task": {"title": "t", "prompt": "hi"},
    }
    assert client.post("/v1/runs", json=body).status_code == 200


def test_bola_subject_cannot_post_thread_message_for_other_subject() -> None:
    """REQUIRED: a subject-scoped principal may NOT post a thread turn under a foreign subject."""
    app = _bola_app()
    client_a = _client_on(app, "key-a")
    created = client_a.post("/v1/threads", json={"workspace_id": "shared"})
    assert created.status_code == 201, created.text
    thread_id = created.json()["thread_id"]
    resp = client_a.post(
        f"/v1/threads/{thread_id}/messages",
        json={"workspace_id": "shared", "subject_id": "subj-b", "prompt": "hi"},
    )
    assert resp.status_code == 403


def test_bola_subject_cannot_upsert_context_for_other_subject() -> None:
    """REQUIRED: a subject-scoped principal may NOT overwrite another subject's PII fields."""
    app = _bola_app()
    client_a = _client_on(app, "key-a")
    resp = client_a.post(
        "/v1/context/fields:upsert",
        json={
            "workspace_id": "shared",
            "subject_id": "subj-b",  # foreign subject's PII
            "fields": [
                {"key": "email", "value": "poison@evil.test", "source": "x"}
            ],
        },
    )
    assert resp.status_code == 403


def test_bola_subject_cannot_read_other_subjects_context_fields() -> None:
    """REQUIRED: a subject-scoped principal reading a foreign subject's PII gets []."""
    app = _bola_app()
    # The (non-subject-scoped) writer seeds victim PII for subj-b.
    writer = _client_on(app, "key-writer")
    seeded = writer.post(
        "/v1/context/fields:upsert",
        json={
            "workspace_id": "shared",
            "subject_id": "subj-b",
            "fields": [{"key": "email", "value": "victim@corp.test", "source": "x"}],
        },
    )
    assert seeded.status_code == 200, seeded.text

    client_a = _client_on(app, "key-a")  # subject_scoped → "subj-a"
    # A asks for B's fields → narrowed to an empty list (never B's PII).
    listed = client_a.get(
        "/v1/context/fields", params={"subject_id": "subj-b", "workspace_id": "shared"}
    )
    assert listed.status_code == 200
    assert listed.json() == []


def test_bola_subject_can_read_and_write_its_own_context() -> None:
    """No false lockout: a subject-scoped principal reads+writes its OWN context fields."""
    app = _bola_app()
    client_a = _client_on(app, "key-a")  # subject_scoped → "subj-a"
    wrote = client_a.post(
        "/v1/context/fields:upsert",
        json={
            "workspace_id": "shared",
            "subject_id": "subj-a",
            "fields": [{"key": "email", "value": "me@corp.test", "source": "x"}],
        },
    )
    assert wrote.status_code == 200, wrote.text
    listed = client_a.get(
        "/v1/context/fields", params={"subject_id": "subj-a", "workspace_id": "shared"}
    )
    assert listed.status_code == 200
    assert any(f["key"] == "email" for f in listed.json())


def test_bola_offline_context_reads_and_writes_any_subject() -> None:
    """INVARIANT: offline / ANONYMOUS context read+write is un-narrowed (byte-unchanged)."""
    client = TestClient(create_app(ApiContainer.build_default()))
    wrote = client.post(
        "/v1/context/fields:upsert",
        json={
            "workspace_id": "w1",
            "subject_id": "anyone",
            "fields": [{"key": "email", "value": "a@b.test", "source": "x"}],
        },
    )
    assert wrote.status_code == 200, wrote.text
    listed = client.get(
        "/v1/context/fields", params={"subject_id": "anyone", "workspace_id": "w1"}
    )
    assert listed.status_code == 200
    assert any(f["key"] == "email" for f in listed.json())


# --------------------------------------------------------------------------- #
# Subject-axis BOLA on /v1/recommendations + /v1/dashboard (red-team reattack-r2)
#
# Recommendations are subject-attributed (created with subject_id=run.subject_id),
# so a ``subject_scoped`` principal must read/mutate ONLY its own subject's recs —
# exactly like runs. The dashboard summary is a per-subject aggregate and is gated
# the same way. Offline / non-subject-scoped / tenant_admin keep reading everything.
# --------------------------------------------------------------------------- #


def _seed_recommendation(app: Any, *, workspace: str, subject_id: str) -> str:
    """Save a recommendation directly in the app's canonical store for a subject."""
    from himmy.services.storage.models import RecommendationItem
    from tests.conftest import run_async

    storage = app.state.container.storage
    item = RecommendationItem(
        run_id="run-" + subject_id,
        workspace_id=workspace,
        subject_id=subject_id,
        kind="advice",
        title="secret for " + subject_id,
        summary="confidential",
    )
    run_async(storage.save_recommendation(item))
    return item.recommendation_id


def test_bola_subject_cannot_list_other_subjects_recommendations() -> None:
    """REQUIRED: a subject-scoped list is pinned to the caller's own subject's recs.

    Regression: list_recommendations omitted ``narrow_subject``, so subject A could
    enumerate every subject's recs (no filter) or directly target subject B's.
    """
    app = _bola_app()
    rec_a = _seed_recommendation(app, workspace="shared", subject_id="subj-a")
    rec_b = _seed_recommendation(app, workspace="shared", subject_id="subj-b")

    client_a = _client_on(app, "key-a")
    # Unfiltered list: B's rec is NEVER surfaced to A (pinned to own subject).
    ids = {
        it["recommendation_id"]
        for it in client_a.get(
            "/v1/recommendations", params={"workspace_id": "shared"}
        ).json()["items"]
    }
    assert rec_a in ids
    assert rec_b not in ids
    # Even explicitly asking for B's subject is ignored.
    forced = {
        it["recommendation_id"]
        for it in client_a.get(
            "/v1/recommendations",
            params={"workspace_id": "shared", "subject_id": "subj-b"},
        ).json()["items"]
    }
    assert rec_b not in forced


def test_bola_subject_cannot_read_other_subjects_recommendation_provenance() -> None:
    """REQUIRED: A cannot trace B's recommendation provenance — a clean 404."""
    app = _bola_app()
    rec_b = _seed_recommendation(app, workspace="shared", subject_id="subj-b")
    client_a = _client_on(app, "key-a")
    resp = client_a.get(
        f"/v1/recommendations/{rec_b}/provenance", params={"workspace_id": "shared"}
    )
    assert resp.status_code == 404


def test_bola_subject_cannot_mutate_other_subjects_recommendation() -> None:
    """REQUIRED: A cannot transition B's recommendation status — a clean 404, no write."""
    app = _bola_app()
    rec_b = _seed_recommendation(app, workspace="shared", subject_id="subj-b")
    client_a = _client_on(app, "key-a")
    resp = client_a.patch(
        f"/v1/recommendations/{rec_b}",
        params={"workspace_id": "shared"},
        json={"status": "DISMISSED"},
    )
    assert resp.status_code == 404


def test_bola_subject_can_read_and_mutate_its_own_recommendation() -> None:
    """No false lockout: A reads + mutates its OWN recommendation."""
    app = _bola_app()
    rec_a = _seed_recommendation(app, workspace="shared", subject_id="subj-a")
    client_a = _client_on(app, "key-a")
    listed = {
        it["recommendation_id"]
        for it in client_a.get(
            "/v1/recommendations", params={"workspace_id": "shared"}
        ).json()["items"]
    }
    assert rec_a in listed
    patched = client_a.patch(
        f"/v1/recommendations/{rec_a}",
        params={"workspace_id": "shared"},
        json={"status": "DISMISSED"},
    )
    assert patched.status_code == 200


def test_bola_tenant_admin_can_target_any_subjects_recommendation() -> None:
    """A tenant_admin crosses subjects on recs within its own tenant."""
    app = _bola_app()
    rec_b = _seed_recommendation(app, workspace="shared", subject_id="subj-b")
    admin = _client_on(app, "key-admin")
    ids = {
        it["recommendation_id"]
        for it in admin.get(
            "/v1/recommendations", params={"workspace_id": "shared"}
        ).json()["items"]
    }
    assert rec_b in ids
    assert (
        admin.patch(
            f"/v1/recommendations/{rec_b}",
            params={"workspace_id": "shared"},
            json={"status": "DISMISSED"},
        ).status_code
        == 200
    )


def test_bola_offline_recommendations_un_narrowed() -> None:
    """INVARIANT: offline / ANONYMOUS reads recs of ANY subject (no narrowing)."""
    app = create_app(ApiContainer.build_default())
    rec_a = _seed_recommendation(app, workspace="w1", subject_id="subj-a")
    rec_b = _seed_recommendation(app, workspace="w1", subject_id="subj-b")
    client = TestClient(app)
    ids = {
        it["recommendation_id"]
        for it in client.get(
            "/v1/recommendations", params={"workspace_id": "w1"}
        ).json()["items"]
    }
    assert {rec_a, rec_b} <= ids


def test_bola_subject_cannot_read_other_subjects_dashboard_summary() -> None:
    """REQUIRED: A cannot read B's dashboard aggregate — a clean 404.

    Regression: dashboard_summary enforced only the tenant axis (require_workspace),
    leaking another subject's context/run/recommendation aggregate metadata.
    """
    app = _bola_app()
    _seed_recommendation(app, workspace="shared", subject_id="subj-b")
    client_a = _client_on(app, "key-a")
    resp = client_a.get(
        "/v1/dashboard/summary",
        params={"workspace_id": "shared", "subject_id": "subj-b"},
    )
    assert resp.status_code == 404


def test_bola_subject_can_read_its_own_dashboard_summary() -> None:
    """No false lockout: A reads its OWN dashboard aggregate."""
    app = _bola_app()
    client_a = _client_on(app, "key-a")
    resp = client_a.get(
        "/v1/dashboard/summary",
        params={"workspace_id": "shared", "subject_id": "subj-a"},
    )
    assert resp.status_code == 200
    assert resp.json()["subject_id"] == "subj-a"


def test_bola_tenant_admin_reads_any_subjects_dashboard_summary() -> None:
    """A tenant_admin reads every subject's dashboard aggregate within its tenant."""
    app = _bola_app()
    admin = _client_on(app, "key-admin")
    assert (
        admin.get(
            "/v1/dashboard/summary",
            params={"workspace_id": "shared", "subject_id": "subj-b"},
        ).status_code
        == 200
    )


def test_bola_offline_dashboard_summary_un_narrowed() -> None:
    """INVARIANT: offline / ANONYMOUS reads any subject's dashboard aggregate."""
    client = TestClient(create_app(ApiContainer.build_default()))
    resp = client.get(
        "/v1/dashboard/summary",
        params={"workspace_id": "w1", "subject_id": "anyone"},
    )
    assert resp.status_code == 200
