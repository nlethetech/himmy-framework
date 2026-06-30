"""Studio hardening: RBAC gating of /api/studio + symlink-safe spec resolution.

Covers the two halves of the Studio trust boundary once the app leaves loopback:

* With an authenticator configured, every Studio route requires its per-surface
  permission (``studio.console:read``, ``studio.connections:write``, …; admin's
  ``*:*`` qualifies for all); without one, the zero-config behavior is unchanged.
* :func:`resolve_spec_path` rejects any symlinked component below the project
  root, in addition to the existing ``..``/containment check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy
from himmy.api.studio_service import resolve_spec_path


# ------------------------------------------------------ studio RBAC gating
def _app_with_key(*roles: str) -> FastAPI:
    """An app whose mapped API key ``"k"`` is bound to ``roles``."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=list(roles), auth_method="apikey"
            )
        }
    )
    return app


def _client_with_key(app: FastAPI) -> TestClient:
    """A client for ``app`` with the mapped key pre-attached."""
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def test_no_auth_studio_unchanged() -> None:
    """Zero-config (no authenticator): Studio answers without any credentials."""
    c = TestClient(create_app(ApiContainer.build_default()))
    assert c.get("/api/studio/health").status_code == 200


def test_auth_configured_unauthenticated_is_401() -> None:
    """With auth on, a request without a key never reaches Studio."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(shared_keys={"s"})
    assert TestClient(app).get("/api/studio/health").status_code == 401


def test_auth_configured_role_without_studio_grant_is_403() -> None:
    """An authenticated principal with NO Studio grant is denied everywhere.

    Studio is now gated per-surface (``studio.console:read``, ``studio.runs:read``,
    …) rather than by one coarse ``studio:use``; a custom role holding only a ``/v1``
    grant carries none of those, so every Studio route 403s.
    """
    app = _app_with_key("v1_only")
    app.state.access_policy = AccessPolicy.from_mapping({"v1_only": ["run:read"]})
    c = _client_with_key(app)
    assert c.get("/api/studio/health").status_code == 403
    assert c.get("/api/studio/connections").status_code == 403
    assert c.get("/api/studio/agents").status_code == 403


def test_auth_configured_admin_passes() -> None:
    """``admin`` (``*:*``) covers ``studio:use``."""
    c = _client_with_key(_app_with_key("admin"))
    assert c.get("/api/studio/health").status_code == 200


def test_shared_internal_key_passes() -> None:
    """The shared trusted-boundary key maps to admin, so Studio keeps working."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(shared_keys={"s"})
    c = TestClient(app)
    r = c.get("/api/studio/health", headers={"x-himmy-internal-key": "s"})
    assert r.status_code == 200


def test_custom_policy_can_grant_studio_read() -> None:
    """An operator-defined role granting ``studio.console:read`` unlocks the console.

    The granular replacement for the old coarse ``studio:use``: the read baseline the
    main router carries is ``studio.console:read``, so a role holding it can browse
    Studio's read surfaces (but not the ``/v1`` resource, nor any mutating route).
    """
    app = _app_with_key("studio_user")
    app.state.access_policy = AccessPolicy.from_mapping(
        {"studio_user": ["studio.console:read", "studio.connections:read"]}
    )
    c = _client_with_key(app)
    assert c.get("/api/studio/health").status_code == 200
    assert c.get("/api/studio/connections").status_code == 200
    # …but not the /v1 surface (no other grants).
    assert c.get("/v1/runs", params={"workspace_id": "t"}).status_code == 403
    # …and not a mutating Studio route (no studio.connections:write).
    assert (
        c.put("/api/studio/connections/email", json={"fields": {}}).status_code == 403
    )


def _app_with_policy(role: str, perms: list[str]) -> FastAPI:
    """An app whose mapped key is bound to ``role`` granting exactly ``perms``."""
    app = _app_with_key(role)
    app.state.access_policy = AccessPolicy.from_mapping({role: perms})
    return app


def test_per_surface_read_isolates_models_from_gmail() -> None:
    """A role granted ONE surface's read cannot read a DIFFERENT surface (r6 split).

    The Studio main-router GET routes no longer collapse to one coarse
    ``studio.console:read``: each carries its OWN ``studio.<surface>:read``. A role
    holding the console baseline + ``studio.modelcatalog:read`` (but NOT the Google grant
    the Gmail/Calendar reads require) is 200 on the benign model catalog yet 403 on the
    Gmail and Calendar reads — proving the surfaces are independently gated, not a single
    bucket. (reattack-r6: the catalog moved off ``studio.models`` to the tenant-grantable
    ``studio.modelcatalog`` so the catalog stays readable while the operator
    provider-credential surface ``studio.models`` is withheld — see the dedicated test.)
    """
    app = _app_with_policy(
        "models_only", ["studio.console:read", "studio.modelcatalog:read"]
    )
    c = _client_with_key(app)
    # Granted surface → 200.
    assert c.get("/api/studio/models").status_code == 200
    # Un-granted sensitive surfaces (folded under studio.google) → 403.
    assert c.get("/api/studio/google/gmail").status_code == 403
    assert c.get("/api/studio/google/calendar").status_code == 403
    assert c.get("/api/studio/google").status_code == 403
    # …and the inverse: a connections-read role cannot read the model catalog.
    app2 = _app_with_policy(
        "conn_only", ["studio.console:read", "studio.connections:read"]
    )
    c2 = _client_with_key(app2)
    assert c2.get("/api/studio/connections").status_code == 200
    assert c2.get("/api/studio/models").status_code == 403


def test_models_providers_credential_surface_withheld_from_browse_roles() -> None:
    """reattack-r6: the operator provider-CREDENTIAL surface is admin-only, not browse.

    GET /api/studio/models/providers (gated by ``studio.models:read``) enumerates every
    key-based provider's ``configured`` + ``detected_via`` ('secret'|'env'), plus
    ``secrets_writable`` and the default provider — operator deployment posture, the same
    reconnaissance class withheld for ``studio.connections``/``studio.google``. It must be
    403 for the default browse roles (viewer/operator/auditor) while the BENIGN model
    catalog GET /api/studio/models (now ``studio.modelcatalog:read``) stays 200, and admin
    reads both.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        # Benign catalog stays readable…
        assert c.get("/api/studio/models").status_code == 200, role
        # …but the provider credential-status surface is withheld (admin-only).
        assert c.get("/api/studio/models/providers").status_code == 403, role
    # A role explicitly granted ``studio.models:read`` (the credential surface) is NOT
    # granted the benign catalog by that alone — the two are independent resources now.
    app = _app_with_policy(
        "creds_only", ["studio.console:read", "studio.models:read"]
    )
    c = _client_with_key(app)
    assert c.get("/api/studio/models/providers").status_code == 200
    assert c.get("/api/studio/models").status_code == 403
    # admin (``*:*``) reads both surfaces.
    ca = _client_with_key(_app_with_key("admin"))
    assert ca.get("/api/studio/models").status_code == 200
    assert ca.get("/api/studio/models/providers").status_code == 200


def test_health_posture_fields_withheld_from_browse_roles() -> None:
    """reattack-r6: GET /api/studio/health withholds operator posture from browse roles.

    The bare readiness fields (``status``/``version``) stay readable under the console
    baseline, but ``secrets_writable`` + the ``providers`` presence map (which LLM binaries
    are installed) are operator deployment posture, included ONLY for a caller that also
    holds ``studio.console:write`` (admin). A tenant browse role gets a clean 200 probe with
    NO posture; admin gets the full payload; OFFLINE (no auth) keeps the full payload.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        body = c.get("/api/studio/health").json()
        assert body["status"] == "ok", role
        assert "secrets_writable" not in body, role
        assert "providers" not in body, role
    # admin sees the operator posture.
    ca = _client_with_key(_app_with_key("admin"))
    abody = ca.get("/api/studio/health").json()
    assert "secrets_writable" in abody
    assert "providers" in abody
    # OFFLINE (no authenticator) is byte-unchanged: full posture payload.
    offline = TestClient(create_app(ApiContainer.build_default())).get(
        "/api/studio/health"
    ).json()
    assert "secrets_writable" in offline
    assert "providers" in offline


def test_per_surface_runs_read_split() -> None:
    """The run readers require ``studio.runs:read``, not just the console baseline.

    Before r6 the ``/runs`` GET family inherited only ``studio.console:read``; now each
    declares ``studio.runs:read``. A role with the console baseline but no runs grant is
    403 on run history/analytics/detail, while one granted the runs read is 200.
    """
    app = _app_with_policy("console_only", ["studio.console:read"])
    c = _client_with_key(app)
    assert c.get("/api/studio/runs").status_code == 403
    assert c.get("/api/studio/runs/analytics").status_code == 403
    assert c.get("/api/studio/runs/does-not-exist").status_code == 403
    # Lineage is run data too: console baseline alone must NOT reach it.
    assert c.get("/api/studio/runs/does-not-exist/lineage").status_code == 403

    app2 = _app_with_policy(
        "runs_reader", ["studio.console:read", "studio.runs:read"]
    )
    c2 = _client_with_key(app2)
    assert c2.get("/api/studio/runs").status_code == 200
    assert c2.get("/api/studio/runs/analytics").status_code == 200
    # A missing run is 404 (past the 403 guard), never 403, for a granted reader.
    assert c2.get("/api/studio/runs/does-not-exist").status_code == 404
    # The runs reader passes the lineage guard too (404 = past it, run not found).
    assert (
        c2.get("/api/studio/runs/does-not-exist/lineage").status_code == 404
    )


def test_admin_reads_every_studio_surface() -> None:
    """``admin`` (``*:*``) satisfies every per-surface read guard."""
    c = _client_with_key(_app_with_key("admin"))
    for path in (
        "/api/studio/models",
        "/api/studio/runs",
        "/api/studio/runs/analytics",
        "/api/studio/connections",
        "/api/studio/google",
        "/api/studio/approvals",
    ):
        assert c.get(path).status_code == 200, path


def test_default_browse_roles_grant_benign_reads_not_sensitive() -> None:
    """viewer/operator/auditor get the benign surface reads; google/connections stay out.

    The DEFAULT_RBAC browse roles are spliced with ``studio.<surface>:read`` for every
    surface EXCEPT the un-partitionable operator/global-store surfaces
    (``_STUDIO_GLOBAL_STORE_RESOURCES``: connections, google, memory, …). So a default
    ``viewer`` reads models/runs/approvals but is 403 on the operator's connection and
    Google surfaces — least-privilege, no per-deployment policy file needed.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        # Benign tenant-facing reads granted by default.
        assert c.get("/api/studio/models").status_code == 200, role
        assert c.get("/api/studio/runs").status_code == 200, role
        # Sensitive operator surfaces withheld by default (admin-only). ``approvals``
        # (reattack-r4: process-wide HITL checkpoint store, no per-tenant partition) and the
        # operator provider-credential surface ``/models/providers`` (reattack-r6) are both
        # in ``_STUDIO_GLOBAL_STORE_RESOURCES``.
        assert c.get("/api/studio/approvals").status_code == 403, role
        assert c.get("/api/studio/models/providers").status_code == 403, role
        assert c.get("/api/studio/connections").status_code == 403, role
        assert c.get("/api/studio/google").status_code == 403, role
        assert c.get("/api/studio/google/gmail").status_code == 403, role


def test_tenant_bound_runs_reader_cannot_cross_tenants() -> None:
    """A tenant-bound principal with ``studio.runs:read`` only sees its own workspace.

    The per-surface read guard is the surface gate, NOT the tenant boundary: the run
    readers stay tenant-filtered (``studio_tenant_filter``), so a principal bound to
    tenant ``t`` cannot fetch a run stamped to another workspace by id — it 404s rather
    than leaking existence, even though the role DOES hold ``studio.runs:read``.
    """
    from himmy.api import ApiContainer, create_app
    from himmy.api.auth.apikey import ApiKeyAuthenticator

    container = ApiContainer.build_default()
    app = create_app(container)
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=["runs_reader"], auth_method="apikey"
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping(
        {"runs_reader": ["studio.console:read", "studio.runs:read"]}
    )
    c = _client_with_key(app)
    # Seed a run owned by a DIFFERENT workspace into the canonical store.
    import anyio

    from himmy.services.storage.models import RunRecord, RunStatus

    other = RunRecord(
        run_id="other-tenant-run",
        workspace_id="other",
        subject_id="someone-else",
        status=RunStatus.SUCCEEDED,
    )
    anyio.run(container.storage.save_run, other)
    # The granted reader is 404 (not 200, not 403) on the cross-tenant run by id.
    assert c.get("/api/studio/runs/other-tenant-run").status_code == 404
    # And the list does not include the other tenant's run.
    body = c.get("/api/studio/runs").json()
    assert all(item["run_id"] != "other-tenant-run" for item in body["items"])


def test_tenant_bound_studio_routine_reader_cannot_cross_tenants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-bound principal with ``studio.routines:read`` cannot read a foreign routine.

    Regression (red-team reattack-r2): the Studio by-id routine readers
    (get/update/delete/run-now) called ``store.get(routine_id)`` with NO workspace filter,
    while the ``/v1`` sibling carries ``store.get(routine_id, workspace_id=...)``. Both
    surfaces share ``.himmy/routines.db``, so a tenant could read another tenant's routine
    row (prompt, agent binding, provider/model, last run preview/error) by id. The by-id
    readers now intersect the routine's owning ``workspace_id`` against the caller's tenants
    via ``authorize_studio_object`` and fold a foreign row into a uniform 404.

    reattack-r7: ``studio.routines:read`` is no longer a default browse grant (it was
    withheld to admin-only via ``_STUDIO_GLOBAL_STORE_RESOURCES``), so this test binds a
    role explicitly granted that read — otherwise the router guard would 403 before the
    by-id BOLA filter ran, masking the behaviour under test (the BOLA 404 PAST the guard).
    """
    from himmy.api import routines as svc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    svc.reset_routines_store()
    svc.reset_scheduler()
    try:
        # Seed a routine stamped to ANOTHER tenant's workspace (as POST /v1/routines would).
        foreign = svc.Routine(
            name="victim",
            agent_id="agt_other",
            prompt="another tenant's secret prompt",
            schedule=svc.Schedule(kind="every", hours=6),
            workspace_id="other",
        )
        svc.get_routines_store().upsert(foreign)

        app = create_app(ApiContainer.build_default())
        app.state.authenticator = ApiKeyAuthenticator(
            key_principals={
                "k": Principal.build(
                    "u", tenant_ids=["t"], roles=["rt_reader"], auth_method="apikey"
                )
            }
        )
        app.state.access_policy = AccessPolicy.from_mapping(
            {"rt_reader": ["studio.console:read", "studio.routines:read"]}
        )
        c = _client_with_key(app)
        # The granted reader is 404 (not 200, not 403) on the cross-tenant routine by id.
        assert c.get(f"/api/studio/routines/{foreign.id}").status_code == 404
        # Mutating by-id paths fold the same way (gated additionally by :write, but the
        # read-level BOLA must already 404 a viewer before any role check on a foreign row).
        assert c.post(f"/api/studio/routines/{foreign.id}/run-now").status_code in (
            403,
            404,
        )
    finally:
        svc.reset_routines_store()
        svc.reset_scheduler()


def test_offline_studio_routine_reader_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INVARIANT: offline / no-auth Studio reads a routine by id regardless of workspace."""
    from himmy.api import routines as svc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    svc.reset_routines_store()
    svc.reset_scheduler()
    try:
        foreign = svc.Routine(
            name="any",
            agent_id="agt_x",
            prompt="p",
            schedule=svc.Schedule(kind="every", hours=6),
            workspace_id="other",
        )
        svc.get_routines_store().upsert(foreign)
        c = TestClient(create_app(ApiContainer.build_default()))
        # No authenticator → ANONYMOUS all-tenants → the by-id read is byte-unchanged.
        assert c.get(f"/api/studio/routines/{foreign.id}").status_code == 200
    finally:
        svc.reset_routines_store()
        svc.reset_scheduler()


def test_studio_routines_list_withheld_from_browse_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reattack-r7: GET /api/studio/routines (LIST) is admin-only, not browse.

    The LIST endpoint hard-scopes to the ``__local__`` workspace and has NO
    ``authorize_studio_object`` / ``studio_tenant_filter`` gate (unlike the by-id paths),
    so it returned EVERY operator-local routine's ``agent_path`` (server filesystem path),
    ``prompt``, provider/model and ``last_preview`` (run output) to any tenant-facing browse
    role. ``studio.routines`` is now in ``_STUDIO_GLOBAL_STORE_RESOURCES`` (single-user-local
    store with no per-tenant axis to intersect on), so ``studio.routines:read`` drops out of
    the default browse roles → 403 for viewer/operator/auditor; admin still reads it.
    """
    from himmy.api import routines as svc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    svc.reset_routines_store()
    svc.reset_scheduler()
    try:
        # Seed an operator-local routine carrying infrastructure recon (agent_path/prompt).
        local = svc.Routine(
            name="ops-secret",
            agent_path="/srv/operator/secret-agent.yaml",
            prompt="the operator's private routine prompt",
            schedule=svc.Schedule(kind="every", hours=6),
            workspace_id=svc.LOCAL_WORKSPACE,
        )
        svc.get_routines_store().upsert(local)

        for role in ("viewer", "operator", "auditor"):
            c = _client_with_key(_app_with_key(role))
            assert c.get("/api/studio/routines").status_code == 403, role
        # admin (``*:*``) still lists, and the local row is visible to it.
        ca = _client_with_key(_app_with_key("admin"))
        resp = ca.get("/api/studio/routines")
        assert resp.status_code == 200
        assert any(r["agent_path"] == "/srv/operator/secret-agent.yaml" for r in resp.json())
        # A deployment can still opt a role back in via an explicit policy grant.
        app = _app_with_policy(
            "rt_reader", ["studio.console:read", "studio.routines:read"]
        )
        assert _client_with_key(app).get("/api/studio/routines").status_code == 200
    finally:
        svc.reset_routines_store()
        svc.reset_scheduler()


def test_studio_routines_list_offline_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INVARIANT: offline / no-auth Studio lists local routines byte-unchanged."""
    from himmy.api import routines as svc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    svc.reset_routines_store()
    svc.reset_scheduler()
    try:
        local = svc.Routine(
            name="r",
            agent_path="/srv/x.yaml",
            prompt="p",
            schedule=svc.Schedule(kind="every", hours=6),
            workspace_id=svc.LOCAL_WORKSPACE,
        )
        svc.get_routines_store().upsert(local)
        c = TestClient(create_app(ApiContainer.build_default()))
        resp = c.get("/api/studio/routines")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
    finally:
        svc.reset_routines_store()
        svc.reset_scheduler()


def test_studio_agent_detail_withheld_from_browse_roles() -> None:
    """reattack-r3: GET /api/studio/agent (DETAIL) leaked the full operator spec.

    The r10 round locked the agent/team INVENTORY (``GET /api/studio/agents`` /
    ``/teams``) to admin-only ``studio.agents:read`` / ``studio.teams:read`` because it
    leaks project-relative server paths. But the DETAIL route ``GET /api/studio/agent?path=``
    carried no route-level dependency, so it inherited only the ``studio.console:read``
    baseline every tenant browse role holds — disclosing strictly MORE than the locked list
    (the full system-prompt body, provider/model, tool packs, skills, the spec path). The
    sibling tool-pack / skill inventories had the same gap. All three now require
    ``studio.agents:read``, so a viewer/operator/auditor is 403 (the RBAC dependency fires
    before any filesystem resolution); admin still reads them.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        assert c.get("/api/studio/agents").status_code == 403, role  # r10 list (control)
        assert (
            c.get("/api/studio/agent", params={"path": "agent.yaml"}).status_code == 403
        ), role
        assert c.get("/api/studio/tools").status_code == 403, role
        assert c.get("/api/studio/skills").status_code == 403, role
    ca = _client_with_key(_app_with_key("admin"))
    # admin reaches the gate (200 for the catalogs; 404 only if no agent.yaml on disk).
    assert ca.get("/api/studio/tools").status_code == 200
    assert ca.get("/api/studio/skills").status_code == 200
    assert ca.get("/api/studio/agent", params={"path": "agent.yaml"}).status_code in (
        200,
        404,
    )
    # OFFLINE (no authenticator) reads them byte-unchanged.
    offline = TestClient(create_app(ApiContainer.build_default()))
    assert offline.get("/api/studio/tools").status_code == 200
    assert offline.get("/api/studio/skills").status_code == 200


def test_studio_eval_suites_withheld_from_browse_roles() -> None:
    """reattack-r7: GET /api/studio/eval/suites (LIST) is admin-only, not browse.

    ``list_suites`` enumerates the operator's local filesystem eval-suite NAMES and PATHS
    (``RunnableSuite.path``/``source``), operator-local FS reconnaissance. ``studio.eval`` is
    now withheld from the default browse roles (in ``_STUDIO_GLOBAL_STORE_RESOURCES``), so a
    viewer/operator/auditor is 403; admin still reads it.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        assert c.get("/api/studio/eval/suites").status_code == 403, role
    ca = _client_with_key(_app_with_key("admin"))
    assert ca.get("/api/studio/eval/suites").status_code == 200
    # OFFLINE (no authenticator) reads it byte-unchanged.
    offline = TestClient(create_app(ApiContainer.build_default()))
    assert offline.get("/api/studio/eval/suites").status_code == 200


def test_studio_evals_discovery_withheld_from_browse_roles() -> None:
    """reattack-r8: GET /api/studio/evals is the un-hardened twin of /eval/suites.

    Both call ``studio_eval.discover_suites()`` (operator-local eval-suite NAMES/PATHS/
    case-counts), but ``GET /api/studio/evals`` previously carried no route-level guard, so
    it inherited only the router-level ``studio.console:read`` baseline every tenant browse
    role holds — bypassing the r7 lockdown. ``studio.evals`` is now in
    ``_STUDIO_GLOBAL_STORE_RESOURCES`` (admin-only) and the route is gated on
    ``studio.evals:read``, so a viewer/operator/auditor is 403; admin still reads it.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        assert c.get("/api/studio/evals").status_code == 403, role
    ca = _client_with_key(_app_with_key("admin"))
    assert ca.get("/api/studio/evals").status_code == 200
    # OFFLINE (no authenticator) reads it byte-unchanged.
    offline = TestClient(create_app(ApiContainer.build_default()))
    assert offline.get("/api/studio/evals").status_code == 200


def test_studio_workflows_discovery_withheld_from_browse_roles() -> None:
    """reattack-r8: GET /api/studio/workflows leaked operator workflow topology.

    ``discover_workflows()`` enumerates the operator's workflow specs (project-relative path
    + name + step/tool graph). The read previously carried no route-level dependency, so it
    was gated only by the router-level ``studio.console:read`` baseline every tenant browse
    role holds. ``studio.workflows`` is now in ``_STUDIO_GLOBAL_STORE_RESOURCES`` (admin-only)
    and the route is gated on ``studio.workflows:read``, so a viewer/operator/auditor is 403;
    admin still reads it.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        assert c.get("/api/studio/workflows").status_code == 403, role
    ca = _client_with_key(_app_with_key("admin"))
    assert ca.get("/api/studio/workflows").status_code == 200
    # OFFLINE (no authenticator) reads it byte-unchanged.
    offline = TestClient(create_app(ApiContainer.build_default()))
    assert offline.get("/api/studio/workflows").status_code == 200


def test_studio_benchmarks_withheld_from_browse_roles() -> None:
    """reattack-r7: GET /api/studio/benchmarks raised to ``studio.console:write`` (admin).

    Consistency with its operator-topology siblings (``/doctor``, ``/benchmarks/probe``)
    raised in r6: a tenant browse role holding only ``studio.console:read`` is 403, while
    admin (and the benign per-model accuracy/latency summary via the model catalog under
    ``studio.modelcatalog:read``) is unaffected.
    """
    for role in ("viewer", "operator", "auditor"):
        c = _client_with_key(_app_with_key(role))
        assert c.get("/api/studio/benchmarks").status_code == 403, role
    ca = _client_with_key(_app_with_key("admin"))
    assert ca.get("/api/studio/benchmarks").status_code == 200
    # OFFLINE (no authenticator) reads it byte-unchanged.
    offline = TestClient(create_app(ApiContainer.build_default()))
    assert offline.get("/api/studio/benchmarks").status_code == 200


def test_escape_hatch_disables_studio_rbac(monkeypatch: pytest.MonkeyPatch) -> None:
    """HIMMY_STUDIO_AUTH=off (dangerous) skips the permission check, not authn."""
    monkeypatch.setenv("HIMMY_STUDIO_AUTH", "off")
    app = _app_with_key("viewer")
    assert _client_with_key(app).get("/api/studio/health").status_code == 200
    # Authentication itself still applies: no key is still a 401.
    assert TestClient(app).get("/api/studio/health").status_code == 401


# ------------------------------------------------- symlink-safe spec paths
def test_resolve_spec_path_plain_file(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("name: a\n")
    assert resolve_spec_path("agent.yaml", tmp_path).name == "agent.yaml"


def test_resolve_spec_path_traversal_rejected(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (tmp_path / "outside.yaml").write_text("name: evil\n")
    with pytest.raises(ValueError, match="escapes project root"):
        resolve_spec_path("../outside.yaml", root)


def test_resolve_spec_path_symlink_to_outside_rejected(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    secret = tmp_path / "secret.yaml"
    secret.write_text("name: evil\n")
    (root / "link.yaml").symlink_to(secret)
    with pytest.raises(ValueError):
        resolve_spec_path("link.yaml", root)


def test_resolve_spec_path_symlink_inside_root_rejected(tmp_path: Path) -> None:
    """Even a link whose destination stays under the root is refused."""
    real = tmp_path / "real.yaml"
    real.write_text("name: ok\n")
    (tmp_path / "alias.yaml").symlink_to(real)
    with pytest.raises(ValueError, match="symlinked path rejected"):
        resolve_spec_path("alias.yaml", tmp_path)


def test_resolve_spec_path_symlinked_dir_component_rejected(tmp_path: Path) -> None:
    """A symlinked directory below the root is refused, even pointing inside."""
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "a.yaml").write_text("name: a\n")
    (tmp_path / "shortcut").symlink_to(agents)
    with pytest.raises(ValueError, match="symlinked path rejected"):
        resolve_spec_path("shortcut/a.yaml", tmp_path)
    # The real path is still fine.
    assert resolve_spec_path("agents/a.yaml", tmp_path).name == "a.yaml"
