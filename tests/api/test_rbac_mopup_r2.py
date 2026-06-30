"""Round-2 red-team mop-up: cross-tenant isolation on the Studio memory/knowledge/chats surfaces.

Each test pins a CONFIRMED cross-tenant vuln found in the second red-team round and asserts the
by-construction fix holds while the offline / single-box invariant stays byte-unchanged:

* memory_edit (PATCH /api/studio/memory/{id}) skipped the ``t:<workspace>:`` TENANT-prefix axis
  its siblings enforce → a tenant-bound ``studio.memory:write`` holder could overwrite ANOTHER
  tenant's memory by id (and over-deny its own namespaced memory). Now 404 cross-tenant.
* chats_save (POST /api/studio/chats) had no cross-tenant id-collision pre-check → reusing a
  foreign session_id overwrote its thread/title/messages. Now 404 cross-tenant.
* the colon-bearing-workspace prefix collision (``acme`` vs ``acme:eu``) on the column-less
  memory store → ``tenant_namespace_segment`` escaping makes the namespaces disjoint.
* Studio knowledge had NO tenant namespacing (fixed ``("studio","local")`` scope) → two
  tenant-bound principals shared one global KB. Now scoped per (tenant, subject).
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

_GRANTS = [
    "studio.console:read",
    "studio.memory:read",
    "studio.memory:write",
    "studio.chats:read",
    "studio.chats:write",
    "studio.knowledge:read",
    "studio.knowledge:write",
    "studio.runs:read",
    "studio.runs:write",
]


def _two_tenant_app(*, subject_scoped: bool = False) -> FastAPI:
    """An app with two mapped keys ``kA``/``kB`` bound to tenants ``A``/``B`` (all_tenants=False)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "kA": Principal.build(
                "userA",
                tenant_ids=["A"],
                roles=["member"],
                auth_method="apikey",
                subject_scoped=subject_scoped,
            ),
            "kB": Principal.build(
                "userB",
                tenant_ids=["B"],
                roles=["member"],
                auth_method="apikey",
                subject_scoped=subject_scoped,
            ),
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping({"member": list(_GRANTS)})
    return app


def _client(app: FastAPI, key: str) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": key})
    return c


@pytest.fixture
def isolated_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh cwd + isolated memory/chats/knowledge singletons reset around the test."""
    from himmy.api import studio_chats, studio_knowledge, studio_memory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_MEMORY_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("HIMMY_CHATS_PATH", str(tmp_path / "conversations.db"))
    studio_memory.reset_memory_service()
    studio_chats.reset_chats_store()
    studio_knowledge.reset_kb_service()
    yield tmp_path
    studio_memory.reset_memory_service()
    studio_chats.reset_chats_store()
    studio_knowledge.reset_kb_service()


# --------------------------------------------------------------- memory_edit tenant axis


def test_memory_edit_cannot_overwrite_another_tenants_memory(isolated_stores: Path) -> None:
    """PATCH /api/studio/memory/{id} by tenant B on tenant A's memory → 404, no write."""
    from himmy.api import studio_memory

    app = _two_tenant_app()
    ca = _client(app, "kA")
    cb = _client(app, "kB")

    # Tenant A writes a memory through the route (stamped under t:A:default).
    made = ca.post("/api/studio/memory", json={"text": "ACME secret roadmap"})
    assert made.status_code == 200, made.text
    a_id = made.json()["memory_id"]
    # Confirm it landed in tenant A's namespace on the shared store.
    rec = studio_memory.get_memory(a_id)
    assert rec is not None and rec.subject_id == "t:A:default"

    # Tenant B tries to overwrite A's memory by id — must fold to 404 (existence not leaked).
    resp = cb.patch(f"/api/studio/memory/{a_id}", json={"text": "POISONED BY B"})
    assert resp.status_code == 404, resp.text

    # A's memory is byte-unchanged.
    rec = studio_memory.get_memory(a_id)
    assert rec is not None and rec.text == "ACME secret roadmap"


def test_memory_edit_rewrites_own_tenants_memory(isolated_stores: Path) -> None:
    """A tenant CAN still edit a memory it wrote through the route (gate not over-broad)."""
    from himmy.api import studio_memory

    app = _two_tenant_app()
    ca = _client(app, "kA")

    a_id = ca.post("/api/studio/memory", json={"text": "first"}).json()["memory_id"]
    resp = ca.patch(f"/api/studio/memory/{a_id}", json={"text": "second"})
    assert resp.status_code == 200, resp.text
    rec = studio_memory.get_memory(a_id)
    assert rec is not None and rec.text == "second"


# --------------------------------------------------------------- chats_save tenant axis


def test_chats_save_cannot_overwrite_another_tenants_thread(isolated_stores: Path) -> None:
    """POST /api/studio/chats reusing tenant A's session_id by tenant B → 404, content intact."""
    from himmy.api.studio_chats import get_chats_store

    app = _two_tenant_app()
    ca = _client(app, "kA")
    cb = _client(app, "kB")

    sid = "shared-session-id"
    saved = ca.post(
        "/api/studio/chats",
        json={"id": sid, "title": "A title", "messages": [{"role": "user", "text": "A msg"}]},
    )
    assert saved.status_code == 200, saved.text

    # Tenant B reuses A's session id — must 404 before clobbering A's thread.
    resp = cb.post(
        "/api/studio/chats",
        json={"id": sid, "title": "B title", "messages": [{"role": "user", "text": "overwritten"}]},
    )
    assert resp.status_code == 404, resp.text

    # A's thread is byte-unchanged (read back as tenant A).
    detail = get_chats_store().get(sid, workspace_id=frozenset({"A"}))
    assert detail is not None
    assert detail.title == "A title"
    assert [m.text for m in detail.messages] == ["A msg"]


def test_chats_save_new_id_still_saves(isolated_stores: Path) -> None:
    """A tenant saving a fresh session id is unaffected (no false 404)."""
    app = _two_tenant_app()
    ca = _client(app, "kA")
    resp = ca.post(
        "/api/studio/chats",
        json={"id": "new-one", "title": "t", "messages": [{"role": "user", "text": "hi"}]},
    )
    assert resp.status_code == 200, resp.text


# --------------------------------------------------------------- colon-bearing workspace ids


def test_colon_bearing_workspace_ids_do_not_collide(isolated_stores: Path) -> None:
    """Tenant ``acme`` cannot read/enumerate tenant ``acme:eu``'s memories via prefix collision."""
    from himmy.api import studio_memory

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k1": Principal.build(
                "u1", tenant_ids=["acme"], roles=["member"], auth_method="apikey"
            ),
            "k2": Principal.build(
                "u2", tenant_ids=["acme:eu"], roles=["member"], auth_method="apikey"
            ),
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping({"member": list(_GRANTS)})

    c1 = _client(app, "k1")
    c2 = _client(app, "k2")

    # acme:eu writes a confidential memory under subject "deals".
    made = c2.post(
        "/api/studio/memory", json={"text": "EU merger price 4.2B", "subject_id": "deals"}
    )
    assert made.status_code == 200, made.text
    eu_id = made.json()["memory_id"]
    # It is stored with the ESCAPED workspace segment so acme's prefix can't straddle it.
    rec = studio_memory.get_memory(eu_id)
    assert rec is not None and rec.subject_id == "t:acme%3Aeu:deals"

    # acme must NOT see acme:eu's subject in enumeration.
    subs = c1.get("/api/studio/memory/subjects").json()
    assert "deals" not in subs and "eu:deals" not in subs

    # acme must NOT read acme:eu's memory text via list or recall.
    listed = c1.get("/api/studio/memory", params={"subject": "deals"}).json()
    assert all("merger" not in i["text"] for i in listed)
    hits = c1.post(
        "/api/studio/memory/recall", json={"query": "merger", "subject_id": "deals", "top_k": 50}
    ).json()
    assert all("merger" not in h["text"] for h in hits)


# --------------------------------------------------------------- knowledge tenancy


def test_knowledge_kb_is_tenant_scoped(isolated_stores: Path) -> None:
    """Two tenant-bound principals do NOT share one global Studio KB."""
    app = _two_tenant_app()
    ca = _client(app, "kA")
    cb = _client(app, "kB")

    kb = ca.post("/api/studio/knowledge", json={"name": "acme-docs"})
    assert kb.status_code == 200, kb.text
    kb_id = kb.json()["kb_id"]
    ing = ca.post(
        f"/api/studio/knowledge/{kb_id}/ingest",
        json={"text": "ACME confidential pricing memo line one two three"},
    )
    assert ing.status_code == 200, ing.text

    # Tenant B cannot see A's KB in the list.
    b_list = cb.get("/api/studio/knowledge").json()
    assert all(k["kb_id"] != kb_id for k in b_list)

    # Tenant B cannot read A's chunks via a raw kb_id search → 404 (existence not leaked).
    resp = cb.post(
        f"/api/studio/knowledge/{kb_id}/search", json={"query": "pricing", "top_k": 5}
    )
    assert resp.status_code == 404, resp.text

    # Tenant A still reads its own.
    own = ca.post(
        f"/api/studio/knowledge/{kb_id}/search", json={"query": "pricing", "top_k": 5}
    )
    assert own.status_code == 200, own.text


_STUB_TEAM = """\
entry: lead
members:
  - name: lead
    description: Answer directly.
    provider: stub
    tool_packs: [memory]
"""


def test_team_run_threads_tenant_scope_into_tool_packs(
    isolated_stores: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/studio/run-team threads the launching tenant into build_team's toolkit_config.

    Without this the team members' memory/KB packs revert to ToolkitConfig.from_env() with no
    tenant/subject scope (the static shared "default" / ("local","local") namespace) — a
    cross-tenant confused-deputy. We capture the config build_team receives and assert it is
    namespaced to the launching tenant.
    """
    import himmy.config.team_spec as team_spec_mod

    (isolated_stores / "demo.team.yaml").write_text(_STUB_TEAM)

    captured: dict[str, object] = {}
    real_build_team = team_spec_mod.build_team

    def _spy_build_team(spec, *, toolkit_config=None, resolve_tools_module=None):
        captured["config"] = toolkit_config
        return real_build_team(
            spec, toolkit_config=toolkit_config, resolve_tools_module=resolve_tools_module
        )

    # stream_team_run imports build_team locally from the module, so patch the module attr.
    monkeypatch.setattr(team_spec_mod, "build_team", _spy_build_team)

    app = _two_tenant_app()
    ca = _client(app, "kA")
    with ca.stream(
        "POST", "/api/studio/run-team", json={"team_path": "demo.team.yaml", "prompt": "hi"}
    ) as r:
        assert r.status_code == 200
        for _ in r.iter_lines():
            pass

    cfg = captured.get("config")
    assert cfg is not None, "team-run did not pass a scoped toolkit_config to build_team"
    assert cfg.tenant_scope == "A"
    # The memory subject the members would use is namespaced to tenant A, not the shared default.
    assert cfg.scoped_memory_subject() == "t:A:default"
    assert cfg.scoped_kb_keys() == ("t:A", "t:A")


def test_team_run_offline_leaves_packs_unscoped(
    isolated_stores: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With NO authenticator the team path passes no scoped config (byte-unchanged)."""
    import himmy.config.team_spec as team_spec_mod

    (isolated_stores / "demo.team.yaml").write_text(_STUB_TEAM)
    captured: dict[str, object] = {}
    real_build_team = team_spec_mod.build_team

    def _spy_build_team(spec, *, toolkit_config=None, resolve_tools_module=None):
        captured["config"] = toolkit_config
        return real_build_team(
            spec, toolkit_config=toolkit_config, resolve_tools_module=resolve_tools_module
        )

    monkeypatch.setattr(team_spec_mod, "build_team", _spy_build_team)

    c = TestClient(create_app())
    with c.stream(
        "POST", "/api/studio/run-team", json={"team_path": "demo.team.yaml", "prompt": "hi"}
    ) as r:
        assert r.status_code == 200
        for _ in r.iter_lines():
            pass
    # Offline → no scoped config (build_team falls back to from_env, the historical path).
    assert captured.get("config") is None


def test_knowledge_offline_scope_is_unchanged(isolated_stores: Path) -> None:
    """With NO authenticator the KB scope stays the historical ("studio","local") (byte-unchanged)."""
    from himmy.api import studio_knowledge

    # No authenticator → scope_keys(None-equivalent principal) returns the historical default.
    app = create_app(ApiContainer.build_default())
    c = TestClient(app)
    kb = c.post("/api/studio/knowledge", json={"name": "docs"})
    assert kb.status_code == 200, kb.text
    kb_id = kb.json()["kb_id"]
    c.post(f"/api/studio/knowledge/{kb_id}/ingest", json={"text": "hello world doc"})
    listed = c.get("/api/studio/knowledge").json()
    assert any(k["kb_id"] == kb_id for k in listed)
    # The KB record is under the historical scope.
    assert studio_knowledge._WORKSPACE == "studio"
