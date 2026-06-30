"""Red-team round 9 regressions for the RBAC hardening branch.

Two confirmed findings, each with a test that FAILS before the fix and PASSES after:

* (vuln 1) the Studio model catalog (``GET /api/studio/models`` -> ``build_model_catalog``)
  leaked the SAME operator deployment posture r6 withheld from ``GET /api/studio/health``:
  the host-derived ``available`` booleans ("which LLM binaries are installed") and the live
  LOCAL Ollama model inventory. A default tenant browse role (viewer/operator/auditor) holds
  ``studio.modelcatalog:read``, so it could read that posture. The catalog is now POSTURE-FREE
  for callers that do NOT hold ``studio.console:write`` (the SAME gate ``/health`` uses), while
  admin and OFFLINE keep the full host-aware catalog.

* (vuln 2) the ``context_fields`` store was keyed by ``(subject_id, key)`` ONLY — the
  owning ``workspace_id`` lived inside the payload metadata, never in the key — so a
  tenant-bound upsert under a SHARED ``subject_id`` collided with another tenant's row via
  ``ON CONFLICT (subject_id, key)``: a cross-tenant BOLA WRITE that destroyed the victim's
  value and re-stamped it into the attacker's workspace (so the victim's own read then
  reported it missing). ``workspace_id`` is now a first-class PRIMARY-KEY column, so a write
  can only ever touch the writer's own tenant partition. OFFLINE (no workspace stamp) is
  byte-unchanged.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.services.context.models import ContextField
from himmy.services.storage.inmemory import InMemoryContextStore
from himmy.services.storage.sqlite import SqliteStorageService


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- vuln 1
def _app_with_key(*roles: str) -> FastAPI:
    """An app whose mapped key ``"k"`` is bound to ``roles`` (default-policy roles)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=list(roles), auth_method="apikey"
            )
        }
    )
    return app


def _client(app: FastAPI) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def test_model_catalog_withholds_host_posture_from_browse_roles() -> None:
    """vuln 1: a tenant browse role gets a POSTURE-FREE catalog (no live host inventory).

    The host-derived ``available`` booleans and the live LOCAL Ollama model inventory are the
    SAME "which LLM binaries are installed" operator posture r6 withheld from these roles on
    ``GET /health``. A viewer/operator/auditor must still get a 200 model picker, but with NO
    live Ollama model inventory disclosed (the per-host reconnaissance). The claude-cli static
    tier aliases (haiku/sonnet/opus) are framework constants, not host posture, so they stay.
    """
    from himmy.services.inference import compare

    real = compare.build_model_catalog
    seen: list[bool] = []

    async def _spy(*, reveal_host_posture: bool = True) -> list[dict[str, object]]:
        seen.append(reveal_host_posture)
        return await real(reveal_host_posture=reveal_host_posture)

    compare.build_model_catalog = _spy  # type: ignore[assignment]
    try:
        for role in ("viewer", "operator", "auditor"):
            seen.clear()
            c = _client(_app_with_key(role))
            r = c.get("/api/studio/models")
            assert r.status_code == 200, role
            # The route MUST request the posture-free catalog for a browse role (the gate
            # fires regardless of whether a local Ollama is running in CI).
            assert seen == [False], role
            catalog = {e["provider"]: e for e in r.json()}
            # The live Ollama inventory (host-derived) is withheld from a browse role.
            assert catalog["ollama"]["models"] == [], role
            # ``available`` no longer reflects "installed on THIS host" — it is the static
            # "provider is supported" advertisement, so it cannot be used as a
            # binary-presence probe (always True for the supported providers).
            assert catalog["ollama"]["available"] is True, role
            assert catalog["claude-cli"]["available"] is True, role
    finally:
        compare.build_model_catalog = real  # type: ignore[assignment]


def test_model_catalog_full_host_posture_for_admin() -> None:
    """vuln 1: admin (``studio.console:write`` via ``*:*``) still gets the host-aware catalog.

    Admin is an operator, so it keeps the full catalog seam — the live Ollama inventory (when
    a local Ollama is running) and the real host-derived ``available`` booleans. We can't
    assume a model is installed in CI, so we assert admin gets at LEAST the same posture-free
    catalog and is NOT additionally stripped (the route passes ``reveal_host_posture=True``).
    """
    from himmy.services.inference import compare

    captured: dict[str, bool] = {}
    real = compare.build_model_catalog

    async def _spy(*, reveal_host_posture: bool = True) -> list[dict[str, object]]:
        captured["reveal"] = reveal_host_posture
        return await real(reveal_host_posture=reveal_host_posture)

    app = _app_with_key("admin")
    # The route imports build_model_catalog lazily from the compare module on each call,
    # so patching the module attribute is observed by the route.
    compare.build_model_catalog = _spy  # type: ignore[assignment]
    try:
        c = _client(app)
        assert c.get("/api/studio/models").status_code == 200
        assert captured["reveal"] is True  # admin sees full host posture
    finally:
        compare.build_model_catalog = real  # type: ignore[assignment]


def test_model_catalog_offline_is_byte_unchanged() -> None:
    """vuln 1 INVARIANT: OFFLINE (no authenticator) keeps the FULL host-aware catalog.

    ``_caller_holds_console_write`` returns True with no authenticator, so the route passes
    ``reveal_host_posture=True`` and the single-box console catalog is byte-unchanged.
    """
    from himmy.services.inference import compare

    captured: dict[str, bool] = {}
    real = compare.build_model_catalog

    async def _spy(*, reveal_host_posture: bool = True) -> list[dict[str, object]]:
        captured["reveal"] = reveal_host_posture
        return await real(reveal_host_posture=reveal_host_posture)

    compare.build_model_catalog = _spy  # type: ignore[assignment]
    try:
        c = TestClient(create_app(ApiContainer.build_default()))
        assert c.get("/api/studio/models").status_code == 200
        assert captured["reveal"] is True  # offline → full posture
    finally:
        compare.build_model_catalog = real  # type: ignore[assignment]


def test_build_model_catalog_posture_free_strips_host_inventory() -> None:
    """vuln 1 (unit): the posture-free catalog drops the host Ollama inventory entirely."""
    from himmy.services.inference.compare import build_model_catalog

    posture_free = _run(build_model_catalog(reveal_host_posture=False))
    by_provider = {e["provider"]: e for e in posture_free}  # type: ignore[union-attr]
    assert by_provider["ollama"]["models"] == []
    # No host probe: ``available`` is the static SUPPORTED advertisement.
    assert by_provider["ollama"]["available"] is True


# --------------------------------------------------------------------------- vuln 2
def test_context_field_upsert_is_tenant_partitioned_inmemory() -> None:
    """vuln 2: a tenant's upsert under a SHARED subject_id cannot clobber another tenant's row.

    Tenant B legitimately stores a field under ``subject_id='default'``. Tenant A upserts the
    SAME ``(subject_id, key)`` with its OWN ``workspace_id``. Before the fix the ON CONFLICT
    on ``(subject_id, key)`` destroyed B's row and re-stamped it into A's workspace; after the
    fix the two land in DISTINCT tenant partitions, so B's value survives and is still
    workspace-attributable to B.
    """
    store = InMemoryContextStore()
    fb = ContextField(
        key="home_address",
        value="B-secret",
        metadata={"subject_id": "default", "workspace_id": "tenantB"},
    )
    fa = ContextField(
        key="home_address",
        value="A-poison",
        metadata={"subject_id": "default", "workspace_id": "tenantA"},
    )
    _run(store.save_context_field(fb))
    _run(store.save_context_field(fa))

    got_b = _run(store.get_context_field("default", "home_address", workspace_id="tenantB"))
    got_a = _run(store.get_context_field("default", "home_address", workspace_id="tenantA"))
    assert got_b is not None and got_b.value == "B-secret"  # not destroyed
    assert got_a is not None and got_a.value == "A-poison"
    # Both rows coexist; the subject listing carries one per tenant partition.
    rows = _run(store.list_context_fields("default"))
    assert len(rows) == 2
    stamps = {(r.metadata or {}).get("workspace_id") for r in rows}
    assert stamps == {"tenantB", "tenantA"}


def test_context_field_upsert_is_tenant_partitioned_sqlite() -> None:
    """vuln 2: the same write-partition holds in the persistent SQLite store + its new PK."""
    with tempfile.TemporaryDirectory() as d:
        store = SqliteStorageService(str(Path(d) / "ctx.db"))
        # The PRIMARY KEY now leads with workspace_id (tenant partition).
        cols = {
            r[1]: r[5]
            for r in store._conn.execute("PRAGMA table_info(context_fields)")
        }
        assert cols["workspace_id"] == 1  # first PK column
        assert cols["subject_id"] == 2
        assert cols["key"] == 3

        fb = ContextField(
            key="home_address",
            value="B-secret",
            metadata={"subject_id": "default", "workspace_id": "tenantB"},
        )
        fa = ContextField(
            key="home_address",
            value="A-poison",
            metadata={"subject_id": "default", "workspace_id": "tenantA"},
        )
        _run(store.save_context_field(fb))
        _run(store.save_context_field(fa))

        got_b = _run(
            store.get_context_field("default", "home_address", workspace_id="tenantB")
        )
        assert got_b is not None and got_b.value == "B-secret"  # victim row intact
        rows = _run(store.list_context_fields("default"))
        assert {r.value for r in rows} == {"B-secret", "A-poison"}


def test_context_field_offline_path_byte_unchanged() -> None:
    """vuln 2 INVARIANT: the offline / single-tenant path (no workspace stamp) is unchanged.

    A field written with NO ``workspace_id`` metadata lands in the blank partition, and an
    unscoped ``get_context_field(subject, key)`` (workspace_id=None) reads it back exactly as
    before — no workspace filter is applied.
    """
    store = InMemoryContextStore()
    f = ContextField(key="k", value="v", metadata={"subject_id": "s1"})
    _run(store.save_context_field(f))
    got = _run(store.get_context_field("s1", "k"))
    assert got is not None and got.value == "v"
    # An upsert of the same (subject, key) with no workspace still overwrites in place.
    f2 = ContextField(key="k", value="v2", metadata={"subject_id": "s1"})
    _run(store.save_context_field(f2))
    assert len(_run(store.list_context_fields("s1"))) == 1
    assert _run(store.get_context_field("s1", "k")).value == "v2"


def test_context_app_upsert_cross_tenant_isolation_e2e() -> None:
    """vuln 2 (service-level): ContextAppService.upsert_fields keeps tenants isolated.

    Mirrors the HTTP ``/v1/context/fields:upsert`` path (the router calls
    ``context_app.upsert_fields(workspace_id, subject_id, fields)``). Two tenants sharing the
    literal thread-default ``subject_id='default'`` write the same key; each keeps its own
    value, and each tenant's workspace-scoped LIST returns only its own field.
    """
    from himmy.application.services import ContextAppService
    from himmy.services.context.service import ContextService
    from himmy.services.storage.service import StorageService

    storage = StorageService()
    app_svc = ContextAppService(
        context_service=ContextService(storage_service=storage),
        storage=storage,
    )
    _run(
        app_svc.upsert_fields(
            "tenantB",
            "default",
            [ContextField(key="home_address", value="B-secret")],
        )
    )
    _run(
        app_svc.upsert_fields(
            "tenantA",
            "default",
            [ContextField(key="home_address", value="A-poison")],
        )
    )
    # Tenant B's workspace-scoped read still sees ITS value (not destroyed, not poisoned).
    b_fields = _run(app_svc.list_fields("default", workspace_id="tenantB"))
    a_fields = _run(app_svc.list_fields("default", workspace_id="tenantA"))
    assert [f.value for f in b_fields] == ["B-secret"]
    assert [f.value for f in a_fields] == ["A-poison"]
