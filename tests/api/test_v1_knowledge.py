"""T3a — /v1 knowledge CRUD + ingest + search, workspace-namespaced (isolation-honest).

These tests pin the acceptance for UNIT 2:

* ingest a document then search it back returns the ingested chunk (round-trip);
* the namespace key derivation is the SINGLE choke point ``"{workspace}:{collection}"``;
* AUTHENTICATED regime: a cross-workspace search does NOT return another workspace's
  docs (the boundary holds with an authenticator configured);
* OFFLINE regime: the boundary is INERT — any caller can name any workspace, so a
  second caller in a different "workspace" CAN read the first's docs offline (the
  caveat is asserted, not pretended away);
* list + delete; RBAC denies a principal lacking ``knowledge:write``;
* ``ingest_file`` (server-path read) is operator-only — a tenant caller is rejected.

The plain round-trip / list / delete tests run offline (stub embedder, in-RAM store,
NO authenticator), the standalone single-user posture. The two-regimes isolation is
asserted explicitly: once against an authenticated multi-tenant deployment (boundary
holds) and once offline (boundary inert).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.services.knowledge.app_service import (
    KnowledgeAppService,
    knowledge_namespace,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """An offline /v1 client over a fresh in-RAM container (no auth → operator)."""
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        yield test_client


def _ingest(
    client: TestClient,
    *,
    collection: str = "notes",
    workspace_id: str = "acme",
    text: str = "The quick brown fox jumps over the lazy dog near the riverbank.",
    title: str | None = "fox",
) -> dict[str, Any]:
    """Ingest one text document; return the response JSON."""
    res = client.post(
        f"/v1/knowledge/{collection}/documents",
        json={"workspace_id": workspace_id, "text": text, "title": title},
    )
    assert res.status_code == 201, res.text
    body: dict[str, Any] = res.json()
    return body


# --------------------------------------------------------------- round-trip


def test_ingest_then_search_returns_the_ingested_doc(client: TestClient) -> None:
    """POST a document then GET search returns the chunk for that document (headline)."""
    ingested = _ingest(client)
    document_id = ingested["document"]["document_id"]
    assert ingested["collection"] == "notes"
    assert ingested["document"]["title"] == "fox"

    res = client.get(
        "/v1/knowledge/notes/search",
        params={"workspace_id": "acme", "q": "brown fox riverbank"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["total"] >= 1
    doc_ids = {h["document_id"] for h in body["hits"]}
    assert document_id in doc_ids
    # The hit carries the chunk text + a positive similarity.
    top = body["hits"][0]
    assert top["similarity"] > 0.0
    assert top["text"]


def test_search_unknown_collection_is_empty_not_error(client: TestClient) -> None:
    """Searching a collection that was never ingested into returns no hits (200, empty)."""
    res = client.get(
        "/v1/knowledge/never-made/search",
        params={"workspace_id": "acme", "q": "anything"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["hits"] == []


def test_search_requires_a_query(client: TestClient) -> None:
    """A blank query is a 422 (no implicit match-all)."""
    res = client.get(
        "/v1/knowledge/notes/search", params={"workspace_id": "acme", "q": "  "}
    )
    assert res.status_code == 422


def test_list_then_delete_document(client: TestClient) -> None:
    """List shows the ingested document; delete removes it; re-delete is 404."""
    document_id = _ingest(client)["document"]["document_id"]

    listing = client.get(
        "/v1/knowledge/notes/documents", params={"workspace_id": "acme"}
    )
    assert listing.status_code == 200
    assert document_id in {d["document_id"] for d in listing.json()["items"]}

    deleted = client.delete(
        f"/v1/knowledge/notes/documents/{document_id}",
        params={"workspace_id": "acme"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"

    # Gone from the listing and a re-delete is a 404.
    listing2 = client.get(
        "/v1/knowledge/notes/documents", params={"workspace_id": "acme"}
    ).json()
    assert document_id not in {d["document_id"] for d in listing2["items"]}
    redelete = client.delete(
        f"/v1/knowledge/notes/documents/{document_id}",
        params={"workspace_id": "acme"},
    )
    assert redelete.status_code == 404


def test_ingest_too_large_is_413(client: TestClient) -> None:
    """A body over the character cap is rejected 413 (size limit enforced)."""
    from himmy.api.routers.knowledge import MAX_INGEST_CHARS

    res = client.post(
        "/v1/knowledge/notes/documents",
        json={"workspace_id": "acme", "text": "x" * (MAX_INGEST_CHARS + 1)},
    )
    assert res.status_code == 413


# --------------------------------------------------- namespace choke point


def test_namespace_key_derivation_is_the_single_choke_point() -> None:
    """The workspace+collection are joined in exactly one place, with the documented shape."""
    assert knowledge_namespace("acme", "notes") == "acme:notes"
    assert knowledge_namespace("globex", "notes") == "globex:notes"
    # Same collection name, different workspace -> different KB identity.
    assert knowledge_namespace("acme", "notes") != knowledge_namespace(
        "globex", "notes"
    )


@pytest.mark.asyncio
async def test_service_namespaces_collections_by_workspace() -> None:
    """The service resolves a different KB per workspace for the same collection name.

    Asserted at the service layer (where the workspace argument is honoured): ingesting
    into ``acme``'s ``notes`` and searching ``globex``'s ``notes`` returns nothing,
    because the namespace key differs — the boundary the authenticated HTTP layer relies
    on.
    """
    from himmy.services.knowledge.service import KnowledgeBase
    from himmy.services.storage.service import StorageService

    svc = KnowledgeAppService(KnowledgeBase(storage=StorageService()))

    doc = await svc.ingest_text(
        workspace_id="acme",
        collection="notes",
        text="acme proprietary roadmap: launch the widget in Q3.",
        title="roadmap",
    )
    assert doc.document_id

    # Same workspace + collection finds it.
    own = await svc.search(
        workspace_id="acme", collection="notes", query="widget roadmap launch"
    )
    assert any(h.document_id == doc.document_id for h in own)

    # A DIFFERENT workspace, SAME collection name, finds nothing (namespace differs).
    other = await svc.search(
        workspace_id="globex", collection="notes", query="widget roadmap launch"
    )
    assert other == []


# --------------------------------------------- authenticated isolation (regime 1)


def _tenant_app() -> FastAPI:
    """An app with two real tenant API keys (the AUTHENTICATED multi-tenant regime)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

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
    return app


def test_cross_workspace_search_is_isolated_authenticated() -> None:
    """With auth configured, B cannot retrieve A's docs even naming the same collection.

    A tenant-bound key may only address its OWN workspace (resolve_workspace 403s a
    foreign workspace_id), and the namespace embeds that trusted workspace — so B's
    search of its own ``notes`` returns nothing of A's, and B asking for tenant-a is 403.
    """
    app = _tenant_app()
    with TestClient(app) as base:
        client_a = TestClient(app)
        client_a.headers.update({"x-himmy-internal-key": "key-a"})
        client_b = TestClient(app)
        client_b.headers.update({"x-himmy-internal-key": "key-b"})

        # A ingests a secret into tenant-a/notes.
        ingested = client_a.post(
            "/v1/knowledge/notes/documents",
            json={
                "workspace_id": "tenant-a",
                "text": "tenant-a confidential merger terms with the riverbank deal.",
                "title": "secret",
            },
        )
        assert ingested.status_code == 201, ingested.text
        doc_id = ingested.json()["document"]["document_id"]

        # A finds its own doc.
        own = client_a.get(
            "/v1/knowledge/notes/search",
            params={"workspace_id": "tenant-a", "q": "merger riverbank deal"},
        ).json()
        assert doc_id in {h["document_id"] for h in own["hits"]}

        # B searching its OWN notes finds nothing of A's.
        b_own = client_b.get(
            "/v1/knowledge/notes/search",
            params={"workspace_id": "tenant-b", "q": "merger riverbank deal"},
        ).json()
        assert b_own["hits"] == []
        assert doc_id not in {h["document_id"] for h in b_own["hits"]}

        # B cannot even ASK for tenant-a (403 at the workspace choke point).
        forbidden = client_b.get(
            "/v1/knowledge/notes/search",
            params={"workspace_id": "tenant-a", "q": "merger"},
        )
        assert forbidden.status_code == 403
        # base only exists to keep the lifespan-managed app from being GC'd early.
        assert base is not None


def test_rbac_denies_write_without_permission() -> None:
    """A viewer (knowledge:read only) is denied 403 on an ingest (knowledge:write)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-v": Principal.build(
                "viewer-user",
                tenant_ids=["tenant-a"],
                roles=["viewer"],
                auth_method="apikey",
            )
        }
    )
    with TestClient(app) as client:
        client.headers.update({"x-himmy-internal-key": "key-v"})
        # Read is allowed (empty result).
        assert (
            client.get(
                "/v1/knowledge/notes/search",
                params={"workspace_id": "tenant-a", "q": "x"},
            ).status_code
            == 200
        )
        # Write is denied.
        denied = client.post(
            "/v1/knowledge/notes/documents",
            json={"workspace_id": "tenant-a", "text": "should be denied"},
        )
        assert denied.status_code == 403


def test_ingest_file_is_operator_only_for_a_tenant() -> None:
    """A tenant-bound caller cannot ingest a SERVER file path (operator-only, no LFI)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-a": Principal.build(
                "user-a",
                tenant_ids=["tenant-a"],
                roles=["operator"],
                auth_method="apikey",
            )
        }
    )
    with TestClient(app) as client:
        client.headers.update({"x-himmy-internal-key": "key-a"})
        res = client.post(
            "/v1/knowledge/notes/documents/file",
            json={"workspace_id": "tenant-a", "file": "/etc/passwd"},
        )
        assert res.status_code == 403, res.text


# ------------------------------------------------- offline isolation INERT (regime 2)


def test_offline_isolation_is_inert(client: TestClient) -> None:
    """OFFLINE (no auth): the workspace namespace is a LABEL, not a boundary.

    With no authenticator the caller chooses ``workspace_id`` freely, so a second caller
    naming the FIRST caller's workspace CAN read its docs. This asserts the documented
    'isolation is authenticated-deployment-only, inert offline' caveat — it is NOT a
    false 'isolation everywhere' pass.
    """
    ingested = client.post(
        "/v1/knowledge/notes/documents",
        json={
            "workspace_id": "acme",
            "text": "offline anyone can read this if they name the workspace.",
            "title": "open",
        },
    )
    assert ingested.status_code == 201
    doc_id = ingested.json()["document"]["document_id"]

    # A different "tenant" simply naming workspace_id=acme reads it — no boundary offline.
    leaked = client.get(
        "/v1/knowledge/notes/search",
        params={"workspace_id": "acme", "q": "anyone read workspace"},
    ).json()
    assert doc_id in {h["document_id"] for h in leaked["hits"]}, (
        "offline isolation is expected to be inert (this is the documented caveat)"
    )
