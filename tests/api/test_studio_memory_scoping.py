"""Subject (BOLA) scoping for the Studio long-term-memory surface.

The Studio memory store is a process-wide singleton keyed by ``subject_id`` only (no
``workspace_id`` column). Once an authenticator opts a principal into ``subject_scoped``,
the read surfaces historically still honored a client-supplied ``subject_id`` — letting a
subject-scoped principal holding ``studio.memory:read`` enumerate every subject and recall
ANOTHER data subject's semantic memories from the shared store:

* ``GET  /api/studio/memory/subjects`` returned every subject globally (the enumeration
  oracle), and ``GET /api/studio/memory?subject=<victim>`` returned that subject's text;
* ``POST /api/studio/memory/recall`` — a READ the GET-only coverage gate could not even
  see — took ``subject_id`` straight from the body and recalled the victim's memories.

These tests pin the by-construction fix: the effective subject is now derived from the
verified principal (``narrow_subject``), so a ``subject_scoped`` caller is pinned to its
OWN subject and the body/query ``subject_id`` for another subject is never honored. The
offline / ``all_tenants`` path is asserted byte-unchanged (no narrowing).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer
from himmy.api.app import create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy


@pytest.fixture
def memory_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh cwd + isolated memory db, with the singleton memory service reset."""
    from himmy.api import studio_memory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_MEMORY_PATH", str(tmp_path / "memory.db"))
    studio_memory.reset_memory_service()
    yield tmp_path
    studio_memory.reset_memory_service()


def _seed(subject_id: str, text: str) -> str:
    from himmy.api import studio_memory

    return studio_memory.add_memory(
        text, subject_id=subject_id, kind="semantic"
    ).memory_id


def _subject_scoped_client(subject: str) -> TestClient:
    """A built app authenticated as a ``subject_scoped`` principal with memory:read/write."""
    container = ApiContainer.build_default()
    app = create_app(container)
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                subject,
                tenant_ids=["t"],
                roles=["reader"],
                auth_method="apikey",
                subject_scoped=True,
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping(
        {"reader": ["studio.console:read", "studio.memory:read", "studio.memory:write"]}
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    return client


# --------------------------------------------------------------- the BOLA leak is closed


def test_subjects_enumeration_is_narrowed_to_own_subject(memory_project: Path) -> None:
    """A subject-scoped caller cannot enumerate OTHER subjects (oracle closed)."""
    _seed("alice", "alice loves espresso")
    _seed("bob", "bob loves tea")

    client = _subject_scoped_client("alice")
    subjects = client.get("/api/studio/memory/subjects").json()
    assert "bob" not in subjects
    assert subjects in (["alice"], ["default"]) or set(subjects) <= {"alice"}


def test_list_ignores_a_foreign_subject_query(memory_project: Path) -> None:
    """``GET /api/studio/memory?subject=bob`` is narrowed to the caller's own subject."""
    _seed("alice", "alice loves espresso")
    _seed("bob", "bob secret tea blend")

    client = _subject_scoped_client("alice")
    listed = client.get("/api/studio/memory", params={"subject": "bob"}).json()
    texts = [m["text"] for m in listed]
    assert all("tea" not in t for t in texts)
    assert all(m["subject_id"] == "alice" for m in listed)


def test_recall_does_not_honor_a_foreign_subject_id_in_the_body(
    memory_project: Path,
) -> None:
    """POST /memory/recall with a victim ``subject_id`` returns ZERO victim memories.

    This is the read-shaped POST the GET-only coverage gate was blind to; the subject is
    now pinned to the verified principal, so the body value for ``bob`` is ignored.
    """
    _seed("alice", "alice loves espresso")
    _seed("bob", "bob secret tea blend xyz")

    client = _subject_scoped_client("alice")
    resp = client.post(
        "/api/studio/memory/recall",
        json={"query": "tea blend", "subject_id": "bob", "top_k": 50},
    )
    assert resp.status_code == 200, resp.text
    hits = resp.json()
    assert all("tea" not in h["text"] for h in hits), hits
    assert all("xyz" not in h["text"] for h in hits), hits


# ----------------------------------------- the in-place edit BOLA write is closed (scope-r3)


def test_edit_cannot_rewrite_another_subjects_memory(memory_project: Path) -> None:
    """PATCH /api/studio/memory/{victim_id} by a subject-scoped caller returns 404, no write.

    The round-3 BOLA: ``memory_edit`` fetched the victim record and saved an updated copy
    with ZERO object-axis gate, so a ``subject_scoped`` caller holding ``studio.memory:write``
    could rewrite ANOTHER data subject's memory in place. It now folds a cross-subject edit
    to a uniform 404 (existence never leaked) and leaves the victim's text untouched.
    """
    from himmy.api import studio_memory

    # Seed in the tenant's OWN namespace (the shape the route/agent stamps for tenant "t":
    # ``t:t:<subject>``) so the cross-SUBJECT axis is what is exercised, not the tenant axis.
    _seed("t:t:alice", "alice loves espresso")
    victim_id = _seed("t:t:bob", "bob secret tea blend xyz")

    client = _subject_scoped_client("alice")
    resp = client.patch(f"/api/studio/memory/{victim_id}", json={"text": "poisoned"})
    assert resp.status_code == 404, resp.text

    # The victim's record is byte-unchanged: not poisoned, still its own subject/text.
    rec = studio_memory.get_memory(victim_id)
    assert rec is not None
    assert rec.subject_id == "t:t:bob"
    assert rec.text == "bob secret tea blend xyz"


def test_edit_can_rewrite_own_subjects_memory(memory_project: Path) -> None:
    """A subject-scoped caller CAN still edit its OWN subject's memory (gate is not over-broad).

    The own-subject record is seeded in the tenant's ``t:t:<subject>`` namespace — the exact
    shape the Studio ``memory_add`` route and the agent memory pack stamp for tenant "t" — so
    the edit gate strips the prefix before the subject comparison and the legitimate owner can
    still rewrite. (Regression for the round-2 over-deny: an unstripped check 404'd one's own
    namespaced memory.)
    """
    from himmy.api import studio_memory

    own_id = _seed("t:t:alice", "alice loves espresso")

    client = _subject_scoped_client("alice")
    resp = client.patch(f"/api/studio/memory/{own_id}", json={"text": "alice loves tea"})
    assert resp.status_code == 200, resp.text

    rec = studio_memory.get_memory(own_id)
    assert rec is not None
    assert rec.text == "alice loves tea"


def test_offline_edit_rewrites_any_memory(memory_project: Path) -> None:
    """With NO authenticator the edit gate is a NO-OP — any memory is editable (byte-unchanged)."""
    from himmy.api import studio_memory

    victim_id = _seed("bob", "bob secret tea blend xyz")

    app = create_app(ApiContainer.build_default())
    client = TestClient(app)
    resp = client.patch(f"/api/studio/memory/{victim_id}", json={"text": "rewritten"})
    assert resp.status_code == 200, resp.text

    rec = studio_memory.get_memory(victim_id)
    assert rec is not None
    assert rec.text == "rewritten"


# --------------------------------------------------------------- offline byte-unchanged


def test_offline_recall_honors_the_body_subject(memory_project: Path) -> None:
    """With NO authenticator the body ``subject_id`` is honored verbatim (byte-unchanged)."""
    _seed("bob", "bob secret tea blend xyz")

    app = create_app(ApiContainer.build_default())
    client = TestClient(app)
    resp = client.post(
        "/api/studio/memory/recall",
        json={"query": "tea blend", "subject_id": "bob", "top_k": 50},
    )
    assert resp.status_code == 200, resp.text
    hits = resp.json()
    assert any("xyz" in h["text"] for h in hits), hits


def test_offline_subjects_enumeration_returns_all(memory_project: Path) -> None:
    """Offline enumeration still returns every subject (no narrowing)."""
    _seed("alice", "a")
    _seed("bob", "b")

    app = create_app(ApiContainer.build_default())
    client = TestClient(app)
    subjects = client.get("/api/studio/memory/subjects").json()
    assert {"alice", "bob"} <= set(subjects)
