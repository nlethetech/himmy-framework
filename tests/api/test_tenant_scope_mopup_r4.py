"""rbac-harden(mopup-r4): the three confirmed round-4 residual leaks.

Each test FAILS before the mopup-r4 fix and PASSES after. Grouped by finding:

* CONTENT-TAMPER on legacy NULL-owned Studio rows: a by-id upsert by a bound tenant onto a
  legacy ``workspace_id IS NULL`` (offline / shared-key) row preserved the NULL owner but
  still CLOBBERED the row's content (title/body/prompt/thread) via ``INSERT OR REPLACE`` /
  ``ON CONFLICT``. The design says those rows are READ-visible but IMMUTABLE; the store-level
  upsert now ABORTS the write (returns the existing row unchanged) for any row the caller may
  not mutate — across notes / cookbook / calendar / the unified conversation store.
* AGENT FILE SANDBOX cross-tenant BOLA: ``GET /api/studio/files`` + ``/download`` serve a
  single process-global ``fs_root`` with NO tenant dimension, so a tenant B ``files:read``
  principal could enumerate + download tenant A's agent ``write_file`` outputs. The two routes
  are now restricted to an UNSCOPED operator/offline (``all_tenants``) principal.
* HITL APPROVAL subject-axis BOLA: ``_guard_approval_scope`` / ``get_detail`` / ``list_pending``
  gated only the TENANT axis, so a ``subject_scoped`` user of a shared tenant could approve/
  resume (and read the detail of) ANOTHER user's paused gated tool call. The approvals reads +
  the resolve guard now also enforce the within-tenant SUBJECT axis.

The OFFLINE invariant (scope ``None`` / ``all_tenants`` → no filtering, legacy rows untouched)
is asserted throughout so the zero-config single-box path stays byte-for-byte unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal

# ============================================================ content-tamper on NULL rows


def test_notes_legacy_null_content_immutable_to_bound_tenant(tmp_path: Path) -> None:
    """A bound tenant upserting onto a legacy NULL note's id must NOT clobber its body."""
    from himmy.api.studio_notes import Note, NotesStore

    store = NotesStore(str(tmp_path / "notes.db"))
    legacy = Note(title="plan", body="original")
    store.upsert(legacy, workspace_id=None)  # NULL-owned (offline / shared key)

    # Tenant t-a re-uses the disclosed id and tries to overwrite the content.
    returned = store.upsert(
        Note(id=legacy.id, title="x", body="OVERWRITTEN BY t-a"), workspace_id="t-a"
    )
    # The upsert is aborted: it returns the row UNCHANGED, owner stays NULL.
    assert returned.body == "original"
    row = store._conn.execute(
        "SELECT title, body, workspace_id FROM notes WHERE id = ?", (legacy.id,)
    ).fetchone()
    assert row["body"] == "original" and row["title"] == "plan"
    assert row["workspace_id"] is None
    # Offline (None) still mutates it — byte-unchanged.
    store.upsert(Note(id=legacy.id, title="plan", body="edited offline"), workspace_id=None)
    assert store.get(legacy.id, workspace_id=None).body == "edited offline"  # type: ignore[union-attr]
    store.close()


def test_notes_foreign_content_immutable_to_bound_tenant(tmp_path: Path) -> None:
    """Cross-tenant: t-b upserting onto t-a's note id clobbers neither owner nor content."""
    from himmy.api.studio_notes import Note, NotesStore

    store = NotesStore(str(tmp_path / "notes.db"))
    a = Note(title="secret", body="A's")
    store.upsert(a, workspace_id="t-a")
    store.upsert(Note(id=a.id, title="secret", body="hijack"), workspace_id="t-b")
    row = store._conn.execute(
        "SELECT body, workspace_id FROM notes WHERE id = ?", (a.id,)
    ).fetchone()
    assert row["body"] == "A's" and row["workspace_id"] == "t-a"
    store.close()


def test_cookbook_legacy_null_content_immutable_to_bound_tenant(tmp_path: Path) -> None:
    """A bound tenant must not rewrite a legacy NULL recipe's prompt via id-reuse."""
    from himmy.api.studio_cookbook import CookbookStore, Recipe

    store = CookbookStore(str(tmp_path / "cookbook.db"))
    legacy = Recipe(name="R", prompt="original-prompt")
    store.upsert(legacy, workspace_id=None)

    store.upsert(Recipe(id=legacy.id, name="R", prompt="steal"), workspace_id="t-a")
    row = store._conn.execute(
        "SELECT prompt, workspace_id FROM recipes WHERE id = ?", (legacy.id,)
    ).fetchone()
    assert row["prompt"] == "original-prompt"  # content NOT clobbered
    assert row["workspace_id"] is None  # owner NOT re-stamped
    store.close()


def test_calendar_legacy_null_content_immutable_to_bound_tenant(tmp_path: Path) -> None:
    """A bound tenant must not rewrite a legacy NULL event's title via id-reuse."""
    from himmy.api.studio_calendar import CalendarEvent, CalendarStore

    store = CalendarStore(str(tmp_path / "cal.db"))
    ev = CalendarEvent(date="2026-07-01", title="legacy-title")
    store.add(ev, workspace_id=None)

    store.add(
        CalendarEvent(id=ev.id, date="2026-07-01", title="OVERWRITTEN"),
        workspace_id="t-a",
    )
    row = store._conn.execute(
        "SELECT title, workspace_id FROM calendar_events WHERE id = ?", (ev.id,)
    ).fetchone()
    assert row["title"] == "legacy-title"  # content NOT clobbered
    assert row["workspace_id"] is None
    store.close()


def test_conversation_legacy_null_content_immutable_to_bound_tenant(tmp_path: Path) -> None:
    """The unified chat store: a bound tenant must not clobber a legacy NULL thread/title."""
    from himmy.services.storage.conversations import ConversationStore

    store = ConversationStore(str(tmp_path / "conversations.db"))
    # A CLI / offline conversation (NULL-owned), saved via the flat Studio ingress.
    summary = store.save_flat(
        conversation_id="conv-legacy",
        title="Legacy title",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "original question"), ("agent", "original answer")],
        workspace_id=None,
    )
    assert summary.title == "Legacy title"

    # Tenant t-a re-uses the id and tries to clobber title + messages.
    store.save_flat(
        conversation_id="conv-legacy",
        title="HIJACK TITLE",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "tampered")],
        workspace_id="t-a",
    )
    # Content untouched: title + the original flat transcript survive, owner stays NULL.
    got = store.get_summary("conv-legacy", workspace_id=None)
    assert got is not None and got.title == "Legacy title"
    msgs = [m.text for m in store.flat_messages("conv-legacy")]
    assert msgs == ["original question", "original answer"]
    row = store._conn.execute(
        "SELECT workspace_id FROM conversations WHERE conversation_id = ?",
        ("conv-legacy",),
    ).fetchone()
    assert row["workspace_id"] is None
    store.close()


def test_conversation_offline_resave_still_mutates(tmp_path: Path) -> None:
    """Offline invariant: an unscoped (None) re-save still rewrites content, byte-unchanged."""
    from himmy.services.storage.conversations import ConversationStore

    store = ConversationStore(str(tmp_path / "conversations.db"))
    store.save_flat(
        conversation_id="c1",
        title="v1",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "a")],
        workspace_id=None,
    )
    store.save_flat(
        conversation_id="c1",
        title="v2",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "b")],
        workspace_id=None,
    )
    got = store.get_summary("c1")
    assert got is not None and got.title == "v2"
    assert [m.text for m in store.flat_messages("c1")] == ["b"]
    store.close()


# ============================================================ agent file sandbox BOLA


def _files_app() -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            # Tenant-bound admin (wildcard ``*:*`` satisfies studio.files:read) — the PoC's
            # cross-tenant ``files:read`` principal on a multi-tenant shared deployment.
            "key-a": Principal.build(
                "user-a",
                tenant_ids=["tenant-a"],
                roles=["admin"],
                auth_method="apikey",
            ),
        }
    )
    return TestClient(app)


def test_sandbox_list_denied_to_tenant_bound_principal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-bound principal cannot enumerate the shared, un-namespaced agent sandbox."""
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "report-A.csv").write_text("sensitive")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_FS_ROOT", str(root))

    client = _files_app()
    client.headers.update({"x-himmy-internal-key": "key-a"})
    # The cross-tenant BOLA is closed: tenant-bound → uniform 404 (not 200 + file list).
    assert client.get("/api/studio/files").status_code == 404
    assert (
        client.get("/api/studio/files/download", params={"path": "report-A.csv"}).status_code
        == 404
    )


def test_sandbox_offline_principal_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline invariant: the anonymous (all_tenants) default still lists + downloads."""
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "f.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_FS_ROOT", str(root))

    client = TestClient(create_app())  # no authenticator → ANONYMOUS / all_tenants
    listing = client.get("/api/studio/files")
    assert listing.status_code == 200
    assert [i["path"] for i in listing.json()["items"]] == ["f.txt"]
    dl = client.get("/api/studio/files/download", params={"path": "f.txt"})
    assert dl.status_code == 200 and dl.content == b"hello"


# ============================================================ HITL approval subject axis


@pytest.fixture
def fresh_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from himmy.api import studio_approvals as sa
    from himmy.api.studio_runs import reset_run_store

    monkeypatch.chdir(tmp_path)
    sa.reset_checkpoint_store()
    reset_run_store()
    yield sa
    sa.reset_checkpoint_store()


def _save_subject_checkpoint(sa, *, workspace: str, subject_scope: str):
    from himmy.runtime.checkpoint import (
        AWAITING_APPROVAL,
        AgentCheckpoint,
        PendingToolCall,
    )

    cp = AgentCheckpoint(
        status=AWAITING_APPROVAL,
        pending_tool_calls=[
            PendingToolCall(tool_call_id="c1", tool_name="send_email", args={})
        ],
        thread={"messages": [{"role": "user", "content": "email bob"}]},
        ctx={"context_metadata": {"workspace_id": workspace, "subject_scope": subject_scope}},
    )
    sa.get_checkpoint_store().save(cp)
    return cp


def test_approval_detail_hidden_from_foreign_subject(fresh_checkpoints) -> None:
    """User B (same tenant) cannot read User A's pending approval detail (subject BOLA)."""
    sa = fresh_checkpoints
    cp = _save_subject_checkpoint(sa, workspace="tenant-T", subject_scope="userA")

    # Same tenant, but a DIFFERENT within-tenant user → folded to None (→404).
    assert (
        sa.get_detail(
            cp.checkpoint_id,
            workspace_filter=frozenset({"tenant-T"}),
            subject_filter="userB",
        )
        is None
    )
    # The OWNING user still sees it.
    assert (
        sa.get_detail(
            cp.checkpoint_id,
            workspace_filter=frozenset({"tenant-T"}),
            subject_filter="userA",
        )
        is not None
    )


def test_approval_list_hides_foreign_subject(fresh_checkpoints) -> None:
    """The pending inbox does not surface another within-tenant user's checkpoint."""
    sa = fresh_checkpoints
    a = _save_subject_checkpoint(sa, workspace="tenant-T", subject_scope="userA")
    b = _save_subject_checkpoint(sa, workspace="tenant-T", subject_scope="userB")

    seen_b = {
        p.checkpoint_id
        for p in sa.list_pending(
            workspace_filter=frozenset({"tenant-T"}), subject_filter="userB"
        )
    }
    assert b.checkpoint_id in seen_b and a.checkpoint_id not in seen_b


def test_approval_subject_axis_noop_offline(fresh_checkpoints) -> None:
    """Offline invariant: no subject_filter (None) surfaces every checkpoint, unchanged."""
    sa = fresh_checkpoints
    a = _save_subject_checkpoint(sa, workspace="tenant-T", subject_scope="userA")
    b = _save_subject_checkpoint(sa, workspace="tenant-T", subject_scope="userB")
    seen = {p.checkpoint_id for p in sa.list_pending()}
    assert {a.checkpoint_id, b.checkpoint_id} <= seen
    # A legacy checkpoint carrying NO subject_scope stays visible to a subject-scoped reader.
    from himmy.runtime.checkpoint import (
        AWAITING_APPROVAL,
        AgentCheckpoint,
        PendingToolCall,
    )

    legacy = AgentCheckpoint(
        status=AWAITING_APPROVAL,
        pending_tool_calls=[PendingToolCall(tool_call_id="c", tool_name="t", args={})],
        ctx={"context_metadata": {"workspace_id": "tenant-T"}},
    )
    sa.get_checkpoint_store().save(legacy)
    seen_a = {
        p.checkpoint_id
        for p in sa.list_pending(
            workspace_filter=frozenset({"tenant-T"}), subject_filter="userA"
        )
    }
    assert legacy.checkpoint_id in seen_a  # subject-less legacy run: visible
