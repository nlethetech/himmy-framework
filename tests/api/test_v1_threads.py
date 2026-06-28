"""T2g — /v1 thread continuation + plan mode over HTTP (UNIT 6 acceptance).

Today ``/v1`` only REPLAYS a thread. These tests drive the new agentic continuation
surface end-to-end:

* ``POST /v1/threads`` creates a workspace-stamped thread;
* ``POST /v1/threads/{id}/messages`` continues it on the PER-RUN AGENTIC runtime so the
  model sees prior turns AND the agent's tools are available (NOT a tool-less single-shot
  chat) — proven by a continuation whose gated tool PAUSES the run at AWAITING_APPROVAL;
* the continued thread is visible via the shared ConversationStore (the same store the
  CLI + Studio read) with the prior turns present;
* ``plan=true`` pauses the run at PLAN-READY with the plan in ``run.metadata['plan']`` and
  is resumable via the SAME ``POST /v1/runs/{id}/approve`` machinery;
* a cross-workspace ``thread_id`` is a 404 (authenticated regime);
* RBAC: a viewer (run:read only) cannot continue a thread (403).

Offline HTTP tests run with NO authenticator (operator path); the cross-tenant + RBAC
cases configure an authenticator so the boundary actually bites.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.services.storage.models import RunStatus
from tests.application import _per_run_tools

_TOOLS_MODULE = "tests.application._per_run_tools"


@pytest.fixture
def client() -> Iterator[TestClient]:
    """An offline /v1 client over a fresh in-RAM container (no auth → operator)."""
    with TestClient(create_app(ApiContainer.build_default())) as test_client:
        yield test_client


def setup_function() -> None:
    """Reset the recording lists before each test (shared module-level state)."""
    _per_run_tools.CALLS.clear()
    _per_run_tools.WIRE_CALLS.clear()


# ----------------------------------------------------------------- helpers


def _store_agent(
    client: TestClient,
    *,
    name: str = "chatter",
    workspace_id: str = "acme",
    tools: list[str] | None = None,
    tools_module: str | None = None,
) -> str:
    """Store an agent (optionally tool-bearing) and return its agent_id."""
    spec: dict = {"name": name}
    if tools_module:
        spec["tools_module"] = tools_module
    if tools is not None:
        spec["tools"] = tools
    res = client.post(
        "/v1/agents", json={"workspace_id": workspace_id, "spec": spec}
    )
    assert res.status_code == 201, res.text
    return res.json()["agent_id"]


def _create_thread(
    client: TestClient, *, agent_id: str | None = None, workspace_id: str = "acme"
) -> str:
    """Create a thread (optionally bound to a default agent) and return its id."""
    body: dict = {"workspace_id": workspace_id}
    if agent_id:
        body["agent_id"] = agent_id
    res = client.post("/v1/threads", json=body)
    assert res.status_code == 201, res.text
    return res.json()["thread_id"]


def _poll_status(
    client: TestClient,
    run_id: str,
    wanted: set[str],
    *,
    workspace_id: str = "acme",
    timeout: float = 6.0,
) -> dict:
    """Poll a /v1 run until its status is in ``wanted`` (or timeout)."""
    deadline = time.monotonic() + timeout
    got: dict = {}
    while time.monotonic() < deadline:
        got = client.get(
            f"/v1/runs/{run_id}", params={"workspace_id": workspace_id}
        ).json()
        if got.get("status") in wanted:
            return got
        time.sleep(0.02)
    return got


def _send(
    client: TestClient,
    thread_id: str,
    prompt: str,
    *,
    workspace_id: str = "acme",
    **extra: object,
) -> dict:
    """POST a message to a thread; return the run JSON."""
    body: dict = {"workspace_id": workspace_id, "prompt": prompt, **extra}
    res = client.post(f"/v1/threads/{thread_id}/messages", json=body)
    assert res.status_code == 200, res.text
    return res.json()


# ----------------------------------------------------------------- create


def test_create_thread_returns_workspace_stamped_thread(client: TestClient) -> None:
    """POST /v1/threads creates a thread owned by the requested workspace (T2g)."""
    agent_id = _store_agent(client)
    res = client.post(
        "/v1/threads", json={"workspace_id": "acme", "agent_id": agent_id}
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["thread_id"]
    assert body["workspace_id"] == "acme"
    assert body["agent_id"] == agent_id


# ----------------------------------------------------------------- continue


def test_continue_thread_runs_and_persists_a_linked_run(client: TestClient) -> None:
    """A message continues the thread and produces a listable linked RunRecord (T2g)."""
    agent_id = _store_agent(client)
    thread_id = _create_thread(client, agent_id=agent_id)

    run = _send(client, thread_id, "hello there")
    assert run["thread_id"] == thread_id
    assert run["metadata"]["conversation_id"] == thread_id
    done = _poll_status(client, run["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED", done

    # The run is listable in the canonical run store (one store all surfaces read).
    listed = client.get("/v1/runs", params={"workspace_id": "acme"}).json()
    assert run["run_id"] in [r["run_id"] for r in listed["items"]]


def test_continued_thread_is_visible_in_conversation_store(client: TestClient) -> None:
    """The continued thread is visible via ConversationStore with prior turns (T2g).

    The acceptance: a thread continued via /v1 is browsable by other surfaces (the shared
    ConversationStore). Two turns land both user prompts in the authoritative thread, and
    the flat projection the CLI/Studio read shows them.
    """
    agent_id = _store_agent(client)
    thread_id = _create_thread(client, agent_id=agent_id)

    run1 = _send(client, thread_id, "first question")
    _poll_status(client, run1["run_id"], {"SUCCEEDED", "FAILED"})
    run2 = _send(client, thread_id, "second question")
    _poll_status(client, run2["run_id"], {"SUCCEEDED", "FAILED"})

    # Read the SAME ConversationStore the CLI + Studio read.
    store = client.app.state.container.conversation_store
    thread = store.load_thread(thread_id)
    assert thread is not None
    user_turns = [m.content for m in thread.messages if m.role.value == "user"]
    # Both user turns are present (the prompt renders into a task template, so match by
    # substring) — proving the second run CONTINUED the thread (the model saw turn one).
    assert any("first question" in t for t in user_turns)
    assert any("second question" in t for t in user_turns)

    # The flat projection the CLI/Studio read carries both turns.
    flat = store.flat_messages(thread_id)
    texts = " || ".join(m.text for m in flat)
    assert "first question" in texts
    assert "second question" in texts

    # And the GET surface replays the same flat transcript.
    got = client.get(
        f"/v1/threads/{thread_id}", params={"workspace_id": "acme"}
    ).json()
    got_texts = " || ".join(m["text"] for m in got["messages"])
    assert "first question" in got_texts
    assert "second question" in got_texts


def test_continuation_has_tools_available_and_pauses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continuation runs AGENTICALLY: a gated tool pauses the run (tools available, T2g).

    The reviewer must_fix: a /v1 conversation must be as capable as a Studio one — tools
    bound + HITL pause — NOT a tool-less single-shot chat. A continuation whose ONLY tool
    is approval-gated reaches AWAITING_APPROVAL, proving the tool was actually bound and
    the agentic loop ran.
    """
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_agent(
        client, name="treasurer", tools_module=_TOOLS_MODULE, tools=["wire_money"]
    )
    thread_id = _create_thread(client, agent_id=agent_id)

    run = _send(
        client, thread_id, "wire 1000 to the vendor", hitl=True
    )
    paused = _poll_status(
        client, run["run_id"], {"AWAITING_APPROVAL", "SUCCEEDED", "FAILED"}
    )
    assert paused["status"] == RunStatus.AWAITING_APPROVAL.value
    assert paused["metadata"].get("checkpoint_id")
    assert not _per_run_tools.WIRE_CALLS  # gated, nothing fired yet

    # Approve via the SAME /v1/runs machinery (T2f); the gated tool fires once.
    res = client.post(
        f"/v1/runs/{run['run_id']}/approve", json={"workspace_id": "acme"}
    )
    assert res.status_code == 200, res.text
    done = _poll_status(client, run["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED", done
    assert len(_per_run_tools.WIRE_CALLS) == 1


def test_continuation_agent_id_override(client: TestClient) -> None:
    """A message's ``agent_id`` overrides the thread's default for that turn (T2g)."""
    bound = _store_agent(client, name="bound")
    other = _store_agent(client, name="other")
    thread_id = _create_thread(client, agent_id=bound)

    run = _send(client, thread_id, "use the other agent", agent_id=other)
    done = _poll_status(client, run["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED"
    assert done["metadata"]["agent_id"] == other


def test_continue_thread_with_no_agent_is_422(client: TestClient) -> None:
    """A thread with no bound/supplied agent cannot be continued (422)."""
    thread_id = _create_thread(client)  # no agent bound
    res = client.post(
        f"/v1/threads/{thread_id}/messages",
        json={"workspace_id": "acme", "prompt": "hi"},
    )
    assert res.status_code == 422, res.text


def test_idempotent_message_returns_prior_run(client: TestClient) -> None:
    """A re-POST with the same idempotency_key returns the prior run (no double turn)."""
    agent_id = _store_agent(client)
    thread_id = _create_thread(client, agent_id=agent_id)

    first = _send(client, thread_id, "once", idempotency_key="k1")
    second = _send(client, thread_id, "once", idempotency_key="k1")
    assert first["run_id"] == second["run_id"]


# ----------------------------------------------------------------- plan mode


def test_plan_mode_pauses_at_plan_ready_and_resumes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plan=true pauses at PLAN-READY with the plan in metadata, resumable via approve (T2g)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    # A tool-bearing agent so a per-run registry exists to host the plan tool.
    agent_id = _store_agent(
        client, name="planner", tools_module=_TOOLS_MODULE, tools=["ping"]
    )
    thread_id = _create_thread(client, agent_id=agent_id)

    run = _send(client, thread_id, "do the thing", plan=True)
    paused = _poll_status(
        client, run["run_id"], {"AWAITING_APPROVAL", "SUCCEEDED", "FAILED"}
    )
    assert paused["status"] == RunStatus.AWAITING_APPROVAL.value, paused
    assert paused["metadata"].get("checkpoint_id")
    # The published plan is surfaced in metadata for review.
    plan = paused["metadata"].get("plan")
    assert isinstance(plan, list) and plan, paused["metadata"]
    assert all("title" in step and "status" in step for step in plan)

    # The pending-approvals read shows the gated planning tool.
    pend = client.get(
        f"/v1/runs/{run['run_id']}/pending-approvals",
        params={"workspace_id": "acme"},
    ).json()
    assert "update_plan" in [p["tool_name"] for p in pend["pending_tool_calls"]]

    # Approve via the SAME /v1/runs machinery → the run resumes and completes.
    res = client.post(
        f"/v1/runs/{run['run_id']}/approve", json={"workspace_id": "acme"}
    )
    assert res.status_code == 200, res.text
    done = _poll_status(client, run["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED", done


def test_plan_mode_on_plain_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /v1/runs plan=true (no thread) also pauses at PLAN-READY (T2g footnote)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_agent(
        client, name="planner2", tools_module=_TOOLS_MODULE, tools=["ping"]
    )
    run = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "agent_id": agent_id,
            "task": {"title": "t", "prompt": "do the thing"},
            "plan": True,
        },
    )
    assert run.status_code == 200, run.text
    paused = _poll_status(
        client, run.json()["run_id"], {"AWAITING_APPROVAL", "SUCCEEDED", "FAILED"}
    )
    assert paused["status"] == RunStatus.AWAITING_APPROVAL.value, paused
    assert paused["metadata"].get("plan")


def test_plan_with_inline_persona_is_422(client: TestClient) -> None:
    """plan=true with an inline persona (no per-run registry) is a 422."""
    res = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "persona": {"name": "plain"},
            "task": {"title": "t", "prompt": "hi"},
            "plan": True,
        },
    )
    assert res.status_code == 422, res.text


def test_plan_with_tool_less_stored_agent_run_is_422(client: TestClient) -> None:
    """plan=true on a stored-but-tool-LESS agent is a 422, not a silent never-pause (T2g).

    A bare persona spec (no tool_packs/tools_module/...) builds NO per-run registry, so
    the gated update_plan tool has nowhere to live — the run would silently complete
    WITHOUT pausing at PLAN-READY. The reviewer must_fix: reject up front (422), exactly
    like the inline-persona case, rather than slipping through to a wrong outcome.
    """
    # Stored agent with no tool surface at all — builds_tool_registry() is False.
    agent_id = _store_agent(client, name="plainstored")
    res = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "agent_id": agent_id,
            "task": {"title": "t", "prompt": "do the thing"},
            "plan": True,
        },
    )
    assert res.status_code == 422, res.text


def test_plan_with_tool_less_stored_agent_thread_is_422(client: TestClient) -> None:
    """plan=true continuation on a tool-less stored agent is a 422, not a silent run (T2g).

    Same gate as the plain-run path, but reached through POST
    /v1/threads/{id}/messages — a tool-less stored agent cannot host the gated plan tool.
    """
    agent_id = _store_agent(client, name="plainstored2")
    thread_id = _create_thread(client, agent_id=agent_id)
    res = client.post(
        f"/v1/threads/{thread_id}/messages",
        json={"workspace_id": "acme", "prompt": "do the thing", "plan": True},
    )
    assert res.status_code == 422, res.text


# ----------------------------------------------------------------- shapes


def test_get_unknown_thread_is_404(client: TestClient) -> None:
    """GET an unknown thread id is a 404."""
    res = client.get("/v1/threads/nope", params={"workspace_id": "acme"})
    assert res.status_code == 404, res.text


def test_message_to_unknown_thread_is_404(client: TestClient) -> None:
    """POST a message to an unknown thread id is a 404."""
    res = client.post(
        "/v1/threads/nope/messages",
        json={"workspace_id": "acme", "prompt": "hi"},
    )
    assert res.status_code == 404, res.text


# --------------------------------------------------------------- security


def test_cross_workspace_thread_access_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different tenant cannot read or continue another tenant's thread — 404 (T2g)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "admin": Principal.build(
                "admin", roles=["admin"], all_tenants=True, auth_method="apikey"
            ),
            "key-b": Principal.build(
                "user-b",
                tenant_ids=["tenant-b"],
                roles=["operator"],
                auth_method="apikey",
            ),
        }
    )
    with TestClient(app) as shared:
        shared.headers.update({"x-himmy-internal-key": "admin"})
        agent_id = _store_agent(shared, workspace_id="tenant-a")
        thread_id = _create_thread(shared, agent_id=agent_id, workspace_id="tenant-a")

        # Tenant B reading tenant-A's thread scoped to its OWN workspace → 404.
        as_b_get = shared.get(
            f"/v1/threads/{thread_id}",
            params={"workspace_id": "tenant-b"},
            headers={"x-himmy-internal-key": "key-b"},
        )
        assert as_b_get.status_code == 404, as_b_get.text

        # Tenant B continuing tenant-A's thread scoped to its OWN workspace → 404.
        as_b_post = shared.post(
            f"/v1/threads/{thread_id}/messages",
            json={"workspace_id": "tenant-b", "prompt": "leak?"},
            headers={"x-himmy-internal-key": "key-b"},
        )
        assert as_b_post.status_code == 404, as_b_post.text


def test_viewer_cannot_continue_thread_with_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A viewer (run:read only) cannot continue a thread — 403 (RBAC, T2g)."""
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "admin": Principal.build(
                "admin", roles=["admin"], all_tenants=True, auth_method="apikey"
            ),
            "viewer": Principal.build(
                "viewer", roles=["viewer"], all_tenants=True, auth_method="apikey"
            ),
        }
    )
    with TestClient(app) as shared:
        shared.headers.update({"x-himmy-internal-key": "admin"})
        agent_id = _store_agent(shared)
        thread_id = _create_thread(shared, agent_id=agent_id)

        denied = shared.post(
            f"/v1/threads/{thread_id}/messages",
            json={"workspace_id": "acme", "prompt": "hi"},
            headers={"x-himmy-internal-key": "viewer"},
        )
        assert denied.status_code == 403, denied.text


def _seed_thread_for_subject(
    container: object, *, conversation_id: str, workspace_id: str, subject_id: str
) -> None:
    """Persist a thread owned by ``workspace_id`` + linked to ``subject_id`` (BOLA setup).

    Drives the same ``ThreadAppService._persist`` the agentic continuation uses, so the
    thread carries the workspace ownership metadata AND the conversation store records the
    ``subject_id`` linkage — without needing a live model turn.
    """
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.application.services import THREAD_WORKSPACE_KEY

    thread = ChatThread(metadata={THREAD_WORKSPACE_KEY: workspace_id})
    container.thread_app._persist(  # type: ignore[attr-defined]
        conversation_id, thread, workspace_id, subject_id=subject_id
    )


def test_bola_subject_scoped_thread_read_is_isolated() -> None:
    """WS-bola: a subject-scoped principal cannot read another subject's thread (404).

    Both subjects share ONE tenant, so workspace isolation does NOT separate them — only
    the subject axis does. A tenant_admin crosses subjects; the un-scoped owner reads all.
    """
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-a": Principal.build(
                "subj-a",
                tenant_ids=["shared"],
                roles=["operator"],
                auth_method="apikey",
                subject_scoped=True,
            ),
            "key-admin": Principal.build(
                "admin-user",
                tenant_ids=["shared"],
                roles=["operator", "tenant_admin"],
                auth_method="apikey",
                subject_scoped=True,
            ),
        }
    )
    _seed_thread_for_subject(
        app.state.container,
        conversation_id="conv-b",
        workspace_id="shared",
        subject_id="subj-b",
    )
    with TestClient(app) as client:
        # Subject A (scoped to subj-a) reading subj-b's thread → clean 404.
        as_a = client.get(
            "/v1/threads/conv-b",
            params={"workspace_id": "shared"},
            headers={"x-himmy-internal-key": "key-a"},
        )
        assert as_a.status_code == 404, as_a.text
        # The tenant_admin crosses subjects within its tenant → 200.
        as_admin = client.get(
            "/v1/threads/conv-b",
            params={"workspace_id": "shared"},
            headers={"x-himmy-internal-key": "key-admin"},
        )
        assert as_admin.status_code == 200, as_admin.text


def test_bola_offline_thread_read_is_unnarrowed(client: TestClient) -> None:
    """INVARIANT: offline (no auth) reads a subject-attributed thread (no narrowing)."""
    _seed_thread_for_subject(
        client.app.state.container,  # type: ignore[attr-defined]
        conversation_id="conv-x",
        workspace_id="acme",
        subject_id="someone",
    )
    res = client.get("/v1/threads/conv-x", params={"workspace_id": "acme"})
    assert res.status_code == 200, res.text
