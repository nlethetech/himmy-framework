"""T2f — /v1 HITL approval/resume over HTTP (UNIT 5 acceptance).

The headline /v1 gap: a gated /v1 run could not be resumed. These tests drive the full
REST surface:

* ``POST /v1/runs`` with ``hitl=true`` + a stored agent whose tool is approval-gated
  reaches ``AWAITING_APPROVAL`` with ``metadata.checkpoint_id``;
* ``GET /v1/runs?status=AWAITING_APPROVAL`` lists the paused run (the inbox);
* ``GET /v1/runs/{id}/pending-approvals`` returns the redacted gated tool;
* ``POST /v1/runs/{id}/approve`` fires the gated tool EXACTLY ONCE and completes;
* a double-approve is a no-op (409, the gated tool never fires twice);
* ``POST /v1/runs/{id}/reject`` records the rejection WITHOUT running the gated tool;
* the paused run is never swept to FAILED;
* hitl=true with an inline persona is a 422;
* RBAC: a viewer (run:read only) is denied approve with 403;
* cross-tenant approve is a 404 (authenticated regime).

Offline HTTP tests run with NO authenticator (operator path); the RBAC + cross-tenant
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


def _store_gated_agent(client: TestClient, *, workspace_id: str = "acme") -> str:
    """Store an operator-provisioned agent whose ONLY tool is approval-gated."""
    res = client.post(
        "/v1/agents",
        json={
            "workspace_id": workspace_id,
            "spec": {
                "name": "treasurer",
                "tools_module": _TOOLS_MODULE,
                "tools": ["wire_money"],
            },
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["agent_id"]


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


def _start_hitl_run(client: TestClient, agent_id: str, *, workspace_id="acme") -> dict:
    """POST a hitl run by the stored agent and poll to the paused/terminal state."""
    run = client.post(
        "/v1/runs",
        json={
            "workspace_id": workspace_id,
            "subject_id": "s1",
            "agent_id": agent_id,
            "task": {"title": "pay", "prompt": "wire 1000 to the vendor"},
            "hitl": True,
        },
    )
    assert run.status_code == 200, run.text
    return _poll_status(
        client,
        run.json()["run_id"],
        {"AWAITING_APPROVAL", "SUCCEEDED", "FAILED"},
        workspace_id=workspace_id,
    )


# ------------------------------------------------------------------ pause


def test_hitl_run_reaches_awaiting_approval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hitl=true run on a gated tool reaches AWAITING_APPROVAL + a checkpoint id (T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)
    assert paused["status"] == RunStatus.AWAITING_APPROVAL.value
    assert paused["metadata"].get("checkpoint_id")
    assert not _per_run_tools.WIRE_CALLS  # nothing fired yet


def test_awaiting_approval_inbox_lists_the_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/runs?status=AWAITING_APPROVAL lists the paused run (the inbox, T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)

    inbox = client.get(
        "/v1/runs",
        params={"workspace_id": "acme", "status": RunStatus.AWAITING_APPROVAL.value},
    )
    assert inbox.status_code == 200, inbox.text
    ids = [r["run_id"] for r in inbox.json()["items"]]
    assert paused["run_id"] in ids


def test_pending_approvals_returns_redacted_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/runs/{id}/pending-approvals returns the gated tool (T2f read side)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)

    res = client.get(
        f"/v1/runs/{paused['run_id']}/pending-approvals",
        params={"workspace_id": "acme"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == RunStatus.AWAITING_APPROVAL.value
    assert [p["tool_name"] for p in body["pending_tool_calls"]] == ["wire_money"]


def test_paused_run_is_not_swept(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sweep_stuck_runs does NOT mark an AWAITING_APPROVAL run FAILED (T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)

    # Drive the sweeper directly on the wired service (ttl=0 → would sweep all
    # non-terminal runs, but the paused one is deliberately skipped).
    run_app = client.app.state.container.run_app
    from tests.conftest import run_async

    swept = run_async(run_app.sweep_stuck_runs(ttl_seconds=0.0))
    assert paused["run_id"] not in swept
    still = client.get(
        f"/v1/runs/{paused['run_id']}", params={"workspace_id": "acme"}
    ).json()
    assert still["status"] == RunStatus.AWAITING_APPROVAL.value


# ------------------------------------------------------------------ approve


def test_approve_fires_tool_once_and_completes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST approve fires the gated tool EXACTLY ONCE and completes (T2f headline)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)

    res = client.post(
        f"/v1/runs/{paused['run_id']}/approve", json={"workspace_id": "acme"}
    )
    assert res.status_code == 200, res.text
    done = _poll_status(client, paused["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED", done
    assert len(_per_run_tools.WIRE_CALLS) == 1


def test_double_approve_is_a_noop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second approve is a 409 no-op; the gated tool never fires twice (T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)

    first = client.post(
        f"/v1/runs/{paused['run_id']}/approve", json={"workspace_id": "acme"}
    )
    assert first.status_code == 200
    second = client.post(
        f"/v1/runs/{paused['run_id']}/approve", json={"workspace_id": "acme"}
    )
    assert second.status_code == 409, second.text
    done = _poll_status(client, paused["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED"
    assert len(_per_run_tools.WIRE_CALLS) == 1


def test_approve_with_default_body_infers_workspace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """approve works with no body (workspace inferred from the principal, offline)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)

    res = client.post(f"/v1/runs/{paused['run_id']}/approve")
    assert res.status_code == 200, res.text
    done = _poll_status(client, paused["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED"


# ------------------------------------------------------------------ reject


def test_reject_records_rejection_without_running_tool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST reject records the rejection and NEVER runs the gated tool (T2f)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    agent_id = _store_gated_agent(client)
    paused = _start_hitl_run(client, agent_id)

    res = client.post(
        f"/v1/runs/{paused['run_id']}/reject", json={"workspace_id": "acme"}
    )
    assert res.status_code == 200, res.text
    done = _poll_status(client, paused["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED", done
    assert not _per_run_tools.WIRE_CALLS  # the gated tool never executed


# ------------------------------------------------------------------ shapes


def test_hitl_with_inline_persona_is_422(client: TestClient) -> None:
    """hitl=true with an inline persona (no tools to gate) is a 422."""
    res = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "persona": {"name": "plain"},
            "task": {"title": "t", "prompt": "hi"},
            "hitl": True,
        },
    )
    assert res.status_code == 422, res.text


def test_approve_unknown_run_is_404(client: TestClient) -> None:
    """approve for an unknown run id is a 404."""
    res = client.post("/v1/runs/nope/approve", json={"workspace_id": "acme"})
    assert res.status_code == 404, res.text


def test_approve_a_non_paused_run_is_409(client: TestClient) -> None:
    """approve for a known but non-AWAITING_APPROVAL run is a 409."""
    # A plain (non-hitl) inline run completes; approving it is a 409.
    run = client.post(
        "/v1/runs",
        json={
            "workspace_id": "acme",
            "subject_id": "s1",
            "persona": {"name": "p"},
            "task": {"title": "t", "prompt": "hi"},
        },
    ).json()
    done = _poll_status(client, run["run_id"], {"SUCCEEDED", "FAILED"})
    assert done["status"] == "SUCCEEDED"
    res = client.post(
        f"/v1/runs/{run['run_id']}/approve", json={"workspace_id": "acme"}
    )
    assert res.status_code == 409, res.text


# --------------------------------------------------------------- security


def test_viewer_is_denied_approve_with_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """A viewer (run:read only) cannot approve — 403 (RBAC, T2f).

    An ``all_tenants`` admin provisions + pauses a gated run (operator path → the
    privileged ``tools_module`` survives); a viewer (``all_tenants`` so workspace scoping
    passes, but only ``run:read``) is then denied approve with a 403 before any resume.
    """
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "admin": Principal.build(
                "admin",
                roles=["admin"],
                all_tenants=True,
                auth_method="apikey",
            ),
            "viewer": Principal.build(
                "viewer",
                roles=["viewer"],
                all_tenants=True,
                auth_method="apikey",
            ),
        }
    )
    with TestClient(app) as shared:
        shared.headers.update({"x-himmy-internal-key": "admin"})
        agent_id = _store_gated_agent(shared)
        paused = _start_hitl_run(shared, agent_id)
        assert paused["status"] == RunStatus.AWAITING_APPROVAL.value

        denied = shared.post(
            f"/v1/runs/{paused['run_id']}/approve",
            json={"workspace_id": "acme"},
            headers={"x-himmy-internal-key": "viewer"},
        )
        assert denied.status_code == 403, denied.text
        # The gated tool never fired.
        assert not _per_run_tools.WIRE_CALLS


def test_cross_tenant_approve_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different tenant cannot approve another tenant's paused run — 404 (T2f).

    An ``all_tenants`` admin provisions + pauses a run in ``tenant-a``; a ``tenant-b``-
    scoped operator (NOT all_tenants) asking for its OWN workspace gets 404 (the run is
    invisible), and asking for ``tenant-a`` gets 403 (workspace denied) — never a resume.
    """
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
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
        agent_id = _store_gated_agent(shared, workspace_id="tenant-a")
        paused = _start_hitl_run(shared, agent_id, workspace_id="tenant-a")
        assert paused["status"] == RunStatus.AWAITING_APPROVAL.value

        # Tenant B asks to approve tenant-A's run scoped to its OWN workspace → 404
        # (the run is invisible to B). Asking for tenant-a is a 403 (workspace denied).
        as_b_own = shared.post(
            f"/v1/runs/{paused['run_id']}/approve",
            json={"workspace_id": "tenant-b"},
            headers={"x-himmy-internal-key": "key-b"},
        )
        assert as_b_own.status_code == 404, as_b_own.text
        as_b_a = shared.post(
            f"/v1/runs/{paused['run_id']}/approve",
            json={"workspace_id": "tenant-a"},
            headers={"x-himmy-internal-key": "key-b"},
        )
        assert as_b_a.status_code == 403, as_b_a.text
        # Nothing fired.
        assert not _per_run_tools.WIRE_CALLS
