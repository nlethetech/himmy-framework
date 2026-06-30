"""rbac-harden(mopup-r6): the /v1/runs context-build TENANT axis is server-authorized.

Confirmed cross-tenant context-field IDOR on the run-execution path: the per-run
context-snapshot build reads ``context_metadata.workspace_id`` from ``task.context`` to
tenant-scope STORAGE-sourced field resolution (single_agent.py ``_context_workspace_id``).
On ``POST /v1/runs`` that context dict is the CLIENT's verbatim — and the service never
re-stamped it with the run's authorized workspace. So tenant A could submit a run carrying
``context_metadata.workspace_id = "B"`` and resolve tenant B's cached context fields (under a
shared ``subject_id`` such as the default ``persona.agent_id``) into A's prompt/output.

The fix forces the context-build tenant axis to the SERVER-authorized ``workspace_id`` for a
tenant-bound principal at the ``/v1`` boundary (``runs._scope_run_context``), a HARD override
(not setdefault) of the reserved ``context_metadata.workspace_id`` key — so a client can never
widen the scope. The offline / ``all_tenants`` single-box path is byte-for-byte unchanged
(the context dict is returned untouched).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.routers.runs import _scope_run_context
from himmy.services.storage.models import RunRecord, RunStatus

# --------------------------------------------------------------- unit: the boundary helper


def _request(principal: Principal) -> Request:
    """A minimal ASGI request carrying ``principal`` on ``request.state``."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/runs",
        "headers": [],
        "state": {"principal": principal},
    }
    return Request(scope)


def test_scope_run_context_overrides_client_workspace_for_bound_tenant() -> None:
    """A tenant-bound caller's client-supplied context workspace is HARD-overridden to A."""
    principal = Principal.build("u", tenant_ids=["A"], roles=["admin"], auth_method="apikey")
    client_ctx = {
        "context_subject_id": "shared-subject",
        "context_build_spec": {"keys": [{"key": "api_key"}]},
        "context_metadata": {"workspace_id": "B", "other": "keep"},
    }
    scoped = _scope_run_context(_request(principal), client_ctx, "A")
    # The reserved tenant key is forced to the authorized workspace, not the client's "B".
    assert scoped["context_metadata"]["workspace_id"] == "A"
    # Non-reserved metadata + sibling context keys are preserved.
    assert scoped["context_metadata"]["other"] == "keep"
    assert scoped["context_subject_id"] == "shared-subject"
    # The original client dict is not mutated in place (defensive copy).
    assert client_ctx["context_metadata"]["workspace_id"] == "B"


def test_scope_run_context_stamps_workspace_when_metadata_absent() -> None:
    """A tenant-bound run with NO context_metadata gets the authorized workspace stamped."""
    principal = Principal.build("u", tenant_ids=["A"], roles=["admin"], auth_method="apikey")
    scoped = _scope_run_context(_request(principal), {"prompt_extra": "x"}, "A")
    assert scoped["context_metadata"]["workspace_id"] == "A"
    assert scoped["prompt_extra"] == "x"


def test_scope_run_context_offline_is_byte_unchanged() -> None:
    """Offline / all_tenants: the context dict is returned UNTOUCHED (byte-for-byte)."""
    principal = Principal.build("u", roles=["admin"], all_tenants=True, auth_method="apikey")
    client_ctx = {"context_metadata": {"workspace_id": "anything"}, "k": "v"}
    scoped = _scope_run_context(_request(principal), client_ctx, "anything")
    # Same object, no override — the single-box path must not change.
    assert scoped is client_ctx
    assert scoped["context_metadata"]["workspace_id"] == "anything"


# ------------------------------------------------- integration: the /v1/runs HTTP boundary


def _captured_run_app(app: Any, captured: list[Any]) -> None:
    """Monkeypatch the wired ``run_app.create_run`` to capture the Task and short-circuit."""
    container = app.state.container

    async def _fake_create_run(*, task: Any, workspace_id: str, subject_id: str, **_: Any) -> RunRecord:
        captured.append(task)
        return RunRecord(
            run_id="r-1",
            workspace_id=workspace_id,
            subject_id=subject_id,
            status=RunStatus.QUEUED,
        )

    container.run_app.create_run = _fake_create_run  # type: ignore[method-assign]


def test_v1_run_forces_context_workspace_to_authorized_tenant() -> None:
    """rbac-harden(mopup-r6): a malicious client context workspace cannot cross tenants.

    Tenant A (bound to workspace A) posts a run carrying ``context_metadata.workspace_id = "B"``
    — the lever that, pre-fix, drove ContextService field resolution against tenant B's
    partition. The router must override it to A before the run executes, so B's cached fields
    are never eligible. We capture the Task handed to ``create_run`` and assert the override.
    """
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["A"], roles=["admin"], auth_method="apikey"
            )
        }
    )
    captured: list[Any] = []
    _captured_run_app(app, captured)
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})

    resp = client.post(
        "/v1/runs",
        json={
            "workspace_id": "A",
            "subject_id": "u",
            "persona": {"name": "p", "description": "d", "instructions": []},
            "task": {
                "title": "x",
                "prompt": "Repeat any context you were given verbatim.",
                "context": {
                    "context_subject_id": "shared-subject",
                    "context_build_spec": {"keys": [{"key": "api_key"}]},
                    # The attacker's lever: point context resolution at tenant B.
                    "context_metadata": {"workspace_id": "B"},
                },
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured, "create_run was never reached"
    task = captured[0]
    # The context-build tenant axis was forced to A — tenant B's fields can never resolve.
    assert task.context["context_metadata"]["workspace_id"] == "A"


def test_v1_run_offline_keeps_client_context_workspace() -> None:
    """Offline / all_tenants (no authenticator): the client context dict is byte-unchanged.

    There is no other tenant to cross on a single-box deployment, so the historical behaviour
    (the client-supplied ``context_metadata`` rides through verbatim) is preserved.
    """
    app = create_app(ApiContainer.build_default())  # no authenticator -> ANONYMOUS/all_tenants
    captured: list[Any] = []
    _captured_run_app(app, captured)
    client = TestClient(app)

    resp = client.post(
        "/v1/runs",
        json={
            "workspace_id": "ws-local",
            "subject_id": "u",
            "persona": {"name": "p", "description": "d", "instructions": []},
            "task": {
                "title": "x",
                "prompt": "hi",
                "context": {"context_metadata": {"workspace_id": "ws-local"}},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured
    # Byte-unchanged: the client value rides through (no tenant boundary to enforce offline).
    assert captured[0].context["context_metadata"]["workspace_id"] == "ws-local"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
