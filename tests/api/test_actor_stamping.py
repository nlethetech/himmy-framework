"""WS1.3 — actor stamping: every run records who launched it (who-did-what)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal

_BODY = {
    "workspace_id": "t",
    "subject_id": "s",
    "persona": {"name": "A"},
    "task": {"title": "t", "prompt": "hi"},
}


def test_run_stamps_the_authenticated_actor() -> None:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "user-a",
                tenant_ids=["t"],
                roles=["operator"],
                auth_method="apikey",
            )
        }
    )
    client = TestClient(app)
    client.headers.update({"x-himmy-internal-key": "k"})
    run = client.post("/v1/runs", json=_BODY).json()
    actor = run["metadata"]["actor"]
    assert actor["subject"] == "user-a"
    assert actor["auth_method"] == "apikey"
    assert "operator" in actor["roles"]


def test_offline_run_records_anonymous_actor() -> None:
    """Even with no auth, the run records the (anonymous) initiator, not nothing."""
    client = TestClient(create_app(ApiContainer.build_default()))
    run = client.post("/v1/runs", json=_BODY).json()
    actor = run["metadata"]["actor"]
    assert actor["subject"] == "anonymous"
    assert actor["auth_method"] == "anonymous"
