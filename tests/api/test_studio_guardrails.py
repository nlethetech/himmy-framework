"""Studio guardrails router — the read-only safety-layer viewer (T3e).

In-process FastAPI app, no network. Covers:

* the built-in catalog: ``GET /api/studio/guardrails`` returns the SAME rows the
  CLI ``himmy guardrails`` prints (:func:`himmy.cli.guardrails_view.builtin_rows`);
* recent firings: a run whose cognition trace recorded ``safety`` steps surfaces
  each firing linked to its ``run_id``, with the same family/action/detail the CLI
  shield line shows — read from the Studio run store (``.himmy/studio.db``), the
  reviewer-corrected data source;
* the clean empty state (no runs ⇒ empty firings, never a 500);
* the scan/limit bounds.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.studio_runs import (
    CognitionStep,
    StudioRun,
    get_run_store,
    reset_run_store,
)
from himmy.cli.guardrails_view import builtin_rows


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[TestClient]:
    """A Studio app whose run store is an isolated, per-test ``.himmy/studio.db``."""
    monkeypatch.chdir(tmp_path)
    reset_run_store()  # the studio.db handle is cwd-relative — drop any cached one
    yield TestClient(create_app(ApiContainer.build_default()))
    reset_run_store()


def _save_run(run_id: str, *, created_at: str, steps: list[CognitionStep]) -> None:
    get_run_store().save(
        StudioRun(
            id=run_id,
            created_at=created_at,
            agent_name="guard-agent",
            provider="claude-cli",
            model="haiku",
            prompt="redact me a@b.com",
            output="ok",
            status="ok",
            steps=steps,
        )
    )


# ------------------------------------------------------------------- catalog


def test_catalog_matches_cli_builtin_rows(client: TestClient) -> None:
    """The viewer lists EXACTLY the built-ins ``himmy guardrails`` prints."""
    r = client.get("/api/studio/guardrails")
    assert r.status_code == 200, r.text
    body = r.json()
    cli_rows = builtin_rows()
    assert body["builtins"] == cli_rows
    # And the canonical families are present (pii/injection/nepal_pii/grounding/dlp).
    names = {row["name"] for row in body["builtins"]}
    assert {"pii", "injection", "grounding"} <= names


def test_empty_firings_is_clean_not_error(client: TestClient) -> None:
    """No runs recorded ⇒ empty firings list, exit 200 (the screen's empty state)."""
    body = client.get("/api/studio/guardrails").json()
    assert body["firings"] == []
    assert body["firing_count"] == 0
    assert body["runs_scanned"] == 0


# ------------------------------------------------------------------- firings


def test_firings_read_from_studio_db_cognition_steps(client: TestClient) -> None:
    """A run's ``safety`` cognition steps surface as firings linked to the run id."""
    _save_run(
        "run-1",
        created_at="2026-06-13T10:00:00Z",
        steps=[
            CognitionStep(seq=1, kind="agent", agent="guard-agent", model="haiku"),
            CognitionStep(
                seq=2,
                kind="safety",
                stage="input",
                redacted=True,
                blocked=False,
                flags=["pii:email"],
            ),
            CognitionStep(
                seq=3,
                kind="safety",
                stage="input",
                blocked=True,
                flags=["injection"],
                reasons=["possible prompt injection"],
            ),
        ],
    )
    body = client.get("/api/studio/guardrails").json()
    assert body["runs_scanned"] == 1
    assert body["firing_count"] == 2
    by_name = {f["name"]: f for f in body["firings"]}
    # The PII redaction: family "pii", action "redacted", detail = the sub-label.
    pii = by_name["pii"]
    assert pii["run_id"] == "run-1"
    assert pii["action"] == "redacted"
    assert pii["detail"] == "email"
    assert pii["stage"] == "input"
    assert pii["redacted"] is True
    # The injection block: family "injection", action "blocked", reason as detail.
    inj = by_name["injection"]
    assert inj["action"] == "blocked"
    assert inj["blocked"] is True
    assert inj["detail"] == "possible prompt injection"


def test_non_safety_steps_are_ignored(client: TestClient) -> None:
    """Only ``safety`` steps are firings — reasoning/tool steps never leak in."""
    _save_run(
        "run-2",
        created_at="2026-06-13T11:00:00Z",
        steps=[
            CognitionStep(seq=1, kind="reason", text="thinking"),
            CognitionStep(seq=2, kind="tool", name="web_search"),
        ],
    )
    body = client.get("/api/studio/guardrails").json()
    assert body["runs_scanned"] == 1
    assert body["firing_count"] == 0


def test_firings_are_newest_run_first(client: TestClient) -> None:
    """Firings sort newest run first (then step order within a run)."""
    _save_run(
        "old",
        created_at="2026-06-13T09:00:00Z",
        steps=[CognitionStep(seq=1, kind="safety", redacted=True, flags=["pii:ssn"])],
    )
    _save_run(
        "new",
        created_at="2026-06-13T12:00:00Z",
        steps=[CognitionStep(seq=1, kind="safety", blocked=True, flags=["injection"])],
    )
    firings = client.get("/api/studio/guardrails").json()["firings"]
    assert [f["run_id"] for f in firings] == ["new", "old"]


def test_limit_bounds_firings(client: TestClient) -> None:
    """``limit`` caps the firings returned (newest first)."""
    for i in range(5):
        _save_run(
            f"r{i}",
            created_at=f"2026-06-13T1{i}:00:00Z",
            steps=[
                CognitionStep(seq=1, kind="safety", redacted=True, flags=["pii:email"])
            ],
        )
    body = client.get("/api/studio/guardrails", params={"limit": 2}).json()
    assert body["firing_count"] == 2
    assert len(body["firings"]) == 2


def test_scan_param_bounded(client: TestClient) -> None:
    """``scan`` is range-validated (a 0 or absurd value is a 422, never a crash)."""
    assert client.get("/api/studio/guardrails", params={"scan": 0}).status_code == 422
    assert (
        client.get("/api/studio/guardrails", params={"scan": 99999}).status_code == 422
    )


# ----------------------------------------------- cross-tenant boundary (red-team r3)


def test_tenant_bound_guardrails_reader_gets_no_cross_tenant_firings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A tenant-bound principal sees the catalog but NO firings (cross-tenant IDOR fix).

    The firings list is read from the process-wide ``.himmy/studio.db`` presentation
    cache, which carries no per-run tenant stamp. Before this fix, a tenant-bound
    principal holding ``studio.guardrails:read`` (granted to viewer/operator/auditor)
    could enumerate EVERY tenant's run_id / agent_name / guardrail flags + reasons via
    ``GET /api/studio/guardrails``. Now — mirroring the runs/analytics reader — such a
    principal gets the catalog only (empty firings), while the offline / all-tenants
    path is byte-unchanged (see :func:`test_firings_read_from_studio_db_cognition_steps`).
    """
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    monkeypatch.chdir(tmp_path)
    reset_run_store()
    try:
        # Seed a run written by ANOTHER tenant (the shared cache has no tenant axis).
        get_run_store().save(
            StudioRun(
                id="tenant-b-run",
                created_at="2026-06-13T10:00:00Z",
                agent_name="tenant-b-agent",
                provider="claude-cli",
                model="haiku",
                prompt="my secret a@b.com",
                output="ok",
                status="ok",
                steps=[
                    CognitionStep(
                        seq=1,
                        kind="safety",
                        stage="input",
                        redacted=True,
                        flags=["pii:email"],
                        reasons=["redacted email a@b.com"],
                    )
                ],
            )
        )

        app = create_app(ApiContainer.build_default())
        # viewer holds studio.guardrails:read; tenant-bound (tenant_ids set) so the
        # multi-tenant boundary engages (studio_tenant_filter is not None).
        app.state.authenticator = ApiKeyAuthenticator(
            key_principals={
                "k": Principal.build(
                    "u", tenant_ids=["t"], roles=["viewer"], auth_method="apikey"
                )
            }
        )
        client = TestClient(app)
        client.headers.update({"x-himmy-internal-key": "k"})

        r = client.get("/api/studio/guardrails", params={"scan": 1000, "limit": 500})
        assert r.status_code == 200, r.text
        body = r.json()
        # Catalog is still served (it is declared in code, not per-tenant).
        assert body["builtins"] == builtin_rows()
        # But NO firings leak across the tenant boundary.
        assert body["firings"] == []
        assert body["firing_count"] == 0
        assert body["runs_scanned"] == 0
        # Defensive: the other tenant's identifiers never appear anywhere in the body.
        assert "tenant-b-run" not in r.text
        assert "tenant-b-agent" not in r.text
        assert "a@b.com" not in r.text
    finally:
        reset_run_store()
