"""T1.3 — the connector management service: catalog/configure/enable/disable/test.

Covers the unifying :class:`~himmy.connectors.manage.ConnectorService` surface: the
catalog (no secret values), configure/enable/disable against a WRITABLE backend, the
fail-loud-on-env-backend path, the capability + live-ping test, and build_outbound/inbound
refusing cleanly when a connector is not configured.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)
from himmy.connectors.manage import (
    CONNECTOR_CATALOG,
    ConnectorService,
    ReadOnlyBackendError,
    ensure_builtin_connectors_registered,
)
from himmy.connectors.sdk import ConnectorError
from tests.conftest import run_async

_TOUCHED = [
    "HIMMY_GITHUB_TOKEN",
    "HIMMY_GITHUB_ALLOWED_REPOS",
    "HIMMY_GITHUB_DEFAULT_REPO",
    "HIMMY_SLACK_BOT_TOKEN",
    "HIMMY_SLACK_ALLOWED_CHANNELS",
    "HIMMY_WEBHOOK_SIGNING_SECRET",
    "HIMMY_WEBHOOK_ALLOWED_SOURCES",
    "HIMMY_TAVILY_API_KEY",
    "HIMMY_CONNECTOR_GITHUB_OUTBOUND_ENABLED",
    "HIMMY_CONNECTOR_WEBHOOK_INBOUND_ENABLED",
]


@pytest.fixture
def writable_backend(tmp_path: Path):
    """A writable FileSecrets backend (configure/enable can persist)."""
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in _TOUCHED:
        os.environ.pop(name, None)
    yield
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


def _svc() -> ConnectorService:
    return ConnectorService()


# ----------------------------------------------------------------- catalog


def test_catalog_lists_every_builtin(writable_backend: None) -> None:
    svc = _svc()
    names = svc.names()
    assert {"slack", "discord", "github", "websearch", "webhook"} <= set(names)
    rows = svc.catalog()
    assert {r.name for r in rows} == set(names)


def test_catalog_never_leaks_a_secret_value(writable_backend: None) -> None:
    svc = _svc()
    svc.configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_supersecret"})
    rows = svc.catalog()
    import json

    blob = json.dumps([r.to_dict() for r in rows])
    assert "ghp_supersecret" not in blob
    gh = next(r for r in rows if r.name == "github")
    # presence is reported, value is not
    assert gh.configured is True
    token = next(s for s in gh.secrets if s.name == "HIMMY_GITHUB_TOKEN")
    assert token.present is True


def test_unknown_connector_raises(writable_backend: None) -> None:
    with pytest.raises(KeyError):
        _svc().status("nope")
    with pytest.raises(KeyError):
        _svc().descriptor("nope")


# --------------------------------------------------------------- configure


def test_configure_writes_required_secret(writable_backend: None) -> None:
    svc = _svc()
    assert svc.status("github").configured is False
    status = svc.configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_x"})
    assert status.configured is True
    # round-trips through a fresh service (persisted, not just in-memory)
    assert _svc().status("github").configured is True


def test_configure_ignores_unknown_keys(writable_backend: None) -> None:
    svc = _svc()
    status = svc.configure(
        "github", {"HIMMY_GITHUB_TOKEN": "ghp_x", "HIMMY_NOT_A_FIELD": "y"}
    )
    assert status.configured is True
    from himmy.config.secrets import get_secret

    assert get_secret("HIMMY_NOT_A_FIELD") is None


def test_configure_allowlist_field(writable_backend: None) -> None:
    svc = _svc()
    svc.configure(
        "github",
        {"HIMMY_GITHUB_TOKEN": "ghp_x", "HIMMY_GITHUB_ALLOWED_REPOS": "o/a,o/b"},
    )
    assert svc.status("github").allowlist_set is True


def test_empty_value_deletes(writable_backend: None) -> None:
    svc = _svc()
    svc.configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_x"})
    assert svc.status("github").configured is True
    svc.configure("github", {"HIMMY_GITHUB_TOKEN": ""})
    assert svc.status("github").configured is False


def test_unconfigure_clears_all(writable_backend: None) -> None:
    svc = _svc()
    svc.configure(
        "github",
        {"HIMMY_GITHUB_TOKEN": "ghp_x", "HIMMY_GITHUB_ALLOWED_REPOS": "o/a"},
    )
    svc.unconfigure("github")
    s = svc.status("github")
    assert s.configured is False
    assert s.allowlist_set is False


# ------------------------------------------------------ fail-loud on env backend


def test_configure_fails_loud_on_env_backend() -> None:
    configure_secrets(EnvSecrets())  # not writable
    try:
        with pytest.raises(ReadOnlyBackendError):
            _svc().configure("github", {"HIMMY_GITHUB_TOKEN": "x"})
    finally:
        configure_secrets(None)


def test_enable_fails_loud_on_env_backend() -> None:
    configure_secrets(EnvSecrets())
    try:
        with pytest.raises(ReadOnlyBackendError):
            _svc().enable("github", "outbound")
        with pytest.raises(ReadOnlyBackendError):
            _svc().disable("github", "outbound")
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- enable/disable


def test_enable_disable_round_trip(writable_backend: None) -> None:
    svc = _svc()
    assert svc.is_enabled("github", "outbound") is False
    svc.enable("github", "outbound")
    assert svc.is_enabled("github", "outbound") is True
    # default absent ⇒ OFF, and the flag is surface-specific
    assert svc.is_enabled("github", "inbound") is False
    # persisted across a fresh service
    assert _svc().is_enabled("github", "outbound") is True
    svc.disable("github", "outbound")
    assert svc.is_enabled("github", "outbound") is False


def test_enable_unknown_surface_rejected(writable_backend: None) -> None:
    with pytest.raises(ValueError, match="surface"):
        _svc().enable("github", "sideways")


def test_status_reflects_enabled_flags(writable_backend: None) -> None:
    svc = _svc()
    svc.enable("webhook", "inbound")
    status = svc.status("webhook")
    assert status.enabled_inbound is True
    assert status.enabled_outbound is False


# --------------------------------------------------------------------- test()


def test_test_reports_missing_credential(writable_backend: None) -> None:
    result = run_async(_svc().test("github"))
    assert result.ok is False
    assert result.checked == "capability"
    assert "HIMMY_GITHUB_TOKEN" in result.detail


def test_test_no_probe_connector(writable_backend: None) -> None:
    svc = _svc()
    svc.configure("slack", {"HIMMY_SLACK_BOT_TOKEN": "xoxb-x"})
    result = run_async(svc.test("slack"))
    # slack is configured + capability-ok, has no live probe → green on capability
    assert result.ok is True
    assert result.checked == "capability"


def test_test_live_probe_runs_after_capability(writable_backend: None) -> None:
    svc = _svc()
    svc.configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_x"})

    async def fake_probe() -> str:
        return "@octocat"

    # Swap the descriptor's probe with a fake so the test stays offline.
    object.__setattr__(CONNECTOR_CATALOG["github"], "probe", fake_probe)
    try:
        result = run_async(svc.test("github"))
    finally:
        from himmy.connectors.manage import _probe_github

        object.__setattr__(CONNECTOR_CATALOG["github"], "probe", _probe_github)
    assert result.ok is True
    assert result.checked == "live ping"
    assert result.detail == "@octocat"


def test_test_live_probe_failure_is_secret_safe(writable_backend: None) -> None:
    svc = _svc()
    svc.configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_secretval"})

    async def boom_probe() -> str:
        raise RuntimeError("boom with ghp_secretval inside")

    object.__setattr__(CONNECTOR_CATALOG["github"], "probe", boom_probe)
    try:
        result = run_async(svc.test("github"))
    finally:
        from himmy.connectors.manage import _probe_github

        object.__setattr__(CONNECTOR_CATALOG["github"], "probe", _probe_github)
    assert result.ok is False
    assert result.checked == "live ping"
    # safe_error reduces an unknown exception to its type → secret never rides out
    assert "ghp_secretval" not in result.detail


# ----------------------------------------------------------------- build


def test_build_outbound_refuses_when_unconfigured(writable_backend: None) -> None:
    with pytest.raises(ConnectorError, match="not configured"):
        _svc().build_outbound("github")


def test_build_outbound_succeeds_when_configured(writable_backend: None) -> None:
    svc = _svc()
    svc.configure(
        "github",
        {"HIMMY_GITHUB_TOKEN": "ghp_x", "HIMMY_GITHUB_ALLOWED_REPOS": "o/a"},
    )
    connector = svc.build_outbound("github")
    assert connector.name == "github"
    assert connector.available() is True


def test_build_outbound_rejects_inbound_kind(writable_backend: None) -> None:
    with pytest.raises(ConnectorError, match="not an outbound"):
        _svc().build_outbound("webhook")


def test_build_inbound_refuses_when_unconfigured(writable_backend: None) -> None:
    async def handler(sender: str, text: str) -> str:
        return "ok"

    with pytest.raises(ConnectorError, match="not configured"):
        _svc().build_inbound("webhook", handler)


def test_build_inbound_succeeds_when_configured(writable_backend: None) -> None:
    svc = _svc()
    svc.configure(
        "webhook",
        {
            "HIMMY_WEBHOOK_SIGNING_SECRET": "a-long-secret",
            "HIMMY_WEBHOOK_ALLOWED_SOURCES": "src-1",
        },
    )

    async def handler(sender: str, text: str) -> str:
        return f"echo:{text}"

    connector = svc.build_inbound("webhook", handler)
    # the configured signing secret + allow-list flow into the connector
    assert connector.is_allowed("src-1") is True
    assert connector.is_allowed("evil") is False


def test_build_inbound_rejects_outbound_kind(writable_backend: None) -> None:
    async def handler(sender: str, text: str) -> str:
        return "ok"

    with pytest.raises(ConnectorError, match="not an inbound"):
        _svc().build_inbound("github", handler)


# --------------------------------------------------------- registration idempotent


def test_ensure_registered_is_idempotent() -> None:
    from himmy.connectors.sdk import default_registry

    ensure_builtin_connectors_registered()
    ensure_builtin_connectors_registered()
    names = default_registry().names()
    assert {"slack", "discord", "github", "websearch"} <= set(names)
