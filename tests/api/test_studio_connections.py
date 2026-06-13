"""Tests for Studio Connections — connect Email/Telegram/Web, secrets never leak."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_connections as sc
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)
from tests.conftest import run_async


@pytest.fixture
def file_backend(tmp_path: Path):
    import os

    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    yield
    configure_secrets(None)
    # apply_connections_to_env writes non-secret fields into os.environ — clear them
    # so they don't leak into other tests.
    for ct in sc.CONNECTION_TYPES.values():
        for f in ct.fields:
            os.environ.pop(f.secret_name, None)


def test_set_get_delete_round_trip(file_backend: None) -> None:
    sc.set_connection("telegram", {"bot_token": "123:abc", "chat_id": "42"})
    status = sc.get_connection("telegram")
    assert status is not None
    assert status.configured is True
    assert status.writable is True
    by = {f.name: f for f in status.fields}
    assert by["bot_token"].present and by["bot_token"].masked_hint == "••••"
    assert by["chat_id"].masked_hint == "42"  # non-secret shown
    sc.delete_connection("telegram")
    assert sc.get_connection("telegram").configured is False  # type: ignore[union-attr]


def test_secret_value_never_in_status(file_backend: None) -> None:
    sc.set_connection(
        "email", {"smtp_host": "smtp.x", "smtp_user": "u", "smtp_password": "sekret"}
    )
    status = sc.get_connection("email")
    assert status is not None
    assert "sekret" not in status.model_dump_json()


def test_read_only_backend_rejects_writes() -> None:
    configure_secrets(EnvSecrets())  # not writable
    try:
        with pytest.raises(sc.ReadOnlyBackendError):
            sc.set_connection("telegram", {"bot_token": "x"})
    finally:
        configure_secrets(None)


def test_unknown_type_raises(file_backend: None) -> None:
    with pytest.raises(KeyError):
        sc.set_connection("nonexistent-connection", {"x": "y"})
    assert sc.get_connection("nonexistent-connection") is None


def test_apply_connections_to_env(
    file_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HIMMY_SMTP_HOST", raising=False)
    sc.set_connection("email", {"smtp_host": "smtp.example.com", "smtp_password": "p"})
    # set_connection already applied; non-secret host is now in the process env
    import os

    assert os.environ.get("HIMMY_SMTP_HOST") == "smtp.example.com"


def test_test_connection_email_uses_smtp_login(
    file_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc.set_connection("email", {"smtp_host": "smtp.x", "smtp_password": "p"})
    called: dict[str, bool] = {}

    def fake_login(cfg: object) -> None:
        called["ok"] = True

    monkeypatch.setattr("himmy.toolkit.comms.smtp_login_check", fake_login)
    result = run_async(sc.test_connection("email"))
    assert result.ok is True
    assert called.get("ok") is True
    assert result.checked == "smtp login"


def test_test_connection_web_search_duckduckgo_is_keyless(file_backend: None) -> None:
    sc.set_connection("web_search", {"backend": "duckduckgo"})
    result = run_async(sc.test_connection("web_search"))
    assert result.ok is True
    assert "keyless" in result.detail.lower()
