"""Tests for ``himmy connectors`` — the CLI front door to the connector service.

Drives the subcommand through the real argparse parser + ``cmd_connectors`` dispatcher
against a writable secrets backend, asserting list/set/enable/disable/test work and that
set/enable FAIL LOUDLY (non-zero, clear message) on the bare env-only backend.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from himmy.cli.__main__ import build_parser
from himmy.cli.connectors_cmd import cmd_connectors
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)

_TOUCHED = [
    "HIMMY_GITHUB_TOKEN",
    "HIMMY_GITHUB_ALLOWED_REPOS",
    "HIMMY_SLACK_BOT_TOKEN",
    "HIMMY_CONNECTOR_GITHUB_OUTBOUND_ENABLED",
]


@pytest.fixture
def writable_backend(tmp_path: Path):
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in _TOUCHED:
        os.environ.pop(name, None)
    yield
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


def _run(*argv: str) -> int:
    args = build_parser().parse_args(["connectors", *argv])
    return cmd_connectors(args)


# ----------------------------------------------------------------- parser wiring


def test_connectors_is_a_registered_subcommand() -> None:
    from himmy.cli.__main__ import _command_names

    assert "connectors" in _command_names(build_parser())


def test_default_action_is_list() -> None:
    args = build_parser().parse_args(["connectors"])
    assert getattr(args, "action", None) is None  # dispatcher defaults to list


# --------------------------------------------------------------------------- list


def test_list_shows_catalog(writable_backend: None, capsys: pytest.CaptureFixture) -> None:
    code = _run("list")
    assert code == 0
    out = capsys.readouterr().out
    # script-friendly stdout rows for every built-in
    for name in ("slack", "discord", "github", "webhook", "websearch"):
        assert name in out


# ---------------------------------------------------------------------------- set


def test_set_writes_secret_and_echoes_only_name(
    writable_backend: None, capsys: pytest.CaptureFixture
) -> None:
    code = _run("set", "github", "--secret", "HIMMY_GITHUB_TOKEN=ghp_secretval")
    captured = capsys.readouterr()
    assert code == 0
    # the value never appears on stdout/stderr
    assert "ghp_secretval" not in captured.out
    assert "ghp_secretval" not in captured.err
    # and it actually persisted
    from himmy.connectors.manage import ConnectorService

    assert ConnectorService().status("github").configured is True


def test_set_unknown_field_is_clean_error(
    writable_backend: None, capsys: pytest.CaptureFixture
) -> None:
    code = _run("set", "github", "--secret", "HIMMY_BOGUS=x")
    assert code == 1
    assert "unknown field" in capsys.readouterr().err


def test_set_unknown_connector(writable_backend: None) -> None:
    assert _run("set", "nope", "--secret", "A=b") == 1


def test_set_malformed_pair(writable_backend: None) -> None:
    assert _run("set", "github", "--secret", "NOEQUALS") == 2


def test_set_allowlist(writable_backend: None) -> None:
    assert _run("set", "github", "--secret", "HIMMY_GITHUB_TOKEN=x") == 0
    assert _run("set", "github", "--allowlist", "o/a,o/b") == 0
    from himmy.connectors.manage import ConnectorService

    assert ConnectorService().status("github").allowlist_set is True


# --------------------------------------------------------------- enable / disable


def test_enable_disable(writable_backend: None) -> None:
    from himmy.connectors.manage import ConnectorService

    assert _run("enable", "github") == 0
    assert ConnectorService().is_enabled("github", "outbound") is True
    assert _run("disable", "github") == 0
    assert ConnectorService().is_enabled("github", "outbound") is False


def test_enable_inbound_surface(writable_backend: None) -> None:
    from himmy.connectors.manage import ConnectorService

    assert _run("enable", "webhook", "--surface", "inbound") == 0
    assert ConnectorService().is_enabled("webhook", "inbound") is True
    assert ConnectorService().is_enabled("webhook", "outbound") is False


# --------------------------------------------------------------------------- test


def test_test_reports_unconfigured(
    writable_backend: None, capsys: pytest.CaptureFixture
) -> None:
    code = _run("test", "github")
    assert code == 1
    assert "HIMMY_GITHUB_TOKEN" in capsys.readouterr().err


# ----------------------------------------------------- fail-loud on env backend


def test_set_fails_loud_on_env_backend(capsys: pytest.CaptureFixture) -> None:
    configure_secrets(EnvSecrets())
    try:
        code = _run("set", "github", "--secret", "HIMMY_GITHUB_TOKEN=x")
    finally:
        configure_secrets(None)
    assert code == 1
    assert "read-only" in capsys.readouterr().err


def test_enable_fails_loud_on_env_backend(capsys: pytest.CaptureFixture) -> None:
    configure_secrets(EnvSecrets())
    try:
        code = _run("enable", "github")
    finally:
        configure_secrets(None)
    assert code == 1
    assert "read-only" in capsys.readouterr().err
