"""Regression: SubprocessSandbox logs a one-time warning where limits can't be enforced.

Friend-audit finding #4: the default ``SubprocessSandbox`` enforces no resource limits on
Windows (no ``setrlimit``) and no memory cap on macOS (``RLIMIT_AS`` ignored). It stays the
documented weak default (real isolation = ``ContainerSandbox``), but now emits a loud,
one-time runtime warning so operators can't miss the gap.
"""

from __future__ import annotations

import logging

import pytest

import himmy.services.sandbox.subprocess_sandbox as sbx


@pytest.fixture(autouse=True)
def _reset_warn_flag() -> None:
    sbx._limits_unenforced_warned = False
    yield
    sbx._limits_unenforced_warned = False


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_warns_on_windows(monkeypatch, caplog) -> None:
    monkeypatch.setattr(sbx.os, "name", "nt")
    with caplog.at_level(logging.WARNING, logger="himmy.sandbox"):
        sbx._warn_if_limits_unenforced()
    msgs = _messages(caplog)
    assert msgs and "setrlimit" in msgs[0] and "ContainerSandbox" in msgs[0]


def test_warns_on_macos(monkeypatch, caplog) -> None:
    monkeypatch.setattr(sbx.os, "name", "posix")
    monkeypatch.setattr(sbx.sys, "platform", "darwin")
    with caplog.at_level(logging.WARNING, logger="himmy.sandbox"):
        sbx._warn_if_limits_unenforced()
    msgs = _messages(caplog)
    assert msgs and "RLIMIT_AS" in msgs[0]


def test_no_warning_on_linux(monkeypatch, caplog) -> None:
    monkeypatch.setattr(sbx.os, "name", "posix")
    monkeypatch.setattr(sbx.sys, "platform", "linux")
    with caplog.at_level(logging.WARNING, logger="himmy.sandbox"):
        sbx._warn_if_limits_unenforced()
    assert _messages(caplog) == []


def test_warns_only_once_per_process(monkeypatch, caplog) -> None:
    monkeypatch.setattr(sbx.os, "name", "nt")
    with caplog.at_level(logging.WARNING, logger="himmy.sandbox"):
        for _ in range(5):
            sbx._warn_if_limits_unenforced()
    assert len(_messages(caplog)) == 1
