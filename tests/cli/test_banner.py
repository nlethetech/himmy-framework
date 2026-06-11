"""Bare ``himmy`` prints the HIMMY/AGENTS splash instead of an argparse error."""

from __future__ import annotations

import pytest

from himmy import __version__
from himmy.cli.__main__ import main
from himmy.cli.banner import render_banner, supports_color


def test_bare_himmy_prints_banner_and_exits_zero(capsys: pytest.CaptureFixture) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "▲" in out  # the mark survives next to the wordmark
    assert "███████║██║██╔████╔██║" in out  # the HIMMY block art
    assert "the local-first agent framework" in out
    assert __version__ in out
    assert "himmy --help" in out  # points new users at the real command list


def test_subcommands_still_dispatch_normally(capsys: pytest.CaptureFixture) -> None:
    # The splash hook must not swallow real invocations.
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_banner_without_color_has_no_ansi_codes() -> None:
    plain = render_banner(color=False)
    assert "\x1b[" not in plain
    assert "╚═╝  ╚═╝ ╚═════╝ ╚══════╝" in plain  # the AGENTS base survives uncolored


def test_banner_truecolor_when_terminal_advertises_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLORTERM", "truecolor")
    colored = render_banner(color=True)
    assert "\x1b[38;2;232;41;74m" in colored  # demo-film crimson HIMMY
    assert "\x1b[38;2;151;145;127m" in colored  # stone AGENTS
    assert colored.count("\x1b[0m") >= colored.count("\x1b[38")


def test_banner_falls_back_to_256_color(monkeypatch: pytest.MonkeyPatch) -> None:
    # macOS Terminal.app has no truecolor; 24-bit codes smear into blocks there.
    monkeypatch.delenv("COLORTERM", raising=False)
    colored = render_banner(color=True)
    assert "\x1b[38;2;" not in colored
    assert "\x1b[38;5;197m" in colored  # 256-color crimson
    assert "\x1b[38;5;246m" in colored  # 256-color stone


def test_no_color_env_disables_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    class _Tty:
        def isatty(self) -> bool:
            return True

    assert supports_color(_Tty()) is False


def test_non_tty_disables_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    class _Pipe:
        def isatty(self) -> bool:
            return False

    assert supports_color(_Pipe()) is False
