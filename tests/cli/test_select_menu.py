"""The interactive arrow-key ``/model`` selector (:mod:`himmy.cli.select_menu`).

Every test drives the selector with NO real terminal: navigation tests inject a
``read_key`` callable that returns key tokens deterministically; the terminal-restore
test fakes ``termios``/``tty`` to prove ``tcsetattr`` runs on every exit path; the
non-tty test asserts an immediate ``None`` (so the REPL falls back to the static list).
A final test drives ``/model`` through :class:`ChatRepl` with an injected selection and
asserts the REPL switches the model and calls ``rebuild()``.
"""

from __future__ import annotations

import argparse
import io
from typing import Any

import pytest

from himmy.cli import select_menu
from himmy.cli.select_menu import select_model

# ---------------------------------------------------------------- sample entries

#: A mixed list: provider HEADERS (not selectable) + selectable MODEL rows. Mirrors
#: the shape of himmy.cli.model_picker.build_picker_entries output.
_ENTRIES: list[dict[str, Any]] = [
    {"kind": "header", "provider": "claude-cli", "model": "", "label": "claude-cli",
     "is_active": False},
    {"kind": "model", "provider": "claude-cli", "model": "haiku",
     "label": "free · local", "is_active": False},
    {"kind": "model", "provider": "claude-cli", "model": "sonnet",
     "label": "free · local", "is_active": True},
    {"kind": "header", "provider": "openrouter", "model": "", "label": "openrouter",
     "is_active": False},
    {"kind": "model", "provider": "openrouter", "model": "openai/gpt-4o-mini",
     "label": "$0.15/$0.60 per 1M", "is_active": False},
]


def _keyer(tokens: list[str]):
    """A read_key callable that yields ``tokens`` in order, then raises if exhausted."""
    it = iter(tokens)

    def _read() -> str:
        return next(it)

    return _read


# ------------------------------------------------------------------- navigation


def test_down_down_enter_selects_skipping_headers() -> None:
    # Active row is 'sonnet' (index 2) → pre-selected. Down clamps there (last of
    # claude-cli)? No — Down should move to the next selectable, gpt-4o-mini.
    stream = io.StringIO()
    read = _keyer(["down", "enter"])
    out = select_model(
        _ENTRIES, active_index=2, palette={}, stream=stream, read_key=read
    )
    # From the active sonnet (index 2), one Down lands on gpt-4o-mini, skipping the
    # 'openrouter' header in between.
    assert out == ("openrouter", "openai/gpt-4o-mini")


def test_starts_at_first_model_then_down_enter() -> None:
    stream = io.StringIO()
    read = _keyer(["down", "down", "enter"])
    # No active_index → starts on the first selectable (haiku, index 1).
    out = select_model(
        _ENTRIES, active_index=None, palette={}, stream=stream, read_key=read
    )
    # haiku -> sonnet -> gpt-4o-mini (header skipped between sonnet and gpt).
    assert out == ("openrouter", "openai/gpt-4o-mini")


def test_vi_keys_and_raw_escape_sequences_navigate() -> None:
    stream = io.StringIO()
    # Mix vi 'j' and raw ANSI down, then up, then Enter. Start at haiku.
    read = _keyer(["j", "\x1b[B", "\x1b[A", "\r"])
    out = select_model(
        _ENTRIES, active_index=None, palette={}, stream=stream, read_key=read
    )
    # haiku -> sonnet(j) -> gpt-4o-mini(down) -> sonnet(up) -> select sonnet.
    assert out == ("claude-cli", "sonnet")


def test_up_at_top_clamps() -> None:
    stream = io.StringIO()
    read = _keyer(["up", "up", "up", "enter"])
    out = select_model(
        _ENTRIES, active_index=None, palette={}, stream=stream, read_key=read
    )
    # Repeated Up from the first model clamps on haiku.
    assert out == ("claude-cli", "haiku")


def test_down_at_bottom_clamps() -> None:
    stream = io.StringIO()
    read = _keyer(["down", "down", "down", "down", "down", "enter"])
    out = select_model(
        _ENTRIES, active_index=None, palette={}, stream=stream, read_key=read
    )
    # Walks to the last selectable (gpt-4o-mini) and clamps there.
    assert out == ("openrouter", "openai/gpt-4o-mini")


@pytest.mark.parametrize("cancel", ["\x1b", "q", "\x03", "cancel"])
def test_cancel_tokens_return_none(cancel: str) -> None:
    stream = io.StringIO()
    read = _keyer([cancel])
    out = select_model(
        _ENTRIES, active_index=None, palette={}, stream=stream, read_key=read
    )
    assert out is None


def test_unknown_keys_are_ignored_until_enter() -> None:
    stream = io.StringIO()
    read = _keyer(["x", "?", "down", "enter"])
    out = select_model(
        _ENTRIES, active_index=None, palette={}, stream=stream, read_key=read
    )
    # The junk keys don't move; Down then Enter → sonnet.
    assert out == ("claude-cli", "sonnet")


def test_no_selectable_rows_returns_none() -> None:
    stream = io.StringIO()
    only_header = [{"kind": "header", "provider": "x", "model": "", "label": "x",
                    "is_active": False}]
    read = _keyer(["enter"])  # never read — no selectables → early None
    out = select_model(only_header, palette={}, stream=stream, read_key=read)
    assert out is None


def test_redraws_in_place_each_keypress() -> None:
    # Each move should emit the cursor-up + clear-to-end sequence before redrawing.
    stream = io.StringIO()
    read = _keyer(["down", "down", "enter"])
    select_model(_ENTRIES, active_index=None, palette={}, stream=stream, read_key=read)
    text = stream.getvalue()
    # Two moves → at least two in-place clear sequences (ESC[<n>A ... ESC[J).
    assert text.count("\x1b[J") >= 2
    assert "\x1b[" in text  # cursor-up control present


# --------------------------------------------------------------- terminal restore


class _FakeTermios:
    """A stand-in ``termios`` recording that tcsetattr (restore) was called."""

    TCSADRAIN = 1

    def __init__(self) -> None:
        self.restored = False
        self.saved_attrs = ["ATTRS"]

    def tcgetattr(self, _fd: int) -> Any:
        return self.saved_attrs

    def tcsetattr(self, _fd: int, _when: int, attrs: Any) -> None:
        # The restore must hand back exactly what tcgetattr returned.
        assert attrs == self.saved_attrs
        self.restored = True


class _FakeTty:
    def setcbreak(self, _fd: int) -> None:
        return None


class _FakeStdin:
    """A fake stdin with a real-looking fd that claims to be a TTY."""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 0


def _install_fake_terminal(monkeypatch: pytest.MonkeyPatch) -> _FakeTermios:
    fake_termios = _FakeTermios()
    fake_tty = _FakeTty()
    monkeypatch.setitem(__import__("sys").modules, "termios", fake_termios)
    monkeypatch.setitem(__import__("sys").modules, "tty", fake_tty)
    monkeypatch.setattr(select_menu.sys, "stdin", _FakeStdin())
    return fake_termios


def test_terminal_restored_on_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_termios = _install_fake_terminal(monkeypatch)
    # No injected read_key → raw-mode path. Feed one logical Enter via a patched
    # raw reader so the loop selects immediately.
    monkeypatch.setattr(select_menu, "_read_key_raw", lambda: "enter")
    out = select_model(_ENTRIES, active_index=1, palette={}, stream=io.StringIO())
    assert out == ("claude-cli", "haiku")
    assert fake_termios.restored is True


def test_terminal_restored_when_key_read_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_termios = _install_fake_terminal(monkeypatch)

    def boom() -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(select_menu, "_read_key_raw", boom)
    # The exception propagates, but the finally/__exit__ restore must still run.
    with pytest.raises(KeyboardInterrupt):
        select_model(_ENTRIES, active_index=1, palette={}, stream=io.StringIO())
    assert fake_termios.restored is True


def test_terminal_restored_on_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_termios = _install_fake_terminal(monkeypatch)
    monkeypatch.setattr(select_menu, "_read_key_raw", lambda: "\x1b")  # Esc
    out = select_model(_ENTRIES, active_index=1, palette={}, stream=io.StringIO())
    assert out is None
    assert fake_termios.restored is True


def test_raw_mode_setup_failure_falls_back_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tcgetattr raising OSError (not a controllable terminal) → graceful None, and the
    # restore path is a no-op (nothing was saved) — never a crash.
    class _ExplodingTermios(_FakeTermios):
        def tcgetattr(self, _fd: int) -> Any:
            raise OSError("not a tty")

    monkeypatch.setitem(__import__("sys").modules, "termios", _ExplodingTermios())
    monkeypatch.setitem(__import__("sys").modules, "tty", _FakeTty())
    monkeypatch.setattr(select_menu.sys, "stdin", _FakeStdin())
    out = select_model(_ENTRIES, active_index=1, palette={}, stream=io.StringIO())
    assert out is None


# ------------------------------------------------------------------ non-tty path


def test_non_tty_returns_none_immediately() -> None:
    # isatty=False → no raw mode attempted, no read_key needed; immediate None so the
    # caller prints the static list. The stream stays untouched.
    stream = io.StringIO()
    out = select_model(_ENTRIES, active_index=1, palette={}, stream=stream, isatty=False)
    assert out is None
    assert stream.getvalue() == ""


# --------------------------------------------------------------- REPL integration


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "file": None,
        "name": "t",
        "instruction": None,
        "provider": "stub",
        "model": None,
        "message": None,
        "session": None,
        "yolo": False,
        "safe": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_repl_model_menu_switches_on_selection(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from himmy.cli.repl import ChatRepl
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(provider="stub"), input_fn=lambda _p: "")
    # Force the interactive branch and inject a selection (bypass the real terminal).
    repl.interactive = True
    monkeypatch.setattr(
        repl, "_interactive_model_menu", lambda: ("claude-cli", "opus")
    )
    rebuilt: list[bool] = []
    monkeypatch.setattr(repl, "rebuild", lambda: rebuilt.append(True))

    repl._slash("/model", thread=object())

    # The chosen target is applied + rebuild() called + a short label printed.
    assert repl.args.model == "opus"
    assert repl.args.provider == "claude-cli"
    assert rebuilt == [True]
    err = capsys.readouterr().err
    assert "model:" in err
    assert "active:" not in err  # NOT the static picker


def test_repl_model_menu_cancel_falls_back_to_static_list(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from himmy.cli.repl import ChatRepl
    from himmy.config.agent_spec import AgentSpec

    for _var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(_var, raising=False)
    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(provider="stub"), input_fn=lambda _p: "")
    repl.interactive = True
    # Menu cancels (None) → the original static picker text prints, unchanged.
    monkeypatch.setattr(repl, "_interactive_model_menu", lambda: None)
    rebuilt: list[bool] = []
    monkeypatch.setattr(repl, "rebuild", lambda: rebuilt.append(True))

    repl._slash("/model", thread=object())

    assert rebuilt == []  # no switch on cancel
    err = capsys.readouterr().err
    assert "active:" in err  # the static picker
    assert "available:" in err


def test_repl_model_menu_exception_falls_back(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from himmy.cli.repl import ChatRepl
    from himmy.config.agent_spec import AgentSpec

    for _var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(_var, raising=False)
    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(provider="stub"), input_fn=lambda _p: "")
    repl.interactive = True

    def boom() -> tuple[str, str] | None:
        raise RuntimeError("menu blew up")

    # Even if the menu raises, the REPL must survive and show the static list.
    monkeypatch.setattr(repl, "_interactive_model_menu", boom)

    repl._slash("/model", thread=object())
    err = capsys.readouterr().err
    assert "active:" in err  # fell back to the static picker, REPL alive


def test_repl_model_menu_ctrl_c_cancels_without_exiting_repl(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C in the menu raises KeyboardInterrupt; it must cancel the menu and
    fall back to the static list, NOT propagate out and kill the whole REPL."""
    from himmy.cli.repl import ChatRepl
    from himmy.config.agent_spec import AgentSpec

    for _var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(_var, raising=False)
    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(provider="stub"), input_fn=lambda _p: "")
    repl.interactive = True

    def interrupt() -> tuple[str, str] | None:
        raise KeyboardInterrupt

    monkeypatch.setattr(repl, "_interactive_model_menu", interrupt)

    # Must NOT raise; must fall back to the static picker and keep the REPL alive.
    repl._slash("/model", thread=object())
    err = capsys.readouterr().err
    assert "active:" in err
