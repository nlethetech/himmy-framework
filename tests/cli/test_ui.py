"""Tests for the desk-style live CLI rendering (himmy.cli.ui)."""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest

from himmy.cli import ui
from himmy.cli.__main__ import main
from himmy.core.events import EventType, RunEvent


class _Tty(io.StringIO):
    """A StringIO that claims to be a terminal."""

    def isatty(self) -> bool:  # noqa: D102
        return True


@pytest.fixture
def truecolor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("HIMMY_NO_ANIM", raising=False)


def _ev(kind: EventType, payload: dict | None = None, **kw: Any) -> RunEvent:
    return RunEvent(event_type=kind, payload=payload or {}, **kw)


def test_styles_empty_off_tty() -> None:
    """Pipes get no escape codes at all."""
    plain = ui.styles(io.StringIO())
    assert all(v == "" for v in plain.values())


def test_styles_respect_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert all(v == "" for v in ui.styles(_Tty()).values())


def test_markdown_lite_renders_bold_and_code(truecolor: None) -> None:
    out = ui.render_markdown_lite("a **bold** and `code` bit", stream=_Tty())
    assert "**" not in out and "`" not in out
    assert "bold" in out and "\x1b[1m" in out


def test_markdown_lite_untouched_when_piped() -> None:
    text = "a **bold** and `code` bit"
    assert ui.render_markdown_lite(text, stream=io.StringIO()) == text


def test_live_run_ui_renders_tool_lines(truecolor: None) -> None:
    """Tool call/done/fail and turn lines land styled on the stream, live."""
    stream = _Tty()
    live = ui.LiveRunUI(model_label="sonnet · claude-cli", stream=stream)
    assert live.enabled

    async def _drive() -> None:
        await live.handle(_ev(EventType.AGENT_RUN_STARTED))
        await live.handle(
            _ev(
                EventType.TOOL_CALLED,
                {"tool_name": "calculator", "tool_args": {"e": 1}},
            )
        )
        await live.handle(_ev(EventType.TOOL_COMPLETED, {"tool_name": "calculator"}))
        await live.handle(
            _ev(EventType.TOOL_FAILED, {"tool_name": "run_python"}, error="blocked")
        )
        await live.handle(_ev(EventType.AGENT_TURN_STARTED, {"turn": 2}))

    asyncio.run(_drive())
    live.finish()
    live.footer()
    out = stream.getvalue()
    assert "⚙ tool call" in out and "calculator" in out
    assert "✓" in out and "✗ tool FAILED" in out and "run_python" in out
    assert "○ turn #2" in out
    assert "5 events · 1 tool call(s)" in out


def test_live_run_ui_spinner_resolves_to_ok_line(truecolor: None) -> None:
    """INFERENCE_REQUESTED starts the spinner; SUCCEEDED replaces it with ✓ + ms."""
    stream = _Tty()
    live = ui.LiveRunUI(model_label="sonnet · claude-cli", stream=stream)

    async def _drive() -> None:
        await live.handle(_ev(EventType.INFERENCE_REQUESTED, {"model_key": "sonnet"}))
        await asyncio.sleep(0.2)  # let a few frames draw
        await live.handle(_ev(EventType.INFERENCE_SUCCEEDED, latency_ms=920.0))

    asyncio.run(_drive())
    out = stream.getvalue()
    assert any(f in out for f in ui.SPINNER_FRAMES)  # it animated
    assert "inference ok" in out and "920ms" in out and "sonnet · claude-cli" in out


def test_live_run_ui_silent_off_tty() -> None:
    """No TTY → not enabled, nothing written, never raises."""
    stream = io.StringIO()
    live = ui.LiveRunUI(model_label="x", stream=stream)
    assert not live.enabled
    asyncio.run(live.handle(_ev(EventType.TOOL_CALLED, {"tool_name": "t"})))
    live.finish()
    live.footer()
    assert stream.getvalue() == ""


def test_live_run_ui_never_raises(truecolor: None) -> None:
    """A malformed event is swallowed — decoration must not break the run."""
    live = ui.LiveRunUI(stream=_Tty())
    asyncio.run(live.handle(object()))  # no event_type at all


def test_compose_event_handlers() -> None:
    seen: list[str] = []

    async def a(e: Any) -> None:
        seen.append("a")

    async def b(e: Any) -> None:
        seen.append("b")

    assert ui.compose_event_handlers(None, None) is None
    assert ui.compose_event_handlers(a, None) is a
    fan = ui.compose_event_handlers(a, b)
    asyncio.run(fan(_ev(EventType.AGENT_RUN_STARTED)))
    assert seen == ["a", "b"]


def test_run_stdout_stays_plain_when_piped(capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end: piped `himmy run` output carries no ANSI escapes."""
    code = main(["run", "--provider", "stub", "--name", "t", "hello"])
    assert code == 0
    out = capsys.readouterr().out
    assert "\x1b[" not in out
