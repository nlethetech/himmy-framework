"""Tests for ``himmy.cli.guardrails_view``: the guardrails subcommand, the live
shield line, and the /guardrails REPL slash.

The shield line is tested against a real ``GUARDRAIL_APPLIED`` :class:`RunEvent`
(the same payload shape ``single_agent._apply_guardrail`` emits), so the renderer
is verified against the actual framework event, not a stub.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from himmy.cli.guardrails_view import (
    builtin_rows,
    cmd_guardrails,
    render_guardrail_event,
    slash_guardrails,
)
from himmy.cli.ui import styles
from himmy.core.events import EventType, RunEvent
from himmy.services.guardrails.builtins import BUILTIN_GUARDRAILS

_PLAIN = {k: "" for k in styles(None)}


def _applied_event(**payload: Any) -> RunEvent:
    """A real GUARDRAIL_APPLIED RunEvent with the emitter's payload shape."""
    base = {
        "stage": "input",
        "blocked": False,
        "redacted": False,
        "flags": [],
        "reasons": [],
    }
    base.update(payload)
    return RunEvent(event_type=EventType.GUARDRAIL_APPLIED, payload=base)


# --------------------------------------------------------------- builtin_rows / cmd


def test_builtin_rows_cover_every_registered_guardrail() -> None:
    rows = builtin_rows()
    names = {r["name"] for r in rows}
    # All five named in the prompt are present...
    assert {"pii", "injection", "nepal_pii", "grounding", "dlp"} <= names
    # ...and nothing registered is silently dropped.
    assert names == set(BUILTIN_GUARDRAILS)
    # Each row has a non-empty one-line description (no embedded newlines).
    for r in rows:
        assert r["description"]
        assert "\n" not in r["description"]


def test_cmd_guardrails_json(capsys: Any) -> None:
    rc = cmd_guardrails(argparse.Namespace(json=True))
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert {d["name"] for d in data} == set(BUILTIN_GUARDRAILS)


def test_cmd_guardrails_human_lists_each_to_stdout(capsys: Any) -> None:
    rc = cmd_guardrails(argparse.Namespace(json=False))
    assert rc == 0
    captured = capsys.readouterr()
    # The listing goes to stdout; only the hint goes to stderr.
    for name in ("pii", "injection", "nepal_pii", "grounding", "dlp"):
        assert name in captured.out
    assert "🛡" in captured.out


def test_cmd_guardrails_missing_json_attr_defaults_to_human(capsys: Any) -> None:
    # An argparse.Namespace without a `json` attr must not crash (getattr default).
    rc = cmd_guardrails(argparse.Namespace())
    assert rc == 0
    assert "pii" in capsys.readouterr().out


# --------------------------------------------------------------- render_guardrail_event


def test_render_redaction_line_plain() -> None:
    event = _applied_event(redacted=True, flags=["pii:email"])
    line = render_guardrail_event(event, _PLAIN)
    assert line.startswith("🛡 guardrail ")
    assert "pii" in line
    assert "redacted" in line
    assert "email" in line
    assert "\n" not in line


def test_render_block_line_uses_crimson_on_styled() -> None:
    c = dict(styles(None))
    c.update({"crimson": "<C>", "faint": "<F>", "gold": "<G>", "reset": "<R>"})
    event = _applied_event(
        blocked=True, flags=["injection"], reasons=["possible prompt injection"]
    )
    line = render_guardrail_event(event, c)
    assert "blocked" in line
    # Block accent is crimson, not faint.
    assert "<C>" in line
    assert "injection" in line


def test_render_grounding_block_uses_reason_when_no_sublabel() -> None:
    # `ungrounded` flag has no sub-label, so the detail falls back to the reason.
    event = _applied_event(
        stage="output",
        blocked=True,
        flags=["ungrounded"],
        reasons=["answer relied on stale built-in knowledge, not a tool"],
    )
    line = render_guardrail_event(event, _PLAIN)
    assert "ungrounded" in line  # name derived from the flag
    assert "blocked" in line
    assert "[output]" in line  # stage surfaced


def test_render_empty_payload_does_not_crash() -> None:
    event = RunEvent(event_type=EventType.GUARDRAIL_APPLIED, payload={})
    line = render_guardrail_event(event, _PLAIN)
    assert line.startswith("🛡 guardrail guardrail")  # generic name fallback
    assert "applied" in line


def test_render_dict_event_payload() -> None:
    # Defensive: a plain dict (not a RunEvent) still renders.
    line = render_guardrail_event(
        {"payload": {"redacted": True, "flags": ["pii:phone"]}}, _PLAIN
    )
    assert "phone" in line


# --------------------------------------------------------------- slash_guardrails


class _FakeRecorder:
    def __init__(self, events: list[Any]) -> None:
        self.events = events


class _FakeSpec:
    def __init__(self, guardrails: list[str]) -> None:
        self.guardrails = guardrails


class _FakeRepl:
    def __init__(self, guardrails: list[str], events: list[Any]) -> None:
        self.spec = _FakeSpec(guardrails)
        self.recorder = _FakeRecorder(events)
        self._c = _PLAIN
        self.lines: list[str] = []

    def _eprint(self, *args: Any) -> None:
        self.lines.append(" ".join(str(a) for a in args))


def test_slash_no_guardrails_active() -> None:
    repl = _FakeRepl(guardrails=[], events=[])
    slash_guardrails(repl)
    body = "\n".join(repl.lines)
    assert "no guardrails active" in body


def test_slash_active_with_firings_counts_only_guardrail_events() -> None:
    events = [
        _applied_event(redacted=True, flags=["pii:email"]),
        RunEvent(event_type=EventType.TOOL_CALLED, payload={}),
        _applied_event(blocked=True, flags=["injection"]),
    ]
    repl = _FakeRepl(guardrails=["pii", "injection"], events=events)
    slash_guardrails(repl)
    body = "\n".join(repl.lines)
    assert "pii, injection" in body
    # 2 guardrail events, the TOOL_CALLED is not counted.
    assert "2 guardrail firing" in body


def test_slash_active_with_no_firings() -> None:
    repl = _FakeRepl(guardrails=["pii"], events=[])
    slash_guardrails(repl)
    body = "\n".join(repl.lines)
    assert "pii" in body
    assert "no guardrails fired" in body


def test_slash_tolerates_missing_recorder_and_spec() -> None:
    # A bare object with only _eprint/_c must not crash (all getattr-guarded).
    class Bare:
        def __init__(self) -> None:
            self._c = _PLAIN
            self.lines: list[str] = []

        def _eprint(self, *args: Any) -> None:
            self.lines.append(" ".join(str(a) for a in args))

    repl = Bare()
    slash_guardrails(repl)
    assert any("no guardrails active" in line for line in repl.lines)
