"""Expanded observability-kernel coverage for AAEO-11/14.

Complements ``test_observability.py`` / ``test_observability_spans.py`` with span
lifecycle paths they do not exercise:

- ``TOOL_FAILED`` closes the matching tool span (an error close, not just the
  happy ``TOOL_COMPLETED`` path).
- A ``WORKFLOW_STARTED``/``WORKFLOW_FINISHED`` pair opens and closes a span.
- A CLOSE event with no matching OPEN span degrades to a log (never raises).
- A re-opened span under the same key closes the stale one first (no leak).
- ``reset_spans`` closes + clears any dangling open spans.
- ``emit_event_span`` is robust to ``None`` / odd payload values and never raises.
- ``instrument_asyncpg`` / ``instrument_fastapi`` are no-ops when off.
- A real-logfire integration test is gated behind a skipif (logfire absent).

The module re-imports ``himmy.services.observability`` per test (via
``importlib.reload``) so its module-level span/configure state is reset, and
injects a fake ``logfire`` module so the span context-manager path is exercised
fully offline.
"""

from __future__ import annotations

import importlib

import pytest

from himmy.core.events import EventType, RunEvent


def _fresh_module():
    import himmy.services.observability as obs

    return importlib.reload(obs)


class _FakeSpan:
    """A minimal stand-in for a logfire span (context manager)."""

    def __init__(self, name: str, attributes: dict) -> None:
        self.name = name
        self.attributes = dict(attributes)
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeLogfire:
    """A fake ``logfire`` module recording span() / info() calls."""

    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self.infos: list[tuple[str, dict]] = []

    def span(self, name, **attributes):
        span = _FakeSpan(name, attributes)
        self.spans.append(span)
        return span

    def info(self, name, **attributes):
        self.infos.append((name, attributes))

    def configure(self, **_):
        pass

    def instrument_pydantic_ai(self):
        pass


def _enabled_obs(monkeypatch):
    """Reload the module with the switch on and a fake logfire installed."""
    monkeypatch.setenv("HIMMY_LOGFIRE_ENABLED", "1")
    monkeypatch.delenv("HIMMY_LOGFIRE_INCLUDE_CONTENT", raising=False)
    obs = _fresh_module()
    fake = _FakeLogfire()
    monkeypatch.setitem(importlib.sys.modules, "logfire", fake)
    obs.configure_observability()
    return obs, fake


# ------------------------------------------------------------------- AAEO-11
def test_tool_failed_closes_tool_span(monkeypatch) -> None:
    """TOOL_FAILED closes the tool span opened by TOOL_CALLED (error close)."""
    obs, fake = _enabled_obs(monkeypatch)
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    obs.emit_event_span(
        RunEvent(event_type=EventType.TOOL_CALLED, trace_id="t1", tool_call_id="c1")
    )
    obs.emit_event_span(
        RunEvent(
            event_type=EventType.TOOL_FAILED,
            trace_id="t1",
            tool_call_id="c1",
            error="boom",
        )
    )
    obs.emit_event_span(
        RunEvent(event_type=EventType.AGENT_RUN_FINISHED, trace_id="t1")
    )
    # run + tool spans both opened and both exited; nothing dangling.
    assert len(fake.spans) == 2
    assert all(s.exited for s in fake.spans)
    assert obs._OPEN_SPANS == {}


def test_workflow_span_lifecycle(monkeypatch) -> None:
    """WORKFLOW_STARTED opens a span that WORKFLOW_FINISHED closes."""
    obs, fake = _enabled_obs(monkeypatch)
    obs.emit_event_span(RunEvent(event_type=EventType.WORKFLOW_STARTED, trace_id="t1"))
    assert obs._OPEN_SPANS  # one workflow span open
    obs.emit_event_span(RunEvent(event_type=EventType.WORKFLOW_FINISHED, trace_id="t1"))
    assert len(fake.spans) == 1
    assert fake.spans[0].exited is True
    assert obs._OPEN_SPANS == {}


def test_close_without_open_logs_and_does_not_raise(monkeypatch) -> None:
    """A CLOSE event with no matching OPEN span degrades to a point-in-time log."""
    obs, fake = _enabled_obs(monkeypatch)
    obs.emit_event_span(
        RunEvent(event_type=EventType.AGENT_RUN_FINISHED, trace_id="orphan")
    )
    # No span opened; it became a log instead.
    assert fake.spans == []
    assert any(name == "agent_run_finished" for name, _ in fake.infos)
    assert obs._OPEN_SPANS == {}


def test_reopen_same_key_closes_stale_span(monkeypatch) -> None:
    """Re-opening a span under the same key closes the stale one (no leak)."""
    obs, fake = _enabled_obs(monkeypatch)
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    # A second STARTED under the same trace replaces the first.
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    # Two spans created; the first was exited when replaced, the second is open.
    assert len(fake.spans) == 2
    assert fake.spans[0].exited is True
    assert fake.spans[1].exited is False
    # Exactly one open span remains under the run key.
    assert len(obs._OPEN_SPANS) == 1


def test_reset_spans_closes_dangling(monkeypatch) -> None:
    """reset_spans() exits + clears any open spans."""
    obs, fake = _enabled_obs(monkeypatch)
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    obs.emit_event_span(
        RunEvent(event_type=EventType.TOOL_CALLED, trace_id="t1", tool_call_id="c1")
    )
    assert len(obs._OPEN_SPANS) == 2
    obs.reset_spans()
    assert obs._OPEN_SPANS == {}
    assert all(s.exited for s in fake.spans)


# ------------------------------------------------------------------- AAEO-14
def test_close_event_drops_content_attributes(monkeypatch) -> None:
    """Content keys are dropped from a CLOSE event's recorded attributes by default."""
    obs, fake = _enabled_obs(monkeypatch)
    obs.emit_event_span(
        RunEvent(event_type=EventType.TOOL_CALLED, trace_id="t1", tool_call_id="c1")
    )
    obs.emit_event_span(
        RunEvent(
            event_type=EventType.TOOL_COMPLETED,
            trace_id="t1",
            tool_call_id="c1",
            payload={"output": "secret result", "status_code": 200},
        )
    )
    tool_span = fake.spans[0]
    # The non-content key was recorded on close; the content key was dropped.
    assert tool_span.attributes.get("himmy.payload.status_code") == 200
    assert "himmy.payload.output" not in tool_span.attributes


def test_emit_robust_to_odd_payload(monkeypatch) -> None:
    """emit_event_span tolerates None/odd payload values without raising."""
    obs, fake = _enabled_obs(monkeypatch)
    # None payload values are skipped; non-content keys retained.
    obs.emit_event_span(
        RunEvent(
            event_type=EventType.AGENT_RUN_STARTED,
            trace_id="t1",
            payload={"keep": 1, "skip": None, "nested": {"a": 1}},
        )
    )
    span = fake.spans[0]
    assert span.attributes.get("himmy.payload.keep") == 1
    assert "himmy.payload.skip" not in span.attributes
    # Cleanly close it.
    obs.emit_event_span(
        RunEvent(event_type=EventType.AGENT_RUN_FINISHED, trace_id="t1")
    )
    assert obs._OPEN_SPANS == {}


# ------------------------------------------------------------------- no-op when off
def test_instrument_helpers_noop_when_off(monkeypatch) -> None:
    """instrument_asyncpg / instrument_fastapi are silent no-ops when disabled."""
    monkeypatch.delenv("HIMMY_LOGFIRE_ENABLED", raising=False)
    obs = _fresh_module()
    # No exception and no logfire import.
    obs.instrument_asyncpg()
    obs.instrument_fastapi(object())


def test_configure_then_emit_noop_after_disable(monkeypatch) -> None:
    """emit is a hard no-op when the switch is off even if logfire is importable."""
    monkeypatch.delenv("HIMMY_LOGFIRE_ENABLED", raising=False)
    obs = _fresh_module()
    fake = _FakeLogfire()
    monkeypatch.setitem(importlib.sys.modules, "logfire", fake)
    # Not configured + switch off -> nothing happens.
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    assert fake.spans == []
    assert obs._OPEN_SPANS == {}


# ------------------------------------------------------------------- gated real logfire
def test_configure_with_real_logfire_or_skip(monkeypatch) -> None:
    """Gated: with the real logfire installed, configure succeeds; else skip.

    Offline (logfire absent) this skips, satisfying the offline-first invariant.
    """
    try:
        import logfire  # type: ignore  # noqa: F401
    except ImportError:
        pytest.skip("logfire not installed; real-provider observability path skipped")
    monkeypatch.setenv("HIMMY_LOGFIRE_ENABLED", "1")
    obs = _fresh_module()
    # Should configure without raising when the package is genuinely present.
    obs.configure_observability()
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    obs.emit_event_span(
        RunEvent(event_type=EventType.AGENT_RUN_FINISHED, trace_id="t1")
    )
    obs.reset_spans()
