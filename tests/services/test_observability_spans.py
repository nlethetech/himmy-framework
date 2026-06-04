"""Tests for the observability span lifecycle + content gating (AAEO-11/14)."""

from __future__ import annotations

import importlib

from himmy.core.events import EventType, RunEvent


def _fresh_module():
    import himmy.services.observability as obs

    return importlib.reload(obs)


class _FakeSpan:
    """A minimal stand-in for a logfire span (context manager)."""

    def __init__(self, name: str, attributes: dict) -> None:
        self.name = name
        self.attributes = dict(attributes)
        self.exited = False

    def __enter__(self):
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


def test_span_attributes_drop_content_when_disabled(monkeypatch) -> None:
    """Content payload keys are dropped from span attributes by default (AAEO-14)."""
    monkeypatch.delenv("HIMMY_LOGFIRE_INCLUDE_CONTENT", raising=False)
    obs = _fresh_module()
    event = RunEvent(
        event_type=EventType.AGENT_RUN_STARTED,
        trace_id="t1",
        payload={"model_key": "default", "rendered_prompt": "secret prompt"},
    )
    attrs = obs._span_attributes(event)
    assert attrs["himmy.payload.model_key"] == "default"
    # Content key dropped.
    assert "himmy.payload.rendered_prompt" not in attrs


def test_span_attributes_include_content_when_enabled(monkeypatch) -> None:
    """When content logging is on, content keys are retained."""
    monkeypatch.setenv("HIMMY_LOGFIRE_INCLUDE_CONTENT", "true")
    obs = _fresh_module()
    event = RunEvent(
        event_type=EventType.AGENT_RUN_STARTED,
        trace_id="t1",
        payload={"rendered_prompt": "visible prompt"},
    )
    attrs = obs._span_attributes(event)
    assert attrs["himmy.payload.rendered_prompt"] == "visible prompt"


def test_span_lifecycle_nests_and_closes(monkeypatch) -> None:
    """run/tool open events start spans; paired close events end them (AAEO-11)."""
    monkeypatch.setenv("HIMMY_LOGFIRE_ENABLED", "1")
    monkeypatch.delenv("HIMMY_LOGFIRE_INCLUDE_CONTENT", raising=False)
    obs = _fresh_module()

    fake = _FakeLogfire()
    monkeypatch.setitem(importlib.sys.modules, "logfire", fake)

    obs.configure_observability()

    # Open a run span, then a nested tool span, then close both.
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    obs.emit_event_span(
        RunEvent(event_type=EventType.TOOL_CALLED, trace_id="t1", tool_call_id="c1")
    )
    obs.emit_event_span(
        RunEvent(event_type=EventType.TOOL_COMPLETED, trace_id="t1", tool_call_id="c1")
    )
    obs.emit_event_span(
        RunEvent(event_type=EventType.AGENT_RUN_FINISHED, trace_id="t1")
    )

    # Two spans were opened (run + tool), and both were exited (closed).
    assert len(fake.spans) == 2
    assert all(s.exited for s in fake.spans)
    # No dangling open spans remain.
    assert obs._OPEN_SPANS == {}


def test_non_lifecycle_event_logs(monkeypatch) -> None:
    """A non-lifecycle event becomes a point-in-time log on the current span."""
    monkeypatch.setenv("HIMMY_LOGFIRE_ENABLED", "1")
    obs = _fresh_module()
    fake = _FakeLogfire()
    monkeypatch.setitem(importlib.sys.modules, "logfire", fake)
    obs.configure_observability()
    obs.emit_event_span(
        RunEvent(event_type=EventType.CONTEXT_SNAPSHOT_BUILT, trace_id="t1")
    )
    assert any(name == "context_snapshot_built" for name, _ in fake.infos)


def test_emit_is_noop_when_disabled(monkeypatch) -> None:
    """With the switch off, span emission is a hard no-op (no logfire import)."""
    monkeypatch.delenv("HIMMY_LOGFIRE_ENABLED", raising=False)
    obs = _fresh_module()
    # Should not raise even for lifecycle events.
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED, trace_id="t1"))
    obs.emit_event_span(
        RunEvent(event_type=EventType.AGENT_RUN_FINISHED, trace_id="t1")
    )
    assert obs._OPEN_SPANS == {}
