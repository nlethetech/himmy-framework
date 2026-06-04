"""Tests for the observability kernel: safe no-op when the switch is off."""

from __future__ import annotations

import importlib

import pytest

from himmy.core.events import EventType, RunEvent


def _fresh_module():
    """Re-import the observability module so its module-level state is reset."""
    import himmy.services.observability as obs

    return importlib.reload(obs)


def test_configure_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the switch unset, configure_observability never imports logfire."""
    monkeypatch.delenv("HIMMY_LOGFIRE_ENABLED", raising=False)
    obs = _fresh_module()
    # Should simply return without raising or importing logfire.
    obs.configure_observability()
    obs.configure_observability()  # idempotent
    # emit/instrument are safe no-ops when off.
    obs.emit_event_span(RunEvent(event_type=EventType.AGENT_RUN_STARTED))
    obs.instrument_asyncpg()


def test_emit_event_span_noop_without_configure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """emit_event_span does nothing (and never raises) when not configured."""
    monkeypatch.delenv("HIMMY_LOGFIRE_ENABLED", raising=False)
    obs = _fresh_module()
    # No exception even with a fully-populated event.
    obs.emit_event_span(
        RunEvent(
            event_type=EventType.TOOL_COMPLETED,
            trace_id="t",
            latency_ms=1.0,
            payload={"k": "v"},
        )
    )


def test_enabled_without_logfire_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled but missing the logfire package is a hard misconfiguration."""
    monkeypatch.setenv("HIMMY_LOGFIRE_ENABLED", "1")
    obs = _fresh_module()
    try:
        import logfire  # type: ignore  # noqa: F401

        has_logfire = True
    except ImportError:
        has_logfire = False

    if has_logfire:
        pytest.skip("logfire is installed; cannot assert the missing-package error")
    with pytest.raises(RuntimeError):
        obs.configure_observability()


def test_instrument_fastapi_noop_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """instrument_fastapi is a no-op when observability is disabled."""
    monkeypatch.delenv("HIMMY_LOGFIRE_ENABLED", raising=False)
    obs = _fresh_module()
    # Passing a dummy 'app' must not raise when the switch is off.
    obs.instrument_fastapi(object())
