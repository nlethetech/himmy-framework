"""Observability kernel: opt-in Logfire/OpenTelemetry tracing (off by default).

When ``HIMMY_LOGFIRE_ENABLED`` is unset or falsy this entire module is a
no-op and ``logfire`` is never imported, keeping the dependency fully optional.
The only hard error is an explicit misconfiguration: the switch is on but the
``logfire`` package is missing.

Span lifecycle (AAEO-11/14): ``emit_event_span`` mirrors the run-event stream
into a real Logfire trace tree using ``logfire.span(...)`` context managers, not
flat ``logfire.info`` records. ``AGENT_RUN_STARTED`` opens a run span keyed by
``trace_id``; ``TOOL_CALLED`` opens a child tool span nested under it; the paired
``*_COMPLETED``/``*_FINISHED`` / ``*_FAILED`` events close them with timing and a
status. Content keys (rendered_prompt/output/etc.) are dropped from span
attributes unless ``HIMMY_LOGFIRE_INCLUDE_CONTENT`` is truthy.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# Values (case-insensitive) that count as "on" for boolean env switches.
_TRUTHY = {"1", "true", "yes", "on"}

# Module-level idempotency flag so configure_observability() is safe to call
# from every entry point without re-importing or re-registering anything.
_CONFIGURED = False

# Open spans keyed by their correlation id, so a STARTED event's span can be
# closed by the paired FINISHED/COMPLETED/FAILED event. Each value is the
# context-manager object returned by ``logfire.span(...).__enter__()``.
_OPEN_SPANS: dict[str, Any] = {}
_SPANS_LOCK = threading.Lock()

# Payload keys considered "content" (prompt/completion text) that must be dropped
# from span attributes when content logging is disabled (PII safety).
_CONTENT_KEYS = frozenset(
    {
        "rendered_prompt",
        "prompt",
        "completion",
        "output",
        "output_text",
        "output_structured",
        "content",
        "tool_args",
        "messages",
        "retrieval_ctx",
    }
)

# RunEvent type names that OPEN a span (lifecycle starts).
_SPAN_OPEN = {"AGENT_RUN_STARTED", "TOOL_CALLED", "WORKFLOW_STARTED"}
# RunEvent type names that CLOSE a span, mapped to the open event they pair with.
_SPAN_CLOSE = {
    "AGENT_RUN_FINISHED": "AGENT_RUN_STARTED",
    "TOOL_COMPLETED": "TOOL_CALLED",
    "TOOL_FAILED": "TOOL_CALLED",
    "WORKFLOW_FINISHED": "WORKFLOW_STARTED",
}


def _env_flag(name: str) -> bool:
    """Return True when env var ``name`` is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _logfire_enabled() -> bool:
    """Whether the observability master switch is on."""
    return _env_flag("HIMMY_LOGFIRE_ENABLED")


def _include_content() -> bool:
    """Whether prompt/completion content may be attached to spans (default off)."""
    return _env_flag("HIMMY_LOGFIRE_INCLUDE_CONTENT")


def configure_observability() -> None:
    """Configure Logfire when enabled; idempotent no-op when off.

    No-op (and never imports ``logfire``) unless ``HIMMY_LOGFIRE_ENABLED`` is
    truthy. When enabled but the ``logfire`` package is missing, raises
    :class:`RuntimeError` — that is a hard misconfiguration, not a soft failure.
    Safe to call repeatedly.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    if not _logfire_enabled():
        return

    try:
        import logfire  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires the [observability] extra
        raise RuntimeError(
            "HIMMY_LOGFIRE_ENABLED is set but the 'logfire' package is not "
            "installed; install the observability extra: "
            "pip install 'himmy[observability]'"
        ) from exc

    service_name = os.environ.get("HIMMY_LOGFIRE_SERVICE_NAME", "himmy")
    include_content = _include_content()
    try:  # pragma: no cover - requires logfire
        logfire.configure(service_name=service_name)
        logfire.instrument_pydantic_ai()
    except Exception:  # pragma: no cover - tolerate partial logfire setups
        pass
    # Stash content policy for span emitters to honor.
    os.environ.setdefault(
        "HIMMY_LOGFIRE_INCLUDE_CONTENT", "true" if include_content else "false"
    )
    _CONFIGURED = True


def _span_key(open_type: str, event: Any) -> str:
    """Stable correlation key for matching a CLOSE event to its OPEN span."""
    trace_id = getattr(event, "trace_id", None) or ""
    if open_type == "TOOL_CALLED":
        return f"tool:{trace_id}:{getattr(event, 'tool_call_id', None) or ''}"
    if open_type == "WORKFLOW_STARTED":
        return f"workflow:{trace_id}"
    return f"run:{trace_id}"


def _span_attributes(event: Any) -> dict[str, Any]:
    """Build the span attribute dict, dropping content keys when disabled."""
    include = _include_content()
    attributes: dict[str, Any] = {}
    for label, value in (
        ("himmy.trace_id", getattr(event, "trace_id", None)),
        ("himmy.thread_id", getattr(event, "thread_id", None)),
        ("himmy.agent_id", getattr(event, "agent_id", None)),
        ("himmy.request_id", getattr(event, "request_id", None)),
        ("himmy.tool_call_id", getattr(event, "tool_call_id", None)),
        ("himmy.latency_ms", getattr(event, "latency_ms", None)),
        ("himmy.cost", getattr(event, "cost", None)),
        ("himmy.error", getattr(event, "error", None)),
    ):
        if value is not None:
            attributes[label] = value
    payload = getattr(event, "payload", {}) or {}
    for key, value in payload.items():
        if value is None:
            continue
        if not include and key in _CONTENT_KEYS:
            continue
        attributes[f"himmy.payload.{key}"] = value
    return attributes


def emit_event_span(event: Any) -> None:
    """Emit a ``RunEvent`` into the Logfire trace tree; safe no-op when off (AAEO-11/14).

    OPEN events (``AGENT_RUN_STARTED``/``TOOL_CALLED``/``WORKFLOW_STARTED``) start
    a context-managed ``logfire.span`` recorded by correlation key; the paired
    CLOSE event ends that span (so tool spans nest under the run span and carry
    real timing). Other events are recorded as point-in-time logs on the current
    span. Content payload keys are dropped unless
    ``HIMMY_LOGFIRE_INCLUDE_CONTENT`` is on. Never raises.
    """
    if not _CONFIGURED or not _logfire_enabled():
        return
    try:  # pragma: no cover - requires logfire
        import logfire  # type: ignore

        event_type = getattr(event, "event_type", None)
        type_name = getattr(event_type, "name", None) or str(event_type)
        span_name = (
            getattr(event_type, "value", str(event_type)) or str(event_type)
        ).lower()
        attributes = _span_attributes(event)

        if type_name in _SPAN_OPEN:
            key = _span_key(type_name, event)
            span_cm = logfire.span(span_name, **attributes)
            span = span_cm.__enter__()
            with _SPANS_LOCK:
                # Close any stale span under the same key before replacing it.
                old = _OPEN_SPANS.pop(key, None)
                _OPEN_SPANS[key] = (span_cm, span)
            if old is not None:
                _safe_exit(old[0])
            return

        if type_name in _SPAN_CLOSE:
            open_type = _SPAN_CLOSE[type_name]
            key = _span_key(open_type, event)
            with _SPANS_LOCK:
                entry = _OPEN_SPANS.pop(key, None)
            if entry is not None:
                span_cm, span = entry
                # Record close attributes on the span before exiting it.
                _record_on_span(span, span_name, attributes)
                _safe_exit(span_cm)
            else:
                # No matching open span (e.g. partial run); record as a log.
                logfire.info(span_name, **attributes)
            return

        # Non-lifecycle events become point-in-time logs nested under the
        # currently-open span (logfire tracks the active span via contextvars).
        logfire.info(span_name, **attributes)
    except Exception:  # pragma: no cover - never let telemetry break the run
        pass


def _record_on_span(span: Any, name: str, attributes: dict[str, Any]) -> None:
    """Attach close-time attributes/message to an open span (best-effort)."""
    try:  # pragma: no cover - requires logfire
        setter = getattr(span, "set_attribute", None)
        if setter is not None:
            for key, value in attributes.items():
                setter(key, value)
        message_setter = getattr(span, "message", None)
        if message_setter is not None and isinstance(message_setter, str):
            pass
    except Exception:  # pragma: no cover - defensive
        pass


def _safe_exit(span_cm: Any) -> None:
    """Exit a span context manager, swallowing any teardown error."""
    try:  # pragma: no cover - requires logfire
        span_cm.__exit__(None, None, None)
    except Exception:  # pragma: no cover - defensive
        pass


def reset_spans() -> None:
    """Close + clear any open spans (test/teardown helper)."""
    with _SPANS_LOCK:
        entries = list(_OPEN_SPANS.values())
        _OPEN_SPANS.clear()
    for entry in entries:
        _safe_exit(entry[0])


def instrument_fastapi(app: Any) -> None:
    """Instrument a FastAPI app when the observability sub-extra is available.

    Soft-degrades with a one-line warning when ``HIMMY_LOGFIRE_ENABLED`` is on
    but the ``logfire[fastapi]`` transport is missing; full no-op when off.
    """
    if not _logfire_enabled():
        return
    try:  # pragma: no cover - requires logfire[fastapi]
        import logfire  # type: ignore

        logfire.instrument_fastapi(app)
    except Exception:  # pragma: no cover - soft-degrade
        print(
            "[himmy.observability] FastAPI instrumentation unavailable "
            "(install 'logfire[fastapi]'); continuing without it."
        )


def instrument_asyncpg() -> None:
    """Instrument asyncpg when the observability sub-extra is available.

    Soft-degrades with a one-line warning when ``HIMMY_LOGFIRE_ENABLED`` is on
    but the ``logfire[asyncpg]`` transport is missing; full no-op when off.
    """
    if not _logfire_enabled():
        return
    try:  # pragma: no cover - requires logfire[asyncpg]
        import logfire  # type: ignore

        logfire.instrument_asyncpg()
    except Exception:  # pragma: no cover - soft-degrade
        print(
            "[himmy.observability] asyncpg instrumentation unavailable "
            "(install 'logfire[asyncpg]'); continuing without it."
        )


__all__ = [
    "configure_observability",
    "emit_event_span",
    "instrument_fastapi",
    "instrument_asyncpg",
    "reset_spans",
]
