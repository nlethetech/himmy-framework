"""Prometheus metrics: a tiny, dependency-free instrument registry + ASGI hook.

This module makes the FastAPI app operable in production by exposing request
counts, latencies, and in-flight concurrency in the Prometheus text exposition
format at ``GET /metrics`` — with **no new hard dependency** and **no change to
the zero-config/offline path** (collecting a few in-process counters costs
nothing and never touches the network).

Design
------
* The metric primitives (:class:`Counter`, :class:`Histogram`, :class:`Gauge`)
  are hand-rolled and self-sufficient: they accumulate in plain dicts and render
  the exposition text themselves. The format is simple and stable, so we own it
  rather than taking a dependency for it.
* If ``prometheus_client`` *is* installed (it lives under the existing
  ``observability`` optional extra), nothing here changes — we deliberately keep
  one in-process registry so behavior is identical with or without the extra and
  the test suite exercises the real code path. The extra remains useful for the
  Logfire/OTel exporters wired in :mod:`himmy.services.observability`.

Cardinality safety
-------------------
Labels are bounded by construction. HTTP request labels are
``method`` (a small fixed verb set), ``route`` (the *route template*, e.g.
``/v1/runs/{run_id}`` — never the filled-in path, so an unbounded id space
cannot explode series), and ``status`` (the status *class*: ``2xx``/``4xx``/...).
Unmatched paths collapse to the literal ``<unmatched>`` template. No secret,
header, query string, or raw path parameter is ever used as a label.

The service-layer instruments (inference / tokens / cost / tools / guardrails /
run outcomes, plus the dispatcher gauges) are fed by :class:`MetricsEventSink`,
which subscribes to the existing ``RunEvent`` stream. Those instruments are
label-FREE or carry only tiny closed enumerations (``stage`` ∈ {input,output},
``status`` ∈ a fixed lifecycle set) — **never** a run id, request id, tool name,
user id, or any free-form string — so they can never explode the series space.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import Response

    from himmy.core.events import RunEvent

logger = logging.getLogger("himmy.observability.metrics")

# Histogram buckets (seconds), Prometheus client defaults — covers sub-ms tool
# calls through multi-second agent turns. ``+Inf`` is appended at render time.
_DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

# Token-count histogram buckets: a single prompt/completion spans roughly 1 token
# to a few hundred thousand, so use a coarse log-ish ladder rather than the
# seconds-oriented latency buckets. ``+Inf`` is appended at render time.
_TOKEN_BUCKETS: tuple[float, ...] = (
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
    262144.0,
)

# Closed lifecycle statuses an agent run can finish with. A status outside this
# frozen set collapses to ``other`` so a forged/garbage payload value cannot
# create an unbounded ``status`` label space.
_KNOWN_RUN_STATUSES = frozenset(
    {"success", "failed", "cancelled", "parked", "awaiting_approval", "resolving"}
)

# A frozen allow-list of HTTP methods so a forged/garbage verb cannot create an
# unbounded ``method`` label space. Anything else collapses to ``OTHER``.
_KNOWN_METHODS = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE"}
)


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition spec."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _render_labels(label_names: tuple[str, ...], label_values: tuple[str, ...]) -> str:
    """Render ``{k="v",...}`` (empty string when there are no labels)."""
    if not label_names:
        return ""
    parts = [
        f'{name}="{_escape_label_value(value)}"'
        for name, value in zip(label_names, label_values, strict=True)
    ]
    return "{" + ",".join(parts) + "}"


class Counter:
    """A monotonically-increasing per-label-set counter."""

    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, labels: tuple[str, ...] = (), amount: float = 1.0) -> None:
        """Increment the series for ``labels`` by ``amount`` (default 1)."""
        with self._lock:
            self._values[labels] = self._values.get(labels, 0.0) + amount

    def value(self, labels: tuple[str, ...] = ()) -> float:
        """Read the current value of one series (0.0 if never incremented)."""
        with self._lock:
            return self._values.get(labels, 0.0)

    def render(self) -> list[str]:
        """Render this counter as Prometheus exposition lines."""
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            items = sorted(self._values.items())
        for label_values, value in items:
            lines.append(
                f"{self.name}{_render_labels(self.labelnames, label_values)} "
                f"{_format_float(value)}"
            )
        return lines


class Gauge:
    """A value that can go up or down per label set (e.g. in-flight requests)."""

    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, labels: tuple[str, ...] = (), amount: float = 1.0) -> None:
        """Add ``amount`` to the series for ``labels``."""
        with self._lock:
            self._values[labels] = self._values.get(labels, 0.0) + amount

    def dec(self, labels: tuple[str, ...] = (), amount: float = 1.0) -> None:
        """Subtract ``amount`` from the series for ``labels``."""
        self.inc(labels, -amount)

    def value(self, labels: tuple[str, ...] = ()) -> float:
        """Read the current value of one series."""
        with self._lock:
            return self._values.get(labels, 0.0)

    def render(self) -> list[str]:
        """Render this gauge as Prometheus exposition lines."""
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            items = sorted(self._values.items())
        for label_values, value in items:
            lines.append(
                f"{self.name}{_render_labels(self.labelnames, label_values)} "
                f"{_format_float(value)}"
            )
        return lines


class Histogram:
    """A cumulative histogram (``_bucket``/``_sum``/``_count``) per label set."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = _DEFAULT_BUCKETS,
    ):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self.buckets = tuple(sorted(buckets))
        # Per label set: (cumulative bucket counts, sum, total count).
        self._buckets: dict[tuple[str, ...], list[float]] = {}
        self._sum: dict[tuple[str, ...], float] = {}
        self._count: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, labels: tuple[str, ...] = ()) -> None:
        """Record one observation of ``value`` for ``labels``."""
        with self._lock:
            counts = self._buckets.get(labels)
            if counts is None:
                counts = [0.0] * len(self.buckets)
                self._buckets[labels] = counts
            for i, upper in enumerate(self.buckets):
                if value <= upper:
                    counts[i] += 1.0
            self._sum[labels] = self._sum.get(labels, 0.0) + value
            self._count[labels] = self._count.get(labels, 0.0) + 1.0

    def count(self, labels: tuple[str, ...] = ()) -> float:
        """Total number of observations recorded for one series."""
        with self._lock:
            return self._count.get(labels, 0.0)

    def render(self) -> list[str]:
        """Render this histogram as Prometheus exposition lines."""
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            keys = sorted(self._buckets)
            snapshot = {
                k: (list(self._buckets[k]), self._sum.get(k, 0.0), self._count.get(k, 0.0))
                for k in keys
            }
        for label_values in keys:
            counts, total_sum, total_count = snapshot[label_values]
            for upper, cum in zip(self.buckets, counts, strict=True):
                le_labels = self.labelnames + ("le",)
                le_values = label_values + (_format_bucket_bound(upper),)
                lines.append(
                    f"{self.name}_bucket{_render_labels(le_labels, le_values)} "
                    f"{_format_float(cum)}"
                )
            inf_labels = self.labelnames + ("le",)
            inf_values = label_values + ("+Inf",)
            lines.append(
                f"{self.name}_bucket{_render_labels(inf_labels, inf_values)} "
                f"{_format_float(total_count)}"
            )
            lbl = _render_labels(self.labelnames, label_values)
            lines.append(f"{self.name}_sum{lbl} {_format_float(total_sum)}")
            lines.append(f"{self.name}_count{lbl} {_format_float(total_count)}")
        return lines


def _format_float(value: float) -> str:
    """Render a float the way Prometheus expects (ints stay int-looking)."""
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value):
        return str(int(value))
    return repr(value)


def _format_bucket_bound(value: float) -> str:
    """Render a bucket upper bound for the ``le`` label."""
    return _format_float(value)


class MetricsRegistry:
    """Holds the app's instruments and renders the full exposition document."""

    def __init__(self) -> None:
        self.http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests, by method, route template, and status class.",
            ("method", "route", "status"),
        )
        self.http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds, by method and route template.",
            ("method", "route"),
        )
        self.http_requests_in_flight = Gauge(
            "http_requests_in_flight",
            "Number of HTTP requests currently being served.",
        )

        # --- service-layer instruments (fed by MetricsEventSink) --------------
        # Inference: calls, failures, retries (label-free; bounded by construction).
        self.inference_requests_total = Counter(
            "himmy_inference_requests_total",
            "Total inference requests started, by streamed/non-streamed.",
            ("streamed",),
        )
        self.inference_failures_total = Counter(
            "himmy_inference_failures_total",
            "Total inference requests that ended FAILED.",
        )
        self.inference_retries_total = Counter(
            "himmy_inference_retries_total",
            "Total inference attempts retried after a transient failure.",
        )
        self.inference_duration_seconds = Histogram(
            "himmy_inference_duration_seconds",
            "Inference latency in seconds (terminal events only).",
        )
        # Token burn + estimated cost (counters; no per-run/user labels).
        self.inference_input_tokens_total = Counter(
            "himmy_inference_input_tokens_total",
            "Cumulative prompt (input) tokens consumed across inferences.",
        )
        self.inference_output_tokens_total = Counter(
            "himmy_inference_output_tokens_total",
            "Cumulative completion (output) tokens produced across inferences.",
        )
        self.inference_cost_usd_total = Counter(
            "himmy_inference_cost_usd_total",
            "Cumulative estimated inference cost in USD.",
        )
        self.inference_tokens = Histogram(
            "himmy_inference_tokens",
            "Distribution of total tokens (input+output) per inference.",
            buckets=_TOKEN_BUCKETS,
        )
        # Tools + guardrails.
        self.tool_calls_total = Counter(
            "himmy_tool_calls_total",
            "Total tool invocations, by outcome (called/completed/failed).",
            ("outcome",),
        )
        self.guardrail_blocks_total = Counter(
            "himmy_guardrail_blocks_total",
            "Total guardrail applications, by stage and action (blocked/redacted).",
            ("stage", "action"),
        )
        # Run outcomes (status is clamped to a closed lifecycle set).
        self.agent_runs_total = Counter(
            "himmy_agent_runs_total",
            "Total agent runs started.",
        )
        self.agent_run_outcomes_total = Counter(
            "himmy_agent_run_outcomes_total",
            "Total agent runs finished, by terminal status.",
            ("status",),
        )

        # --- dispatcher instruments (set directly by the dispatcher loop) -----
        self.dispatcher_in_flight = Gauge(
            "himmy_dispatcher_in_flight_runs",
            "Number of claimed runs currently executing on dispatcher workers.",
        )
        self.dispatcher_claims_total = Counter(
            "himmy_dispatcher_claims_total",
            "Total runs claimed off the leased queue by this dispatcher.",
        )
        self.dispatcher_reaps_total = Counter(
            "himmy_dispatcher_reaps_total",
            "Total expired-lease runs re-queued by the reaper (crashed-worker recovery).",
        )
        self.dispatcher_run_duration_seconds = Histogram(
            "himmy_dispatcher_run_duration_seconds",
            "Wall-clock latency of a dispatched run on its worker, in seconds.",
        )

    def render(self) -> str:
        """Render the whole registry as a Prometheus exposition document."""
        lines: list[str] = []
        for instrument in (
            self.http_requests_total,
            self.http_request_duration_seconds,
            self.http_requests_in_flight,
            self.inference_requests_total,
            self.inference_failures_total,
            self.inference_retries_total,
            self.inference_duration_seconds,
            self.inference_input_tokens_total,
            self.inference_output_tokens_total,
            self.inference_cost_usd_total,
            self.inference_tokens,
            self.tool_calls_total,
            self.guardrail_blocks_total,
            self.agent_runs_total,
            self.agent_run_outcomes_total,
            self.dispatcher_in_flight,
            self.dispatcher_claims_total,
            self.dispatcher_reaps_total,
            self.dispatcher_run_duration_seconds,
        ):
            lines.extend(instrument.render())
        # The exposition format requires a trailing newline.
        return "\n".join(lines) + "\n"


# Process-wide registry. One per process keeps series cumulative across requests
# and matches how a Prometheus scrape expects a target to behave.
_REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry."""
    return _REGISTRY


def get_metrics_sink() -> MetricsEventSink:
    """Return the process-wide :class:`MetricsEventSink` (resolves the live registry)."""
    return _METRICS_SINK


CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


def _status_class(status_code: int) -> str:
    """Bucket a status code into its class (``2xx``/``4xx``/...) to bound cardinality."""
    return f"{status_code // 100}xx"


def _normalize_method(method: str) -> str:
    """Clamp the HTTP method to the known verb set (else ``OTHER``)."""
    upper = method.upper()
    return upper if upper in _KNOWN_METHODS else "OTHER"


def route_template(request: Request) -> str:
    """Resolve the matched route *template* for ``request`` (never raw params).

    Starlette stores the matched ``Route`` on ``request.scope['route']`` after
    routing; its ``path_format`` (e.g. ``/v1/runs/{run_id}``) is the low-cardinality
    label we want. Falls back to ``<unmatched>`` when no route matched (404s on
    unknown paths) so an attacker probing random URLs cannot inflate series.
    """
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None) or getattr(route, "path", None)
    if isinstance(path_format, str) and path_format:
        return path_format
    return "<unmatched>"


def install_metrics(app: FastAPI) -> None:
    """Wire request metrics collection + the ``GET /metrics`` endpoint onto ``app``.

    Adds a lightweight ASGI middleware that, per request, increments the in-flight
    gauge, times the handler, and records the request counter + duration histogram
    keyed by **low-cardinality** labels (method, route template, status class).
    Registered after the other middleware so it is the outermost layer and observes
    every request (including ones short-circuited by inner guards). The endpoint
    itself exposes no secrets and is excluded from the OpenAPI schema. This is a
    pure in-process add: the zero-config/offline surface is byte-unchanged.
    """
    import time

    from starlette.responses import PlainTextResponse

    registry = _REGISTRY

    @app.middleware("http")
    async def _collect_metrics(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        registry.http_requests_in_flight.inc()
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            registry.http_requests_in_flight.dec()
            method = _normalize_method(request.method)
            template = route_template(request)
            registry.http_requests_total.inc(
                (method, template, _status_class(status_code))
            )
            registry.http_request_duration_seconds.observe(
                elapsed, (method, template)
            )

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus scrape endpoint (text exposition; no secrets, low cardinality)."""
        return PlainTextResponse(
            registry.render(), media_type=CONTENT_TYPE_LATEST
        )


def reset_registry() -> None:
    """Replace the process-wide registry with a fresh one (test helper)."""
    global _REGISTRY
    _REGISTRY = MetricsRegistry()


def _clamp_run_status(status: object) -> str:
    """Clamp a run terminal status to the closed lifecycle set (bound cardinality)."""
    text = str(status or "").strip().lower()
    return text if text in _KNOWN_RUN_STATUSES else "other"


def _as_float(value: object) -> float:
    """Best-effort numeric coercion of a payload value (0.0 on non-numeric/None)."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


class MetricsEventSink:
    """An :class:`~himmy.core.events.EventSink` that fans the run-event stream into
    bounded-cardinality Prometheus instruments.

    This is the bridge between the lifecycle events the framework already emits
    (inference, tools, guardrails, run outcomes) and the operator-facing
    ``/metrics`` scrape. It is composed alongside the durable/trace sinks (it does
    not replace them), translates each event into counter/histogram updates on the
    process-wide :class:`MetricsRegistry`, and — crucially — uses **no per-run-id,
    per-user, or free-form labels**, so the series space stays bounded no matter how
    many runs flow through. It never raises: observability must never break a run.
    """

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        """Bind to ``registry`` (defaults to the process-wide one at call time)."""
        self._registry = registry

    @property
    def registry(self) -> MetricsRegistry:
        """The registry this sink writes to (resolved lazily so test resets apply)."""
        return self._registry or get_registry()

    async def append_event(self, event: RunEvent) -> None:
        """Translate one :class:`RunEvent` into metric updates (best-effort)."""
        try:
            self._record(event)
        except Exception:  # noqa: BLE001 - metrics must never break a run
            logger.warning("metrics event sink failed to record event", exc_info=True)

    def _record(self, event: RunEvent) -> None:
        reg = self.registry
        name = event.event_type.value
        payload = event.payload or {}
        if name == "INFERENCE_REQUESTED":
            streamed = "true" if payload.get("streamed") else "false"
            reg.inference_requests_total.inc((streamed,))
        elif name == "INFERENCE_SUCCEEDED":
            self._record_inference_usage(reg, event, payload)
        elif name == "INFERENCE_FAILED":
            reg.inference_failures_total.inc()
            if event.latency_ms is not None:
                reg.inference_duration_seconds.observe(event.latency_ms / 1000.0)
        elif name == "TOOL_CALLED":
            reg.tool_calls_total.inc(("called",))
        elif name == "TOOL_COMPLETED":
            reg.tool_calls_total.inc(("completed",))
        elif name == "TOOL_FAILED":
            reg.tool_calls_total.inc(("failed",))
        elif name == "GUARDRAIL_APPLIED":
            self._record_guardrail(reg, payload)
        elif name == "AGENT_RUN_STARTED":
            reg.agent_runs_total.inc()
        elif name == "AGENT_RUN_FINISHED":
            reg.agent_run_outcomes_total.inc((_clamp_run_status(payload.get("status")),))

    @staticmethod
    def _record_inference_usage(
        reg: MetricsRegistry, event: RunEvent, payload: dict[str, object]
    ) -> None:
        if event.latency_ms is not None:
            reg.inference_duration_seconds.observe(event.latency_ms / 1000.0)
        if event.cost:
            reg.inference_cost_usd_total.inc((), float(event.cost))
        input_tokens = _as_float(payload.get("input_tokens"))
        output_tokens = _as_float(payload.get("output_tokens"))
        if input_tokens:
            reg.inference_input_tokens_total.inc((), input_tokens)
        if output_tokens:
            reg.inference_output_tokens_total.inc((), output_tokens)
        reg.inference_tokens.observe(input_tokens + output_tokens)

    @staticmethod
    def _record_guardrail(reg: MetricsRegistry, payload: dict[str, object]) -> None:
        stage = "input" if payload.get("stage") == "input" else "output"
        if payload.get("blocked"):
            reg.guardrail_blocks_total.inc((stage, "blocked"))
        if payload.get("redacted"):
            reg.guardrail_blocks_total.inc((stage, "redacted"))


# Process-wide sink. It holds no registry of its own (``registry`` resolves the
# live process-wide one each call), so ``reset_registry()`` in tests is honored
# without re-creating the sink.
_METRICS_SINK = MetricsEventSink()


__all__ = [
    "CONTENT_TYPE_LATEST",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsEventSink",
    "MetricsRegistry",
    "get_metrics_sink",
    "get_registry",
    "install_metrics",
    "reset_registry",
    "route_template",
]
