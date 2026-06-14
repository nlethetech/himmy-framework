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
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import Response

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

    def render(self) -> str:
        """Render the whole registry as a Prometheus exposition document."""
        lines: list[str] = []
        lines.extend(self.http_requests_total.render())
        lines.extend(self.http_request_duration_seconds.render())
        lines.extend(self.http_requests_in_flight.render())
        # The exposition format requires a trailing newline.
        return "\n".join(lines) + "\n"


# Process-wide registry. One per process keeps series cumulative across requests
# and matches how a Prometheus scrape expects a target to behave.
_REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry."""
    return _REGISTRY


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


__all__ = [
    "CONTENT_TYPE_LATEST",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "get_registry",
    "install_metrics",
    "reset_registry",
    "route_template",
]
