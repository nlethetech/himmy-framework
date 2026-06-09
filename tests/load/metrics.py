"""Load-test measurement primitives: typed result models + small math helpers.

These are pure, dependency-free (pydantic only) value objects + functions used by
the load/profiling tests to summarise a batch of timed requests. They carry **no**
latency assertions themselves — the tests assert on counts / error-rates (which are
deterministic) and merely *record* the percentile/throughput numbers for visibility.

* :class:`LoadTestMetrics` — the rolled-up result of one concurrent batch.
* :class:`ProfileResult` — a parsed, top-N view of a :mod:`cProfile` run.
* :func:`compute_percentile` / :func:`throughput` — the math behind the metrics.
* :func:`render_profile_top_n` — turn :mod:`pstats` into a stable text table.
"""

from __future__ import annotations

import io
import pstats
from collections.abc import Sequence
from cProfile import Profile

from pydantic import BaseModel, Field


def compute_percentile(samples: Sequence[float], pct: float) -> float:
    """Return the ``pct`` percentile (0..100) of ``samples`` by nearest-rank.

    Uses the nearest-rank method on a sorted copy: deterministic, dependency-free,
    and well-defined for tiny sample sets (no interpolation, no numpy). An empty
    sample set yields ``0.0``; ``pct`` is clamped to ``[0, 100]``.
    """
    if not samples:
        return 0.0
    pct = max(0.0, min(100.0, pct))
    ordered = sorted(samples)
    if pct <= 0:
        return float(ordered[0])
    # Nearest-rank: rank = ceil(pct/100 * N), 1-based, clamped to [1, N].
    rank = int((pct / 100.0) * len(ordered) + 0.999999)
    rank = max(1, min(len(ordered), rank))
    return float(ordered[rank - 1])


def throughput(count: int, elapsed_seconds: float) -> float:
    """Return requests-per-second for ``count`` requests over ``elapsed_seconds``.

    Returns ``0.0`` when no time elapsed (or a non-positive duration is supplied),
    so the value is always finite and safe to record.
    """
    if elapsed_seconds <= 0:
        return 0.0
    return count / elapsed_seconds


class LoadTestMetrics(BaseModel):
    """The rolled-up outcome of one fan-out batch of timed requests.

    ``latencies_ms`` is the per-request wall time (perf-counter based); the derived
    percentile / throughput fields are computed once via :meth:`from_samples` so the
    object is a stable, JSON-serialisable record of the batch. ``error_rate`` is the
    fraction of requests that did **not** return an expected status — the only field
    the tests assert on (it must be ``0.0``).
    """

    label: str = ""
    concurrency: int = 0
    total_requests: int = 0
    error_count: int = 0
    elapsed_seconds: float = 0.0
    latencies_ms: list[float] = Field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_rps: float = 0.0

    @property
    def error_rate(self) -> float:
        """Fraction of requests that failed (``0.0`` when none, or when no requests)."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    @classmethod
    def from_samples(
        cls,
        *,
        label: str,
        concurrency: int,
        latencies_ms: Sequence[float],
        error_count: int,
        elapsed_seconds: float,
    ) -> LoadTestMetrics:
        """Build a metrics record from raw per-request latencies + a batch timer.

        Computes p50/p95/p99 (nearest-rank) and throughput from the supplied samples.
        ``total_requests`` is taken from ``len(latencies_ms) + error_count`` so a
        failed request that produced no latency still counts toward the total.
        """
        latency_list = [float(x) for x in latencies_ms]
        total = len(latency_list) + error_count
        return cls(
            label=label,
            concurrency=concurrency,
            total_requests=total,
            error_count=error_count,
            elapsed_seconds=elapsed_seconds,
            latencies_ms=latency_list,
            p50_ms=compute_percentile(latency_list, 50),
            p95_ms=compute_percentile(latency_list, 95),
            p99_ms=compute_percentile(latency_list, 99),
            throughput_rps=throughput(total, elapsed_seconds),
        )

    def summary_line(self) -> str:
        """A single compact human-readable line for test output / logs."""
        return (
            f"[{self.label}] n={self.total_requests} c={self.concurrency} "
            f"err={self.error_count} ({self.error_rate:.0%}) "
            f"p50={self.p50_ms:.1f}ms p95={self.p95_ms:.1f}ms "
            f"p99={self.p99_ms:.1f}ms rps={self.throughput_rps:.0f}"
        )


class ProfileFrame(BaseModel):
    """One row of a parsed cProfile table (a single function's aggregate cost)."""

    function: str
    ncalls: int = 0
    tottime_s: float = 0.0
    cumtime_s: float = 0.0


class ProfileResult(BaseModel):
    """A parsed, top-N view of a :mod:`cProfile` run, ranked by cumulative time.

    ``frames`` holds the top functions (already truncated to ``top_n``); ``raw_text``
    is the full :mod:`pstats` dump so a human can read the long tail. This is a value
    object — it performs no profiling itself; build it via :func:`render_profile_top_n`.
    """

    label: str = ""
    top_n: int = 0
    total_calls: int = 0
    frames: list[ProfileFrame] = Field(default_factory=list)
    raw_text: str = ""

    def table(self) -> str:
        """Render the ranked frames as an aligned, deterministic text table."""
        header = f"{'cumtime':>10}  {'tottime':>10}  {'ncalls':>9}  function"
        rows = [header, "-" * len(header)]
        for frame in self.frames:
            rows.append(
                f"{frame.cumtime_s:>10.4f}  {frame.tottime_s:>10.4f}  "
                f"{frame.ncalls:>9}  {frame.function}"
            )
        return "\n".join(rows)


def render_profile_top_n(
    profile: Profile, *, label: str, top_n: int = 20
) -> ProfileResult:
    """Summarise a finished :class:`cProfile.Profile` into a top-N :class:`ProfileResult`.

    Sorts by cumulative time, captures the full :mod:`pstats` dump as ``raw_text``,
    and parses the leading rows into typed :class:`ProfileFrame`s. Parsing is best
    effort and tolerant of the (stable but textual) pstats layout — any unparsable
    row is skipped rather than raising, so a profile always yields a usable result.
    """
    buffer = io.StringIO()
    stats = pstats.Stats(profile, stream=buffer)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    raw_text = buffer.getvalue()

    frames: list[ProfileFrame] = []
    for line in raw_text.splitlines():
        frame = _parse_pstats_line(line)
        if frame is not None:
            frames.append(frame)
        if len(frames) >= top_n:
            break

    total_calls = int(getattr(stats, "total_calls", 0) or 0)
    return ProfileResult(
        label=label,
        top_n=top_n,
        total_calls=total_calls,
        frames=frames,
        raw_text=raw_text,
    )


def _parse_pstats_line(line: str) -> ProfileFrame | None:
    """Parse one pstats data row into a :class:`ProfileFrame`, or ``None`` if not a row.

    A pstats data row looks like::

        ncalls  tottime  percall  cumtime  percall filename:lineno(function)

    where ``ncalls`` may be ``"primitive/total"``. Header / blank / summary lines do
    not start with a numeric ``ncalls`` token and are rejected.
    """
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split(None, 5)
    if len(parts) < 6:
        return None
    ncalls_token = parts[0]
    # ncalls can be "123" or "123/45" (recursive) — take the total (the second).
    primary = ncalls_token.split("/")[-1]
    try:
        ncalls = int(primary)
        tottime = float(parts[1])
        cumtime = float(parts[3])
    except ValueError:
        return None
    return ProfileFrame(
        function=parts[5].strip(),
        ncalls=ncalls,
        tottime_s=tottime,
        cumtime_s=cumtime,
    )


__all__ = [
    "LoadTestMetrics",
    "ProfileFrame",
    "ProfileResult",
    "compute_percentile",
    "render_profile_top_n",
    "throughput",
]
