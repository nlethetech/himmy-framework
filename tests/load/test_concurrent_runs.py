"""Concurrency load tests: fan out N requests against the run + lineage + list APIs.

Each test drives the in-process app with ``asyncio.gather`` over an httpx
``AsyncClient`` and asserts the **deterministic** invariant — ``error_rate == 0`` and
the expected response counts — while *recording* (never asserting) p50/p95/p99 and
throughput via :class:`~tests.load.metrics.LoadTestMetrics`. N is kept modest (≤100)
and the assertions are generous so the suite is fast and never flaky in CI.

All requests hit the offline default stack (in-memory storage + stub inference), so
no network or API key is involved.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tests.conftest import run_async
from tests.load.metrics import LoadTestMetrics

# Concurrency levels exercised by the fan-out tests. Kept modest so the offline
# stub-inference stack completes quickly and CI stays green.
CONCURRENCY_LEVELS = [10, 50, 100]

_RUN_BODY: dict[str, Any] = {
    "workspace_id": "w1",
    "subject_id": "s1",
    "persona": {"name": "Analyst", "description": "careful"},
    "task": {"title": "brief", "prompt": "Summarize ACME"},
}


async def _timed_gather(
    label: str,
    concurrency: int,
    make_request: Callable[[int], Awaitable[bool]],
) -> LoadTestMetrics:
    """Run ``concurrency`` requests concurrently and roll them up into metrics.

    ``make_request(i)`` performs one request and returns ``True`` on success; it must
    not raise (any exception is caught and counted as an error so a single bad request
    cannot abort the whole batch). Per-request wall time is measured with a perf
    counter and reported as latency; only the success/error count is asserted on.
    """
    latencies_ms: list[float] = []
    errors = 0

    async def _one(index: int) -> tuple[float | None, bool]:
        start = time.perf_counter()
        try:
            ok = await make_request(index)
        except Exception:  # noqa: BLE001 - count as an error, never abort the batch
            return None, False
        return (time.perf_counter() - start) * 1000.0, ok

    batch_start = time.perf_counter()
    results = await asyncio.gather(*(_one(i) for i in range(concurrency)))
    elapsed = time.perf_counter() - batch_start

    for latency_ms, ok in results:
        if ok and latency_ms is not None:
            latencies_ms.append(latency_ms)
        else:
            errors += 1

    return LoadTestMetrics.from_samples(
        label=label,
        concurrency=concurrency,
        latencies_ms=latencies_ms,
        error_count=errors,
        elapsed_seconds=elapsed,
    )


async def _drain_runs(client: Any, run_ids: list[str], *, timeout: float = 20.0) -> int:
    """Poll the given runs until all reach a terminal status; return the count seen.

    Mirrors the suite's ``_await_run`` polling but for a batch: background run
    execution lives on the same event loop as the client, so a short cooperative
    poll lets every queued run finish. Returns how many runs reached a terminal
    state (used only to confirm the batch drained, not as a latency assertion).
    """
    deadline = time.monotonic() + timeout
    pending = set(run_ids)
    terminal = 0
    while pending and time.monotonic() < deadline:
        for run_id in list(pending):
            body = (await client.get(f"/v1/runs/{run_id}")).json()
            if body.get("status") in ("SUCCEEDED", "FAILED"):
                pending.discard(run_id)
                terminal += 1
        if pending:
            await asyncio.sleep(0.02)
    return terminal


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_concurrent_run_creates_have_zero_errors(
    client_factory: Callable[..., Any], concurrency: int
) -> None:
    """Fanning out N concurrent POST /v1/runs yields N runs with a 0% error rate."""

    async def _body() -> LoadTestMetrics:
        async with client_factory() as client:
            created: list[str] = []

            async def _create(_: int) -> bool:
                resp = await client.post("/v1/runs", json=_RUN_BODY)
                if resp.status_code != 200:
                    return False
                created.append(resp.json()["run_id"])
                return True

            metrics = await _timed_gather(
                f"create-runs-c{concurrency}", concurrency, _create
            )
            # Drain so background tasks complete before the client/loop tears down
            # (keeps the event loop clean for xdist + avoids dangling-task warnings).
            await _drain_runs(client, created)
            assert len(created) == concurrency
            return metrics

    metrics = run_async(_body())
    print(metrics.summary_line())
    assert metrics.error_rate == 0.0
    assert metrics.total_requests == concurrency
    assert metrics.error_count == 0


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_concurrent_lineage_reads_have_zero_errors(
    client_factory: Callable[..., Any], concurrency: int
) -> None:
    """Concurrent lineage reads of completed runs all return a 200 graph, 0% errors."""

    async def _body() -> LoadTestMetrics:
        async with client_factory() as client:
            # Seed a pool of completed runs to read lineage for, round-robin.
            seed_n = min(concurrency, 10)
            seeded: list[str] = []
            for _ in range(seed_n):
                resp = await client.post("/v1/runs", json=_RUN_BODY)
                seeded.append(resp.json()["run_id"])
            await _drain_runs(client, seeded)

            async def _read_lineage(index: int) -> bool:
                run_id = seeded[index % len(seeded)]
                resp = await client.get(
                    f"/v1/runs/{run_id}/lineage", params={"workspace_id": "w1"}
                )
                if resp.status_code != 200:
                    return False
                graph = resp.json()
                return "nodes" in graph and "edges" in graph

            metrics = await _timed_gather(
                f"lineage-reads-c{concurrency}", concurrency, _read_lineage
            )
            return metrics

    metrics = run_async(_body())
    print(metrics.summary_line())
    assert metrics.error_rate == 0.0
    assert metrics.total_requests == concurrency
    assert metrics.error_count == 0


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_concurrent_list_pagination_have_zero_errors(
    client_factory: Callable[..., Any], concurrency: int
) -> None:
    """Concurrent paginated list reads all return a well-formed paged envelope."""

    async def _body() -> LoadTestMetrics:
        async with client_factory() as client:
            # Seed enough runs that pagination has multiple pages to walk.
            seeded: list[str] = []
            for _ in range(12):
                resp = await client.post("/v1/runs", json=_RUN_BODY)
                seeded.append(resp.json()["run_id"])
            await _drain_runs(client, seeded)

            page_limit = 5

            async def _list_page(index: int) -> bool:
                offset = (index % 3) * page_limit
                resp = await client.get(
                    "/v1/runs",
                    params={
                        "workspace_id": "w1",
                        "limit": page_limit,
                        "offset": offset,
                    },
                )
                if resp.status_code != 200:
                    return False
                body = resp.json()
                # Paged envelope shape (AAEO-8): items + total + bounded page size.
                return (
                    "items" in body
                    and "total" in body
                    and len(body["items"]) <= page_limit
                )

            metrics = await _timed_gather(
                f"list-pages-c{concurrency}", concurrency, _list_page
            )
            return metrics

    metrics = run_async(_body())
    print(metrics.summary_line())
    assert metrics.error_rate == 0.0
    assert metrics.total_requests == concurrency
    assert metrics.error_count == 0


def test_mixed_workload_fan_out_zero_errors(
    client_factory: Callable[..., Any],
) -> None:
    """A mixed fan-out (creates + reads + lists interleaved) keeps a 0% error rate.

    Exercises the endpoints together under concurrency to surface any shared-state
    race the per-endpoint tests would miss; the assertion stays deterministic
    (every request returns its expected status).
    """

    async def _body() -> LoadTestMetrics:
        async with client_factory() as client:
            seeded: list[str] = []
            for _ in range(6):
                resp = await client.post("/v1/runs", json=_RUN_BODY)
                seeded.append(resp.json()["run_id"])
            await _drain_runs(client, seeded)

            async def _mixed(index: int) -> bool:
                bucket = index % 3
                if bucket == 0:
                    resp = await client.post("/v1/runs", json=_RUN_BODY)
                    return bool(resp.status_code == 200)
                if bucket == 1:
                    run_id = seeded[index % len(seeded)]
                    resp = await client.get(
                        f"/v1/runs/{run_id}/lineage",
                        params={"workspace_id": "w1"},
                    )
                    return bool(resp.status_code == 200)
                resp = await client.get(
                    "/v1/runs", params={"workspace_id": "w1", "limit": 5}
                )
                return bool(resp.status_code == 200)

            metrics = await _timed_gather("mixed-workload-c30", 30, _mixed)
            # Drain any runs created by the mixed fan-out.
            listed = (
                await client.get("/v1/runs", params={"workspace_id": "w1", "limit": 50})
            ).json()
            await _drain_runs(client, [r["run_id"] for r in listed["items"]])
            return metrics

    metrics = run_async(_body())
    print(metrics.summary_line())
    assert metrics.error_rate == 0.0
    assert metrics.error_count == 0
