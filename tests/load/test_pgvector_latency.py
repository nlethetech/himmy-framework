"""pgvector / Postgres latency probe (``@pytest.mark.slow``), skipped offline.

This test only runs when a real Postgres is reachable: it requires BOTH the optional
``asyncpg`` driver (the ``[postgres]`` extra) AND ``HIMMY_DATABASE_URL`` to be set.
On the default offline path neither is present, so the test ``pytest.skip``s with a
clear, actionable reason — preserving the framework's zero-config / offline-first
guarantee (no DB is ever required to run the suite).

When it does run, it builds a Postgres-backed container, registers a synthetic graph,
and times a batch of lineage traversals (the ``WITH RECURSIVE`` path on Postgres),
recording p50/p95 via :class:`~tests.load.metrics.LoadTestMetrics`. The only
assertions are deterministic (the traversal returns the expected subgraph) — never a
latency threshold.
"""

from __future__ import annotations

import importlib.util
import os
import time

import pytest

from tests.conftest import run_async
from tests.load.fixtures import SyntheticDataFactory
from tests.load.metrics import LoadTestMetrics

pytestmark = pytest.mark.slow


def _asyncpg_available() -> bool:
    """True when the optional ``asyncpg`` driver is importable (``[postgres]`` extra)."""
    return importlib.util.find_spec("asyncpg") is not None


def _database_url() -> str | None:
    """Return ``HIMMY_DATABASE_URL`` if set + non-empty, else ``None``."""
    dsn = os.environ.get("HIMMY_DATABASE_URL")
    return dsn if dsn else None


def _require_postgres() -> str:
    """Skip cleanly unless asyncpg is importable AND a DSN is configured.

    Returns the DSN when both conditions hold; otherwise calls ``pytest.skip`` with a
    specific reason naming exactly what is missing so a developer enabling this test
    knows what to provide.
    """
    if not _asyncpg_available():
        pytest.skip(
            "pgvector latency test needs the optional 'asyncpg' driver "
            "(install the [postgres] extra: pip install -e '.[postgres]')."
        )
    dsn = _database_url()
    if dsn is None:
        pytest.skip(
            "pgvector latency test needs HIMMY_DATABASE_URL pointing at a reachable "
            "Postgres (e.g. postgresql://user:pass@localhost:5432/himmy)."
        )
    return dsn


def test_pgvector_lineage_latency() -> None:
    """Time lineage traversals against a Postgres registry; skip cleanly when offline."""
    dsn = _require_postgres()

    async def _body() -> LoadTestMetrics:
        from himmy.entities.postgres import PostgresEntityRegistry

        registry = await PostgresEntityRegistry.connect(dsn)
        try:
            await registry.create_schema()

            # Build a deterministic synthetic graph in memory, then mirror it into PG.
            graph = SyntheticDataFactory(seed=2024).build_graph(personas=5)
            for record in graph.personas + graph.prompts + graph.snapshots:
                await registry.register(record)
            for link in graph.registry._links:  # noqa: SLF001 - test mirrors edges
                await registry.link(
                    from_record_id=link.from_record_id,
                    to_record_id=link.to_record_id,
                    relation=link.relation,
                )

            latencies_ms: list[float] = []
            errors = 0
            batch_start = time.perf_counter()
            for root_id in graph.root_ids:
                start = time.perf_counter()
                traced = await registry.trace(root_id, max_depth=8)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                # Deterministic: each persona root reaches 13 nodes (1 + 4 + 4*2).
                if traced.node_count != 13:
                    errors += 1
            elapsed = time.perf_counter() - batch_start

            return LoadTestMetrics.from_samples(
                label="pgvector-lineage",
                concurrency=len(graph.root_ids),
                latencies_ms=latencies_ms,
                error_count=errors,
                elapsed_seconds=elapsed,
            )
        finally:
            closer = getattr(registry, "close", None)
            if closer is not None:
                await closer()

    metrics = run_async(_body())
    print(metrics.summary_line())
    assert metrics.error_count == 0
    assert metrics.total_requests == 5
