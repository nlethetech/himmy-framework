"""Deterministic, offline load/concurrency + profiling harness for Himmy.

The suite drives the in-process FastAPI app (httpx ``AsyncClient`` over an
``ASGITransport``) and the in-memory :class:`~himmy.entities.registry.EntityRegistry`
to measure concurrency behaviour, throughput, and lineage-traversal cost — with
**zero** network or API keys. Heavy profiling / Postgres-vector tests are marked
``@pytest.mark.slow`` and excluded from the default run (see the project Makefile).

Assertions are deterministic (counts / error-rates), never latency thresholds, so
the suite is CI-friendly and parallel-safe under ``pytest-xdist``.
"""

from __future__ import annotations

__all__: list[str] = []
