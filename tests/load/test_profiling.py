"""Profiling tests (``@pytest.mark.slow``): where does the time go under load?

These run :mod:`cProfile` around (a) a batch of concurrent run creates against the
in-process app and (b) a deep lineage traversal over a ~500-node synthetic graph,
then print the **top-20 functions by cumulative time** via :mod:`pstats`. They are
marked ``slow`` so the default run (``-m "not slow"``) skips them; ``make test-load``
includes them.

Assertions stay deterministic: the profile must capture *some* calls and the
synthetic graph/traversal must have the expected node counts — never a latency or
cumtime threshold, which could flake on a busy CI box. The profile dump is for human
inspection only.
"""

from __future__ import annotations

from collections.abc import Callable
from cProfile import Profile
from typing import Any

import pytest

from tests.conftest import run_async
from tests.load.fixtures import SyntheticDataFactory
from tests.load.metrics import render_profile_top_n

pytestmark = pytest.mark.slow

# The synthetic graph size profiled for lineage traversal. With the default fan-out
# this resolves to a deterministic node count >= 500.
TARGET_NODES = 500

# How many concurrent creates to profile. Modest so the slow suite still finishes
# quickly while exercising the async create path under fan-out.
PROFILE_CONCURRENCY = 25


def test_profile_concurrent_creates_prints_top20(
    client_factory: Callable[..., Any],
) -> None:
    """cProfile a batch of concurrent run creates and print the top-20 by cumtime."""
    import asyncio

    async def _create_batch() -> int:
        async with client_factory() as client:
            body = {
                "workspace_id": "w1",
                "subject_id": "s1",
                "persona": {"name": "Analyst"},
                "task": {"title": "t", "prompt": "hi"},
            }
            resps = await asyncio.gather(
                *(
                    client.post("/v1/runs", json=body)
                    for _ in range(PROFILE_CONCURRENCY)
                )
            )
            created = sum(1 for r in resps if r.status_code == 200)
            # Drain the background runs so the loop is clean before profiling ends.
            run_ids = [r.json()["run_id"] for r in resps if r.status_code == 200]
            deadline_polls = 1000
            pending = set(run_ids)
            while pending and deadline_polls > 0:
                deadline_polls -= 1
                for run_id in list(pending):
                    j = (await client.get(f"/v1/runs/{run_id}")).json()
                    if j.get("status") in ("SUCCEEDED", "FAILED"):
                        pending.discard(run_id)
                if pending:
                    await asyncio.sleep(0.01)
            return created

    profiler = Profile()
    profiler.enable()
    created = run_async(_create_batch())
    profiler.disable()

    result = render_profile_top_n(profiler, label="concurrent-creates", top_n=20)
    print("\n=== PROFILE: concurrent run creates (top 20 by cumtime) ===")
    print(result.table())

    assert created == PROFILE_CONCURRENCY
    assert result.total_calls > 0
    assert len(result.frames) > 0


def test_profile_lineage_traversal_500_nodes(tmp_path: Any) -> None:
    """cProfile a deep lineage traversal over a >=500-node synthetic DAG; print top-20.

    ``tmp_path`` is wired in so any artifact (the raw pstats dump) is written under
    the test's temp dir, never the repo root.
    """
    factory = SyntheticDataFactory(seed=4242)
    graph = factory.build_graph_of_size(TARGET_NODES)
    assert graph.node_count >= TARGET_NODES

    registry = graph.registry
    roots = graph.root_ids

    def _traverse_all() -> int:
        total_nodes = 0
        # Walk the lineage from every persona root deep enough to reach snapshots.
        for root_id in roots:
            traced = registry.trace(root_id, max_depth=8)
            total_nodes += traced.node_count
        return total_nodes

    profiler = Profile()
    profiler.enable()
    reached = _traverse_all()
    profiler.disable()

    result = render_profile_top_n(profiler, label="lineage-traversal", top_n=20)
    print(
        f"\n=== PROFILE: lineage traversal over {graph.node_count} nodes "
        f"(top 20 by cumtime) ==="
    )
    print(result.table())

    # Persist the raw profile text for human inspection (tmp_path, not repo root).
    artifact = tmp_path / "lineage_profile.txt"
    artifact.write_text(result.raw_text, encoding="utf-8")
    assert artifact.is_file()

    # Deterministic structural assertions: every root reaches its whole subtree.
    # Each persona owns prompts_per_persona prompts, each with snapshots_per_prompt
    # snapshots, so a downward walk reaches 1 + 4 + 4*2 = 13 nodes per root.
    assert reached == len(roots) * 13
    assert result.total_calls > 0
    assert len(result.frames) > 0


def test_profile_result_is_deterministic_in_shape() -> None:
    """Two identical seeded graphs yield the same node/edge counts (determinism check)."""
    a = SyntheticDataFactory(seed=99).build_graph(personas=5)
    b = SyntheticDataFactory(seed=99).build_graph(personas=5)
    assert a.node_count == b.node_count
    assert a.edge_count == b.edge_count
    assert a.root_ids == b.root_ids
