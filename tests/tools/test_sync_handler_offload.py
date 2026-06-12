"""Sync local tool handlers must run OFF the event loop (concurrency-scale audit).

A synchronous handler that does blocking IO (the web/file/sqlite packs are sync)
used to run inline on the single shared event loop, so one slow tool froze every
concurrent run/stream/scheduler tick — and the per-tool ``wait_for`` timeout could
not even fire while the loop was blocked. These tests pin the fix: sync handlers are
offloaded via ``asyncio.to_thread`` so the loop stays responsive and the timeout
works; async handlers still run directly on the loop.
"""

from __future__ import annotations

import asyncio
import threading
import time

from himmy.services.tools import (
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_local_tool,
)
from tests.conftest import run_async


def test_sync_handler_does_not_block_the_event_loop() -> None:
    """While a slow sync tool runs, other loop coroutines keep making progress."""
    started = threading.Event()

    def slow_sync(args: dict) -> dict:
        started.set()
        time.sleep(0.4)  # blocking IO stand-in
        return {"ok": True}

    registry = ToolRegistry()
    register_local_tool(registry, name="slow", handler=slow_sync)
    service = ToolService(registry)

    async def scenario() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            # The loop must keep scheduling this while `slow` is blocked in a thread.
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        done = asyncio.Event()
        tick_task = asyncio.create_task(ticker())
        result = await service.execute(ToolInvocation(tool_name="slow", args={}))
        done.set()
        await tick_task
        assert result.outcome == "success"
        return ticks

    ticks = run_async(scenario())
    # If the sync handler had blocked the loop, the ticker would have advanced 0–1
    # times; offloaded, it advances many times across the ~0.4s sleep.
    assert ticks >= 5


def test_sync_handler_timeout_fires() -> None:
    """The per-tool timeout preempts a blocking sync handler (loop stays free).

    Because the handler is offloaded to a thread, the ``wait_for`` ceiling can fire
    and hand back a TIMEOUT result ~0.1s in — the loop is not held hostage by the
    handler's 2s ``sleep``. (Python cannot kill the orphaned worker thread, so the
    elapsed time is measured INSIDE the loop, where the timeout actually preempts;
    ``asyncio.run`` would otherwise block on executor shutdown until the thread ends,
    which is a property of threads, not of the dispatch fix.)
    """

    def hang(args: dict) -> dict:
        time.sleep(2.0)
        return {"never": True}

    registry = ToolRegistry()
    register_local_tool(registry, name="hang", handler=hang, timeout_seconds=0.1)
    service = ToolService(registry)

    async def scenario() -> tuple:
        start = time.perf_counter()
        result = await service.execute(ToolInvocation(tool_name="hang", args={}))
        return result, time.perf_counter() - start

    result, elapsed = run_async(scenario())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT
    # The timeout fired ~0.1s in — the loop did NOT wait out the 2s sleep.
    assert elapsed < 1.0


def test_two_concurrent_sync_tools_run_in_parallel() -> None:
    """Two slow sync tools overlap (run on threads) instead of serializing."""

    def slow_sync(args: dict) -> dict:
        time.sleep(0.3)
        return {"ok": True}

    registry = ToolRegistry()
    register_local_tool(registry, name="slow", handler=slow_sync)
    service = ToolService(registry)

    async def scenario() -> float:
        start = time.perf_counter()
        await asyncio.gather(
            service.execute(ToolInvocation(tool_name="slow", args={})),
            service.execute(ToolInvocation(tool_name="slow", args={})),
        )
        return time.perf_counter() - start

    elapsed = run_async(scenario())
    # Serialized on the loop they'd take ~0.6s; overlapped on threads, ~0.3s.
    assert elapsed < 0.55


def test_async_handler_still_runs_directly() -> None:
    """A coroutine handler is awaited on the loop (no behavioral change)."""
    ran_on: dict[str, int] = {}

    async def async_handler(args: dict) -> dict:
        ran_on["thread"] = threading.get_ident()
        return {"echo": args.get("v")}

    registry = ToolRegistry()
    register_local_tool(registry, name="ah", handler=async_handler)
    service = ToolService(registry)

    main_thread = threading.get_ident()
    result = run_async(
        service.execute(ToolInvocation(tool_name="ah", args={"v": 7}))
    )
    assert result.outcome == "success"
    assert result.result == {"echo": 7}
    # Async handlers are NOT offloaded — they run on the loop's own thread.
    assert ran_on["thread"] == main_thread
