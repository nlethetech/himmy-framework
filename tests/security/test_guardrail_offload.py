"""Regression test for the redos-sync-offload residual (DoS amplifier).

Finding (residual of the guardrails-redos bundle): the catastrophic ReDoS in the
PII regex was fixed, but ``SingleAgentRuntime._apply_guardrail`` still called
``guardrail.inspect(...)`` SYNCHRONOUSLY on the async agent turn path. A ~2 MiB
input that takes ~1.5 s to inspect therefore pinned the event loop for the entire
inspection, starving every other coroutine on the loop — a denial-of-service
amplifier even without catastrophic backtracking.

The fix offloads the inspection with ``await asyncio.to_thread(...)`` so the work
runs on a worker thread and the event loop stays free.

This test registers a guardrail whose ``inspect`` BLOCKS (``time.sleep``) and
proves that a concurrent asyncio task keeps ticking *during* the inspection —
which is only possible if the blocking call no longer runs on the event loop.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

from himmy.runtime.single_agent import SingleAgentRuntime

# How long the guardrail blocks for. Long enough to be unambiguous, short enough
# to keep the test fast.
_BLOCK_SECONDS = 0.3


class _SleepingGuardrail:
    """A guardrail whose synchronous ``inspect`` blocks the calling thread.

    Stands in for a real guardrail processing a multi-megabyte input: the CPU/IO
    work is modelled as a blocking ``time.sleep``. Semantics are a clean pass so
    the runtime returns the text unchanged.
    """

    def inspect(self, text: str, context: dict[str, Any] | None = None) -> Any:
        time.sleep(_BLOCK_SECONDS)
        return SimpleNamespace(
            text=text,
            allowed=True,
            flags=[],
            reasons=[],
            block_replacement=False,
        )


class _EmitStub:
    """Minimal stand-in exposing the one collaborator ``_apply_guardrail`` uses."""

    async def _emit(self, event: Any) -> None:  # noqa: D401 - test stub
        return None


def test_guardrail_inspection_does_not_block_the_event_loop() -> None:
    """A blocking ``inspect`` must run off the loop so other coroutines keep running.

    A background ticker counts how many times it gets scheduled while a guardrail
    inspection that blocks for ``_BLOCK_SECONDS`` is in flight. If the inspection
    ran ON the event loop (the pre-fix synchronous call) the ticker would be
    frozen for the whole sleep and rack up ~0 ticks. With the work offloaded via
    ``asyncio.to_thread`` the ticker keeps advancing throughout.
    """
    # Bind the REAL fixed method to a lightweight stub so we exercise the exact
    # code path at single_agent.py without standing up a full runtime.
    apply = SingleAgentRuntime._apply_guardrail.__get__(_EmitStub())  # type: ignore[attr-defined]
    guardrail = _SleepingGuardrail()

    async def scenario() -> tuple[int, float, str]:
        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        ticker_task = asyncio.create_task(ticker())
        # Yield once so the ticker is actually running before we start blocking.
        await asyncio.sleep(0)

        start = time.perf_counter()
        result = await apply(
            guardrail,
            "payload",
            stage="input",
            agent_id="a",
        )
        elapsed = time.perf_counter() - start

        stop.set()
        await ticker_task
        return ticks, elapsed, result

    ticks, elapsed, result = asyncio.run(scenario())

    # The inspection really did take its full blocking duration...
    assert elapsed >= _BLOCK_SECONDS, f"inspect returned too fast: {elapsed:.3f}s"
    # ...yet the event loop kept scheduling the ticker the whole time. At 5 ms per
    # tick over a 0.3 s block we expect dozens of ticks; pre-fix (loop blocked)
    # this is ~0-1. A conservative floor proves concurrency without flakiness.
    expected_min = int((_BLOCK_SECONDS / 0.005) * 0.5)
    assert ticks >= expected_min, (
        f"event loop was blocked during inspect: only {ticks} ticks "
        f"(expected >= {expected_min}); inspection ran ON the loop"
    )
    # Semantics unchanged: a clean pass returns the text untouched.
    assert result == "payload"
