"""Parallel read-only tool fan-out (eff-p1: parallel-read-tools).

``_execute_tool_calls`` runs independent READ-ONLY calls concurrently under a bounded
semaphore (``HIMMY_TOOL_PARALLELISM``) while keeping write/side-effecting/ambiguous
tools strictly sequential — results stay in original order, per-call errors stay
isolated, and the authz/approval gate still fires per call. These tests lock the
behaviour + output invariants (concurrency is a latency win, not a semantics change).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from himmy.services.inference.local import _execute_tool_calls
from himmy.services.inference.models import (
    BoundTool,
    ToolCallRecord,
    ToolReturnRecord,
)


def _read_tool(name: str) -> BoundTool:
    return BoundTool(name=name, description=f"{name} tool", read_only=True)


def _write_tool(name: str) -> BoundTool:
    return BoundTool(name=name, description=f"{name} tool", read_only=False)


def _call(name: str, **args: object) -> ToolCallRecord:
    return ToolCallRecord(tool_call_id=f"tc_{name}", tool_name=name, args=dict(args))


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ------------------------------------------------------------ concurrency win


def test_read_only_calls_run_concurrently() -> None:
    """Three sleepy read-only calls finish in ~1 slice, not the serial sum."""
    delay = 0.15

    async def handler(name: str, args: dict) -> ToolReturnRecord:
        await asyncio.sleep(delay)
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        return await handler(name, args)

    tools = [_read_tool("a"), _read_tool("b"), _read_tool("c")]
    calls = [_call("a"), _call("b"), _call("c")]

    started = time.perf_counter()
    returns = _run(_execute_tool_calls(tools, calls, executor))
    elapsed = time.perf_counter() - started

    assert [r.content for r in returns] == ["a", "b", "c"]  # original order
    # Concurrent: wall-clock well under the 3x serial sum (0.45s); allow headroom.
    assert elapsed < delay * 3 * 0.8


# -------------------------------------------------------------- original order


def test_results_returned_in_original_order_despite_finish_order() -> None:
    """A fast-then-slow completion order must NOT reorder the returns."""
    delays = {"a": 0.20, "b": 0.01, "c": 0.10}

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        await asyncio.sleep(delays[name])
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [_read_tool("a"), _read_tool("b"), _read_tool("c")]
    calls = [_call("a"), _call("b"), _call("c")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["a", "b", "c"]
    assert [r.tool_call_id for r in returns] == ["tc_a", "tc_b", "tc_c"]


# ----------------------------------------------------- write stays sequential


def test_write_tool_stays_sequential_and_ordered() -> None:
    """A write tool in the batch executes strictly in order (never concurrently)."""
    events: list[str] = []
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        events.append(f"start:{name}")
        await asyncio.sleep(0.02)
        events.append(f"end:{name}")
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    # write between two reads.
    tools = [_read_tool("r1"), _write_tool("w1"), _read_tool("r2")]
    calls = [_call("r1"), _call("w1"), _call("r2")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["r1", "w1", "r2"]  # original order
    # The write must have run in isolation: at its start/end no other call overlaps it.
    w_start = events.index("start:w1")
    w_end = events.index("end:w1")
    assert w_end == w_start + 1  # nothing interleaved inside the write


def test_parallelism_one_restores_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """HIMMY_TOOL_PARALLELISM=1 => fully serial (no two calls ever overlap)."""
    monkeypatch.setenv("HIMMY_TOOL_PARALLELISM", "1")
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.02)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [_read_tool("a"), _read_tool("b"), _read_tool("c")]
    calls = [_call("a"), _call("b"), _call("c")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["a", "b", "c"]
    assert max_concurrent == 1  # serial


# ------------------------------------------------------- exception isolation


def test_one_failing_read_is_isolated_and_reraised() -> None:
    """A raising read must not corrupt siblings; the earliest failure is re-raised."""
    ran: set[str] = set()

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        await asyncio.sleep(0.01)
        ran.add(name)
        if name == "b":
            raise RuntimeError("boom in b")
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [_read_tool("a"), _read_tool("b"), _read_tool("c")]
    calls = [_call("a"), _call("b"), _call("c")]

    with pytest.raises(RuntimeError, match="boom in b"):
        _run(_execute_tool_calls(tools, calls, executor))
    # Siblings still executed (isolation) — the failure of b did not cancel a/c.
    assert {"a", "c"} <= ran


def test_earliest_failure_is_the_one_raised() -> None:
    """When two reads raise, the earliest-in-original-order exception surfaces."""

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        await asyncio.sleep(0.01)
        raise RuntimeError(f"fail:{name}")

    tools = [_read_tool("a"), _read_tool("b")]
    calls = [_call("a"), _call("b")]

    with pytest.raises(RuntimeError, match="fail:a"):
        _run(_execute_tool_calls(tools, calls, executor))


# ------------------------------------------------------ authz/approval gating


def test_authz_and_approval_gate_fires_per_call() -> None:
    """Every call — read or write — goes through the executor (the authz/HITL seam)."""
    gated: list[str] = []

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        # Stand-in for the capability-authz + approval gate: it must run once per call.
        gated.append(name)
        if name == "w1":
            # Simulate a denied write surfacing as a failed record (the gate's job),
            # exactly as the serial path would.
            return ToolReturnRecord(
                tool_call_id="", tool_name=name, content="denied", outcome="failed"
            )
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [_read_tool("r1"), _write_tool("w1"), _read_tool("r2")]
    calls = [_call("r1"), _call("w1"), _call("r2")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert sorted(gated) == ["r1", "r2", "w1"]  # gate fired for each
    assert returns[1].outcome == "failed"  # the denied write kept its gate verdict


# --------------------------------------------------- classifier fallback path


def test_unset_read_only_uses_name_classifier() -> None:
    """read_only unset => name-based classifier decides concurrency eligibility."""
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.03)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    # "get_*" classifies read-only via classify_read_only; flag left as None.
    tools = [
        BoundTool(name="get_price", description="lookup"),
        BoundTool(name="get_weather", description="lookup"),
    ]
    calls = [_call("get_price"), _call("get_weather")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["get_price", "get_weather"]
    assert max_concurrent == 2  # both ran concurrently via the classifier


def test_ambiguous_name_stays_sequential() -> None:
    """An unclassifiable name with no read_only flag is treated as NOT read-only."""
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.02)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [
        BoundTool(name="frobnicate", description="?"),
        BoundTool(name="wibble", description="?"),
    ]
    calls = [_call("frobnicate"), _call("wibble")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["frobnicate", "wibble"]
    assert max_concurrent == 1  # ambiguous => sequential (safe default)
