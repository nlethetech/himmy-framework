"""Structural read/write intent gates parallelism (eff-p2: structural-tool-intent).

Read/write intent is a FIRST-CLASS, STRUCTURAL property of a tool. The parallel-safety
gate prefers the DECLARED intent (an authoritative author ``read_only`` flag) and only
falls back to the name heuristic — strictly fail-closed — when intent is undeclared or
merely method-inferred. This closes the P1 accepted residual: a side-effecting HTTP GET
(non-authoritative ``read_only=True``) or a read-named mutator must NOT auto-parallelise.

These tests lock the gate itself (``_is_parallel_safe``) and the end-to-end concurrency
behaviour, plus the definition→BoundTool threading of the authoritative flag.
"""

from __future__ import annotations

import asyncio

from himmy.services.inference.local import _execute_tool_calls, _is_parallel_safe
from himmy.services.inference.models import (
    BoundTool,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.tools.registry import (
    ToolRegistry,
    register_http_tool,
    register_local_tool,
)
from himmy.services.tools.service import ToolService


def _call(name: str, **args: object) -> ToolCallRecord:
    return ToolCallRecord(tool_call_id=f"tc_{name}", tool_name=name, args=dict(args))


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


# --------------------------------------------------------------- unit: the gate


def test_declared_read_only_is_parallel_safe() -> None:
    """An authoritative ``read_only=True`` declaration is parallel-safe regardless of name."""
    # An odd, un-read-looking name still parallelises because intent is DECLARED.
    tool = BoundTool(name="zzz", read_only=True, read_only_authoritative=True)
    assert _is_parallel_safe(tool) is True


def test_declared_write_is_a_barrier_even_with_read_name() -> None:
    """A declared ``read_only=False`` writer is a barrier even when its name reads read-only."""
    # A read-NAMED mutator: the name says "get", the author says it writes.
    tool = BoundTool(name="get_then_charge", read_only=False)
    assert _is_parallel_safe(tool) is False


def test_side_effecting_get_is_not_parallel_when_non_authoritative() -> None:
    """A method-derived ``read_only=True`` (HTTP GET) is NOT trusted — fail-closed name gate.

    ``GET /trigger`` may fire a server-side side effect, so a non-authoritative read-only
    must route through the strict name gate. ``trigger`` carries a write verb → barrier.
    """
    tool = BoundTool(
        name="trigger",
        read_only=True,
        read_only_authoritative=False,
    )
    assert _is_parallel_safe(tool) is False


def test_non_authoritative_get_with_read_name_still_uses_name_gate() -> None:
    """A non-authoritative read-only with a genuine read name passes the strict name gate."""
    tool = BoundTool(
        name="get_status",
        read_only=True,
        read_only_authoritative=False,
    )
    # first token "get" is an unambiguous read verb -> name gate says parallel-safe.
    assert _is_parallel_safe(tool) is True


def test_undeclared_ambiguous_name_is_fail_closed_barrier() -> None:
    """An UNDECLARED tool with an ambiguous name is a sequential WRITE barrier (P1 invariant)."""
    tool = BoundTool(name="frobnicate")  # read_only=None, name unclassifiable
    assert _is_parallel_safe(tool) is False


def test_undeclared_read_verb_prefixed_mutator_is_fail_closed() -> None:
    """An undeclared read-verb-prefixed writer (``fetch_and_delete``) stays a barrier."""
    tool = BoundTool(name="fetch_and_delete")  # read_only=None
    assert _is_parallel_safe(tool) is False


def test_sequential_forces_barrier_even_when_declared_read_only() -> None:
    """``sequential=True`` beats even an authoritative read-only declaration."""
    tool = BoundTool(
        name="get_status",
        read_only=True,
        read_only_authoritative=True,
        sequential=True,
    )
    assert _is_parallel_safe(tool) is False


# ------------------------------------------------- threading: definition -> BoundTool


def test_local_read_only_flag_is_authoritative_in_bound_tool() -> None:
    """``register_local_tool(read_only=True)`` yields an AUTHORITATIVE BoundTool."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="odd_reader",
        handler=lambda args: {"ok": True},
        read_only=True,
    )
    bound = {t.name: t for t in ToolService(registry).bound_tools()}["odd_reader"]
    assert bound.read_only is True
    assert bound.read_only_authoritative is True
    assert _is_parallel_safe(bound) is True  # declared -> parallel despite odd name


def test_local_computed_read_only_can_be_non_authoritative() -> None:
    """A caller COMPUTING ``read_only`` (e.g. a method inference) passes ``read_only_authoritative=False``.

    This is the declarative-connector path: the flag is method-derived, not a hand-written
    author assertion, so it must NOT license parallel hoisting / retry / the ``:write`` waiver.
    """
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="trigger_deploy",  # a side-effecting GET fronted as a local tool
        handler=lambda args: {"ok": True},
        read_only=("GET" in {"GET", "HEAD"}),  # computed inference -> literal True
        read_only_authoritative=False,
    )
    bound = {t.name: t for t in ToolService(registry).bound_tools()}["trigger_deploy"]
    assert bound.read_only is True  # method-derived hint preserved
    assert bound.read_only_authoritative is False  # but NOT authoritative
    assert _is_parallel_safe(bound) is False  # falls to strict name gate -> barrier


def test_declarative_connector_get_is_fail_closed_end_to_end() -> None:
    """A declarative-spec GET connector must register NON-authoritatively (parallel gate barrier)."""
    from himmy.connectors.spec import ConnectorSpec

    registry = ToolRegistry()
    spec = ConnectorSpec(
        name="beacon",
        description="A side-effecting GET beacon.",
        base_url="https://x",
        egress_allow_hosts=["x"],
        tools=[{"name": "trigger", "method": "GET", "path": "/trigger"}],
    )
    spec.build(fetcher=None).register_tools(registry)
    definition = registry.get("trigger")
    # Definition: method-derived read-only, NOT authoritative.
    assert definition.read_only is True
    assert definition.read_only_authoritative is False
    # Parallel gate: falls to strict name gate ("trigger" is not a read verb) -> barrier.
    bound = {t.name: t for t in ToolService(registry).bound_tools()}["trigger"]
    assert _is_parallel_safe(bound) is False


def test_http_get_is_non_authoritative_read_only_in_bound_tool() -> None:
    """A GET connector derives ``read_only=True`` but NON-authoritatively (fail-closed gate)."""
    from himmy.services.tools.models import HttpToolConfig

    registry = ToolRegistry()
    # A GET whose NAME carries a write verb (``trigger``) — a side-effecting GET.
    register_http_tool(
        registry,
        name="trigger",
        http_config=HttpToolConfig(method="GET", base_url="https://x", path_template="/trigger"),
    )
    bound = {t.name: t for t in ToolService(registry).bound_tools()}["trigger"]
    assert bound.read_only is True  # method-derived hint
    assert bound.read_only_authoritative is False  # but NOT authoritative
    # So the parallel gate falls back to the strict name gate -> barrier.
    assert _is_parallel_safe(bound) is False


def test_http_get_with_explicit_read_only_is_authoritative() -> None:
    """An explicit ``read_only=True`` on a GET connector IS authoritative (parallel-safe)."""
    from himmy.services.tools.models import HttpToolConfig

    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="trigger",  # odd/write-y name, but the author asserts read-only
        http_config=HttpToolConfig(method="GET", base_url="https://x", path_template="/trigger"),
        read_only=True,
    )
    bound = {t.name: t for t in ToolService(registry).bound_tools()}["trigger"]
    assert bound.read_only_authoritative is True
    assert _is_parallel_safe(bound) is True


# --------------------------------------------------------- e2e: concurrency effect


def test_declared_read_only_runs_concurrently_e2e() -> None:
    """Two authoritatively-declared readers with odd names run CONCURRENTLY."""
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.03)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [
        BoundTool(name="zzz1", read_only=True, read_only_authoritative=True),
        BoundTool(name="zzz2", read_only=True, read_only_authoritative=True),
    ]
    calls = [_call("zzz1"), _call("zzz2")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["zzz1", "zzz2"]
    assert max_concurrent == 2  # declared read-only -> parallel


def test_side_effecting_get_stays_sequential_e2e() -> None:
    """A side-effecting GET (non-authoritative read-only, write-y name) is a barrier."""
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
        BoundTool(name="trigger", read_only=True, read_only_authoritative=False),
        BoundTool(name="dispatch", read_only=True, read_only_authoritative=False),
    ]
    calls = [_call("trigger"), _call("dispatch")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["trigger", "dispatch"]
    assert max_concurrent == 1  # fail-closed name gate -> sequential


def test_read_named_mutator_stays_sequential_e2e() -> None:
    """A declared write with a read-only-looking name never joins a read-batch."""
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.02)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    # both LOOK like reads but are declared writers -> both barriers.
    tools = [
        BoundTool(name="get_and_delete", read_only=False),
        BoundTool(name="list_and_purge", read_only=False),
    ]
    calls = [_call("get_and_delete"), _call("list_and_purge")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["get_and_delete", "list_and_purge"]
    assert max_concurrent == 1
