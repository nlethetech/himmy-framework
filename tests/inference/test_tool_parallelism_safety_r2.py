"""Red-team round-2 regressions for the parallel/retry tool-safety fixes (eff-p1 safe-r2).

These lock the CONFIRMED correctness bugs from the efficiency work:

* ``classify_parallel_safe`` was fail-OPEN for a large class of action verbs missing from
  ``_WRITE_VERBS`` (``charge``/``publish``/``deploy``/``sync``/``notify``/…). A read-verb-
  prefixed side-effecting name (``get_and_charge``, ``list_and_publish``,
  ``describe_and_deploy``) was wrongly judged parallel-safe AND re-fired on timeout.
* an HTTP GET tool with a server-side side effect was auto-classified read-only and re-fired
  on TIMEOUT: only an AUTHORITATIVE (explicit) ``read_only=True`` may re-fire a timed-out call.
"""

from __future__ import annotations

import asyncio

from himmy.config.http_tool_spec import HttpToolSpec
from himmy.services.inference.local import _execute_tool_calls, _is_parallel_safe
from himmy.services.inference.models import BoundTool, ToolCallRecord, ToolReturnRecord
from himmy.services.tools.access import classify_parallel_safe
from himmy.services.tools.models import HttpToolConfig
from himmy.services.tools.registry import ToolRegistry, register_http_tool


def _call(name: str, **args: object) -> ToolCallRecord:
    return ToolCallRecord(tool_call_id=f"tc_{name}", tool_name=name, args=dict(args))


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


# --------------------------------------------------- bug #1: fail-closed classifier


def test_read_prefixed_action_verbs_are_not_parallel_safe() -> None:
    """A read verb paired with a side-effecting action verb must stay SEQUENTIAL.

    These are exactly the names the old deny-list missed (the action verb wasn't in
    ``_WRITE_VERBS``), so they were fail-OPEN: parallel-safe AND retried on timeout.
    """
    for name in (
        "get_and_charge",
        "list_and_publish",
        "describe_and_deploy",
        "get_weather_and_notify",
        "get_and_sync",
        "get_and_refund",
        "read_and_sign",
        "fetch_and_dispatch",
        "list_and_purge",
        "get_and_reset",
        "get_and_revoke",
        "list_and_invite",
        "read_and_share",
        "get_and_trigger",
        "get_and_email",
        "get_and_flush",
        "get_and_drop",
        "get_and_mint",
        "get_and_enqueue",
        "get_and_apply",
        "get_and_commit",
        "get_and_push",
        "get_and_emit",
        "get_and_clear",
        "get_and_migrate",
        "get_and_provision",
        "get_and_activate",
        "get_and_archive",
    ):
        assert classify_parallel_safe(name) is False, name


def test_pure_readers_still_parallel_safe_after_expansion() -> None:
    """Expanding the write-verb set must NOT regress genuine readers (efficiency win kept)."""
    for name in (
        "get_price",
        "get_weather",
        "list_flocks",
        "egg_totals",
        "sql_query",
        "describe_table",
        "read_config",
        "get_summary",
        "list_recent",
    ):
        assert classify_parallel_safe(name) is True, name


def test_action_prefixed_read_stays_sequential_and_ordered() -> None:
    """[get_and_charge, get_balance] must NOT run concurrently / hoist the charge."""
    fired: list[str] = []
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        fired.append(name)
        await asyncio.sleep(0.02)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [
        BoundTool(name="get_and_charge", description="fetch then charge"),
        BoundTool(name="get_balance", description="look up a balance"),
    ]
    calls = [_call("get_and_charge", x=1), _call("get_balance", x=1)]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["get_and_charge", "get_balance"]
    assert fired == ["get_and_charge", "get_balance"]  # charge ran first, serially
    assert max_concurrent == 1  # never overlapped


def test_action_prefixed_read_is_not_parallel_eligible() -> None:
    """The bound-tool gate agrees: an unset read_only + action name is a barrier."""
    tool = BoundTool(name="describe_and_deploy", description="describe then deploy")
    assert _is_parallel_safe(tool) is False


# -------------------------------- bug #2: method-derived read_only is not authoritative


def test_http_get_read_only_is_not_authoritative() -> None:
    """A GET tool is read_only (parallel hint) but NOT authoritative (no timeout re-fire)."""
    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="ping_beacon",
        http_config=HttpToolConfig(base_url="https://svc", method="GET"),
    )
    definition = registry.get("ping_beacon")
    assert definition is not None
    assert definition.read_only is True  # method-derived hint (parallelism ok)
    assert definition.read_only_authoritative is False  # NOT safe to re-fire on timeout


def test_http_explicit_read_only_is_authoritative() -> None:
    """An explicit read_only=True IS authoritative — the author asserts no side effect."""
    registry = ToolRegistry()
    HttpToolSpec(
        name="search_es",
        method="POST",
        base_url="https://api.example.com",
        path="/_search",
        read_only=True,
    ).register(registry)
    definition = registry.get("search_es")
    assert definition is not None
    assert definition.read_only is True
    assert definition.read_only_authoritative is True


def test_http_post_read_only_authoritative_defaults_false() -> None:
    """A POST tool (method-derived writer) is neither read_only nor authoritative."""
    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="submit_order",
        http_config=HttpToolConfig(base_url="https://svc", method="POST"),
    )
    definition = registry.get("submit_order")
    assert definition is not None
    assert definition.read_only is False
    assert definition.read_only_authoritative is False


def test_local_explicit_read_only_is_authoritative() -> None:
    """A LOCAL tool marked read_only=True is authoritative; unset is not."""
    from himmy.services.tools.registry import register_local_tool

    registry = ToolRegistry()

    def handler(_args: dict) -> str:
        return "ok"

    register_local_tool(
        registry, name="get_thing", handler=handler, read_only=True
    )
    register_local_tool(
        registry, name="lookup_other", handler=handler
    )
    explicit = registry.get("get_thing")
    inferred = registry.get("lookup_other")
    assert explicit is not None and explicit.read_only_authoritative is True
    assert inferred is not None and inferred.read_only_authoritative is False
