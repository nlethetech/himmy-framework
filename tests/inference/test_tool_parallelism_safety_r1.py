"""Red-team round-1 regressions for the parallel read-tool fan-out (eff-p1 safe-r1).

These lock the CORRECTNESS fixes for the confirmed bugs where the name-classifier /
HTTP method / sequential flag could let a WRITE be hoisted into a concurrent read-batch
ahead of a dependent read (ordering / idempotency break):

* a side-effecting tool whose NAME leads with a read verb (``report_incident``,
  ``fetch_and_delete``, ``get_or_create_session``) must stay SEQUENTIAL when its
  ``read_only`` flag is unset — the strict :func:`classify_parallel_safe` gate;
* a declarative HTTP POST/PUT/PATCH/DELETE tool must inherit ``read_only=False`` from its
  METHOD (not its name) and stay sequential;
* an explicit ``sequential=True`` tool must never be batched even when read-only.
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.config.http_tool_spec import HttpToolSpec
from himmy.services.inference.local import _execute_tool_calls
from himmy.services.inference.models import BoundTool, ToolCallRecord, ToolReturnRecord
from himmy.services.tools.access import classify_parallel_safe, classify_read_only
from himmy.services.tools.registry import ToolRegistry, register_http_tool


def _call(name: str, **args: object) -> ToolCallRecord:
    return ToolCallRecord(tool_call_id=f"tc_{name}", tool_name=name, args=dict(args))


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ------------------------------------------------------- strict classifier gate


def test_classify_parallel_safe_rejects_read_verb_prefixed_writers() -> None:
    """A write verb anywhere in the name → NOT parallel-safe, despite a read-verb prefix.

    These are exactly the names first-token-wins ``classify_read_only`` mis-tags as
    read-only. The strict concurrency gate must refuse them.
    """
    for name in (
        "report_incident",
        "check_out",
        "find_or_create",
        "fetch_and_delete",
        "get_or_create_session",
        "count_and_reset",
        "view_and_ack",
        "history_purge",
        "search_replace",
        "status_update",
    ):
        # The lenient hint still (deliberately) mis-classifies these...
        assert classify_read_only(name) is True, name
        # ...but the strict concurrency gate must NOT trust them.
        assert classify_parallel_safe(name) is False, name


def test_classify_parallel_safe_allows_pure_readers() -> None:
    """Unambiguous readers (no write verb) stay parallel-safe (keeps the efficiency win)."""
    for name in ("get_price", "get_weather", "list_flocks", "egg_totals", "sql_query"):
        assert classify_parallel_safe(name) is True, name


def test_classify_parallel_safe_rejects_ambiguous() -> None:
    """A name with no read verb at all is not parallel-safe (fail closed)."""
    assert classify_parallel_safe("frobnicate") is False
    assert classify_parallel_safe("calculator") is False


def test_read_verb_prefixed_writer_stays_sequential_and_ordered() -> None:
    """[report_incident, get_incident] must NOT run concurrently / reorder the write.

    ``report_incident`` (read-verb prefix, side-effecting, ``read_only`` unset) must act as
    an ordering barrier so ``get_incident`` observes its effect — the pre-fix bug hoisted
    both into one concurrent batch.
    """
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

    # Both left read_only=None so the classifier fallback decides.
    tools = [
        BoundTool(name="report_incident", description="file an incident"),
        BoundTool(name="get_incident", description="look up an incident"),
    ]
    calls = [_call("report_incident", x=1), _call("get_incident", x=1)]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["report_incident", "get_incident"]
    assert fired == ["report_incident", "get_incident"]  # write ran first, serially
    assert max_concurrent == 1  # never overlapped


# ---------------------------------------------------- HTTP method drives intent


def _http_spec(name: str, method: str) -> BoundTool:
    """Register a declarative HTTP tool via HttpToolSpec and return its bound form."""
    registry = ToolRegistry()
    HttpToolSpec(
        name=name,
        method=method,
        base_url="https://api.example.com",
        path="/x",
    ).register(registry)
    definition = registry.get(name)
    assert definition is not None
    return BoundTool(
        name=definition.name,
        description=definition.description,
        read_only=definition.read_only,
    )


def test_http_post_tool_inherits_write_intent_from_method() -> None:
    """A POST tool named with a read verb (``search_orders``) is read_only=False by METHOD."""
    registry = ToolRegistry()
    HttpToolSpec(
        name="search_orders",  # read-verb name...
        method="POST",  # ...but a POST (writer)
        base_url="https://api.example.com",
        path="/orders/search",
    ).register(registry)
    definition = registry.get("search_orders")
    assert definition is not None
    assert definition.read_only is False  # method truth beats the name


def test_http_get_tool_is_read_only_by_method() -> None:
    for method in ("GET", "HEAD", "OPTIONS"):
        registry = ToolRegistry()
        HttpToolSpec(
            name="fetch_thing",
            method=method,
            base_url="https://api.example.com",
            path="/t",
        ).register(registry)
        definition = registry.get("fetch_thing")
        assert definition is not None
        assert definition.read_only is True, method


def test_http_read_only_override_beats_method() -> None:
    """An explicit read_only override wins (e.g. a side-effect-free POST search)."""
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


def test_register_http_tool_derives_read_only_from_method() -> None:
    """The low-level register_http_tool derives read_only when unset."""
    from himmy.services.tools.models import HttpToolConfig

    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="report_event",
        http_config=HttpToolConfig(base_url="https://x.example", method="POST"),
    )
    definition = registry.get("report_event")
    assert definition is not None
    assert definition.read_only is False


def test_http_post_tool_stays_sequential_in_fanout() -> None:
    """A read-verb-named POST must not be parallelised with a following read."""
    active = 0
    max_concurrent = 0
    fired: list[str] = []

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        fired.append(name)
        await asyncio.sleep(0.02)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    tools = [_http_spec("search_orders", "POST"), _http_spec("get_order", "GET")]
    calls = [_call("search_orders"), _call("get_order")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["search_orders", "get_order"]
    assert fired == ["search_orders", "get_order"]
    assert max_concurrent == 1  # the POST barrier stopped concurrency


# --------------------------------------------------- explicit sequential flag


def test_sequential_read_only_tool_is_not_batched() -> None:
    """read_only=True AND sequential=True → forced one-at-a-time (non-reentrant client)."""
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
        BoundTool(name="a", description="read", read_only=True, sequential=True),
        BoundTool(name="b", description="read", read_only=True, sequential=True),
    ]
    calls = [_call("a"), _call("b")]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["a", "b"]
    assert max_concurrent == 1  # sequential honoured despite read_only


def test_sequential_flag_threads_from_definition_to_bound_tool() -> None:
    """ToolService binds ToolDefinition.sequential onto BoundTool (no longer dropped)."""
    from himmy.services.tools.registry import register_local_tool

    registry = ToolRegistry()

    def handler(_args: dict) -> str:
        return "ok"

    register_local_tool(
        registry,
        name="get_thing",
        handler=handler,
        read_only=True,
        sequential=True,
    )
    definition = registry.get("get_thing")
    assert definition is not None
    assert definition.sequential is True
    bound = BoundTool(
        name=definition.name,
        description=definition.description,
        read_only=definition.read_only,
        sequential=definition.sequential,
    )
    # A sequential read-only bound tool is not parallel-eligible.
    from himmy.services.inference.local import _is_parallel_safe

    assert _is_parallel_safe(bound) is False


@pytest.mark.parametrize("value", [True, False, None])
def test_pure_reader_still_parallelises(value: bool | None) -> None:
    """A plain read-only tool (not sequential) keeps the concurrency win."""
    active = 0
    max_concurrent = 0

    async def executor(name: str, args: dict) -> ToolReturnRecord:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.03)
        active -= 1
        return ToolReturnRecord(tool_call_id="", tool_name=name, content=name)

    ro = True if value is None else value
    tools = [
        BoundTool(name="get_a", description="r", read_only=ro),
        BoundTool(name="get_b", description="r", read_only=ro),
    ]
    calls = [_call("get_a"), _call("get_b")]
    _run(_execute_tool_calls(tools, calls, executor))
    if ro:
        assert max_concurrent == 2
    else:
        assert max_concurrent == 1
