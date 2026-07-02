"""Red-team round-3 regression for parallel tool-safety fail-open (eff-p1 safe-r3).

Confirmed bug: ``classify_parallel_safe`` was fail-OPEN for COMPOSITE read+write names
whose write verb was not in ``_WRITE_VERBS``. Any name carrying an unambiguous read verb
(``get``/``list``/``read``/``describe``) AND an unlisted mutating verb (``actuate``,
``toggle``, ``move``, ``copy``, ``open``, ``close``, ``invoke``, ``generate``, ``build``,
``forward``, ``place``, ``wire``, ``process``, …) was judged parallel-safe and hoisted into
a concurrent read-batch ahead of a dependent read — breaking serial happens-before and
firing the mutation even when the turn errors.

The fix (1) adds those mutating verbs to the deny-list and (2) adds a structural tail-guard:
a name that is NOT entirely read verbs is parallel-safe ONLY when its FIRST token is an
unambiguous read verb OR its LAST token is itself a read verb — so a future unlisted
mutating tail (``foo_get_franflam``) also barriers instead of failing open.
"""

from __future__ import annotations

import asyncio

from himmy.services.inference.local import _execute_tool_calls, _is_parallel_safe
from himmy.services.inference.models import BoundTool, ToolCallRecord, ToolReturnRecord
from himmy.services.tools.access import classify_parallel_safe


def _call(name: str, **args: object) -> ToolCallRecord:
    return ToolCallRecord(tool_call_id=f"tc_{name}", tool_name=name, args=dict(args))


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ------------------------------------------ the 17 confirmed fail-open composite names


def test_composite_read_write_names_are_not_parallel_safe() -> None:
    """Every confirmed side-effecting composite must now stay SEQUENTIAL (fail-closed)."""
    for name in (
        "get_and_actuate",
        "list_and_move",
        "read_and_copy",
        "get_and_invoke",
        "read_and_toggle",
        "list_and_open",
        "get_and_generate",
        "read_and_build",
        "list_and_forward",
        "get_and_place",
        "get_and_wire",
        "pump_read_actuate",
        "valve_get_open",
        "sensor_read_toggle",
        "file_get_move",
        "account_get_close",
        "read_sensor_and_actuate",
    ):
        assert classify_parallel_safe(name) is False, name


def test_bare_mutating_verbs_still_sequential() -> None:
    """Single-verb mutators remain barriers (no read verb ⇒ never parallel-safe)."""
    for name in (
        "open_valve",
        "close_ticket",
        "move_file",
        "copy_file",
        "toggle_switch",
        "invoke_lambda",
        "generate_report",
        "actuate_pump",
        "build_image",
    ):
        assert classify_parallel_safe(name) is False, name


def test_genuine_readers_still_parallel_safe() -> None:
    """The deny-list expansion + tail-guard must NOT regress real readers (win kept)."""
    for name in (
        "get_valve_state",
        "list_documents",
        "read_file",
        "get_user",
        "egg_totals",
        "describe_table",
        "get_summary",
        "list_recent",
        "account_summary",
        "sql_query",
    ):
        assert classify_parallel_safe(name) is True, name


def test_unlisted_mutating_tail_still_barriers_via_tail_guard() -> None:
    """A noun-first composite with a NOVEL (unknown) mutating tail must fail CLOSED.

    ``foo_get_franflam`` carries the read verb ``get`` but ends in a token that is neither a
    read verb nor (yet) in the deny-list. The structural tail-guard treats it as a possible
    action rather than fail-open — the name classifier is not an authorization boundary.
    """
    assert classify_parallel_safe("valve_get_franflam") is False
    assert classify_parallel_safe("device_read_zorp") is False


def test_composite_read_write_not_parallel_eligible_as_bound_tool() -> None:
    """The bound-tool gate agrees: unset read_only + a mutating composite is a barrier."""
    for name in ("valve_get_open", "file_get_move", "get_and_actuate"):
        assert _is_parallel_safe(BoundTool(name=name, description="x")) is False


def test_mutating_composite_not_hoisted_into_read_batch() -> None:
    """[valve_get_open, get_valve_state] must run SERIALLY, never overlapped/reordered.

    The dependent read must observe post-mutation state (serial happens-before), and the
    mutation must never fire concurrently with the read.
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

    tools = [
        BoundTool(name="valve_get_open", description="open the valve"),
        BoundTool(name="get_valve_state", description="read the valve state"),
    ]
    calls = [_call("valve_get_open", x=1), _call("get_valve_state", x=1)]

    returns = _run(_execute_tool_calls(tools, calls, executor))
    assert [r.tool_name for r in returns] == ["valve_get_open", "get_valve_state"]
    assert fired == ["valve_get_open", "get_valve_state"]  # mutation ran first, serially
    assert max_concurrent == 1  # never overlapped
