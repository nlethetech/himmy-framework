"""Tests for tolerant tool-call parsing + schema-guided name repair (Tier 1)."""

from __future__ import annotations

import asyncio

from himmy.services.inference.local import _execute_tool_calls
from himmy.services.inference.models import (
    BoundTool,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.tool_protocol import parse_text_tool_calls
from himmy.services.tools.repair import resolve_tool_name, unknown_tool_message
from tests.conftest import executor_from

KNOWN = {"egg_totals", "pond_status", "add_task", "list_flocks"}


# ---- tolerant parser ----------------------------------------------------


def test_marker_with_prose_and_dedup() -> None:
    text = (
        "Let me check the eggs.\n"
        'TOOL_CALL egg_totals {"days": 7}\n\n'
        'TOOL_CALL egg_totals {"days": 7}'  # repeated → deduped
    )
    calls = parse_text_tool_calls(text, KNOWN)
    assert [(c.tool_name, c.args) for c in calls] == [("egg_totals", {"days": 7})]


def test_fenced_json_block() -> None:
    text = '```json\n{"tool": "add_task", "args": {"title": "net litchi"}}\n```'
    calls = parse_text_tool_calls(text, KNOWN)
    assert calls[0].tool_name == "add_task"
    assert calls[0].args == {"title": "net litchi"}


def test_bare_json_name_arguments() -> None:
    calls = parse_text_tool_calls('{"name": "pond_status", "arguments": {}}', KNOWN)
    assert calls[0].tool_name == "pond_status"


def test_bare_json_list_multiple() -> None:
    text = '[{"tool":"egg_totals","args":{"days":3}},{"tool":"pond_status","args":{}}]'
    calls = parse_text_tool_calls(text, KNOWN)
    assert [c.tool_name for c in calls] == ["egg_totals", "pond_status"]


def test_function_nested_arguments_string() -> None:
    # OpenAI-ish: function.name + arguments as a JSON string.
    text = '{"function": {"name": "egg_totals", "arguments": "{\\"days\\": 5}"}}'
    calls = parse_text_tool_calls(text, KNOWN)
    assert calls[0].tool_name == "egg_totals"
    assert calls[0].args == {"days": 5}


def test_final_answer_json_is_not_a_call() -> None:
    # A prose answer that happens to contain JSON must not be parsed as a tool call.
    text = 'Here is your summary: {"eggs": 245, "status": "ok"}'
    assert parse_text_tool_calls(text, KNOWN) == []


def test_unknown_name_in_ambiguous_json_rejected() -> None:
    # Bare JSON whose name isn't a known/near tool is not a call (false-positive guard).
    assert (
        parse_text_tool_calls('{"tool": "wildly_unrelated", "args": {}}', KNOWN) == []
    )


def test_marker_accepts_unknown_name_for_repair() -> None:
    # An explicit TOOL_CALL marker is always parsed (the repair layer fixes the name).
    calls = parse_text_tool_calls('TOOL_CALL egg_total {"days": 7}', KNOWN)
    assert calls[0].tool_name == "egg_total"


# ---- name repair --------------------------------------------------------


def test_resolve_exact_and_typo_and_case() -> None:
    avail = ["egg_totals", "pond_status", "add_task"]
    assert resolve_tool_name("egg_totals", avail).name == "egg_totals"
    r = resolve_tool_name("egg_total", avail)
    assert r.name == "egg_totals" and r.auto_corrected
    assert resolve_tool_name("Pond_Status", avail).name == "pond_status"


def test_resolve_ambiguous_does_not_autocorrect() -> None:
    avail = ["list_flocks", "list_trees", "list_tasks"]
    r = resolve_tool_name("list", avail)
    assert r.name is None
    assert set(r.suggestions) <= set(avail) and r.suggestions


def test_unknown_tool_message_is_actionable() -> None:
    avail = ["egg_totals", "pond_status"]
    msg = unknown_tool_message(resolve_tool_name("eg_total", avail), avail)
    assert "Unknown tool" in msg and "egg_totals" in msg


# ---- repair wired into execution ----------------------------------------


def _bound() -> list[BoundTool]:
    return [BoundTool(name="egg_totals", args_json_schema={"type": "object"})]


async def _egg_handler(args: dict) -> ToolReturnRecord:
    return ToolReturnRecord(
        tool_call_id="x",
        tool_name="egg_totals",
        content={"total": 245},
        outcome="success",
    )


_EXECUTOR = executor_from({"egg_totals": _egg_handler})


def test_execute_autocorrects_typo_and_runs() -> None:
    calls = [ToolCallRecord(tool_call_id="c1", tool_name="egg_total", args={"days": 7})]
    rets = asyncio.run(_execute_tool_calls(_bound(), calls, _EXECUTOR))
    assert rets[0].outcome == "success"
    assert rets[0].tool_name == "egg_totals"
    assert rets[0].metadata["repaired_from"] == "egg_total"


def test_execute_unknown_tool_returns_correction() -> None:
    calls = [ToolCallRecord(tool_call_id="c2", tool_name="frobnicate", args={})]
    rets = asyncio.run(_execute_tool_calls(_bound(), calls, _EXECUTOR))
    assert rets[0].outcome == "failed"
    assert "Unknown tool" in rets[0].content
    assert rets[0].metadata["error_code"] == "UNKNOWN_TOOL"
