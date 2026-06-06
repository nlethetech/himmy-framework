"""Tests for read/write tool-access intent (reader/writer disambiguation)."""

from __future__ import annotations

from himmy.services.inference.local import _ollama_tools, _react_tool_manifest
from himmy.services.inference.models import BoundTool
from himmy.services.tools.access import classify_read_only, describe_for_model


def test_classify_read_verbs() -> None:
    for name in ("egg_totals", "list_flocks", "pond_status", "finance_summary",
                 "farm_overview", "recent_harvests", "get_order", "sql_query"):
        assert classify_read_only(name) is True, name


def test_classify_write_verbs() -> None:
    for name in ("log_eggs", "add_task", "record_expense", "complete_task",
                 "write_file", "send_email", "kb_ingest", "remember"):
        assert classify_read_only(name) is False, name


def test_classify_ambiguous_is_none() -> None:
    # No clear verb → no tag rather than a wrong one.
    assert classify_read_only("calculator") is None
    assert classify_read_only("http_request") is None


def test_explicit_flag_overrides_heuristic() -> None:
    # A "log_" tool that's actually a reader can be corrected by the author.
    out = describe_for_model("log_viewer", "Show recent logs", read_only=True)
    assert "[read-only:" in out
    # And a read-looking name forced to write.
    out2 = describe_for_model("list_send", "Queue a mailing list send", read_only=False)
    assert "[WRITE:" in out2


def test_describe_is_idempotent() -> None:
    once = describe_for_model("egg_totals", "Egg totals.")
    twice = describe_for_model("egg_totals", once)
    assert once == twice


def test_ollama_tool_descriptions_carry_intent() -> None:
    tools = [
        BoundTool(name="egg_totals", description="Egg totals."),
        BoundTool(name="log_eggs", description="Record eggs."),
    ]
    specs = {t["function"]["name"]: t["function"]["description"] for t in _ollama_tools(tools)}
    assert "[read-only:" in specs["egg_totals"]
    assert "[WRITE:" in specs["log_eggs"]


def test_text_manifest_carries_intent() -> None:
    tools = [BoundTool(name="log_eggs", description="Record eggs.")]
    manifest = _react_tool_manifest(tools)
    assert "[WRITE:" in manifest


def test_explicit_read_only_field_on_bound_tool_wins() -> None:
    # A write-looking name the author marked read-only is presented as read-only.
    tools = [BoundTool(name="log_search", description="Search logs", read_only=True)]
    specs = _ollama_tools(tools)
    assert "[read-only:" in specs[0]["function"]["description"]
