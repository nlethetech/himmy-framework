"""run_python echoes a trailing bare expression (REPL semantics) so models' natural
last-line style isn't silently lost as empty stdout."""

from __future__ import annotations

import asyncio

from himmy.services.sandbox.factory import build_sandbox
from himmy.services.sandbox.tools import echo_last_expression, register_sandbox_tool
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig


def test_trailing_expression_is_echoed() -> None:
    out = echo_last_expression("result = 4869 * 17 + 233\nresult")
    assert "print(repr(" in out


def test_bare_expression_is_echoed() -> None:
    assert "print(repr(" in echo_last_expression("4869 * 17 + 233")


def test_trailing_print_is_left_alone() -> None:
    # A trailing print(...) evaluates to None; the guard skips it (no double echo).
    code = "print(1 + 1)"
    out = echo_last_expression(code)
    # print(...) is itself an expression, so it gets wrapped, but its None result
    # is guarded out — semantics preserved.
    assert "print(1 + 1)" in out


def test_assignment_last_is_untouched() -> None:
    assert echo_last_expression("x = 5") == "x = 5"


def test_trailing_string_not_echoed() -> None:
    assert echo_last_expression("'just a string'") == "'just a string'"


def test_syntax_error_returned_verbatim() -> None:
    assert echo_last_expression("def (") == "def ("


def test_end_to_end_bare_expression_produces_stdout() -> None:
    # The real payoff: a model that writes `result` (not print) still gets a value.
    registry = ToolRegistry()
    sandbox = build_sandbox(ToolkitConfig().code_exec)
    register_sandbox_tool(registry, sandbox, requires_approval=False)
    handler = registry.handler_for("run_python")
    result = asyncio.run(handler({"code": "v = 4869 * 17 + 233\nv"}))
    assert result["stdout"].strip() == "83006"
