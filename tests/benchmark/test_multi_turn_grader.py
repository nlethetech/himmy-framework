"""Tests for the multi-turn ``result_feeds_arg`` data-flow grader (FEATURE #4).

``result_feeds_arg`` is the BFCL multi-turn AST-check: a tool's RESULT must be carried
into a LATER tool call's argument. It catches the failure where a model calls both tools
but re-derives / hallucinates the bridging value instead of passing on the real one.
"""

from __future__ import annotations

import pytest

from himmy.benchmark.graders import grade_trajectory
from himmy.benchmark.trajectory import RecordedCall, Trajectory


def _traj(*turns: list[RecordedCall]) -> Trajectory:
    return Trajectory(turns=list(turns))


def _spec(**kw: object) -> dict[str, object]:
    base = {
        "type": "result_feeds_arg",
        "from_tool": "sql_query",
        "to_tool": "calculator",
        "arg": "expression",
    }
    base.update(kw)
    return base


# --- happy path: a result value flows into the next call's arg ----------------------


def test_result_feeds_arg_carries_value_forward() -> None:
    # sql returns 30; the calculator's expression carries that 30.
    traj = _traj(
        [RecordedCall("sql_query", {"sql": "select quantity ..."}, result={"rows": [{"quantity": 30}]}, has_result=True)],
        [RecordedCall("calculator", {"expression": "30*2"})],
    )
    assert grade_trajectory(_spec(value="30"), traj)


def test_result_feeds_arg_substring_mode_without_token() -> None:
    # No `value` token given: the arg's text must be a substring of the result's text.
    traj = _traj(
        [RecordedCall("read_file", {"path": "base.txt"}, result="count=21", has_result=True)],
        [RecordedCall("calculator", {"expression": "21"})],
    )
    assert grade_trajectory(
        {"type": "result_feeds_arg", "from_tool": "read_file", "to_tool": "calculator", "arg": "expression"},
        traj,
    )


# --- failure modes -------------------------------------------------------------------


def test_result_feeds_arg_fails_when_value_not_in_result() -> None:
    # The calculator used 99, which the sql result never produced — a hallucinated bridge.
    traj = _traj(
        [RecordedCall("sql_query", {"sql": "x"}, result={"rows": [{"quantity": 30}]}, has_result=True)],
        [RecordedCall("calculator", {"expression": "99*2"})],
    )
    assert not grade_trajectory(_spec(value="30"), traj)


def test_result_feeds_arg_requires_producer_before_consumer() -> None:
    # The calculator call comes BEFORE the sql result — the data could not have flowed.
    traj = _traj(
        [RecordedCall("calculator", {"expression": "30*2"})],
        [RecordedCall("sql_query", {"sql": "x"}, result={"rows": [{"quantity": 30}]}, has_result=True)],
    )
    assert not grade_trajectory(_spec(value="30"), traj)


def test_result_feeds_arg_ignores_producer_without_captured_result() -> None:
    # The producing call's result was never captured — it cannot have fed the consumer.
    traj = _traj(
        [RecordedCall("sql_query", {"sql": "x"}, result=None, has_result=False)],
        [RecordedCall("calculator", {"expression": "30*2"})],
    )
    assert not grade_trajectory(_spec(value="30"), traj)


def test_result_feeds_arg_fails_when_consumer_never_called() -> None:
    traj = _traj(
        [RecordedCall("sql_query", {"sql": "x"}, result={"rows": [{"quantity": 30}]}, has_result=True)],
    )
    assert not grade_trajectory(_spec(value="30"), traj)


def test_result_feeds_arg_fails_when_arg_missing() -> None:
    # Right tools, right order, but the consuming call didn't pass the bridged arg.
    traj = _traj(
        [RecordedCall("sql_query", {"sql": "x"}, result={"rows": [{"quantity": 30}]}, has_result=True)],
        [RecordedCall("calculator", {"mode": "int"})],
    )
    assert not grade_trajectory(_spec(value="30"), traj)


def test_result_feeds_arg_token_must_be_in_both_result_and_arg() -> None:
    # The token is in the result but the arg carried a different (also-present) number.
    traj = _traj(
        [RecordedCall("sql_query", {"sql": "x"}, result={"rows": [{"quantity": 30}, {"q": 12}]}, has_result=True)],
        [RecordedCall("calculator", {"expression": "12*2"})],  # used 12, not the asked-for 30
    )
    assert not grade_trajectory(_spec(value="30"), traj)


def test_result_feeds_arg_empty_arg_value_does_not_match() -> None:
    traj = _traj(
        [RecordedCall("read_file", {}, result="anything", has_result=True)],
        [RecordedCall("calculator", {"expression": ""})],
    )
    assert not grade_trajectory(
        {"type": "result_feeds_arg", "from_tool": "read_file", "to_tool": "calculator", "arg": "expression"},
        traj,
    )


def test_result_feeds_arg_on_flat_list_cannot_match() -> None:
    # A flat name-only list carries no args/results, so the data-flow grader finds nothing.
    assert not grade_trajectory(_spec(value="30"), ["sql_query", "calculator"])


# --- composition + registration ------------------------------------------------------


def test_result_feeds_arg_under_all_of() -> None:
    traj = _traj(
        [RecordedCall("sql_query", {"sql": "x"}, result={"quantity": 30}, has_result=True)],
        [RecordedCall("calculator", {"expression": "30*2"})],
    )
    spec = {
        "type": "all_of",
        "of": [
            {"type": "tool_sequence", "value": ["sql_query", "calculator"]},
            _spec(value="30"),
        ],
    }
    assert grade_trajectory(spec, traj)


def test_result_feeds_arg_is_registered_as_arg_trajectory_type() -> None:
    from himmy.benchmark.graders import ARG_TRAJECTORY_TYPES, TRAJECTORY_TYPES

    assert "result_feeds_arg" in ARG_TRAJECTORY_TYPES
    assert "result_feeds_arg" in TRAJECTORY_TYPES


def test_result_feeds_arg_rejected_in_answer_block() -> None:
    from himmy.benchmark.graders import grade

    with pytest.raises(ValueError):
        grade(_spec(value="30"), "answer")
