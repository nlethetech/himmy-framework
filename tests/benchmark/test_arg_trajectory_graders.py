"""Tests for argument-level + parallelism trajectory graders (FOUNDATION B).

These graders run against the structured :class:`Trajectory` (call args, results, and
turn grouping), not the flat name-only sequence — the BFCL AST-check equivalent for
"right tool, WRONG args". The name-only graders must still accept either input.
"""

from __future__ import annotations

import pytest

from himmy.benchmark.graders import grade, grade_trajectory
from himmy.benchmark.trajectory import RecordedCall, Trajectory


def _traj(*turns: list[RecordedCall]) -> Trajectory:
    return Trajectory(turns=list(turns))


# --- tool_called_with ---------------------------------------------------------------


def test_tool_called_with_matches_subset_of_args() -> None:
    traj = _traj([RecordedCall("calculator", {"expression": "17*23", "mode": "int"})])
    # Only the listed arg must match; extra args (mode) are ignored.
    assert grade_trajectory(
        {"type": "tool_called_with", "tool": "calculator", "args": {"expression": "17*23"}},
        traj,
    )


def test_tool_called_with_wrong_arg_value_fails() -> None:
    traj = _traj([RecordedCall("calculator", {"expression": "1+1"})])
    assert not grade_trajectory(
        {"type": "tool_called_with", "tool": "calculator", "args": {"expression": "17*23"}},
        traj,
    )


def test_tool_called_with_missing_arg_fails() -> None:
    traj = _traj([RecordedCall("calculator", {"mode": "int"})])
    assert not grade_trajectory(
        {"type": "tool_called_with", "tool": "calculator", "args": {"expression": "x"}},
        traj,
    )


def test_tool_called_with_wrong_tool_fails() -> None:
    traj = _traj([RecordedCall("web_search", {"query": "x"})])
    assert not grade_trajectory(
        {"type": "tool_called_with", "tool": "calculator", "args": {"expression": "x"}},
        traj,
    )


# --- arg_equals (with forgiving scalar coercion) ------------------------------------


def test_arg_equals_exact() -> None:
    traj = _traj([RecordedCall("sql_query", {"limit": 5})])
    assert grade_trajectory(
        {"type": "arg_equals", "tool": "sql_query", "arg": "limit", "value": 5}, traj
    )


def test_arg_equals_coerces_stringified_number() -> None:
    # Models often stringify numeric args; "5" should match 5 (lossless coercion).
    traj = _traj([RecordedCall("sql_query", {"limit": "5"})])
    assert grade_trajectory(
        {"type": "arg_equals", "tool": "sql_query", "arg": "limit", "value": 5}, traj
    )


def test_arg_equals_coerces_stringified_bool() -> None:
    traj = _traj([RecordedCall("toggle", {"on": "true"})])
    assert grade_trajectory(
        {"type": "arg_equals", "tool": "toggle", "arg": "on", "value": True}, traj
    )


def test_arg_equals_wrong_value_fails() -> None:
    traj = _traj([RecordedCall("sql_query", {"limit": 10})])
    assert not grade_trajectory(
        {"type": "arg_equals", "tool": "sql_query", "arg": "limit", "value": 5}, traj
    )


def test_arg_equals_picks_the_matching_call_among_repeats() -> None:
    # Two calls to the same tool with different args; one matches.
    traj = _traj(
        [RecordedCall("calc", {"x": 1})],
        [RecordedCall("calc", {"x": 2})],
    )
    assert grade_trajectory(
        {"type": "arg_equals", "tool": "calc", "arg": "x", "value": 2}, traj
    )


# --- arg_matches (regex) ------------------------------------------------------------


def test_arg_matches_regex_case_insensitive() -> None:
    traj = _traj([RecordedCall("web_search", {"query": "Capital of FRANCE"})])
    assert grade_trajectory(
        {"type": "arg_matches", "tool": "web_search", "arg": "query", "value": "france"},
        traj,
    )


def test_arg_matches_no_match_fails() -> None:
    traj = _traj([RecordedCall("web_search", {"query": "weather today"})])
    assert not grade_trajectory(
        {"type": "arg_matches", "tool": "web_search", "arg": "query", "value": "^france$"},
        traj,
    )


# --- result_contains ----------------------------------------------------------------


def test_result_contains_matches_string_result() -> None:
    traj = _traj(
        [RecordedCall("read_file", {"path": "x"}, result="code word MARIGOLD", has_result=True)]
    )
    assert grade_trajectory(
        {"type": "result_contains", "tool": "read_file", "value": "marigold"}, traj
    )


def test_result_contains_matches_structured_result() -> None:
    traj = _traj(
        [RecordedCall("sql_query", {}, result={"rows": [{"name": "Bardaghat"}]}, has_result=True)]
    )
    assert grade_trajectory(
        {"type": "result_contains", "tool": "sql_query", "value": "bardaghat"}, traj
    )


def test_result_contains_ignores_calls_without_captured_result() -> None:
    # A call whose result was never captured can't satisfy result_contains.
    traj = _traj([RecordedCall("read_file", {}, result=None, has_result=False)])
    assert not grade_trajectory(
        {"type": "result_contains", "tool": "read_file", "value": "x"}, traj
    )


# --- parallel_in_turn / max_parallel ------------------------------------------------


def test_parallel_in_turn_all_in_one_turn() -> None:
    traj = _traj([RecordedCall("a"), RecordedCall("b"), RecordedCall("c")])
    assert grade_trajectory(
        {"type": "parallel_in_turn", "value": ["a", "b"]}, traj
    )


def test_parallel_in_turn_split_across_turns_fails() -> None:
    traj = _traj([RecordedCall("a")], [RecordedCall("b")])
    assert not grade_trajectory(
        {"type": "parallel_in_turn", "value": ["a", "b"]}, traj
    )


def test_max_parallel_pass_and_fail() -> None:
    one_per_turn = _traj([RecordedCall("a")], [RecordedCall("b")], [RecordedCall("c")])
    assert grade_trajectory({"type": "max_parallel", "value": 1}, one_per_turn)
    three_in_turn = _traj([RecordedCall("a"), RecordedCall("b"), RecordedCall("c")])
    assert not grade_trajectory({"type": "max_parallel", "value": 1}, three_in_turn)
    assert grade_trajectory({"type": "max_parallel", "value": 3}, three_in_turn)


def test_max_parallel_no_calls_is_zero() -> None:
    assert grade_trajectory({"type": "max_parallel", "value": 0}, _traj())


# --- back-compat: name-only graders still work on a Trajectory AND a flat list -------


def test_name_only_graders_accept_structured_trajectory() -> None:
    traj = _traj([RecordedCall("sql_schema", {"db": "x"})], [RecordedCall("sql_query", {})])
    assert grade_trajectory({"type": "first_tool", "value": "sql_schema"}, traj)
    assert grade_trajectory(
        {"type": "tool_sequence", "value": ["sql_schema", "sql_query"]}, traj
    )
    assert grade_trajectory({"type": "max_tool_calls", "value": 2}, traj)
    assert grade_trajectory({"type": "tool_called", "value": "sql_query"}, traj)


def test_name_only_graders_still_accept_flat_list() -> None:
    # The legacy flat name-only input is unchanged.
    assert grade_trajectory({"type": "first_tool", "value": "a"}, ["a", "b"])
    assert grade_trajectory({"type": "max_tool_calls", "value": 0}, [])


# --- combinators over the new graders -----------------------------------------------


def test_all_of_over_arg_graders() -> None:
    traj = _traj([RecordedCall("calc", {"x": 17, "y": 23})])
    spec = {
        "type": "all_of",
        "of": [
            {"type": "tool_called_with", "tool": "calc", "args": {"x": 17}},
            {"type": "arg_equals", "tool": "calc", "arg": "y", "value": 23},
            {"type": "max_parallel", "value": 1},
        ],
    }
    assert grade_trajectory(spec, traj)


def test_any_of_over_arg_graders() -> None:
    traj = _traj([RecordedCall("calc", {"x": 1})])
    spec = {
        "type": "any_of",
        "of": [
            {"type": "arg_equals", "tool": "calc", "arg": "x", "value": 99},
            {"type": "arg_equals", "tool": "calc", "arg": "x", "value": 1},
        ],
    }
    assert grade_trajectory(spec, traj)


# --- fail-loud: arg graders require structure; answer block rejects them ------------


def test_arg_grader_on_flat_list_cannot_match() -> None:
    # A flat name-only list carries no args, so an arg grader simply finds no matching
    # arg — it fails (does not crash, does not spuriously pass).
    assert not grade_trajectory(
        {"type": "arg_equals", "tool": "calc", "arg": "x", "value": 1}, ["calc"]
    )


def test_answer_block_rejects_arg_trajectory_grader() -> None:
    # An arg trajectory predicate placed in an answer `grade` block fails loud.
    with pytest.raises(ValueError):
        grade({"type": "arg_equals", "tool": "c", "arg": "x", "value": 1}, "answer")
    with pytest.raises(ValueError):
        grade({"type": "max_parallel", "value": 1}, "answer")


def test_unknown_trajectory_type_still_raises() -> None:
    with pytest.raises(ValueError):
        grade_trajectory({"type": "telepathy", "value": 1}, _traj())
