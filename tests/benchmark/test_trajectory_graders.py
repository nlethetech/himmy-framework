"""Tests for the trajectory (tool-call sequence) graders."""

from __future__ import annotations

import pytest

from himmy.benchmark.graders import grade, grade_trajectory


def test_first_tool_pass_and_fail() -> None:
    spec = {"type": "first_tool", "value": "sql_schema"}
    assert grade_trajectory(spec, ["sql_schema", "sql_query"])
    assert not grade_trajectory(spec, ["sql_query", "sql_schema"])


def test_first_tool_empty_sequence_fails() -> None:
    # Nothing was called first, so the predicate can't be satisfied.
    assert not grade_trajectory({"type": "first_tool", "value": "calculator"}, [])


def test_max_tool_calls_pass_and_fail() -> None:
    spec = {"type": "max_tool_calls", "value": 2}
    assert grade_trajectory(spec, ["a", "b"])
    assert not grade_trajectory(spec, ["a", "b", "c"])  # runaway loop


def test_max_tool_calls_zero_requires_no_tools() -> None:
    spec = {"type": "max_tool_calls", "value": 0}
    assert grade_trajectory(spec, [])
    assert not grade_trajectory(spec, ["calculator"])


def test_tool_called_pass_and_fail() -> None:
    spec = {"type": "tool_called", "value": "kb_search"}
    assert grade_trajectory(spec, ["kb_ingest", "kb_search"])
    assert not grade_trajectory(spec, ["kb_ingest"])


def test_tool_not_called_catches_hallucinated_tool() -> None:
    # An attempted call to a tool the agent shouldn't use (or one that doesn't exist,
    # whose attempted name the runner still records) is caught here.
    assert grade_trajectory(
        {"type": "tool_not_called", "value": "web_search"}, ["calc"]
    )
    assert not grade_trajectory(
        {"type": "tool_not_called", "value": "web_search"}, ["calc", "web_search"]
    )
    # A hallucinated/nonexistent tool name the runner recorded is detectable too.
    assert not grade_trajectory(
        {"type": "tool_not_called", "value": "made_up_tool"}, ["made_up_tool"]
    )


def test_tool_sequence_ordered_subsequence() -> None:
    spec = {"type": "tool_sequence", "value": ["sql_schema", "sql_query"]}
    # Exact order.
    assert grade_trajectory(spec, ["sql_schema", "sql_query"])
    # Gaps allowed (other calls interleaved), order preserved.
    assert grade_trajectory(spec, ["sql_schema", "noise", "sql_query"])
    # Wrong order fails.
    assert not grade_trajectory(spec, ["sql_query", "sql_schema"])
    # Missing a required tool fails.
    assert not grade_trajectory(spec, ["sql_schema"])


def test_trajectory_all_of_composition() -> None:
    spec = {
        "type": "all_of",
        "of": [
            {"type": "first_tool", "value": "scratchpad_set"},
            {"type": "max_tool_calls", "value": 2},
            {"type": "tool_not_called", "value": "web_search"},
        ],
    }
    assert grade_trajectory(spec, ["scratchpad_set", "scratchpad_get"])
    # First tool wrong → all_of fails.
    assert not grade_trajectory(spec, ["scratchpad_get", "scratchpad_set"])
    # Too many calls → all_of fails.
    assert not grade_trajectory(
        spec, ["scratchpad_set", "scratchpad_get", "scratchpad_get"]
    )


def test_trajectory_any_of_composition() -> None:
    spec = {
        "type": "any_of",
        "of": [
            {"type": "first_tool", "value": "read_file"},
            {"type": "tool_called", "value": "calculator"},
        ],
    }
    assert grade_trajectory(spec, ["read_file"])  # first branch
    assert grade_trajectory(spec, ["sql_query", "calculator"])  # second branch
    assert not grade_trajectory(spec, ["sql_query"])  # neither


def test_trajectory_empty_of_raises() -> None:
    with pytest.raises(ValueError):
        grade_trajectory({"type": "all_of", "of": []}, ["a"])
    with pytest.raises(ValueError):
        grade_trajectory({"type": "any_of"}, ["a"])


def test_trajectory_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        grade_trajectory({"type": "telekinesis", "value": "x"}, ["a"])


def test_answer_grader_rejects_trajectory_type() -> None:
    # A trajectory predicate in an answer-grade block fails loud, not silently.
    with pytest.raises(ValueError):
        grade({"type": "first_tool", "value": "calculator"}, "the answer")


def test_trajectory_grader_rejects_answer_type() -> None:
    # The mirror: an answer predicate in a trajectory block also fails loud. The two
    # grader families do NOT mix under one combinator (the corrected docstring) — each
    # combinator recurses only into its own family and raises on the other's leaf.
    with pytest.raises(ValueError):
        grade_trajectory({"type": "contains", "value": "x"}, ["calculator"])


def test_combinators_do_not_mix_families() -> None:
    # all_of in a `grade` block raises on a trajectory leaf; all_of in a `trajectory`
    # block raises on an answer leaf — the families cannot be combined under one
    # combinator (regression for the graders.py docstring claim, findings 10/15).
    with pytest.raises(ValueError):
        grade(
            {"type": "all_of", "of": [{"type": "tool_called", "value": "calc"}]},
            "the answer",
        )
    with pytest.raises(ValueError):
        grade_trajectory(
            {"type": "all_of", "of": [{"type": "contains", "value": "x"}]},
            ["calc"],
        )


def test_trajectory_handles_none_sequence() -> None:
    # grade_trajectory coalesces a None sequence to [] (the `list(tool_sequence or [])`
    # guard), so a None trajectory is treated as "no tools called". Actually pass None —
    # the empty-list case is already covered elsewhere. (Finding 21.)
    assert grade_trajectory({"type": "max_tool_calls", "value": 1}, None)  # type: ignore[arg-type]
    assert grade_trajectory({"type": "max_tool_calls", "value": 0}, None)  # type: ignore[arg-type]
    # first_tool on a None (empty) sequence fails — nothing was called first.
    assert not grade_trajectory({"type": "first_tool", "value": "calc"}, None)  # type: ignore[arg-type]
