"""Deterministic correctness graders for benchmark tasks.

A grader is a small declarative spec (``{"type": ..., ...}``) checked against the
agent's answer — pure, fast, and reproducible (no model calls), so a task's pass/fail
is objective and a benchmark is comparable run-to-run. Composable via ``all_of`` /
``any_of``. For genuinely open-ended tasks, an ``llm_judge`` grader can be layered on
top by the caller, but the defaults here are rule-based on purpose.

Two families of grader share one ``{"type": ...}`` syntax:

* **Answer graders** (``contains``/``regex``/``exact``/``numeric``/…) check the agent's
  final *text* via :func:`grade`.
* **Trajectory graders** (``first_tool``/``max_tool_calls``/``tool_called``/
  ``tool_not_called``/``tool_sequence``) check the agent's ordered *tool-call sequence*
  via :func:`grade_trajectory`. They catch process failures a final-answer check
  misses — wrong first tool, a runaway tool loop, a hallucinated/unknown tool call, or a
  required tool ordering.

Each family composes under its own ``all_of`` / ``any_of`` combinator, but the two
families do **not** mix under one combinator: :func:`grade`'s combinators recurse only
into :func:`grade` (and raise on a trajectory-type leaf), and :func:`grade_trajectory`'s
combinators recurse only into :func:`grade_trajectory` (and raise on an answer-type
leaf). Declare answer graders in the task's ``grade`` block and trajectory graders in its
separate ``trajectory`` block — putting one family's leaf under the other combinator
raises ``ValueError`` (recorded as a trial error), by design, so a mis-declared grader
fails loud rather than silently mis-scoring.
"""

from __future__ import annotations

import re
from typing import Any

#: Grader ``type`` values that grade the ordered tool-call trajectory rather than the
#: final answer text. :func:`grade` uses this set to reject a trajectory-type leaf placed
#: in an answer ``grade`` block (fail-loud), not to silently route it to the other family.
TRAJECTORY_TYPES = frozenset(
    {
        "first_tool",
        "max_tool_calls",
        "tool_called",
        "tool_not_called",
        "tool_sequence",
    }
)


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for forgiving exact comparison."""
    return " ".join((text or "").lower().split())


def _numbers(text: str) -> list[float]:
    """Extract numeric literals (handles thousands separators) from ``text``."""
    out: list[float] = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*", text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:  # pragma: no cover - defensive
            continue
    return out


def grade(spec: dict[str, Any], answer: str) -> bool:
    """Return whether ``answer`` satisfies the grader ``spec``.

    Types: ``contains`` / ``not_contains`` (case-insensitive substring), ``regex``
    (case-insensitive search), ``exact`` (normalized equality), ``numeric`` (any number
    in the answer within ``tolerance`` of ``value``), and the combinators ``all_of`` /
    ``any_of`` over a list under ``of``.
    """
    answer = answer or ""
    gtype = spec.get("type", "contains")

    if gtype == "contains":
        return str(spec["value"]).lower() in answer.lower()
    if gtype == "not_contains":
        return str(spec["value"]).lower() not in answer.lower()
    if gtype == "regex":
        return re.search(str(spec["value"]), answer, re.IGNORECASE) is not None
    if gtype == "exact":
        return _normalize(answer) == _normalize(str(spec["value"]))
    if gtype == "numeric":
        target = float(spec["value"])
        tol = float(spec.get("tolerance", 0.0))
        return any(abs(n - target) <= tol for n in _numbers(answer))
    if gtype in ("all_of", "any_of"):
        sub = spec.get("of")
        if not sub:
            # An empty/missing `of` would make all_of vacuously True (a silent
            # false-positive). Fail loud so a malformed grade can't pass a task.
            raise ValueError(f"{gtype} grader requires a non-empty 'of' list")
        combiner = all if gtype == "all_of" else any
        return combiner(grade(s, answer) for s in sub)
    if gtype in TRAJECTORY_TYPES:
        # A trajectory predicate placed in an answer-grade block has no tool sequence to
        # check against here. Fail loud rather than silently treat it as the default
        # `contains` — declare trajectory graders under a task's `trajectory` block.
        raise ValueError(
            f"trajectory grader {gtype!r} cannot be used as an answer grader; "
            "declare it under the task's 'trajectory' block"
        )
    raise ValueError(f"unknown grader type {gtype!r}")


def grade_trajectory(spec: dict[str, Any], tool_sequence: list[str]) -> bool:
    """Return whether ``tool_sequence`` satisfies the trajectory grader ``spec``.

    ``tool_sequence`` is the agent's tools called *in order, with repeats* (a benchmark
    trial's :attr:`~himmy.benchmark.models.TrialResult.tools_called`). Predicate types:

    * ``first_tool`` — the first tool called must equal ``spec["value"]`` (fails on an
      empty sequence — nothing was called first).
    * ``max_tool_calls`` — at most ``spec["value"]`` total tool calls (bounds runaway
      tool loops). ``0`` requires that no tool was called.
    * ``tool_called`` — ``spec["value"]`` appears somewhere in the sequence.
    * ``tool_not_called`` — ``spec["value"]`` does **not** appear (hallucinated-/unknown-
      tool detection: an attempted call to a tool the agent shouldn't use, or one that
      does not exist, shows up here as long as the runner records attempted names).
    * ``tool_sequence`` — ``spec["value"]`` (a list of tool names) appears as an ordered
      **subsequence** (gaps allowed, order preserved) of the trajectory.

    Composes under ``all_of`` / ``any_of`` exactly like :func:`grade`, over a list of
    trajectory sub-specs under ``of`` (a missing/empty ``of`` raises — fail loud).
    """
    calls = list(tool_sequence or [])
    gtype = spec.get("type")

    if gtype == "first_tool":
        target = str(spec["value"])
        return bool(calls) and calls[0] == target
    if gtype == "max_tool_calls":
        return len(calls) <= int(spec["value"])
    if gtype == "tool_called":
        return str(spec["value"]) in calls
    if gtype == "tool_not_called":
        return str(spec["value"]) not in calls
    if gtype == "tool_sequence":
        expected = [str(x) for x in spec["value"]]
        return _is_subsequence(expected, calls)
    if gtype in ("all_of", "any_of"):
        sub = spec.get("of")
        if not sub:
            raise ValueError(f"{gtype} grader requires a non-empty 'of' list")
        combiner = all if gtype == "all_of" else any
        return combiner(grade_trajectory(s, calls) for s in sub)
    raise ValueError(f"unknown trajectory grader type {gtype!r}")


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Whether ``needle`` appears in ``haystack`` in order (gaps allowed)."""
    it = iter(haystack)
    return all(item in it for item in needle)


__all__ = ["grade", "grade_trajectory", "TRAJECTORY_TYPES"]
