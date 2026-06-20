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
from typing import TYPE_CHECKING, Any

from himmy.benchmark.trajectory import RecordedCall, Trajectory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Argument-/result-/parallelism-level trajectory grader types. These require the full
#: structured :class:`~himmy.benchmark.trajectory.Trajectory` (call args, results, and
#: turn grouping), not just the flat name-only sequence the legacy graders use. A flat
#: ``list[str]`` carries no args/results, so :func:`grade_trajectory` returns ``False``
#: (no matching arg/result — never crashes) if one is graded against a name-only
#: sequence. In practice the runner always passes a full :class:`Trajectory`.
ARG_TRAJECTORY_TYPES = frozenset(
    {
        "tool_called_with",
        "arg_equals",
        "arg_matches",
        "result_contains",
        "parallel_in_turn",
        "max_parallel",
        "result_feeds_arg",
    }
)

#: Grader ``type`` values that grade the ordered tool-call trajectory rather than the
#: final answer text. :func:`grade` uses this set to reject a trajectory-type leaf placed
#: in an answer ``grade`` block (fail-loud), not to silently route it to the other family.
TRAJECTORY_TYPES = (
    frozenset(
        {
            "first_tool",
            "max_tool_calls",
            "tool_called",
            "tool_not_called",
            "tool_sequence",
        }
    )
    | ARG_TRAJECTORY_TYPES
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


def grade_trajectory(
    spec: dict[str, Any],
    tool_sequence: Sequence[str] | Trajectory | None,
) -> bool:
    """Return whether the tool-call trajectory satisfies the trajectory grader ``spec``.

    ``tool_sequence`` is EITHER the agent's tools called *in order, with repeats* (the
    flat name-only ``list[str]`` — a benchmark trial's
    :attr:`~himmy.benchmark.models.TrialResult.tools_called`) OR the richer structured
    :class:`~himmy.benchmark.trajectory.Trajectory` (call args, results, and turn
    grouping). The name-only graders accept either; the argument-/result-/parallelism
    graders need a :class:`Trajectory` (a flat list carries no args/results, so they
    return ``False`` — no matching arg — rather than crashing or mis-scoring). In
    practice the runner always passes a full :class:`Trajectory`.

    Name-only predicate types (work on either input):

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

    Argument-/result-/parallelism predicate types (require a :class:`Trajectory` — the
    BFCL AST-check equivalent for "right tool, WRONG args", the dominant small-model
    failure):

    * ``tool_called_with`` — ``spec["tool"]`` was called somewhere with arguments
      MATCHING ``spec["args"]`` (each listed arg present and equal; extra args ignored).
    * ``arg_equals`` — some call to ``spec["tool"]`` passed ``spec["arg"]`` exactly equal
      to ``spec["value"]`` (with forgiving scalar coercion: ``"42"`` matches ``42``).
    * ``arg_matches`` — some call to ``spec["tool"]`` passed ``spec["arg"]`` whose string
      form matches the regex ``spec["value"]`` (case-insensitive).
    * ``result_contains`` — some call to ``spec["tool"]`` returned a result whose string
      form contains ``spec["value"]`` (case-insensitive substring). Only calls whose
      result was captured count.
    * ``parallel_in_turn`` — ``spec["value"]`` (a list of tool names) were ALL called
      within a single assistant turn (within-turn parallelism).
    * ``max_parallel`` — at most ``spec["value"]`` tool calls in any single turn.
    * ``result_feeds_arg`` — the BFCL multi-turn data-flow check: a tool result feeds the
      NEXT call's argument. Some call to ``spec["from_tool"]`` returned a result, and a
      LATER call to ``spec["to_tool"]`` passed ``spec["arg"]`` whose value was carried from
      that result — i.e. the model used the captured result downstream rather than
      re-deriving or hallucinating it. "Carried" means the later arg's string form and the
      earlier result's string form share the token ``spec["value"]`` (case-insensitive)
      when ``value`` is given; otherwise it means the arg's (non-empty) string form is a
      substring of the result's string form (the result literally contains what was passed
      on). Ordering matters: the producing call (``from_tool``) must come before the
      consuming call (``to_tool``) in trajectory order. Only ``from_tool`` calls whose
      result was captured count (``has_result``).

    Composes under ``all_of`` / ``any_of`` exactly like :func:`grade`, over a list of
    trajectory sub-specs under ``of`` (a missing/empty ``of`` raises — fail loud).
    """
    traj = _as_trajectory(tool_sequence)
    calls = traj.names
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
    if gtype in ARG_TRAJECTORY_TYPES:
        return _grade_arg_trajectory(spec, traj, gtype)
    if gtype in ("all_of", "any_of"):
        sub = spec.get("of")
        if not sub:
            raise ValueError(f"{gtype} grader requires a non-empty 'of' list")
        combiner = all if gtype == "all_of" else any
        return combiner(grade_trajectory(s, traj) for s in sub)
    raise ValueError(f"unknown trajectory grader type {gtype!r}")


def _as_trajectory(value: Sequence[str] | Trajectory | None) -> Trajectory:
    """Coerce the grader input to a :class:`Trajectory`.

    A :class:`Trajectory` passes through. A flat name-only sequence (the legacy input,
    or ``None``) is wrapped as a single turn of arg-less calls — enough for the name-only
    graders; the arg/result graders then find no matching arg/result and return ``False``
    (never crash) on such an input.
    """
    if isinstance(value, Trajectory):
        return value
    names = list(value or [])
    return Trajectory(turns=[[RecordedCall(name=str(n)) for n in names]] if names else [])


def _grade_arg_trajectory(spec: dict[str, Any], traj: Trajectory, gtype: str) -> bool:
    """Argument-/result-/parallelism graders over the structured trajectory."""
    if gtype == "parallel_in_turn":
        expected = {str(x) for x in spec["value"]}
        return any(expected <= {c.name for c in turn} for turn in traj.turns)
    if gtype == "max_parallel":
        return traj.max_parallel <= int(spec["value"])
    if gtype == "result_feeds_arg":
        return _grade_result_feeds_arg(spec, traj)
    # The remaining graders inspect a named tool's calls.
    tool = str(spec["tool"])
    matched = [c for c in traj.calls if c.name == tool]
    if gtype == "tool_called_with":
        want = dict(spec.get("args") or {})
        return any(_args_match(c.args, want) for c in matched)
    if gtype == "arg_equals":
        arg = str(spec["arg"])
        want = spec["value"]
        return any(arg in c.args and _scalar_equal(c.args[arg], want) for c in matched)
    if gtype == "arg_matches":
        arg = str(spec["arg"])
        pattern = str(spec["value"])
        return any(
            arg in c.args
            and re.search(pattern, _as_text(c.args[arg]), re.IGNORECASE) is not None
            for c in matched
        )
    if gtype == "result_contains":
        needle = str(spec["value"]).lower()
        return any(
            c.has_result and needle in _as_text(c.result).lower() for c in matched
        )
    raise ValueError(f"unknown trajectory grader type {gtype!r}")  # pragma: no cover


def _grade_result_feeds_arg(spec: dict[str, Any], traj: Trajectory) -> bool:
    """Whether a ``from_tool`` result was carried into a LATER ``to_tool`` argument.

    The multi-turn data-flow predicate: scans the flat call sequence in trajectory order
    and, for each consuming call to ``to_tool`` that passed ``arg``, checks whether any
    EARLIER producing call to ``from_tool`` (with a captured result) supplied that value.
    When ``spec["value"]`` is given, both the earlier result and the later arg must contain
    that token (the data-flow carried a known value); otherwise the later arg's string form
    must be a non-empty substring of the earlier result's string form (the result literally
    contained what was passed on). Order is enforced over the flat call list so a result can
    only feed a call that comes after it.
    """
    from_tool = str(spec["from_tool"])
    to_tool = str(spec["to_tool"])
    arg = str(spec["arg"])
    token = spec.get("value")
    calls = traj.calls
    # Producing calls (with a captured result), paired with their flat index for ordering.
    producers = [
        (i, c)
        for i, c in enumerate(calls)
        if c.name == from_tool and c.has_result
    ]
    if not producers:
        return False
    for j, consumer in enumerate(calls):
        if consumer.name != to_tool or arg not in consumer.args:
            continue
        arg_text = _as_text(consumer.args[arg]).strip()
        if not arg_text:
            continue
        for i, producer in producers:
            if i >= j:
                break  # producer must precede the consumer
            result_text = _as_text(producer.result)
            if token is not None:
                tok = _as_text(token).lower()
                if (
                    tok in result_text.lower()
                    and tok in arg_text.lower()
                ):
                    return True
            elif arg_text.lower() in result_text.lower():
                return True
    return False


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Whether every ``expected`` key is present in ``actual`` and scalar-equal.

    Extra keys in ``actual`` are ignored (the model passing additional optional args is
    not a failure). Scalar values compare with forgiving coercion (``"42"`` == ``42``);
    non-scalar (list/dict) values compare structurally for equality.
    """
    for key, want in expected.items():
        if key not in actual:
            return False
        if not _scalar_equal(actual[key], want):
            return False
    return True


def _scalar_equal(actual: Any, expected: Any) -> bool:
    """Forgiving equality: exact match, else compare string forms (``"42"`` == ``42``).

    Models often stringify numeric/boolean args, so a literal ``==`` would spuriously
    fail "right tool, right value, wrong type". We accept equality if the values are
    equal OR their normalized string forms are equal.
    """
    if actual == expected:
        return True
    return _as_text(actual).strip().lower() == _as_text(expected).strip().lower()


def _as_text(value: Any) -> str:
    """Stable string form of an arg/result value for substring/regex/coercion checks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, list, dict)):
        import json

        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Whether ``needle`` appears in ``haystack`` in order (gaps allowed)."""
    it = iter(haystack)
    return all(item in it for item in needle)


__all__ = [
    "grade",
    "grade_trajectory",
    "TRAJECTORY_TYPES",
    "ARG_TRAJECTORY_TYPES",
]
