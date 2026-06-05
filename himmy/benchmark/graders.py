"""Deterministic correctness graders for benchmark tasks.

A grader is a small declarative spec (``{"type": ..., ...}``) checked against the
agent's answer — pure, fast, and reproducible (no model calls), so a task's pass/fail
is objective and a benchmark is comparable run-to-run. Composable via ``all_of`` /
``any_of``. For genuinely open-ended tasks, an ``llm_judge`` grader can be layered on
top by the caller, but the defaults here are rule-based on purpose.
"""

from __future__ import annotations

import re
from typing import Any


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
    if gtype == "all_of":
        return all(grade(s, answer) for s in spec.get("of", []))
    if gtype == "any_of":
        return any(grade(s, answer) for s in spec.get("of", []))
    raise ValueError(f"unknown grader type {gtype!r}")


__all__ = ["grade"]
