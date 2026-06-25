"""Regression test for the guardrail aggregate-scan-budget fix (Phase-4 re-attack).

The per-leaf PII scan cap (``_MAX_PII_SCAN_LEN`` = 128 KiB) bounds ONE string, but the
post-execution hook's ``_guard_value`` walk used to inspect an *unbounded* container of
capped leaves synchronously on the event loop. A benign-looking tool result such as
``{"items": [<128 KiB string> × N]}`` therefore multiplied the GIL-held regex CPU by the
leaf count and froze the loop for seconds (re-attack measured 50 leaves → 199 ms, 300 →
1.19 s, 1000 → 3.97 s).

The fix threads an aggregate byte budget (``_MAX_TOTAL_SCAN_BYTES`` = 512 KiB) through the
whole walk: once exhausted, remaining leaves pass through unscanned, so total inspect CPU
per tool result is bounded regardless of leaf count. Early leaves are still scanned (PII up
to the budget is redacted); only the far tail of an enormous multi-leaf result is skipped.
"""

from __future__ import annotations

import time

from himmy.services.guardrails import (
    build_guardrail_pipeline,
    build_guardrail_post_hook,
)
from himmy.services.guardrails.tool_hook import _MAX_TOTAL_SCAN_BYTES
from himmy.services.tools.models import (
    ToolBackendKind,
    ToolDefinition,
    ToolExecutionResult,
)
from tests.conftest import run_async

_LEAF = "1." * 65_536  # ~128 KiB of dotted digits (worst-case amplifier for the rules)


def _hook():  # type: ignore[no-untyped-def]
    return build_guardrail_post_hook(build_guardrail_pipeline(["pii"]))


def _result(value: object) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="c1", tool_name="t", outcome="success", result=value
    )


def _defn() -> ToolDefinition:
    return ToolDefinition(name="t", kind=ToolBackendKind.LOCAL)


def test_many_leaf_container_is_bounded_and_does_not_freeze() -> None:
    """A 200-leaf container of 128 KiB pathological strings completes fast.

    Without the aggregate budget this is ~200 × per-leaf-scan ≈ tens of seconds of
    GIL-held regex on the event loop. With the 512 KiB aggregate budget only the first
    ~4 leaves are scanned and the rest pass through, so it completes in well under a
    second. Budget here: < 3 s (generous; the real target is ~0.5 s).
    """
    payload = {"items": [_LEAF for _ in range(200)]}
    hook = _hook()
    start = time.perf_counter()
    out = run_async(hook(_result(payload), _defn()))
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"aggregate scan not bounded: took {elapsed:.2f}s"
    # The structure is preserved (200 leaves returned).
    assert isinstance(out, dict) and len(out["items"]) == 200


def test_early_pii_redacted_late_pii_passes_through_after_budget() -> None:
    """PII within the aggregate budget is redacted; PII far past it passes through.

    This documents the deliberate tradeoff: the guard still protects the first
    ``_MAX_TOTAL_SCAN_BYTES`` of a tool result, but a giant multi-leaf result's far tail
    is no longer scanned (the availability/DoS bound). An early email is redacted; an
    email placed well beyond the budget (after >512 KiB of leaves) is not.
    """
    big = "1." * 65_536  # ~128 KiB each; 8 of them = ~1 MiB > 512 KiB budget
    items = ["contact early: a@early.com"] + [big for _ in range(8)]
    items += ["contact late: z@late.com"]
    out = run_async(_hook()(_result({"items": items}), _defn()))
    leaves = out["items"]
    # Early leaf: inside the budget -> email redacted.
    assert "a@early.com" not in leaves[0]
    assert "[REDACTED-EMAIL]" in leaves[0]
    # Late leaf: budget exhausted by the 8×128 KiB middle -> passed through unscanned.
    assert leaves[-1] == "contact late: z@late.com"


def test_tiny_leaf_flood_is_bounded() -> None:
    """A flood of hundreds of thousands of *tiny* leaves is also bounded.

    Each leaf costs ~0 bytes but a full ``inspect`` call, so a byte-only budget could be
    bypassed by leaf count. The per-leaf minimum charge (``_MIN_LEAF_CHARGE``) plus the
    short-circuit-the-whole-walk once exhausted bound the number of inspect calls, so a
    300k-leaf container completes fast instead of running 300k regex scans on the loop.
    """
    payload = {"items": ["x" for _ in range(300_000)]}
    start = time.perf_counter()
    out = run_async(_hook()(_result(payload), _defn()))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"tiny-leaf flood not bounded: took {elapsed:.2f}s"
    assert isinstance(out, dict) and len(out["items"]) == 300_000


def test_wide_container_walk_is_bounded() -> None:
    """A wide container of MILLIONS of leaves (incl. non-string) can't stall the walk.

    The byte budget never charges non-string (int) leaves, and the per-child
    short-circuit still let the parent comprehension iterate + rebuild every element —
    an O(N) GIL-held walk (re-attack measured a 20M-key int dict at 6.4 s). The node
    budget + fanout cap (``len(value) > remaining`` → return the container unwalked)
    makes worst-case time independent of element count. Each case must finish fast.
    """
    cases = [
        {"items": list(range(5_000_000))},  # 5M int leaves (byte budget irrelevant)
        dict.fromkeys((str(i) for i in range(3_000_000)), 0),  # 3M-key int-val dict
        {"rows": [{"id": i, "v": "a"} for i in range(2_000_000)]},  # wide dump shape
    ]
    for payload in cases:
        start = time.perf_counter()
        run_async(_hook()(_result(payload), _defn()))
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"wide-container walk not bounded: {elapsed:.2f}s"


def test_deeply_nested_container_does_not_raise() -> None:
    """A deeply nested narrow container is bounded (no RecursionError → dropped result).

    Each level is len 1 so the fanout cap never trips; without a depth cap the walk would
    recurse to Python's limit and raise, which the tool service turns into an EXECUTION_ERROR
    that drops the whole result (a crafted-result denial). The walk-depth cap returns the
    over-deep subtree unwalked instead. 5000 deep must complete fast and not raise.
    """
    deep: list = []
    cur = deep
    for _ in range(5000):
        nxt: list = []
        cur.append(nxt)
        cur = nxt
    start = time.perf_counter()
    out = run_async(_hook()(_result({"d": deep}), _defn()))
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0 and isinstance(out, dict)


def test_normal_sized_result_is_fully_scanned() -> None:
    """A realistic small/medium result stays fully scanned (no coverage regression).

    Total content well under the 512 KiB budget, so every leaf is inspected and all PII
    is redacted — the cap only degrades coverage for pathologically large results.
    """
    assert _MAX_TOTAL_SCAN_BYTES == 512 * 1024
    items = [f"row {i}: mail user{i}@corp.com" for i in range(50)]
    out = run_async(_hook()(_result({"rows": items}), _defn()))
    joined = "\n".join(out["rows"])
    assert "@corp.com" not in joined
    assert joined.count("[REDACTED-EMAIL]") == 50
