"""LIVE self-learning proof (Layer 3): a REAL model is steered away from a flagged tool.

This is the for-real end of the loop, and it is the pytest twin of
``examples/self_learning_live_study.py``. Layers 1-2 prove himmy's internal bookkeeping
(reputation -> reorder + hint + ``LEARNING_APPLIED``) deterministically; this drives a
REAL OpenRouter model and asserts the pre-registered causal effect: with the tool history
held constant, turning ``self_learning`` ON measurably reduces how often the model reaches
for the flagged tool.

Why this is NOT the old hollow proof (the three confounds it kills):

1. NEUTRAL symmetric tools (``tool_alpha`` / ``tool_beta``, byte-identical descriptions)
   and a ZERO-preference instruction — so a pass cannot come from the model reading
   "reliable" in a tool name or "prefer the reliable tool" in the persona.
2. Condition B (history present, learning OFF) vs condition C (history present, learning
   ON): the clean causal contrast holds history constant and toggles only the feature.
3. Every turn is driven through the PRODUCTION ``spec.make_task(prompt)`` path, and a
   pre-flight invariant asserts the ``<context key="learned_hints">`` block actually
   reaches the system prompt — the old test hand-built ``Task`` and the hint never reached
   the model.

Counterbalancing (which tool is flagged x declared order) cancels name + first-position
bias. Real provider token usage > 0 is asserted (no stub produces real usage). Nothing is
stubbed, no assertion is weakened. N is modest for runtime; the full Hermes-grade study
(N=24/condition + the ablation + escalation) lives in the example script.

Skipped automatically when ``OPENROUTER_API_KEY`` is absent, so the offline suite is
unaffected.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# himmy does NOT auto-load .env — load the repo .env so OPENROUTER_API_KEY reaches
# get_secret()'s EnvSecrets backend (which reads os.environ) before we build anything.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

_HAS_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))
_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

pytestmark = pytest.mark.skipif(
    not _HAS_KEY, reason="requires OPENROUTER_API_KEY (live OpenRouter test)"
)

# Reuse the study harness: neutral tools, production-path runner, counterbalancing, and
# the by-hand Fisher/Newcombe statistics (no scipy dependency). Importing it here keeps the
# test and the example provably in lockstep — same tools, same wiring, same metric.
from examples.self_learning_live_study import (  # noqa: E402
    ALL,
    fisher_one_sided_less,
    newcombe_diff_ci,
    preflight_treatment,
    run_condition,
)


def _isolate_durable_state(tmp_path: Path) -> None:
    """Point the durable spine + run store at a fresh temp dir for this test.

    The content-addressed entity spine (``.himmy/spine.db``) is durable and shared across
    the project tree; the neutral tools register the same stable_id every trial, so a stale
    on-disk tool_definition with a different payload would raise a content-address violation.
    Isolating the spine + store keeps every registration self-consistent. Must run BEFORE any
    runtime build (the spine path is resolved eagerly).
    """
    os.environ["HIMMY_SPINE_PATH"] = str(tmp_path / "spine.db")
    os.environ["HIMMY_STORE_PATH"] = str(tmp_path / "store.db")
    os.environ.pop("HIMMY_DATABASE_URL", None)
    os.environ.pop("HIMMY_SPINE_DATABASE_URL", None)


def test_real_model_steered_away_from_flagged_tool(tmp_path: Path) -> None:
    """B vs C on a REAL model: self_learning ON reduces flagged-tool selection.

    Bounded for runtime (N=4 per arrangement -> 16 trials per condition, 4 conditions =
    64 live turns) but rigorous: counterbalanced, production make_task path, pre-flight
    invariants verified, real token usage asserted, pre-registered pass criterion enforced.
    """
    _isolate_durable_state(tmp_path)

    async def _run() -> None:
        # --- Pre-flight: reorder + annotation + hint + hint-in-system-prompt MUST all
        # fire before any live call counts. A failure here is INVALID (the treatment was
        # not applied), so we fail loudly rather than silently testing nothing.
        preflight = await preflight_treatment(_MODEL)
        assert preflight.ok, (
            "pre-flight invariants failed (treatment not actually applied):\n  "
            + "\n  ".join(preflight.details)
        )

        n = 4  # per arrangement; 4 arrangements => 16 trials per condition
        sem = asyncio.Semaphore(6)

        # A (baseline), B (history, learning OFF), C (history, learning ON).
        cond_a = await run_condition(
            _MODEL, name="A", self_learning=False, seed=False, lever=None,
            n_per_arrangement=n, semaphore=sem,
        )
        cond_b = await run_condition(
            _MODEL, name="B", self_learning=False, seed=True, lever=None,
            n_per_arrangement=n, semaphore=sem,
        )
        cond_c = await run_condition(
            _MODEL, name="C", self_learning=True, seed=True, lever=ALL,
            n_per_arrangement=n, semaphore=sem,
        )

        # --- Real round-trip proof: a stub cannot produce real provider usage.
        for cond in (cond_a, cond_b, cond_c):
            assert cond.input_tokens > 0, (
                f"condition {cond.name}: no input tokens — the live call did not happen"
            )
            assert cond.output_tokens > 0, (
                f"condition {cond.name}: no output tokens — the live call did not happen"
            )

        # --- Tool-call coverage: a condition where the model rarely calls a tool is a
        # measurement failure, not a result. Require >= 75% call rate at this modest N.
        for cond in (cond_b, cond_c):
            call_rate = cond.call_rate or 0.0
            assert call_rate >= 0.75, (
                f"condition {cond.name}: only {call_rate:.0%} of trials called a tool "
                f"(measurement failure, not a result)"
            )

        pr_a = cond_a.pickrate
        pr_b = cond_b.pickrate
        pr_c = cond_c.pickrate
        assert pr_b is not None and pr_c is not None

        # --- Surface the real numbers (visible with `pytest -s`) for transparency.
        print(
            "\n[self-learning live] flagged-tool first-call rate:"
            f"\n  A (baseline)      = {pr_a}"
            f"\n  B (history, off)  = {pr_b}  (flagged={cond_b.flagged_hits}, "
            f"unflagged={cond_b.unflagged_hits})"
            f"\n  C (treatment, on) = {pr_c}  (flagged={cond_c.flagged_hits}, "
            f"unflagged={cond_c.unflagged_hits})"
            f"\n  tokens B in/out   = {cond_b.input_tokens}/{cond_b.output_tokens}"
            f"\n  tokens C in/out   = {cond_c.input_tokens}/{cond_c.output_tokens}"
        )

        # --- Pre-registered statistics on the B -> C contrast.
        a = cond_b.flagged_hits
        b = cond_b.unflagged_hits
        c = cond_c.flagged_hits
        d = cond_c.unflagged_hits
        p_value = fisher_one_sided_less(a, b, c, d)
        reduction = pr_b - pr_c
        ci = newcombe_diff_ci(a, a + b, c, c + d)
        directional = 0
        b_by_label = {cell.arrangement.label: cell for cell in cond_b.cells}
        for c_cell in cond_c.cells:
            b_cell = b_by_label[c_cell.arrangement.label]
            b_rate = (
                b_cell.flagged_hits / b_cell.trials_with_call
                if b_cell.trials_with_call
                else None
            )
            c_rate = (
                c_cell.flagged_hits / c_cell.trials_with_call
                if c_cell.trials_with_call
                else None
            )
            if b_rate is not None and c_rate is not None and c_rate <= b_rate:
                directional += 1

        print(
            f"\n  reduction (B-C)   = {reduction:+.3f}"
            f"\n  Newcombe 95% CI   = [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            f"\n  Fisher 1-sided p  = {p_value:.4f}"
            f"\n  directional cells = {directional}/4"
        )

        # --- Pre-registered PASS criterion (the same one the example evaluates):
        #   significant (p < 0.05) AND a meaningful drop (>= 0.20) AND it survives
        #   counterbalancing (C <= B in >= 3 of 4 arrangements).
        assert p_value < 0.05, (
            f"B vs C not significant (Fisher one-sided p={p_value:.4f}); on this run the "
            f"real model showed NO detectable steering (pickrate B={pr_b}, C={pr_c}). "
            "This is an honest null — re-run, raise N via the example study, or escalate "
            "to a stronger model; do NOT weaken this assertion."
        )
        assert reduction >= 0.20, (
            f"B vs C significant but the effect is small/weak (reduction={reduction:+.3f} "
            f"< 0.20; pickrate B={pr_b}, C={pr_c})."
        )
        assert directional >= 3, (
            f"the reduction does not survive counterbalancing (held in only {directional}/4 "
            "arrangements) — likely a single name/position fluke, not real steering."
        )

        # --- Sanity: B approx A (seeded history alone, with the feature off, does not move
        # the model). A weak guard at this modest N; the example study runs the full test.
        if pr_a is not None:
            assert abs(pr_a - pr_b) < 0.25, (
                f"history alone (no feature) shifted the model: A={pr_a}, B={pr_b} — the "
                "B-vs-C effect may be contaminated by a history leak."
            )

    asyncio.run(_run())
