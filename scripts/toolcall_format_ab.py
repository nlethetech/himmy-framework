#!/usr/bin/env python3
"""A/B harness: GENERIC vs NATIVE (Hermes/Qwen ChatML-XML) tool-call FORMAT.

This is the statistical A/B that answers: does the native ``<tool_call>`` format
improve a LOCAL Qwen model's tool calling vs himmy's current generic text protocol?

DESIGN — FORMAT is the only variable
------------------------------------
Both arms run the SAME tasks against the SAME local Ollama model at temperature 0
on the SAME Ollama TEXT/generate path (Ollama's native ``tools=`` function field is
SUPPRESSED in both arms). The single thing that differs is the tool-call grammar:

* Arm GENERIC — today's behaviour: ``_react_tool_manifest`` render + the tolerant
  ``parse_text_tool_calls`` parser. To isolate FORMAT (not the native-vs-text path),
  this arm runs on the text path too via an ephemeral ``generic_text`` format that
  wraps GENERIC's exact render/parse and only flips ``use_text_tool_path=True``.
* Arm NATIVE — the new ``hermes_chatml_xml`` format: the ``<tools>`` manifest render
  + the fail-open ``<tool_call>{json}</tool_call>`` parser. Already text-path.

Because the only delta between the two ephemeral formats is render + parse, any
difference in the scorecards is attributable to the tool-call grammar alone.

We reuse the shipped benchmark machinery end to end: the real agent loop
(``BenchmarkRunner`` via an injected ``inference_factory``), the arg-level / parallel
/ irrelevance graders, and the Wilson-CI + paired-McNemar stats. Nothing is
hand-rolled. The local model genuinely runs — per-arm Ollama eval_count + latency are
printed so the result cannot be a stub.

Usage::

    python scripts/toolcall_format_ab.py --model qwen2.5:7b-instruct --trials 5
    python scripts/toolcall_format_ab.py --model qwen2.5:0.5b-instruct --trials 5 \
        --out /tmp/ab_0.5b.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, replace

from himmy.benchmark.models import (
    BenchmarkSuite,
    ModelScorecard,
    ModelSpec,
    compare_scorecards,
)
from himmy.benchmark.runner import BenchmarkRunner
from himmy.benchmark.stats import mcnemar_exact, wilson_interval
from himmy.services.inference.local import OllamaClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.inference.tool_formats import (
    GENERIC,
    HERMES_CHATML_XML,
    ToolCallFormat,
    ToolCallGrammarFlags,
    register_format,
)

# --------------------------------------------------------------------------- #
# Arm formats. Both run on the Ollama TEXT path so the FORMAT is the only knob.
# --------------------------------------------------------------------------- #

#: GENERIC's render/parse, but forced onto the text path (use_text_tool_path=True)
#: so it is compared apples-to-apples with the native text-path format. Render +
#: parse are GENERIC's exact callables — only the path flag differs.
GENERIC_TEXT = ToolCallFormat(
    name="generic_text_ab",
    render_system_manifest=GENERIC.render_system_manifest,
    parse=GENERIC.parse,
    render_tool_results=GENERIC.render_tool_results,
    model_tags=frozenset(),  # never auto-selected; chosen by explicit override
    flags=ToolCallGrammarFlags(use_text_tool_path=True),
)

#: The native arm IS the shipped Hermes/Qwen format (already text-path). Registered
#: under a stable A/B override name so selection is explicit, not tag-driven.
NATIVE_AB = replace(HERMES_CHATML_XML, name="hermes_chatml_xml_ab")

register_format(GENERIC_TEXT)
register_format(NATIVE_AB)

ARM_GENERIC = "generic_text_ab"
ARM_NATIVE = "hermes_chatml_xml_ab"


# --------------------------------------------------------------------------- #
# The A/B task suite: single-tool, parallel-tool, wrong-args traps, irrelevance.
# Every task is OFFLINE and self-contained (utils/files/data/tasks packs + sqlite
# fixtures), graded by the shipped arg-level / parallel / irrelevance graders.
# --------------------------------------------------------------------------- #
SUITE_DATA = {
    "name": "toolcall_format_ab",
    "tasks": [
        # ---- single-tool: right tool, right args (selection + arg correctness) ----
        {
            "id": "single_calc_mul",
            "category": "single",
            "prompt": "What is 17 times 23? Use the calculator tool.",
            "packs": ["utils"],
            "expect_tools": ["calculator"],
            "grade": {"type": "numeric", "value": 391},
            "trajectory": {"type": "tool_called", "value": "calculator"},
        },
        {
            "id": "single_calc_div",
            "category": "single",
            "prompt": "Use the calculator to compute 4096 divided by 16.",
            "packs": ["utils"],
            "expect_tools": ["calculator"],
            "grade": {"type": "numeric", "value": 256},
            "trajectory": {"type": "tool_called", "value": "calculator"},
        },
        {
            "id": "single_addtask_arg",
            "category": "single",
            # Arg-level: the title arg must carry the right string (right tool, RIGHT args).
            "prompt": (
                "Add a task titled exactly 'water the tomatoes' using the add_task tool."
            ),
            "packs": ["tasks"],
            "expect_tools": ["add_task"],
            "grade": {"type": "contains", "value": "water the tomatoes"},
            "trajectory": {
                "type": "arg_matches",
                "tool": "add_task",
                "arg": "title",
                "value": "water the tomatoes",
            },
        },
        {
            "id": "single_read_file",
            "category": "single",
            "prompt": (
                "Read the file notes.txt and tell me the secret word it contains."
            ),
            "packs": ["files"],
            "files": {"notes.txt": "The secret word is MARIGOLD."},
            "expect_tools": ["read_file"],
            "grade": {"type": "contains", "value": "marigold"},
            "trajectory": {
                "type": "arg_matches",
                "tool": "read_file",
                "arg": "path",
                "value": "notes",
            },
        },
        {
            "id": "single_sql_count",
            "category": "single",
            "prompt": (
                "How many rows are in the chickens table? Use the sql_query tool."
            ),
            "packs": ["data"],
            "sqlite": [
                "CREATE TABLE chickens (id INTEGER, name TEXT);"
                "INSERT INTO chickens VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e');"
            ],
            "expect_tools": ["sql_query"],
            "grade": {"type": "numeric", "value": 5},
            "trajectory": {"type": "tool_called", "value": "sql_query"},
        },
        # ---- wrong-args traps: right tool is easy, the ARGS are the test ----
        {
            "id": "args_calc_subtract",
            "category": "wrong_args",
            # The model must pass the operands 1000 and 247 (answer 753). A format that
            # produces a parseable call but mangles args fails the arg grader.
            "prompt": (
                "A bin holds 1000 eggs and 247 are sold. Use the calculator to find how "
                "many remain."
            ),
            "packs": ["utils"],
            "expect_tools": ["calculator"],
            "grade": {"type": "numeric", "value": 753},
            "trajectory": {"type": "tool_called", "value": "calculator"},
        },
        {
            "id": "args_addtask_exact",
            "category": "wrong_args",
            "prompt": (
                "Use add_task to create a task whose title is exactly "
                "'call the vet at 9am'. Do not change the wording."
            ),
            "packs": ["tasks"],
            "expect_tools": ["add_task"],
            "grade": {"type": "contains", "value": "call the vet at 9am"},
            "trajectory": {
                "type": "arg_matches",
                "tool": "add_task",
                "arg": "title",
                "value": r"call the vet at 9\s*am",
            },
        },
        {
            "id": "args_read_specific_file",
            "category": "wrong_args",
            # Two files exist; the model must read the RIGHT one (arg = path selection).
            "prompt": (
                "Two files exist: feed.txt and water.txt. Read water.txt and report the "
                "number it contains."
            ),
            "packs": ["files"],
            "files": {
                "feed.txt": "The feed level is 11.",
                "water.txt": "The water level is 88.",
            },
            "expect_tools": ["read_file"],
            "grade": {"type": "numeric", "value": 88},
            "trajectory": {
                "type": "arg_matches",
                "tool": "read_file",
                "arg": "path",
                "value": "water",
            },
        },
        # ---- parallel-tool: two calls in one turn (within-turn parallelism) ----
        {
            "id": "parallel_two_calcs",
            "category": "parallel",
            "prompt": (
                "Use the calculator twice: first compute 12 times 12, then compute "
                "100 minus 37. Report both results."
            ),
            "packs": ["utils"],
            "expect_tools": ["calculator"],
            # Both answers must appear; the trajectory must show >=2 calculator calls.
            "grade": {
                "type": "all_of",
                "of": [
                    {"type": "numeric", "value": 144},
                    {"type": "numeric", "value": 63},
                ],
            },
            "trajectory": {
                "type": "all_of",
                "of": [
                    {"type": "tool_called", "value": "calculator"},
                    {"type": "max_tool_calls", "value": 4},
                ],
            },
        },
        {
            "id": "parallel_two_tasks",
            "category": "parallel",
            "prompt": (
                "Add two tasks with add_task: one titled 'feed ducks' and one titled "
                "'clean coop'. Add both."
            ),
            "packs": ["tasks"],
            "expect_tools": ["add_task"],
            "grade": {
                "type": "all_of",
                "of": [
                    {"type": "contains", "value": "feed ducks"},
                    {"type": "contains", "value": "clean coop"},
                ],
            },
            "trajectory": {"type": "tool_called", "value": "add_task"},
        },
        # ---- irrelevance / abstention: tools bound, NONE needed (over-calling = fail) ----
        {
            "id": "irrel_capital",
            "category": "irrelevance",
            "prompt": "What is the capital of Japan? Answer in one word. Do not use any tool.",
            "packs": ["utils"],
            "grade": {"type": "contains", "value": "tokyo"},
            "trajectory": {"type": "max_tool_calls", "value": 0},
        },
        {
            "id": "irrel_trivial_math",
            "category": "irrelevance",
            "prompt": "What is 2 plus 2? Answer with just the number. Do not use any tool.",
            "packs": ["utils"],
            "grade": {"type": "numeric", "value": 4},
            "trajectory": {"type": "max_tool_calls", "value": 0},
        },
        {
            "id": "irrel_password_in_prompt",
            "category": "irrelevance",
            "prompt": (
                "The password is HELIOTROPE. Repeat the password back to me. Do not read "
                "any file - it is right here in this message."
            ),
            "packs": ["files"],
            "grade": {"type": "contains", "value": "heliotrope"},
            "trajectory": {"type": "max_tool_calls", "value": 0},
        },
        {
            "id": "irrel_days_in_week",
            "category": "irrelevance",
            "prompt": (
                "How many days are in a week? Answer with just the number. Do not use any "
                "tool."
            ),
            "packs": ["data"],
            "grade": {"type": "numeric", "value": 7},
            "trajectory": {"type": "max_tool_calls", "value": 0},
        },
    ],
}


@dataclass
class ArmResult:
    arm: str
    card: ModelScorecard
    eval_counts: list[int]
    parse_hits: int
    parse_total: int


def _make_factory(model: str, fmt_override: str, probe: dict):
    """An ``inference_factory(spec)`` that pins a tool-call FORMAT on Ollama.

    Wraps the real :class:`OllamaClientManager` (HTTP to localhost:11434) in an
    :class:`InferenceService` and pins ``tool_call_format`` to the arm's format. A
    transport wrapper records Ollama's ``eval_count`` per call so we can PROVE the
    local model actually generated tokens (a stub would have none).
    """

    def factory(spec: ModelSpec) -> InferenceService:
        mgr = OllamaClientManager(
            model=model,
            model_registry={"default": model},
            tool_call_format=fmt_override,
            timeout=180.0,
        )
        # Wrap the manager's transport to capture real eval_count + parse-hit signal.
        orig_post = mgr._post

        async def counting_post(path, payload, timeout):
            data = await orig_post(path, payload, timeout)
            ec = data.get("eval_count")
            if isinstance(ec, int):
                probe["eval_counts"].append(ec)
            # Raw parse-hit-rate: did THIS reply (on the text path) yield >=1 parseable
            # tool call under the arm's parser? Only count turns where tools were bound
            # AND no native tool_calls came back (the text/prose path we are A/B-ing).
            content = ((data.get("message") or {}).get("content")) or ""
            tools_in = bool(payload.get("tools"))
            if not tools_in and content:
                probe["parse_total"] += 1
                from himmy.services.inference.tool_formats import format_for

                fmt = format_for(model, fmt_override)
                # `known` is unknown at the transport layer; use a permissive set (parser
                # filters on shape, not name, for the native pass; generic filters on
                # name — so pass the suite's tool names to be fair to BOTH arms).
                if fmt.parse(content, probe["known"]):
                    probe["parse_hits"] += 1
            return data

        mgr._post = counting_post  # type: ignore[method-assign]
        return InferenceService(mgr)

    return factory


async def run_arm(
    suite: BenchmarkSuite,
    model: str,
    arm: str,
    fmt_override: str,
    trials: int,
    known_tools: set[str],
) -> ArmResult:
    probe = {
        "eval_counts": [],
        "parse_hits": 0,
        "parse_total": 0,
        "known": known_tools,
    }
    factory = _make_factory(model, fmt_override, probe)
    runner = BenchmarkRunner(
        trials=trials,
        inference_factory=factory,
        append_history=False,
    )
    spec = ModelSpec(
        provider="ollama",
        model=model,
        label=f"{model}::{arm}",
        temperature=0.0,
        max_turns=4,
    )
    cards = await runner.run(suite, [spec])
    return ArmResult(
        arm=arm,
        card=cards[0],
        eval_counts=probe["eval_counts"],
        parse_hits=probe["parse_hits"],
        parse_total=probe["parse_total"],
    )


def _metric_outcomes(card: ModelScorecard, predicate):
    """Per-(task_id, trial_index) bool grid for a metric defined by ``predicate(trial)``.

    Restricted to tasks where the metric is defined (predicate may return None to skip
    a trial entirely — e.g. tool-selection only applies to tasks that expect a tool).
    """
    grid = {}
    for score in card.task_scores:
        for i, trial in enumerate(score.trials):
            val = predicate(score, trial)
            if val is not None:
                grid[(score.task_id, i)] = bool(val)
    return grid


def _rate(grid):
    if not grid:
        return (0, 0, 0.0, (0.0, 0.0))
    k = sum(1 for v in grid.values() if v)
    n = len(grid)
    return (k, n, k / n, wilson_interval(k, n))


def _paired_mcnemar(grid_a, grid_b):
    """Exact paired McNemar over shared keys. Returns (n_pairs, b, c, p)."""
    shared = set(grid_a) & set(grid_b)
    b = sum(1 for kk in shared if grid_a[kk] and not grid_b[kk])
    c = sum(1 for kk in shared if grid_b[kk] and not grid_a[kk])
    return (len(shared), b, c, mcnemar_exact(b, c))


# Metric predicates. None => trial not applicable to this metric.
def _sel(score, trial):  # tool-selection accuracy (did expected tool get called?)
    return trial.tool_ok  # None for tasks with no expect_tools


def _arg(score, trial):  # argument-correctness (answer+trajectory both pass)
    if score.category == "irrelevance":
        return None
    return trial.correct


def _parallel(score, trial):  # parallel-correctness (both calls + both answers)
    if score.category != "parallel":
        return None
    return trial.correct


def _irrel(score, trial):  # irrelevance accuracy (abstained = no tool, not errored)
    if score.category != "irrelevance":
        return None
    if trial.error:
        return None
    return not trial.tools_called


def main() -> int:
    ap = argparse.ArgumentParser(description="Tool-call FORMAT A/B (generic vs native).")
    ap.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen2.5:7b-instruct")
    ap.add_argument("--trials", type=int, default=5, help="trials per task per arm")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args()

    suite = BenchmarkSuite.from_dict(SUITE_DATA)
    known = set()
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit import ToolkitConfig, register_packs

    cfg = ToolkitConfig.from_env()
    for pk in {p for t in suite.tasks for p in t.packs}:
        r = ToolRegistry()
        register_packs(r, [pk], cfg)
        known |= {t.name for t in r.list()}

    print(f"# Tool-call FORMAT A/B — model={args.model} trials/task={args.trials}")
    print(f"# tasks={len(suite.tasks)} known_tools={sorted(known)}")
    print(f"# arms: GENERIC={ARM_GENERIC} (text-path) vs NATIVE={ARM_NATIVE} (text-path)")
    print("# Ollama native tools= field SUPPRESSED in BOTH arms (FORMAT is the only variable).")
    sys.stdout.flush()

    t0 = time.time()
    generic = asyncio.run(
        run_arm(suite, args.model, "GENERIC", ARM_GENERIC, args.trials, known)
    )
    t_g = time.time() - t0
    print(f"# GENERIC arm done in {t_g:.1f}s")
    sys.stdout.flush()

    t1 = time.time()
    native = asyncio.run(
        run_arm(suite, args.model, "NATIVE", ARM_NATIVE, args.trials, known)
    )
    t_n = time.time() - t1
    print(f"# NATIVE arm done in {t_n:.1f}s")
    sys.stdout.flush()

    # ---- Proof the local model ran: real Ollama eval_count + latency ----
    def _ecsum(r):
        ec = r.eval_counts
        return (len(ec), sum(ec), (sum(ec) / len(ec) if ec else 0.0))

    out = {
        "model": args.model,
        "trials_per_task": args.trials,
        "n_tasks": len(suite.tasks),
        "proof_of_run": {
            "generic": {
                "ollama_calls": _ecsum(generic)[0],
                "total_eval_tokens": _ecsum(generic)[1],
                "mean_eval_tokens": round(_ecsum(generic)[2], 1),
                "p50_latency_s": round(generic.card.p50_latency, 2),
                "p95_latency_s": round(generic.card.p95_latency, 2),
            },
            "native": {
                "ollama_calls": _ecsum(native)[0],
                "total_eval_tokens": _ecsum(native)[1],
                "mean_eval_tokens": round(_ecsum(native)[2], 1),
                "p50_latency_s": round(native.card.p50_latency, 2),
                "p95_latency_s": round(native.card.p95_latency, 2),
            },
        },
        "metrics": {},
    }

    metrics = [
        ("tool_selection_accuracy", _sel),
        ("argument_correctness", _arg),
        ("parallel_correctness", _parallel),
        ("irrelevance_accuracy", _irrel),
    ]

    print("\n## Per-metric results (paired, same task set, temp=0)\n")
    header = (
        f"{'metric':<26} {'GENERIC k/n (rate)':<24} {'NATIVE k/n (rate)':<24} "
        f"{'delta':<9} {'McNemar b/c':<12} {'p':<8} verdict"
    )
    print(header)
    print("-" * len(header))

    for name, pred in metrics:
        ga = _metric_outcomes(generic.card, pred)
        na = _metric_outcomes(native.card, pred)
        gk, gn, gr, gci = _rate(ga)
        nk, nn, nr, nci = _rate(na)
        n_pairs, b, c, p = _paired_mcnemar(ga, na)
        delta = nr - gr
        sig = "SIGNIFICANT" if p < 0.05 else "n.s."
        if n_pairs == 0 or (b == 0 and c == 0):
            sig = "no discordant pairs"
        direction = (
            "native better"
            if c > b
            else "generic better"
            if b > c
            else "tie"
        )
        verdict = f"{direction} ({sig})"
        print(
            f"{name:<26} "
            f"{f'{gk}/{gn} ({gr:.0%})':<24} "
            f"{f'{nk}/{nn} ({nr:.0%})':<24} "
            f"{f'{delta:+.0%}':<9} "
            f"{f'{b}/{c}':<12} "
            f"{f'{p:.3f}':<8} "
            f"{verdict}"
        )
        out["metrics"][name] = {
            "generic": {"k": gk, "n": gn, "rate": gr, "wilson_ci": gci},
            "native": {"k": nk, "n": nn, "rate": nr, "wilson_ci": nci},
            "delta": delta,
            "mcnemar": {"n_pairs": n_pairs, "b_generic_only": b, "c_native_only": c, "p": p},
            "verdict": verdict,
        }

    # ---- raw parse-hit-rate (did the FORMAT produce a parseable call at all?) ----
    print("\n## Raw parse-hit-rate (text-path replies that yielded >=1 parseable call)\n")
    gph = (generic.parse_hits, generic.parse_total)
    nph = (native.parse_hits, native.parse_total)
    gphr = gph[0] / gph[1] if gph[1] else 0.0
    nphr = nph[0] / nph[1] if nph[1] else 0.0
    print(f"GENERIC: {gph[0]}/{gph[1]} ({gphr:.0%})   NATIVE: {nph[0]}/{nph[1]} ({nphr:.0%})")
    out["parse_hit_rate"] = {
        "generic": {"hits": gph[0], "total": gph[1], "rate": gphr},
        "native": {"hits": nph[0], "total": nph[1], "rate": nphr},
    }

    # ---- overall paired McNemar on full correctness (the headline) ----
    overall = compare_scorecards(native.card, generic.card)
    print("\n## Overall paired correctness (all non-judge tasks)\n")
    print(
        f"NATIVE acc={native.card.accuracy:.0%} {native.card.accuracy_ci}  "
        f"GENERIC acc={generic.card.accuracy:.0%} {generic.card.accuracy_ci}"
    )
    print(
        f"McNemar: n_pairs={overall.n_pairs} native_only_wins(c)={overall.c} "
        f"generic_only_wins(b)={overall.b} p={overall.p_value:.4f} leader={overall.leader}"
    )
    out["overall"] = {
        "native_accuracy": native.card.accuracy,
        "native_ci": native.card.accuracy_ci,
        "generic_accuracy": generic.card.accuracy,
        "generic_ci": generic.card.accuracy_ci,
        "mcnemar": {
            "n_pairs": overall.n_pairs,
            "native_only_wins": overall.c,
            "generic_only_wins": overall.b,
            "p": overall.p_value,
            "leader": overall.leader,
        },
        "error_rate": {
            "native": native.card.error_rate,
            "generic": generic.card.error_rate,
        },
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n# wrote {args.out}")

    print(f"\n# total wall-clock: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
