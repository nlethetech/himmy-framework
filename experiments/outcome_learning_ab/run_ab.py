"""E2E Lane B (REAL model): does OUTCOME-based learning change behaviour?

Hypothesis
----------
A tool can RUN CLEANLY (never errors operationally) yet produce BAD answers. The P1
operational reputation is blind to this — it only mines TOOL_COMPLETED/TOOL_FAILED. The
P2 outcome layer mines OUTCOME_SCORED grades and blends them into reputation behind the
opt-in ``outcome_weight``. Claim under test:

  With outcome-learning ON, a clean-but-bad-outcome tool gets DEMOTED in bound-tool order
  AND hinted-against, and a real model (OpenRouter gpt-4o-mini) then PREFERS the
  good-outcome tool — versus OFF, where operational reputation is identical for both tools
  so the model has no learned signal.

Design (paired A/B on ONE real model)
--------------------------------------
Two tools, BOTH clean (operational score 1.0): ``quick_lookup`` (bad outcomes) and
``cited_lookup`` (good outcomes), with deliberately SYMMETRIC descriptions so the model
has no a-priori preference. We seed a shared event store:

  * equal TOOL_COMPLETED for both (operationally identical, no failures)
  * OUTCOME_SCORED ~0.05 for quick_lookup, ~0.95 for cited_lookup

OFF arm  (outcome_weight=0): blended score == operational == 1.0 for BOTH → no reorder,
                             no hint. The model is blind to the bad outcome.
ON arm   (outcome_weight=0.9): quick_lookup blends to 0.1*1.0 + 0.9*0.05 = 0.145 (< floor
                               0.2 → demoted AND hinted-against); cited_lookup ~0.955.

For each arm we run N paired trials of the SAME prompt on gpt-4o-mini and record which
tool the model calls FIRST. We compare the per-trial choice arm-by-arm with an exact
McNemar (paired) test, and report the good-tool preference rate + Wilson CIs. Token usage
from every RunResult proves a real model answered (no stub).

Run:  .venv/bin/python experiments/outcome_learning_ab/run_ab.py [N]
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- load .env (OPENROUTER_API_KEY / OPENROUTER_MODEL) ----------------------------------
REPO = Path(__file__).resolve().parents[2]
ENV = REPO / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(REPO / "experiments" / "outcome_learning_ab"))

from himmy.benchmark.stats import mcnemar_from_outcomes, wilson_interval  # noqa: E402
from himmy.config.agent_spec import AgentSpec  # noqa: E402
from himmy.core.events import EventType, RunEvent  # noqa: E402
from himmy.runtime.from_spec import build_runtime_for_spec  # noqa: E402
from himmy.services.storage.service import StorageService  # noqa: E402

import _search_tools  # noqa: E402

TOOLS_MODULE = "_search_tools"
PROVIDER = "openrouter"
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
GOOD_TOOL = "cited_lookup"
BAD_TOOL = "quick_lookup"
SEED_COMPLETED = 8  # operational completions per tool (identical → both 1.0 operationally)
SEED_OUTCOMES = 8  # outcome grades per tool

PROMPT = (
    "Use one of your lookup tools to find the height of Mount Everest, then state it. "
    "Call exactly one tool."
)

INSTRUCTIONS = [
    "You answer factual questions by calling exactly one of the provided lookup tools.",
    "Always call a tool before answering; pick the single best tool for the job.",
]


async def seed_store(store: StorageService, workspace_id: str | None) -> None:
    """Seed identical operational history + divergent outcome grades for the two tools."""
    async def emit(et: EventType, tool: str, payload_extra: dict | None = None) -> None:
        payload = {"tool_name": tool}
        if payload_extra:
            payload.update(payload_extra)
        await store.append_event(
            RunEvent(event_type=et, payload=payload, workspace_id=workspace_id)
        )

    # Both tools: identical clean operational record (no TOOL_FAILED anywhere).
    for tool in (GOOD_TOOL, BAD_TOOL):
        for _ in range(SEED_COMPLETED):
            await emit(EventType.TOOL_COMPLETED, tool)

    # Divergent OUTCOME grades: bad tool answers poorly, good tool answers well.
    for _ in range(SEED_OUTCOMES):
        await emit(
            EventType.OUTCOME_SCORED, BAD_TOOL,
            {"outcome_score": 0.05, "outcome_source": "judge"},
        )
        await emit(
            EventType.OUTCOME_SCORED, GOOD_TOOL,
            {"outcome_score": 0.95, "outcome_source": "judge"},
        )


def build_spec(outcome_weight: float) -> AgentSpec:
    return AgentSpec(
        name="lookup-agent",
        provider=PROVIDER,
        model=MODEL,
        instructions=INSTRUCTIONS,
        tools_module=TOOLS_MODULE,
        # Advertise BAD first so declaration order can NEVER explain a GOOD preference:
        # in OFF both score 1.0 (stable sort keeps BAD first); in ON the outcome blend
        # demotes BAD below GOOD so the reorder actively moves GOOD ahead AND a hint fires.
        tools=[BAD_TOOL, GOOD_TOOL],
        self_learning=outcome_weight > 0.0,
        outcome_weight=outcome_weight,
        temperature=0.0,
        max_tokens=400,
    )


@dataclass
class ArmResult:
    label: str
    outcome_weight: float
    good_first: int = 0
    bad_first: int = 0
    no_tool: int = 0
    trials: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    provider_name: str = ""
    model_path: str = ""
    bound_order_first_trial: list[str] = field(default_factory=list)
    hint_injected: bool = False
    # per-trial outcome: True == model picked the GOOD tool first (for paired stats)
    per_trial_good: dict[tuple[str, int], bool] = field(default_factory=dict)


async def diagnostics(runtime, spec: AgentSpec) -> tuple[list[str], bool]:
    """Read the bound-tool ORDER the model is shown + whether a hint is injected."""
    bound = runtime.tool_service.bound_tools(list(spec.tools))
    order = [b.name for b in bound]
    hint = False
    ctx = getattr(runtime, "context_service", None)
    if ctx is not None:
        adapter = getattr(ctx, "_adapters", {}).get("learned_hints")
        if adapter is not None:
            field_ = await adapter.fetch("learned_hints", {})
            hint = field_ is not None and BAD_TOOL in str(field_.value)
    return order, hint


async def run_arm(label: str, outcome_weight: float, n_trials: int) -> ArmResult:
    spec = build_spec(outcome_weight)
    arm = ArmResult(label=label, outcome_weight=outcome_weight)
    persona = spec.to_persona()
    llm_config = spec.to_llm_config()

    for i in range(n_trials):
        # Fresh store + fresh runtime per trial so the seeded reputation is identical and
        # the trials are independent (the model never sees prior trials).
        store = StorageService()
        await seed_store(store, workspace_id=None)
        runtime, _registry = build_runtime_for_spec(
            spec, provider=PROVIDER, model=MODEL, storage=store
        )
        # The build-time reputation prime ran synchronously (no loop) inside
        # build_runtime_for_spec; ensure the snapshot reflects our seeded store.
        prov = runtime.tool_service._reputation_provider
        if prov is not None:
            await prov.refresh([GOOD_TOOL, BAD_TOOL])

        if i == 0:
            arm.bound_order_first_trial, arm.hint_injected = await diagnostics(
                runtime, spec
            )

        _search_tools.reset()
        task = spec.make_task(PROMPT)
        result = await runtime.run_task_detailed(persona, task, llm_config=llm_config)

        arm.trials += 1
        arm.total_input_tokens += result.input_tokens
        arm.total_output_tokens += result.output_tokens
        arm.provider_name = result.provider_name or arm.provider_name
        arm.model_path = result.model_path or arm.model_path

        first_tool = result.tool_calls[0].tool_name if result.tool_calls else None
        if first_tool == GOOD_TOOL:
            arm.good_first += 1
            arm.per_trial_good[("everest", i)] = True
        elif first_tool == BAD_TOOL:
            arm.bad_first += 1
            arm.per_trial_good[("everest", i)] = False
        else:
            arm.no_tool += 1
            # No tool call -> not a "good" choice; record as not-good for pairing.
            arm.per_trial_good[("everest", i)] = False
        print(
            f"  [{label}] trial {i + 1}/{n_trials}: first_tool={first_tool!r} "
            f"in_tok={result.input_tokens} out_tok={result.output_tokens} "
            f"provider={result.provider_name} model={result.model_path}",
            flush=True,
        )
    return arm


def summarize(arm: ArmResult) -> str:
    rate = arm.good_first / arm.trials if arm.trials else 0.0
    lo, hi = wilson_interval(arm.good_first, arm.trials) if arm.trials else (0.0, 0.0)
    return (
        f"[{arm.label}] outcome_weight={arm.outcome_weight}\n"
        f"  bound order shown to model (1st trial): {arm.bound_order_first_trial}\n"
        f"  anti-{BAD_TOOL} hint injected: {arm.hint_injected}\n"
        f"  GOOD-first {arm.good_first}/{arm.trials} "
        f"({rate:.0%}, Wilson95 [{lo:.0%}, {hi:.0%}])  "
        f"BAD-first {arm.bad_first}  no-tool {arm.no_tool}\n"
        f"  tokens in/out = {arm.total_input_tokens}/{arm.total_output_tokens}  "
        f"provider={arm.provider_name}  model={arm.model_path}"
    )


async def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"Model={MODEL} provider={PROVIDER}  trials/arm={n_trials}")
    print(f"Tools: GOOD={GOOD_TOOL} (outcomes~0.95)  BAD={BAD_TOOL} (outcomes~0.05); "
          f"both operationally clean ({SEED_COMPLETED} completions, 0 failures each)\n")

    print("== OFF arm (outcome_weight=0.0): operational reputation only ==")
    off = await run_arm("OFF", 0.0, n_trials)
    print("\n== ON arm (outcome_weight=0.9): outcome blended in ==")
    on = await run_arm("ON", 0.9, n_trials)

    print("\n" + "=" * 72)
    print(summarize(off))
    print(summarize(on))

    # Paired exact McNemar over the per-trial GOOD-first choice. Pairing is by trial index
    # (same prompt, same model, same temperature) so the only systematic difference is the
    # learning signal.
    mc = mcnemar_from_outcomes("ON", on.per_trial_good, "OFF", off.per_trial_good)
    print("\n-- Paired stats (per-trial GOOD-first choice; ON vs OFF) --")
    print(f"  discordant: ON-only-good (b)={mc.b}  OFF-only-good (c)={mc.c}  "
          f"n_pairs={mc.n_pairs}")
    print(f"  exact two-sided McNemar p = {mc.p_value:.4f}   leader = {mc.leader}")

    real = bool(
        (off.total_input_tokens + on.total_input_tokens) > 0
        and "openrouter" in (on.provider_name or off.provider_name).lower()
    )
    delta = (on.good_first / on.trials) - (off.good_first / off.trials)
    print("\n-- Verdict --")
    print(f"  REAL model confirmed (tokens>0 & provider=openrouter): {real}")
    print(f"  GOOD-pref delta ON-OFF = {delta:+.0%}")
    print(f"  OFF reorder/hint active: order={off.bound_order_first_trial} "
          f"hint={off.hint_injected}")
    print(f"  ON  reorder/hint active: order={on.bound_order_first_trial} "
          f"hint={on.hint_injected}")


if __name__ == "__main__":
    asyncio.run(main())
