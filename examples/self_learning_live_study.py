"""LIVE self-learning study (Layer 3): does himmy's loop steer a REAL model?

This is the Hermes-grade, pre-registered experiment. Layers 1-2
(``examples/self_learning_demo.py`` + ``tests/runtime/test_self_learning_e2e.py``)
prove himmy's internal bookkeeping with a deterministic stub. This script proves the
*for-real* end: against a REAL OpenRouter model, does seeding a tool's reputation and
turning ``self_learning=True`` measurably steer the model AWAY from the flagged tool —
with every known confound controlled?

It is honest by construction:

* NEUTRAL symmetric tools (``tool_alpha`` / ``tool_beta``, byte-identical descriptions)
  and a zero-preference instruction kill the name/persona leaks that made the old proof
  hollow (the old test passed with the feature OFF).
* Three conditions hold history constant: A (no history, learning off) vs
  B (history present, learning off) vs C (history present, learning ON). The clean
  causal contrast is B -> C; A -> B is the sanity check that history alone is inert.
* Every run drives the PRODUCTION ``spec.make_task(prompt)`` path so the learned-hints
  context block actually reaches the system prompt (the old test hand-built ``Task`` and
  the hint never reached the model — confound 3).
* Counterbalancing crosses *which* tool is flagged x *declared order*, so name and
  first-position bias appear identically in A/B/C and cancel in B -> C.
* Pre-flight invariants in C assert reorder + annotation + hint + hint-in-system-prompt
  ALL fired before any live call counts — a pass cannot be a neutral-toolset coincidence.
* An ablation (reorder-only / annotation-only / hint-only / all) isolates which lever
  moves the model.
* Real token usage > 0 is asserted per condition — no stub can produce real usage.
* NULL / WEAK results are printed honestly with the real numbers. Nothing is tuned to
  manufacture a pass.

Run it (needs ``OPENROUTER_API_KEY`` in the repo ``.env``)::

    .venv/bin/python examples/self_learning_live_study.py
    .venv/bin/python examples/self_learning_live_study.py --n 12 --concurrency 4
    .venv/bin/python examples/self_learning_live_study.py --escalate   # also run sonnet
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# himmy does NOT auto-load .env — load the repo .env so OPENROUTER_API_KEY reaches
# get_secret()'s EnvSecrets backend (which reads os.environ) before we build anything.
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

from examples.self_learning_neutral_tools import (  # noqa: E402
    TOOL_ALPHA,
    TOOL_BETA,
)
from himmy.agents.personas.persona import Persona  # noqa: E402
from himmy.config.agent_spec import AgentSpec  # noqa: E402
from himmy.core.events import EventType, RunEvent  # noqa: E402
from himmy.runtime.from_spec import build_runtime_for_spec  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    InferenceRequest,
    InferenceResponse,
)
from himmy.services.inference.service import InferenceService  # noqa: E402
from himmy.services.learning import LearningService  # noqa: E402
from himmy.services.learning.service import ToolReputationProvider  # noqa: E402
from himmy.services.storage.service import StorageService  # noqa: E402

TOOLS_MODULE = "examples.self_learning_neutral_tools"

#: Identical, zero-preference instruction across EVERY condition. No mention of
#: reliability, quality, ordering, or tool names.
INSTRUCTION = "Answer the user's question by calling exactly one lookup tool."
#: Identical prompt across every trial.
PROMPT = "Look up the value for key 'x'. Use a lookup tool."

#: The exact annotation suffix ToolService appends to an unreliable tool's description
#: (himmy/services/tools/service.py:_annotated_description). Pre-flight asserts it.
ANNOTATION_SUFFIX = "(note: this tool has failed frequently recently)"

#: Default model (OpenRouter passthrough id; the '/' is part of the name).
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
#: The one stronger model to escalate to if gpt-4o-mini is NULL/WEAK (protocol §Escalation).
ESCALATION_MODEL = "anthropic/claude-3.5-sonnet"

#: Seed exactly 4 TOOL_FAILED for the flagged tool: >= min_samples (3) so has_min_samples
#: is True, 0 completions so score == 0.0 < floor (0.2).
SEED_FAILURES = 4

HAS_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))


def isolate_durable_state() -> None:
    """Point the durable spine + run store at a FRESH temp dir for this process.

    The entity spine (``.himmy/spine.db``) is content-addressed and durable, shared by
    every run in the project tree. Our two neutral tools register the SAME stable_id every
    trial; if a PRIOR run of this study (or an earlier code revision of the tools) left a
    tool_definition with a DIFFERENT payload in the project spine, the re-registration
    raises a content-address violation. Isolating the spine + store per process keeps every
    trial's registration self-consistent and never collides with stale on-disk state. This
    must run BEFORE any ``build_runtime_for_spec`` (which resolves the spine path eagerly).
    """
    tmp = Path(tempfile.mkdtemp(prefix="himmy-self-learning-study-"))
    os.environ["HIMMY_SPINE_PATH"] = str(tmp / "spine.db")
    os.environ["HIMMY_STORE_PATH"] = str(tmp / "store.db")
    os.environ.pop("HIMMY_DATABASE_URL", None)
    os.environ.pop("HIMMY_SPINE_DATABASE_URL", None)


# --------------------------------------------------------------------------------------
# Counterbalancing: 4 arrangements = which-tool-flagged x declared-order.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Arrangement:
    """One counterbalanced cell: which tool is flagged + the declared tool order."""

    flagged: str  # F
    unflagged: str  # U
    declared: tuple[str, str]  # spec.tools order

    @property
    def label(self) -> str:
        first_is_flagged = self.declared[0] == self.flagged
        return f"flag={self.flagged} order={'F,U' if first_is_flagged else 'U,F'}"


def arrangements() -> list[Arrangement]:
    """The 4 fully-crossed counterbalanced arrangements."""
    out: list[Arrangement] = []
    for flagged, unflagged in ((TOOL_ALPHA, TOOL_BETA), (TOOL_BETA, TOOL_ALPHA)):
        out.append(Arrangement(flagged, unflagged, (flagged, unflagged)))  # F,U
        out.append(Arrangement(flagged, unflagged, (unflagged, flagged)))  # U,F
    return out


# --------------------------------------------------------------------------------------
# Ablation levers (which self-learning signal is live).
# --------------------------------------------------------------------------------------
ALL = "all"  # = condition C, full treatment
REORDER_ONLY = "reorder_only"
ANNOTATION_ONLY = "annotation_only"
HINT_ONLY = "hint_only"


# --------------------------------------------------------------------------------------
# Recording manager wrapper: capture the EXACT system prompt sent to the model.
# --------------------------------------------------------------------------------------
class _RecordingManager:
    """Wrap a real ClientManager, recording the system message of every request.

    Delegates everything (resolve / generate / generate_stream / cache_capability_for)
    to the wrapped manager via ``__getattr__`` so it is a faithful passthrough — the
    REAL model still serves the call. We only peek at ``request.messages`` to capture
    the assembled system prompt, which lets condition C assert the learned-hints block
    actually reached the model (closing confound 3) without faking anything.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.system_prompts: list[str] = []

    @property
    def provider_name(self) -> str:
        return getattr(self._inner, "provider_name", "openrouter")

    def resolve(self, model_key: str) -> str:
        return self._inner.resolve(model_key)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        for message in request.messages:
            if message.role == "system":
                self.system_prompts.append(message.content)
                break
        return await self._inner.generate(request)

    def __getattr__(self, name: str) -> Any:
        # Delegate any other protocol surface (generate_stream, cache_capability_for,
        # cache_capability, etc.) straight to the wrapped real manager.
        return getattr(self._inner, name)


# --------------------------------------------------------------------------------------
# Ablation provider wrappers (isolate one lever at a time, at runtime, no from_spec edit).
# --------------------------------------------------------------------------------------
class _ReorderOnlyProvider:
    """Reorder fires, annotation suppressed: real score_for, is_unreliable forced False."""

    def __init__(self, provider: ToolReputationProvider) -> None:
        self._provider = provider

    def score_for(self, name: str) -> float:
        return self._provider.score_for(name)

    def is_unreliable(self, name: str) -> bool:
        return False


class _AnnotationOnlyProvider:
    """Annotation fires, reorder suppressed: constant score_for, real is_unreliable."""

    def __init__(self, provider: ToolReputationProvider) -> None:
        self._provider = provider

    def score_for(self, name: str) -> float:
        # Constant score => stable sort is a no-op => declared order preserved.
        return 1.0

    def is_unreliable(self, name: str) -> bool:
        return self._provider.is_unreliable(name)


# --------------------------------------------------------------------------------------
# Statistics: Fisher's exact one-sided + Newcombe CI on the risk difference. No scipy.
# --------------------------------------------------------------------------------------
def _log_factorial_table(n: int) -> list[float]:
    table = [0.0] * (n + 1)
    acc = 0.0
    for i in range(1, n + 1):
        acc += math.log(i)
        table[i] = acc
    return table


def fisher_one_sided_less(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p for the 2x2 table [[a,b],[c,d]].

    Rows = condition {B, C}; columns = outcome {picked flagged, picked unflagged}.
    a = B-picked-flagged, b = B-picked-unflagged, c = C-picked-flagged, d = C-picked-unflagged.
    Alternative: the flagged-pick rate is LOWER in C than in B (i.e. ``a/(a+b) > c/(c+d)``),
    so we sum hypergeometric probabilities for tables at least as extreme (c as small or
    smaller, given fixed margins). Returns 1.0 for a degenerate table.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 1.0
    lf = _log_factorial_table(n)

    def log_prob(c_val: int) -> float:
        # Given fixed margins, the table is determined by c (C-picked-flagged):
        # a = col1 - c, b = row1 - a, d = row2 - c.
        a_val = col1 - c_val
        b_val = row1 - a_val
        d_val = row2 - c_val
        if min(a_val, b_val, c_val, d_val) < 0:
            return float("-inf")
        return (
            lf[row1] + lf[row2] + lf[col1] + lf[col2]
            - lf[n] - lf[a_val] - lf[b_val] - lf[c_val] - lf[d_val]
        )

    # c can range over [max(0, col1 - row1), min(col1, row2)].
    lo = max(0, col1 - row1)
    hi = min(col1, row2)
    # One-sided "less in C": sum probability mass for c <= observed c.
    total = 0.0
    for c_val in range(lo, hi + 1):
        lp = log_prob(c_val)
        if lp != float("-inf") and c_val <= c:
            total += math.exp(lp)
    return min(1.0, total)


def _wilson_bounds(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a proportion (used by Newcombe's method)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Newcombe 95% CI for the risk difference p1 - p2 (B minus C).

    Combines the two Wilson intervals (method 10 of Newcombe 1998), valid for small,
    possibly-zero counts where the normal approximation breaks.
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1 = k1 / n1
    p2 = k2 / n2
    l1, u1 = _wilson_bounds(k1, n1, z)
    l2, u2 = _wilson_bounds(k2, n2, z)
    diff = p1 - p2
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lower, upper)


# --------------------------------------------------------------------------------------
# Trial result accounting.
# --------------------------------------------------------------------------------------
@dataclass
class CellResult:
    """Aggregated outcomes for one (condition/lever, arrangement) cell."""

    arrangement: Arrangement
    flagged_hits: int = 0  # first tool call == flagged
    unflagged_hits: int = 0  # first tool call == unflagged
    no_call: int = 0  # no tool call at all
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def trials_with_call(self) -> int:
        return self.flagged_hits + self.unflagged_hits

    @property
    def total(self) -> int:
        return self.flagged_hits + self.unflagged_hits + self.no_call


@dataclass
class ConditionResult:
    """All cells for one condition/lever, plus aggregate metrics."""

    name: str
    model: str
    cells: list[CellResult] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def flagged_hits(self) -> int:
        return sum(c.flagged_hits for c in self.cells)

    @property
    def unflagged_hits(self) -> int:
        return sum(c.unflagged_hits for c in self.cells)

    @property
    def trials_with_call(self) -> int:
        return sum(c.trials_with_call for c in self.cells)

    @property
    def no_call(self) -> int:
        return sum(c.no_call for c in self.cells)

    @property
    def total(self) -> int:
        return sum(c.total for c in self.cells)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.cells)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.cells)

    @property
    def pickrate(self) -> float | None:
        n = self.trials_with_call
        return (self.flagged_hits / n) if n else None

    @property
    def call_rate(self) -> float | None:
        return (self.trials_with_call / self.total) if self.total else None


# --------------------------------------------------------------------------------------
# Spec + seeding + runtime construction.
# --------------------------------------------------------------------------------------
def make_spec(model: str, *, self_learning: bool, declared: tuple[str, str]) -> AgentSpec:
    return AgentSpec(
        name="lookup-agent",
        instructions=[INSTRUCTION],
        model=model,
        provider="openrouter",
        self_learning=self_learning,
        tools_module=TOOLS_MODULE,
        tools=list(declared),
    )


def seed_failures(store: StorageService, tool: str, n: int) -> None:
    for _ in range(n):
        asyncio.run(
            store.append_event(
                RunEvent(event_type=EventType.TOOL_FAILED, payload={"tool_name": tool})
            )
        )


async def _seed_failures_async(store: StorageService, tool: str, n: int) -> None:
    for _ in range(n):
        await store.append_event(
            RunEvent(event_type=EventType.TOOL_FAILED, payload={"tool_name": tool})
        )


def _wrap_inference(store: StorageService, model: str) -> tuple[InferenceService, _RecordingManager]:
    """Build the REAL OpenRouter inference service, wrapped to record system prompts."""
    from himmy.cli.provider import build_manager_for

    inner_manager = build_manager_for("openrouter", model)
    recorder = _RecordingManager(inner_manager)
    service = InferenceService(recorder, event_sink=store)
    return service, recorder


def _apply_ablation(runtime: Any, lever: str) -> None:
    """Mutate a self_learning=True runtime so only ``lever`` is the live signal.

    Done on the BUILT runtime (no edit to from_spec.py): swap the tool_service's
    reputation provider and/or drop the learned_hints adapter. For HINT_ONLY the hint
    must still flow through spec.make_task(), so we keep the adapter but remove the
    provider (no reorder, no annotation).
    """
    tool_service = runtime.tool_service
    context_service = runtime.context_service
    provider = getattr(tool_service, "_reputation_provider", None)

    if lever == ALL:
        return  # full treatment, untouched

    if lever == REORDER_ONLY:
        if provider is not None:
            tool_service._reputation_provider = _ReorderOnlyProvider(provider)
        # Drop the hint adapter so no learned_hints block is injected.
        context_service._adapters.pop("learned_hints", None)
    elif lever == ANNOTATION_ONLY:
        if provider is not None:
            tool_service._reputation_provider = _AnnotationOnlyProvider(provider)
        context_service._adapters.pop("learned_hints", None)
    elif lever == HINT_ONLY:
        # No reorder, no annotation: drop the provider entirely. Keep the hint adapter
        # so spec.make_task()'s learned_hints block still reaches the system prompt.
        tool_service._reputation_provider = None
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown ablation lever: {lever}")


# --------------------------------------------------------------------------------------
# Pre-flight invariants for condition C / treatment levers (cheap, deterministic).
# --------------------------------------------------------------------------------------
@dataclass
class PreflightReport:
    ok: bool
    details: list[str] = field(default_factory=list)


async def preflight_treatment(model: str) -> PreflightReport:
    """Assert reorder + annotation + hint + hint-in-system-prompt ALL fire in C.

    Runs ONE arrangement (flag=alpha, declared [F,U]) deterministically against a stub:
    no live call needed for the structural checks; the system-prompt check uses the
    actual snapshot the production make_task() path builds. Returns ok=False (INVALID,
    not failed) if any invariant is violated.
    """
    details: list[str] = []
    ok = True
    flagged, unflagged = TOOL_ALPHA, TOOL_BETA

    store = StorageService()
    await _seed_failures_async(store, flagged, SEED_FAILURES)

    # (a) reputation gate: has_min_samples True, score 0.0, failed 4, completed 0.
    rep = (await LearningService(store).get_tool_reputation([flagged]))[flagged]
    cond_a = rep.has_min_samples and rep.score < 0.2 and rep.failed == SEED_FAILURES and rep.completed == 0
    details.append(
        f"reputation: has_min_samples={rep.has_min_samples} score={rep.score} "
        f"failed={rep.failed} completed={rep.completed} -> {'OK' if cond_a else 'FAIL'}"
    )
    ok = ok and bool(cond_a)

    spec = make_spec(model, self_learning=True, declared=(flagged, unflagged))
    inference, recorder = _wrap_inference(store, model)
    runtime, _registry = build_runtime_for_spec(spec, inference=inference, storage=store)
    await runtime.reprime_self_learning()

    # (b) reorder fired: unflagged comes before flagged.
    bound = runtime.tool_service.bound_tools([flagged, unflagged])
    order = [bt.name for bt in bound]
    cond_b = order == [unflagged, flagged]
    details.append(f"reorder: bound order={order} -> {'OK' if cond_b else 'FAIL'}")
    ok = ok and cond_b

    # (c) annotation fired: flagged tool's description carries the caution suffix.
    flagged_bt = next((bt for bt in bound if bt.name == flagged), None)
    cond_c = flagged_bt is not None and ANNOTATION_SUFFIX in flagged_bt.description
    details.append(
        f"annotation: flagged description ends with caution -> {'OK' if cond_c else 'FAIL'}"
    )
    ok = ok and cond_c

    # (d) hint resolvable: adapter.fetch returns a non-None field naming the flagged tool.
    adapter = runtime.context_service._adapters["learned_hints"]
    field_obj = await adapter.fetch("learned_hints", {})
    cond_d = field_obj is not None and f'tool "{flagged}"' in str(field_obj.value)
    details.append(f"hint: adapter field names flagged tool -> {'OK' if cond_d else 'FAIL'}")
    ok = ok and cond_d

    # (e) hint reaches the SYSTEM PROMPT via the production make_task path. Build a
    # snapshot exactly as the runtime would for a real turn and project it, then confirm
    # the <context key='learned_hints'> block naming the flagged tool is present.
    cond_e = await _hint_in_system_prompt(runtime, spec, flagged)
    details.append(
        f"hint-in-system-prompt: <context key='learned_hints'> names flagged tool "
        f"-> {'OK' if cond_e else 'FAIL'}"
    )
    ok = ok and cond_e

    return PreflightReport(ok=ok, details=details)


async def _hint_in_system_prompt(runtime: Any, spec: AgentSpec, flagged: str) -> bool:
    """True if the assembled system prompt for spec.make_task(PROMPT) carries the hint.

    Builds the snapshot from the task's context_build_spec (the production path) and
    projects it via the ContextPromptMapper exactly as the runtime does, then checks the
    rendered system text for the learned_hints context block naming the flagged tool.
    """
    task = spec.make_task(PROMPT)
    build_spec = task.context.get("context_build_spec")
    map_spec = task.context.get("context_prompt_map_spec")
    if not build_spec or not map_spec:
        return False
    from himmy.services.context.models import ContextBuildSpec
    from himmy.services.prompts.mapper import ContextPromptMapper, ContextPromptMapSpec

    snapshot = await runtime.context_service.build_snapshot(
        subject_id="preflight",
        task_id=task.task_id,
        build_spec=ContextBuildSpec.model_validate(build_spec),
        metadata={"query": PROMPT},
    )
    system_block, _task_block, _missing = ContextPromptMapper().project(
        snapshot, ContextPromptMapSpec.model_validate(map_spec)
    )
    # The mapper fences each key as <context key="learned_hints">...</context>.
    return 'key="learned_hints"' in system_block and flagged in system_block


# --------------------------------------------------------------------------------------
# Running one live trial.
# --------------------------------------------------------------------------------------
async def run_one_trial(
    model: str,
    *,
    arrangement: Arrangement,
    self_learning: bool,
    seed: bool,
    lever: str | None,
) -> tuple[str | None, int, int]:
    """Run ONE live turn. Returns (first_tool_or_None, input_tokens, output_tokens).

    Fresh isolated store per call so no history bleeds across cells.
    """
    store = StorageService()
    if seed:
        await _seed_failures_async(store, arrangement.flagged, SEED_FAILURES)

    spec = make_spec(model, self_learning=self_learning, declared=arrangement.declared)
    inference, _recorder = _wrap_inference(store, model)
    runtime, _registry = build_runtime_for_spec(spec, inference=inference, storage=store)

    if self_learning:
        await runtime.reprime_self_learning()
        if lever is not None:
            _apply_ablation(runtime, lever)
            # HINT_ONLY removed the provider; the order/annotation no longer reflect
            # reputation. reorder/annotation-only kept their (wrapped) provider. No
            # re-prime needed: the snapshot already primed; wrappers read it lazily.

    task = spec.make_task(PROMPT)
    loop = await runtime.run_agent_loop(
        Persona(name="lookup", instructions=[INSTRUCTION]),
        task,
        max_turns=4,
    )
    called = [c.tool_name for turn in loop.turns for c in turn.tool_calls]
    first = called[0] if called else None
    return first, loop.total_input_tokens, loop.total_output_tokens


async def run_condition(
    model: str,
    *,
    name: str,
    self_learning: bool,
    seed: bool,
    lever: str | None,
    n_per_arrangement: int,
    semaphore: asyncio.Semaphore,
) -> ConditionResult:
    """Run all 4 arrangements x N trials for one condition/lever, bounded by ``semaphore``."""
    result = ConditionResult(name=name, model=model)
    arr_list = arrangements()
    cells = {arr.label: CellResult(arrangement=arr) for arr in arr_list}
    start = time.monotonic()

    async def _trial(arr: Arrangement) -> None:
        async with semaphore:
            first, in_tok, out_tok = await run_one_trial(
                model,
                arrangement=arr,
                self_learning=self_learning,
                seed=seed,
                lever=lever,
            )
        cell = cells[arr.label]
        cell.input_tokens += in_tok
        cell.output_tokens += out_tok
        if first is None:
            cell.no_call += 1
        elif first == arr.flagged:
            cell.flagged_hits += 1
        elif first == arr.unflagged:
            cell.unflagged_hits += 1
        else:
            # Defensive: an unexpected tool name counts as no measurable pick.
            cell.no_call += 1

    tasks = [
        _trial(arr) for arr in arr_list for _ in range(n_per_arrangement)
    ]
    await asyncio.gather(*tasks)
    result.cells = [cells[arr.label] for arr in arr_list]
    result.wall_seconds = time.monotonic() - start
    return result


# --------------------------------------------------------------------------------------
# Verdict.
# --------------------------------------------------------------------------------------
@dataclass
class Verdict:
    label: str  # WORKED / WEAK / NULL / INVALID
    p_value: float | None
    reduction: float | None
    ci: tuple[float, float] | None
    arrangements_directional: int
    sanity_ok: bool
    lines: list[str] = field(default_factory=list)


def evaluate(
    cond_a: ConditionResult,
    cond_b: ConditionResult,
    cond_c: ConditionResult,
    preflight: PreflightReport,
) -> Verdict:
    """Apply the pre-registered pass criterion to A/B/C, honestly."""
    lines: list[str] = []

    if not preflight.ok:
        return Verdict(
            label="INVALID",
            p_value=None,
            reduction=None,
            ci=None,
            arrangements_directional=0,
            sanity_ok=False,
            lines=["pre-flight invariants failed; the treatment was not actually applied"],
        )

    pr_a = cond_a.pickrate
    pr_b = cond_b.pickrate
    pr_c = cond_c.pickrate

    # Pooled B-vs-C 2x2: rows {B, C}, cols {flagged, unflagged}.
    a = cond_b.flagged_hits
    b = cond_b.unflagged_hits
    c = cond_c.flagged_hits
    d = cond_c.unflagged_hits
    p_value = fisher_one_sided_less(a, b, c, d)
    reduction = (pr_b - pr_c) if (pr_b is not None and pr_c is not None) else None
    ci = newcombe_diff_ci(a, a + b, c, c + d) if (a + b) and (c + d) else None

    # Per-arrangement directional check: C <= B in >= 3/4 arrangements.
    directional = 0
    b_by_label = {cell.arrangement.label: cell for cell in cond_b.cells}
    for c_cell in cond_c.cells:
        b_cell = b_by_label.get(c_cell.arrangement.label)
        if b_cell is None:
            continue
        b_rate = (b_cell.flagged_hits / b_cell.trials_with_call) if b_cell.trials_with_call else None
        c_rate = (c_cell.flagged_hits / c_cell.trials_with_call) if c_cell.trials_with_call else None
        if b_rate is not None and c_rate is not None and c_rate <= b_rate:
            directional += 1

    # Sanity: B approx A and B-vs-A not significant.
    sanity_ok = True
    if pr_a is not None and pr_b is not None:
        sanity_ok = abs(pr_a - pr_b) < 0.15
        # B vs A two-proportion (A flagged-first vs B): use the same Fisher test, two
        # directions matter so we report both tails' min*2 (conservative two-sided).
        ba_p_less = fisher_one_sided_less(
            cond_a.flagged_hits, cond_a.unflagged_hits,
            cond_b.flagged_hits, cond_b.unflagged_hits,
        )
        ba_p_more = fisher_one_sided_less(
            cond_b.flagged_hits, cond_b.unflagged_hits,
            cond_a.flagged_hits, cond_a.unflagged_hits,
        )
        ba_two_sided = min(1.0, 2 * min(ba_p_less, ba_p_more))
        sanity_ok = sanity_ok and (ba_two_sided >= 0.05)
        lines.append(f"sanity B vs A two-sided p = {ba_two_sided:.4f}")

    # Classify.
    label = "NULL"
    if reduction is not None:
        if p_value < 0.05 and reduction >= 0.20 and directional >= 3 and sanity_ok:
            label = "WORKED"
        elif p_value < 0.05 and (0.10 <= reduction < 0.20 or directional <= 2):
            label = "WEAK"
        elif p_value >= 0.05 or reduction < 0.10:
            label = "NULL"
        elif p_value < 0.05 and reduction >= 0.20 and not sanity_ok:
            label = "WEAK"  # significant + meaningful but sanity broke -> not a clean WORKED
        else:
            label = "WEAK"

    return Verdict(
        label=label,
        p_value=p_value,
        reduction=reduction,
        ci=ci,
        arrangements_directional=directional,
        sanity_ok=sanity_ok,
        lines=lines,
    )


# --------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------
def _fmt_rate(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _print_condition(c: ConditionResult) -> None:
    print(
        f"  {c.name:18s} pickrate(F)={_fmt_rate(c.pickrate)} "
        f"(flagged={c.flagged_hits}, unflagged={c.unflagged_hits}, "
        f"no-call={c.no_call}, N_called={c.trials_with_call}/{c.total}) "
        f"call_rate={_fmt_rate(c.call_rate)} "
        f"tokens(in/out)={c.input_tokens}/{c.output_tokens} "
        f"wall={c.wall_seconds:.1f}s"
    )
    for cell in c.cells:
        n = cell.trials_with_call
        rate = (cell.flagged_hits / n) if n else None
        print(
            f"      [{cell.arrangement.label:24s}] pick(F)={_fmt_rate(rate)} "
            f"(F={cell.flagged_hits} U={cell.unflagged_hits} none={cell.no_call})"
        )


async def run_study_for_model(
    model: str, *, n_per_arrangement: int, concurrency: int
) -> None:
    print("=" * 78)
    print(f"MODEL: {model}   (N={n_per_arrangement} per arrangement x 4 = "
          f"{n_per_arrangement * 4} trials per condition)")
    print("=" * 78)

    semaphore = asyncio.Semaphore(concurrency)

    print("\nPRE-FLIGHT (condition C invariants — reorder + annotation + hint + system prompt):")
    preflight = await preflight_treatment(model)
    for line in preflight.details:
        print(f"  {line}")
    print(f"  => pre-flight {'PASS' if preflight.ok else 'FAIL (run is INVALID)'}")

    print("\nRunning conditions A / B / C ...")
    cond_a = await run_condition(
        model, name="A baseline", self_learning=False, seed=False, lever=None,
        n_per_arrangement=n_per_arrangement, semaphore=semaphore,
    )
    cond_b = await run_condition(
        model, name="B history-hidden", self_learning=False, seed=True, lever=None,
        n_per_arrangement=n_per_arrangement, semaphore=semaphore,
    )
    cond_c = await run_condition(
        model, name="C treatment", self_learning=True, seed=True, lever=ALL,
        n_per_arrangement=n_per_arrangement, semaphore=semaphore,
    )

    print("\nPRIMARY CONDITIONS (flagged-tool first-call rate):")
    for cond in (cond_a, cond_b, cond_c):
        _print_condition(cond)

    print("\nRunning ablation levers (learning ON, one signal at a time) ...")
    abl_reorder = await run_condition(
        model, name="C-reorder-only", self_learning=True, seed=True, lever=REORDER_ONLY,
        n_per_arrangement=n_per_arrangement, semaphore=semaphore,
    )
    abl_annot = await run_condition(
        model, name="C-annotation-only", self_learning=True, seed=True, lever=ANNOTATION_ONLY,
        n_per_arrangement=n_per_arrangement, semaphore=semaphore,
    )
    abl_hint = await run_condition(
        model, name="C-hint-only", self_learning=True, seed=True, lever=HINT_ONLY,
        n_per_arrangement=n_per_arrangement, semaphore=semaphore,
    )

    print("\nABLATION (which lever moves the model; compare to B):")
    print(f"  {'B history-hidden':18s} pickrate(F)={_fmt_rate(cond_b.pickrate)}")
    for cond in (cond_c, abl_reorder, abl_annot, abl_hint):
        _print_condition(cond)

    # Token reality check.
    print("\nREAL PROVIDER TOKEN USAGE (a stub cannot produce these):")
    for cond in (cond_a, cond_b, cond_c, abl_reorder, abl_annot, abl_hint):
        ok = cond.input_tokens > 0 and cond.output_tokens > 0
        print(
            f"  {cond.name:18s} in={cond.input_tokens} out={cond.output_tokens} "
            f"-> {'real round-trip' if ok else 'NO USAGE (suspect)'}"
        )

    # Verdict.
    print("\n" + "-" * 78)
    verdict = evaluate(cond_a, cond_b, cond_c, preflight)
    print("VERDICT (pre-registered B -> C contrast):")
    print(f"  pickrate(F|A) = {_fmt_rate(cond_a.pickrate)}  (intrinsic name+position bias)")
    print(f"  pickrate(F|B) = {_fmt_rate(cond_b.pickrate)}  (history present, learning OFF)")
    print(f"  pickrate(F|C) = {_fmt_rate(cond_c.pickrate)}  (treatment)")
    if verdict.reduction is not None:
        print(f"  reduction B-C = {verdict.reduction:+.3f}")
    if verdict.ci is not None:
        print(f"  Newcombe 95% CI on reduction = "
              f"[{verdict.ci[0]:+.3f}, {verdict.ci[1]:+.3f}]")
    if verdict.p_value is not None:
        # Use scientific notation for very small p so a real 3.9e-05 doesn't render
        # as a misleading "0.0000" (the pass logic uses the float, not this string).
        p_str = (
            f"{verdict.p_value:.4f}" if verdict.p_value >= 1e-4 else f"{verdict.p_value:.2e}"
        )
        print(f"  Fisher one-sided p (B vs C, C lower) = {p_str}")
    print(f"  directional arrangements (C <= B) = {verdict.arrangements_directional}/4")
    for line in verdict.lines:
        print(f"  {line}")
    print(f"\n  >>> {model}: {verdict.label}", end="")
    if verdict.label == "WORKED":
        print(" — self-learning measurably steers this model away from the flagged tool.")
    elif verdict.label == "WEAK":
        print(" — significant but small / order-dependent steering.")
    elif verdict.label == "NULL":
        print(" — NO detectable steering on this model (an honest, reportable outcome).")
    else:
        print(" — run did not validate; treatment was not actually applied.")
    print("-" * 78)

    # Stash the verdict on the function for the escalation decision.
    run_study_for_model.last_verdict = verdict  # type: ignore[attr-defined]


async def main_async(args: argparse.Namespace) -> None:
    if not HAS_KEY:
        print("OPENROUTER_API_KEY not set (looked in repo .env). Cannot run the live study.")
        print("Set the key and re-run: this study makes REAL model calls.")
        return

    isolate_durable_state()
    overall_start = time.monotonic()
    await run_study_for_model(
        args.model, n_per_arrangement=args.n, concurrency=args.concurrency
    )
    verdict: Verdict = run_study_for_model.last_verdict  # type: ignore[attr-defined]

    if args.escalate and verdict.label in {"NULL", "WEAK"}:
        print(f"\n{args.model} was {verdict.label}; escalating to {ESCALATION_MODEL} "
              "(identical protocol, model string only) ...\n")
        await run_study_for_model(
            ESCALATION_MODEL, n_per_arrangement=args.n, concurrency=args.concurrency
        )

    print(f"\nTotal wall time: {time.monotonic() - overall_start:.1f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n", type=int, default=6,
        help="trials per arrangement (default 6 -> 24 per condition, per the protocol)",
    )
    p.add_argument(
        "--concurrency", type=int, default=6,
        help="max concurrent live turns (default 6; <=8 min total)",
    )
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"OpenRouter model id (default {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--escalate", action="store_true",
        help=f"if NULL/WEAK, re-run identical protocol on {ESCALATION_MODEL}",
    )
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
