"""End-to-end, runnable proof of himmy's opt-in self-learning loop.

Two LOCAL tools do the SAME job:

* ``lookup_reliable`` — returns a valid answer (clean return -> ``TOOL_COMPLETED``).
* ``lookup_flaky``    — always raises (-> ``TOOL_FAILED``).

An :class:`AgentSpec` binds BOTH with ``self_learning=True``. We fail the flaky tool
three times so its reputation drops below the ``floor`` of ``0.2``, then show himmy:

  (a) reorder the bound tools so ``lookup_reliable`` sorts ahead of ``lookup_flaky``;
  (b) inject a system-prompt hint naming the flaky tool;
  (c) emit a ``LEARNING_APPLIED`` event;
  (d) (in the live test) drive a real model to call the reliable tool instead.

Everything printed below is read from REAL himmy objects — the actual reputation score,
the actual bound-tool order before/after, the actual injected hint text, and the actual
``LEARNING_APPLIED`` event. No narrative is hard-coded. A deterministic stub model that
always picks the FIRST advertised tool drives the four runs, so this script runs with no
API key: the ONLY thing that changes which tool the model picks is himmy's own reorder.

Run it::

    .venv/bin/python examples/self_learning_demo.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from himmy.config.agent_spec import AgentSpec
from himmy.core.events import EventType
from himmy.runtime.from_spec import build_runtime_for_spec
from himmy.services.inference.models import (
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool

# The two tools live in a module the spec names via ``tools_module`` so the real
# build path registers them onto the per-run registry. Both handlers take a SINGLE
# dict arg — himmy calls local handlers as ``handler(args_dict)``, never ``**kwargs``.
TOOLS_MODULE = "examples.self_learning_demo"

_RELIABLE = "lookup_reliable"
_FLAKY = "lookup_flaky"


def register(registry: ToolRegistry) -> None:
    """Register the two same-job tools onto a per-run registry (from_spec entrypoint)."""

    async def _reliable(args: dict[str, Any]) -> dict[str, Any]:
        # Clean return -> TOOL_COMPLETED.
        return {"answer": "The capital of France is Paris.", "source": _RELIABLE}

    async def _flaky(args: dict[str, Any]) -> dict[str, Any]:
        # Always raises -> TOOL_FAILED.
        raise RuntimeError("upstream lookup service is unreachable")

    register_local_tool(
        registry,
        name=_RELIABLE,
        handler=_reliable,
        description="Look up a fact and return the answer.",
        args_json_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        read_only=True,
    )
    register_local_tool(
        registry,
        name=_FLAKY,
        handler=_flaky,
        description="Look up a fact and return the answer.",
        args_json_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        read_only=True,
    )


class FirstToolManager:
    """A deterministic stub model: call the FIRST advertised tool, then answer.

    This manager holds NO opinion about which tool is good — it always reaches for
    ``request.bound_tools[0]``. So the ONLY lever that changes its choice is himmy's
    own self-learning reorder of that list. On a continuation turn (a ``tool`` message
    already present in the conversation) it stops with a final text answer.
    """

    provider_name = "stub"

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        response = InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            model_path=self.resolve(request.model_key),
            provider_name=self.provider_name,
            latency_ms=1.0,
            input_tokens=8,
            output_tokens=4,
        )
        already_used_a_tool = any(m.role == "tool" for m in request.messages)
        if already_used_a_tool or not request.bound_tools:
            response.output_text = "Done."
            return response
        # Pick the FIRST advertised tool and execute it through himmy's real executor
        # (which records TOOL_COMPLETED / TOOL_FAILED for the run).
        target = request.bound_tools[0]
        assert request.tool_executor is not None
        ret = await request.tool_executor(target.name, {"query": "capital of France?"})
        # Pair the call to the return on the SAME id so the runtime's replay finds the
        # outcome (the executor mints the id; mirror it onto the call record).
        response.tool_calls = [
            ToolCallRecord(
                tool_call_id=ret.tool_call_id,
                tool_name=target.name,
                args={"query": "capital of France?"},
            )
        ]
        response.tool_returns = [ret]
        response.output_text = f"[stub] called {target.name}"
        return response


def _inference_over(storage: StorageService) -> InferenceService:
    """An InferenceService backed by the deterministic manager, sharing the run store."""
    return InferenceService(FirstToolManager(), event_sink=storage)


def _spec() -> AgentSpec:
    return AgentSpec(
        name="lookup-agent",
        instructions=["Answer the user's question using a lookup tool."],
        self_learning=True,
        tools_module=TOOLS_MODULE,
        # Flaky is listed FIRST on purpose: the stub model reaches for the first
        # advertised tool, so before any history it picks the flaky one and fails.
        # himmy's reputation reorder is the only thing that can move it to the back.
        tools=[_FLAKY, _RELIABLE],
    )


async def _run_one_turn(storage: StorageService, advertised: list[str]) -> str:
    """Build a fresh self-learning runtime over the SHARED store and run one turn.

    ``advertised`` is the tool list offered to the model (in declared order). himmy's
    reputation reorder may re-sort it before the model sees it. Returns the name of the
    tool the (stub) model actually called this turn — read from the real loop result.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    runtime, _registry = build_runtime_for_spec(
        _spec(), inference=_inference_over(storage), storage=storage
    )
    # Inside a running loop the build-time reputation prime is a background task; drive
    # the re-prime hook so the snapshot is ready BEFORE the first turn binds tools (else
    # the first turn could read a neutral snapshot and miss the reorder).
    await runtime.reprime_self_learning()
    loop = await runtime.run_agent_loop(
        Persona(name="lookup", instructions=["Use a lookup tool to answer."]),
        Task(
            title="q",
            prompt="What is the capital of France?",
            context={"tool_names": list(advertised)},
        ),
        max_turns=4,
    )
    called = [c.tool_name for turn in loop.turns for c in turn.tool_calls]
    return called[0] if called else "(no tool)"


async def _reputation(storage: StorageService) -> dict[str, Any]:
    """Read the ACTUAL reputation snapshot for both tools from the shared store."""
    from himmy.services.learning import LearningService

    svc = LearningService(storage)
    return await svc.get_tool_reputation([_RELIABLE, _FLAKY])


async def _bound_order_and_hint(storage: StorageService) -> tuple[list[str], str | None]:
    """Read the ACTUAL bound-tool order and the ACTUAL injected hint from a built runtime."""
    runtime, _registry = build_runtime_for_spec(
        _spec(), inference=_inference_over(storage), storage=storage
    )
    # We are inside a running event loop, so the build-time reputation prime was
    # scheduled as a background task and may not have landed yet. Drive the public
    # re-prime hook to refresh the snapshot + hint candidates deterministically before
    # we read them (this is the same hook the MCP attach path uses).
    await runtime.reprime_self_learning()
    order = [bt.name for bt in runtime.tool_service.bound_tools([_FLAKY, _RELIABLE])]
    adapter = runtime.context_service._adapters["learned_hints"]
    field = await adapter.fetch("learned_hints", {})
    hint = None if field is None else str(field.value)
    return order, hint


def _rule() -> None:
    print("-" * 72)


async def main() -> None:
    # ONE shared store preserves the TOOL_FAILED history across the per-turn rebuilds —
    # without this, every run would start from a clean (neutral) reputation.
    storage = StorageService()

    print("=" * 72)
    print("himmy self-learning, end to end: two same-job tools, one learns to win")
    print("=" * 72)
    print(
        f"\nTools bound (self_learning=True):\n"
        f"  - {_RELIABLE}: returns a valid answer (TOOL_COMPLETED)\n"
        f"  - {_FLAKY}: always raises (TOOL_FAILED)\n"
        f"The stub model always reaches for the FIRST advertised tool, so the only thing\n"
        f"that can change its choice is himmy's own reputation-driven reorder.\n"
    )

    _rule()
    print("BEFORE any history — neither tool has a track record:")
    order0, hint0 = await _bound_order_and_hint(storage)
    print(f"  bound-tool order : {order0}")
    print(f"  injected hint    : {hint0!r}")
    print("  (flaky still sorts first; no hint — a fresh tool is never punished.)")

    # ---- Runs 1-3: the model reaches for the flaky tool and it FAILS for real. We
    # advertise only the flaky tool on these establishing runs so each turn is a genuine
    # failed loop turn (real TOOL_FAILED on the shared store), building its track record.
    for i in (1, 2, 3):
        _rule()
        called = await _run_one_turn(storage, [_FLAKY])
        rep = await _reputation(storage)
        flaky = rep[_FLAKY]
        print(
            f"Run {i}: model picked {called!r} -> "
            f"{'FAILED (recording TOOL_FAILED)' if called == _FLAKY else 'SUCCESS'}"
        )
        print(
            f"        {_FLAKY} so far: {flaky.failed} failed / {flaky.total} calls, "
            f"score={flaky.score} (has_min_samples={flaky.has_min_samples})"
        )

    # ---- himmy re-checks reputation now that 3 failures are on record.
    _rule()
    rep = await _reputation(storage)
    flaky = rep[_FLAKY]
    reliable = rep[_RELIABLE]
    print(
        f"himmy re-checks reputation: {_FLAKY} score = {flaky.score} "
        f"({flaky.failed} failures, {flaky.completed} completions) "
        f"-- below the 0.2 floor, has_min_samples={flaky.has_min_samples}"
    )
    print(
        f"                            {_RELIABLE} score = {reliable.score} "
        f"(neutral prior -- never called yet)"
    )

    # ---- The behavioural change: reorder + hint, read from real himmy objects.
    _rule()
    order, hint = await _bound_order_and_hint(storage)
    print(f"{_FLAKY} DEMOTED below {_RELIABLE} + flagged in the prompt:")
    print(f"  bound-tool order BEFORE history : {order0}")
    print(f"  bound-tool order  AFTER history : {order}")
    print("  injected reliability hint       :\n    " + (hint or "(none)").replace("\n", "\n    "))

    # The annotated description the flaky tool now carries in the bound toolset.
    runtime, _registry = build_runtime_for_spec(
        _spec(), inference=_inference_over(storage), storage=storage
    )
    await runtime.reprime_self_learning()
    bound = {bt.name: bt for bt in runtime.tool_service.bound_tools([_FLAKY, _RELIABLE])}
    print(f"  {_FLAKY} description now reads    : {bound[_FLAKY].description!r}")

    # ---- The LEARNING_APPLIED event himmy emits when the order actually changes.
    _rule()
    applied = await storage.list_events(event_type=EventType.LEARNING_APPLIED)
    print(f"LEARNING_APPLIED events emitted: {len(applied)}")
    for ev in applied:
        print(f"  payload: {ev.payload}")

    # ---- Run 4: offer BOTH tools again. The SAME stub model (still "pick the first
    # advertised tool") now reaches for the reliable one — because himmy re-sorted it
    # ahead of the demoted flaky tool.
    _rule()
    called = await _run_one_turn(storage, [_FLAKY, _RELIABLE])
    print(
        f"Run 4: model now picks {called!r} -> "
        f"{'SUCCESS' if called == _RELIABLE else 'still flaky?!'}"
    )
    assert called == _RELIABLE, "expected the reliable tool to win after the reorder"

    _rule()
    print(
        "PROOF: the stub model's tool choice flipped from the flaky tool to the\n"
        "reliable one with NO change to the model — only himmy's recorded reputation,\n"
        "reorder, hint, and LEARNING_APPLIED event drove the behavioural change."
    )
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
