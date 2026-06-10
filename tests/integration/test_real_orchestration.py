"""WS10 — CI-gated real-model coverage for MULTI-AGENT, MULTI-TURN, and HYBRID-RAG.

Before WS10 the CI integration lane (``.github/workflows/integration.yml``) exercised
only single-agent tool/plain runs (``tests/integration/test_real_provider.py``). The
richer paths — agent-to-agent handoff/group-chat/fan-out, multi-turn conversations that
must carry context across calls, and hybrid (BM25+dense) RAG grounding — were verified
only by the standalone ``tests/live/*`` ``__main__`` scripts, which pytest never collects
and CI never runs. This module promotes that coverage into the gated lane: a regression in
any of those paths now blocks a merge.

Like ``test_real_provider.py`` these are marked ``integration`` and skipped when Ollama is
unreachable, so the normal offline suite stays green; the CI integration job provides an
Ollama service container + a small model, which makes them effectively required.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Literal

import httpx
import pytest

from tests.conftest import run_async

if TYPE_CHECKING:
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.tools.registry import ToolRegistry

_OLLAMA_URL = os.environ.get("HIMMY_OLLAMA_URL", "http://localhost:11434")
_MODEL = os.environ.get("HIMMY_INTEGRATION_MODEL", "qwen2.5:3b-instruct")


def _ollama_available() -> bool:
    try:
        return httpx.get(f"{_OLLAMA_URL}/api/tags", timeout=3.0).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable"),
]


def _new_runtime() -> tuple[SingleAgentRuntime, ToolRegistry]:
    """Build a runtime + shared :class:`ToolRegistry` bound to the real Ollama model."""
    from himmy import build_runtime
    from himmy.cli.provider import build_inference_for
    from himmy.services.tools.registry import ToolRegistry

    registry = ToolRegistry()
    inference = build_inference_for("ollama", _MODEL)
    runtime, _inf, _tools = build_runtime(inference=inference, tool_registry=registry)
    return runtime, registry


# --------------------------------------------------------------------------- #
# (a) MULTI-AGENT end-to-end (handoff + fan-out), mirroring                    #
#     tests/live/live_multiagent_ollama.py                                     #
# --------------------------------------------------------------------------- #


def test_multi_agent_handoff_runs_through_two_agents_on_a_real_model() -> None:
    """A handoff team must transfer control to a second agent on a real model.

    Mirrors the live handoff check: the entry persona calls ``transfer_to_<peer>``,
    after which the peer runs under ITS OWN system prompt. We assert a non-empty
    final answer, that the run touched >=2 distinct agents, and the framework-level
    proof that the shared thread's SYSTEM message was swapped to the target persona.
    """
    from himmy.agents.base_agent.thread import MessageRole
    from himmy.agents.personas.persona import Persona
    from himmy.core.events import EventType
    from himmy.orchestrators import AgentTeam, TeamMember
    from himmy.orchestrators.multi_agent import MultiAgentOrchestrator

    pirate = Persona(
        name="pirate",
        description="A swashbuckling pirate captain who speaks in pirate slang.",
        instructions=[
            "You are a pirate. Speak in heavy pirate slang (arr, matey, ye).",
            "You MUST end EVERY message with the exact token: SIGNOFF_PIRATE",
            "Keep replies to one or two short sentences.",
        ],
    )
    robot = Persona(
        name="robot",
        description="A cold, precise industrial robot that speaks in machine logs.",
        instructions=[
            "You are a robot. Speak in clipped, mechanical phrases prefixed 'UNIT:'.",
            "You MUST end EVERY message with the exact token: SIGNOFF_ROBOT",
            "Keep replies to one or two short sentences.",
        ],
    )
    team = AgentTeam(
        members=[
            TeamMember(name="pirate", persona=pirate, handoffs=["robot"]),
            TeamMember(name="robot", persona=robot),
        ],
        entry="pirate",
    )

    # Live-model BEHAVIOR retry policy: whether a 3B model decides to call the
    # handoff tool is sampling-marginal even at temperature 0 (CPU kernels round
    # differently across hosts). One retry with a fresh team distinguishes "the
    # framework broke handoffs" (fails twice) from a single model coin-flip.
    # Unit/offline tests never get retries — this applies to live-model behavior only.
    res = None
    events: list[object] = []
    for _attempt in range(2):
        runtime, registry = _new_runtime()
        events = []

        async def on_event(ev: object, _sink: list[object] = events) -> None:
            _sink.append(ev)

        orch = MultiAgentOrchestrator(
            runtime,
            team,
            registry,
            on_event=on_event,
            max_turns=4,
            default_temperature=0.0,
        )

        res = run_async(
            orch.run(
                "You are the pirate. Briefly greet, then you MUST call the "
                "transfer_to_robot tool to hand the conversation to the robot so "
                "the robot can give a status report. Do not answer the robot's "
                "part yourself."
            )
        )
        if any(
            getattr(e, "event_type", None) == EventType.AGENT_HANDOFF for e in events
        ):
            break

    assert res is not None
    answer = (res.output_text or "").strip()
    assert answer, "multi-agent run returned an EMPTY final answer"

    handoff_events = [
        e for e in events if getattr(e, "event_type", None) == EventType.AGENT_HANDOFF
    ]
    assert handoff_events, (
        "no AGENT_HANDOFF event fired in 2 attempts (handoff never happened)"
    )

    # End-to-end through >=2 agents: both personas must have spoken / be in the chain.
    speakers = {speaker for speaker, _ in res.turns}
    assert len(speakers) >= 2 or "robot" in res.handoff_chain, (
        f"expected >=2 distinct agents, saw speakers={speakers} "
        f"chain={res.handoff_chain}"
    )

    # FRAMEWORK-LEVEL PROOF: the shared thread's final SYSTEM message belongs to the
    # TARGET (robot) persona, not the source (pirate) — i.e. the handoff swapped it.
    systems = [m for m in res.thread.messages if m.role == MessageRole.SYSTEM]
    assert systems, "no system message on the shared thread"
    final_system = systems[-1]
    persona_meta = final_system.metadata.get("persona")
    assert persona_meta == "robot" or (
        "SIGNOFF_ROBOT" in final_system.content
        and "robot" in final_system.content.lower()
    ), f"shared-thread system prompt was not swapped to robot (meta={persona_meta!r})"


def test_multi_agent_fan_out_runs_workers_concurrently_on_a_real_model() -> None:
    """Parallel fan-out runs N workers and joins their typed results end-to-end.

    Mirrors the live fan-out check: typed ``SubtaskSpec`` -> typed ``SubtaskResult``,
    join order == submission order, every worker returns a non-empty grounded answer
    on the real model, and the FANOUT_* lifecycle events fire.
    """
    from himmy.agents.personas.persona import Persona
    from himmy.core.events import EventType
    from himmy.orchestrators import AgentTeam, TeamMember
    from himmy.orchestrators.group_chat import SubtaskResult, SubtaskSpec, fan_out

    analyst = Persona(
        name="analyst",
        description="A data analyst who reasons about numbers and trends.",
        instructions=["You analyze data and cite figures.", "Be concise."],
    )
    writer = Persona(
        name="writer",
        description="A copywriter who turns findings into clear prose.",
        instructions=["You write clear, engaging summaries.", "Be concise."],
    )
    critic = Persona(
        name="critic",
        description="A skeptical critic who finds flaws and risks.",
        instructions=["You find weaknesses and risks in any plan.", "Be concise."],
    )
    team = AgentTeam(
        members=[
            TeamMember(name="analyst", persona=analyst),
            TeamMember(name="writer", persona=writer),
            TeamMember(name="critic", persona=critic),
        ],
        entry="analyst",
    )

    runtime, _registry = _new_runtime()
    fan_events: list[object] = []

    async def on_event(ev: object) -> None:
        et = getattr(ev, "event_type", None)
        if et in (
            EventType.FANOUT_STARTED,
            EventType.FANOUT_WORKER_COMPLETED,
            EventType.FANOUT_JOINED,
        ):
            fan_events.append(et)

    specs = [
        SubtaskSpec(
            worker="analyst",
            objective="State the single biggest risk of high inflation in one sentence.",
        ),
        SubtaskSpec(
            worker="writer",
            objective="Write a one-sentence tagline for a green energy startup.",
        ),
        SubtaskSpec(
            worker="critic",
            objective="Name one flaw in the plan 'lend at 0% to everyone'.",
        ),
    ]

    # Live-model retry policy (see the handoff test): a slow host can time out a
    # worker's generation; one retry distinguishes a framework fan-out bug (fails
    # twice) from environmental tail latency. Live-model behavior tests only.
    t0 = time.perf_counter()
    res = None
    attempts_used = 1
    for attempt in range(2):
        attempts_used = attempt + 1
        res = run_async(
            fan_out(
                runtime,
                team,
                specs,
                on_event=on_event,
                max_worker_turns=2,
                temperature=0.0,
            )
        )
        if all(r.answer.strip() for r in res.results):
            break
    wall = time.perf_counter() - t0

    assert res is not None
    assert len(res.results) == len(specs), "fan-out dropped a worker result"
    assert all(isinstance(r, SubtaskResult) for r in res.results), (
        "fan-out returned a non-typed result"
    )
    assert [r.worker for r in res.results] == [s.worker for s in specs], (
        "fan-out join order did not match submission order"
    )
    assert sum(1 for r in res.results if r.answer.strip()) == len(specs), (
        "a fan-out worker produced an EMPTY answer on the real model (2 attempts)"
    )
    assert EventType.FANOUT_STARTED in fan_events, "FANOUT_STARTED never fired"
    assert EventType.FANOUT_JOINED in fan_events, "FANOUT_JOINED never fired"
    # Sanity bound: a real concurrent fan-out of three short tasks should not take an
    # absurd amount of wall-clock; a hard serialization/hang regression blows past
    # this. Scaled by attempts so the live-model retry doesn't trip it.
    budget = 600.0 * attempts_used
    assert wall < budget, f"fan-out wall-clock unexpectedly high: {wall:.1f}s"


# --------------------------------------------------------------------------- #
# (b) MULTI-TURN conversation end-to-end, mirroring the live agent-loop path.  #
# --------------------------------------------------------------------------- #


def test_multi_turn_conversation_carries_context_on_a_real_model() -> None:
    """A 2-turn exchange where turn 2 can ONLY be answered from turn 1's context.

    The same :class:`ChatThread` is threaded from turn 1 into turn 2, so the model
    must remember a fact stated in turn 1 to answer turn 2 correctly. This is the
    real-model multi-turn memory contract that the live scripts cover but CI did not.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    runtime, _registry = _new_runtime()
    persona = Persona(
        name="assistant",
        instructions=[
            "You are a helpful assistant with a good memory.",
            "Answer in one short sentence.",
        ],
    )

    # Turn 1: establish a fact the model could not otherwise know.
    turn1 = run_async(
        runtime.run_task_detailed(
            persona,
            Task(
                title="t1",
                prompt=(
                    "Remember this for later: my favorite number is 4471. "
                    "Just acknowledge that you'll remember it."
                ),
            ),
        )
    )
    assert (turn1.output_text or "").strip(), "turn 1 returned an empty answer"

    # Turn 2: reuse turn 1's thread. The answer is recoverable ONLY from context.
    turn2 = run_async(
        runtime.run_task_detailed(
            persona,
            Task(
                title="t2",
                prompt="What is my favorite number? Reply with the number only.",
            ),
            thread=turn1.thread,
        )
    )
    answer = (turn2.output_text or "").strip()
    assert answer, "turn 2 returned an empty answer (multi-turn loop broke)"
    assert "4471" in answer, (
        f"turn 2 did not carry turn 1's context; expected 4471, got: {answer!r}"
    )


# --------------------------------------------------------------------------- #
# (c) HYBRID-RAG end-to-end, mirroring tests/live/live_rag_hybrid.py           #
# --------------------------------------------------------------------------- #


def test_hybrid_rag_grounds_a_real_model_answer() -> None:
    """Hybrid (BM25+dense) retrieval feeds a real-model grounded answer.

    A rare code buried in one doc among common distractors is retrieved via the
    hybrid path; the winning chunk grounds a real Ollama answer. Mirrors the live
    RAG script's discriminator: on the rare-token query the lexical (BM25) leg lets
    hybrid rank the needle at least as well as pure dense, so we assert hybrid wins
    retrieval, then assert the real-model answer is non-empty AND grounded in the
    retrieved chunk (mentions recovery facts only present in that chunk).
    """
    from himmy.cli.provider import build_inference_for
    from himmy.services.inference.models import InferenceMessage, InferenceRequest
    from himmy.services.knowledge import (
        DeterministicEmbedder,
        KnowledgeBase,
        RetrievalConfig,
        SemanticChunker,
    )
    from himmy.services.knowledge.models import RetrievedChunk

    rare_code = "XK9-ZQ7-4471"
    docs = [
        (
            "incident_runbook",
            (
                "When the gateway begins dropping connections during a deploy, the "
                "on-call engineer should first check the load balancer health. If the "
                "service refuses to start, the most common root cause is the boot "
                "secret rotation. The recovery procedure is documented under error "
                "code " + rare_code + ", which maps to a stale credential cache that "
                "must be flushed before the service will accept traffic again. After "
                "flushing the cache, restart the workers and confirm the queue drains."
            ),
        ),
        (
            "deploy_guide",
            (
                "Deploys roll out gradually. The load balancer routes traffic to the "
                "new workers once their health checks pass. If a deploy stalls, check "
                "the queue depth and the worker logs. Most deploy issues come from a "
                "service that fails its readiness probe or a connection pool that "
                "exhausts under load during the gradual rollout."
            ),
        ),
        (
            "onboarding",
            (
                "New engineers should set up their laptop, install the toolchain, and "
                "request access to the staging environment. Pair with a buddy for the "
                "first week and read the architecture overview before touching prod."
            ),
        ),
    ]

    chunker = SemanticChunker(max_chars=600, overlap=50)
    # The retrieval query is the lexically-decisive rare-token query: this is the case
    # where the BM25 leg earns its keep, exactly as the live RAG script demonstrates.
    retrieval_query = f"error code {rare_code}"

    async def _build_and_search() -> str:
        needle_doc_id = ""

        async def _ingest(
            mode: Literal["dense", "hybrid"],
        ) -> tuple[KnowledgeBase, str]:
            kb = KnowledgeBase(
                storage=None,
                embedder=DeterministicEmbedder(),
                chunker=chunker,
                retrieval=RetrievalConfig(mode=mode, candidate_pool=20),
            )
            rec = await kb.create_kb(
                workspace_id="ws_it", client_id="client_it", name=f"ops_kb_{mode}"
            )
            nonlocal needle_doc_id
            for title, text in docs:
                doc = await kb.ingest_text(
                    rec.kb_id, text, title=title, source_uri=title
                )
                if title == "incident_runbook":
                    needle_doc_id = doc.document_id
            return kb, rec.kb_id

        dense_kb, dense_id = await _ingest("dense")
        hybrid_kb, hybrid_id = await _ingest("hybrid")

        dense_hits = await dense_kb.search(dense_id, retrieval_query, top_k=2)
        hybrid_hits = await hybrid_kb.search(hybrid_id, retrieval_query, top_k=2)
        assert hybrid_hits, "hybrid search returned no hits"
        assert hybrid_hits[0].document_id == needle_doc_id, (
            "hybrid retrieval did not rank the needle doc first"
        )

        # Hybrid must do at least as well as dense at surfacing the needle (the live
        # script's contract): the needle's rank under hybrid is <= its rank under dense.
        def _needle_rank(hits: list[RetrievedChunk]) -> int:
            for i, h in enumerate(hits):
                if h.document_id == needle_doc_id:
                    return i
            return len(hits)  # not found -> worst possible rank

        assert _needle_rank(hybrid_hits) <= _needle_rank(dense_hits), (
            "hybrid ranked the rare-token needle worse than pure dense"
        )

        assert hybrid_hits[0].text, "winning chunk had no text to ground the model"
        return hybrid_hits[0].text

    context = run_async(_build_and_search())

    inference = build_inference_for("ollama", _MODEL)
    question = (
        f"According to the context, what does error code {rare_code} indicate, "
        "and what is the recovery step?"
    )
    req = InferenceRequest(
        messages=[
            InferenceMessage(
                role="system",
                content=(
                    "You answer ONLY using the provided context. Be concise. "
                    "If the answer is not in the context, say you don't know."
                ),
            ),
            InferenceMessage(
                role="user",
                content=f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            ),
        ],
        generation_params={"temperature": 0.0, "max_tokens": 200},
        timeout_seconds=120.0,
    )
    resp = run_async(inference.run(req))

    answer = (resp.output_text or "").strip()
    assert answer, "RAG-grounded model answer was empty"
    lowered = answer.lower()
    assert any(tok in lowered for tok in ("stale", "credential", "cache", "flush")), (
        "model answer was not grounded in the retrieved chunk; "
        f"expected stale/credential/cache/flush, got: {answer!r}"
    )
