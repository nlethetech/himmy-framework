"""Build + drive a team/workflow orchestration for a /v1 run (T3b).

This is the BODY a :meth:`RunAppService._execute_orchestration_run` runs on its background
task — it reuses the existing orchestrators (``himmy/orchestrators/``), it does not
reimplement them. Given the ordered, pre-resolved member :class:`AgentDefRecord`s it:

1. rehydrates each member's :class:`AgentSpec` and RE-SANITIZES it under the run's operator
   status (defense-in-depth: a stored tenant spec already had its privileged tools stripped
   at write, but a per-run re-sanitize guarantees a tenant orchestration can never reach
   ``tools_module``/``http_tools``/``mcp_servers`` — T0.3);
2. builds a single tool-bearing team runtime that SHARES the run service's storage (so the
   orchestration's thread/events land in the one canonical store) and reuses the shared
   inference when no member pins a provider (so the offline stub / configured gateway is
   preserved with no surprise provider switch);
3. drives the matching engine:

   * ``multi_agent`` — :class:`MultiAgentOrchestrator` (handoff + delegation). With members
     that declare no edges (the common stored-agent case) the entry member runs and
     produces the answer; an agent that DOES declare handoffs/delegates routes as usual.
   * ``group_chat`` — :class:`GroupChatOrchestrator` (round-robin panel over a shared
     thread, ending on ``final_answer`` or ``max_rounds``).
   * ``graph`` — a durable linear :class:`StateGraph` (one node per member, output threaded
     to the next) compiled against the caller's :class:`SqliteGraphCheckpointStore`, so a
     long run resumes after a restart from the last completed superstep.

The returned :class:`OrchestrationOutcome` is projected onto the canonical
:class:`~himmy.services.storage.models.RunRecord` by the run service.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.config.agent_spec import AgentSpec
    from himmy.services.storage.models import AgentDefRecord

#: Bounds on the team/group-chat orchestration loops (kept modest so a misconfigured
#: team cannot spin the provider). A graph workflow runs one node per member, so it is
#: bounded by the member count.
_MULTI_AGENT_MAX_TURNS = 12
_GROUP_CHAT_MAX_ROUNDS = 8


@dataclass
class OrchestrationOutcome:
    """The result of a team/workflow run, projected onto the canonical run record."""

    thread_id: str | None = None
    output_text: str | None = None
    stopped_reason: str = "final"
    route: list[str] = field(default_factory=list)
    failed: bool = False
    error: str | None = None
    #: For the durable ``graph`` kind: the checkpoint id a long run resumes from.
    graph_checkpoint_id: str | None = None


def _member_specs(
    members: list[AgentDefRecord], *, operator_provisioned: bool
) -> list[tuple[str, AgentSpec]]:
    """Rehydrate + re-sanitize each member's spec, returning ``(name, spec)`` in order.

    The names are de-duplicated (two stored agents may share a ``name``) by suffixing a
    1-based index on a collision, so the orchestrator's member-name lookups stay unique
    while the first/entry member keeps its plain name.
    """
    from himmy.config.spec_sanitizer import sanitize_tenant_spec

    out: list[tuple[str, AgentSpec]] = []
    seen: dict[str, int] = {}
    for record in members:
        spec = sanitize_tenant_spec(
            record.agent_spec(), operator_provisioned=operator_provisioned
        ).spec
        base = spec.name or record.agent_id
        if base in seen:
            seen[base] += 1
            name = f"{base}-{seen[base]}"
        else:
            seen[base] = 1
            name = base
        out.append((name, spec))
    return out


def _build_team_spec(named: list[tuple[str, AgentSpec]], *, kind: str) -> Any:
    """Project the ordered member specs into a :class:`TeamSpec` for ``build_team``.

    Each stored agent contributes its persona (name/description/instructions/role), its
    declarative tool surface (``tool_packs``/``tools``), and its model/provider. For the
    ``multi_agent`` kind the ENTRY member is given handoff edges to every peer so a stored
    team (whose members declare no edges of their own) can still route control to a
    specialist; ``group_chat`` needs no edges (the selector drives turns).
    """
    from himmy.config.team_spec import TeamMemberSpec, TeamSpec

    names = [n for n, _ in named]
    members: list[TeamMemberSpec] = []
    for index, (name, spec) in enumerate(named):
        handoffs: list[str] = []
        if kind == "multi_agent" and index == 0 and len(named) > 1:
            handoffs = [n for n in names if n != name]
        members.append(
            TeamMemberSpec(
                name=name,
                description=spec.description,
                instructions=list(spec.instructions),
                role=spec.role,
                provider=spec.provider,
                model=spec.model,
                tools=list(spec.tools),
                tool_packs=list(spec.tool_packs),
                handoffs=handoffs,
            )
        )
    return TeamSpec(members=members, entry=names[0])


def _build_team_runtime(team_spec: Any, *, storage: Any, shared_inference: Any) -> Any:
    """Build ``(team, registry, runtime)`` sharing the run service's storage.

    The team registry is pre-loaded with every member's toolkit packs (by ``build_team``);
    the runtime is wired over the SAME ``storage`` the run record lives in so the
    orchestration's thread + events land in the one canonical store. The shared inference
    is reused unless a member pins its own provider (then a multi-provider manager is built
    by ``build_team_inference``), preserving the offline-stub / configured-gateway default.
    """
    from himmy.config.team_spec import build_team, build_team_inference
    from himmy.runtime.builder import build_runtime

    team, registry = build_team(team_spec)
    member_pins_provider = any(m.provider for m in team_spec.members)
    inference = (
        build_team_inference(team_spec)
        if member_pins_provider or shared_inference is None
        else shared_inference
    )
    runtime, _inference, _tools = build_runtime(
        inference=inference,
        tool_registry=registry,
        storage=storage,
    )
    return team, registry, runtime


async def _run_multi_agent(
    named: list[tuple[str, AgentSpec]],
    prompt: str,
    *,
    storage: Any,
    shared_inference: Any,
) -> OrchestrationOutcome:
    """Drive the handoff/delegation orchestrator over the member team."""
    from himmy.orchestrators import MultiAgentOrchestrator

    team_spec = _build_team_spec(named, kind="multi_agent")
    team, registry, runtime = await asyncio.to_thread(
        _build_team_runtime, team_spec, storage=storage, shared_inference=shared_inference
    )
    orch = MultiAgentOrchestrator(
        runtime, team, registry, max_turns=_MULTI_AGENT_MAX_TURNS
    )
    result = await orch.run(prompt)
    return OrchestrationOutcome(
        thread_id=result.thread.thread_id,
        output_text=result.output_text,
        stopped_reason=result.stopped_reason,
        route=list(result.handoff_chain),
    )


async def _run_group_chat(
    named: list[tuple[str, AgentSpec]],
    prompt: str,
    *,
    storage: Any,
    shared_inference: Any,
) -> OrchestrationOutcome:
    """Drive the selector-driven group chat over the member team."""
    from himmy.orchestrators import GroupChatOrchestrator

    team_spec = _build_team_spec(named, kind="group_chat")
    team, registry, runtime = await asyncio.to_thread(
        _build_team_runtime, team_spec, storage=storage, shared_inference=shared_inference
    )
    orch = GroupChatOrchestrator(
        runtime, team, registry, max_rounds=_GROUP_CHAT_MAX_ROUNDS
    )
    result = await orch.run(prompt)
    return OrchestrationOutcome(
        thread_id=result.thread.thread_id,
        output_text=result.output_text,
        stopped_reason=result.stopped_reason,
        route=list(result.speaker_order),
    )


async def _run_graph(
    named: list[tuple[str, AgentSpec]],
    prompt: str,
    *,
    storage: Any,
    shared_inference: Any,
    graph_checkpoint_store: Any,
    graph_resume_id: str | None,
) -> OrchestrationOutcome:
    """Drive a durable linear state-graph: one node per member, output threaded forward.

    Each node runs its agent on the SHARED runtime, reading the prior node's text from the
    graph state (``last_output``) and writing its own back, so the members form a pipeline.
    The graph compiles against the caller's :class:`SqliteGraphCheckpointStore`, so an
    interrupted long run resumes from its last completed superstep via ``graph_resume_id``.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.orchestrators import END, StateGraph
    from himmy.runtime.checkpoint import InMemoryGraphCheckpointStore

    team_spec = _build_team_spec(named, kind="group_chat")
    _team, _registry, runtime = await asyncio.to_thread(
        _build_team_runtime, team_spec, storage=storage, shared_inference=shared_inference
    )

    graph = StateGraph(name="workflow")
    node_names = [name for name, _ in named]

    def _make_node(node_name: str, spec: AgentSpec) -> Any:
        async def _node(state: dict[str, Any]) -> dict[str, Any]:
            prior = state.get("last_output") or ""
            base_prompt = state.get("prompt") or ""
            turn_prompt = (
                f"{base_prompt}\n\nPrevious step output:\n{prior}" if prior else base_prompt
            )
            task = Task(title=f"{node_name}-step", prompt=turn_prompt, context={})
            result = await runtime.run_task_detailed(
                spec.to_persona(), task, llm_config=spec.to_llm_config()
            )
            text = result.output_text or ""
            outputs = dict(state.get("outputs") or {})
            outputs[node_name] = text
            return {"last_output": text, "outputs": outputs}

        return _node

    for index, (name, spec) in enumerate(named):
        graph.add_node(name, _make_node(name, spec))
        if index == 0:
            graph.set_entry_point(name)
        else:
            graph.add_edge(node_names[index - 1], name)
    graph.add_edge(node_names[-1], END)

    compiled = graph.compile(
        checkpoint_store=graph_checkpoint_store or InMemoryGraphCheckpointStore(),
        memory_store=storage,
    )
    result = await compiled.invoke(
        {"prompt": prompt, "last_output": "", "outputs": {}},
        resume=graph_resume_id,
    )
    final_state = result.final_state or {}
    failed = result.status == "failed"
    return OrchestrationOutcome(
        thread_id=f"graph:{result.checkpoint_id}",
        output_text=str(final_state.get("last_output") or "") or None,
        stopped_reason=result.status,
        route=list(result.node_sequence),
        failed=failed,
        error=result.error if failed else None,
        graph_checkpoint_id=result.checkpoint_id,
    )


async def run_orchestration(
    *,
    kind: str,
    members: list[AgentDefRecord],
    prompt: str,
    resource_kind: str,
    storage: Any,
    shared_inference: Any,
    operator_provisioned: bool,
    graph_checkpoint_store: Any = None,
    graph_resume_id: str | None = None,
) -> OrchestrationOutcome:
    """Resolve the member specs and drive the orchestrator for ``kind`` (T3b entry point).

    A WORKFLOW (``resource_kind == 'workflow'``) always runs as the durable linear graph
    pipeline regardless of ``kind`` (a workflow IS an ordered pipeline); a TEAM runs the
    orchestrator its ``kind`` names. Returns the :class:`OrchestrationOutcome` the run
    service projects onto the canonical run record.
    """
    named = _member_specs(members, operator_provisioned=operator_provisioned)

    if resource_kind == "workflow":
        return await _run_graph(
            named,
            prompt,
            storage=storage,
            shared_inference=shared_inference,
            graph_checkpoint_store=graph_checkpoint_store,
            graph_resume_id=graph_resume_id,
        )
    if kind == "graph":
        return await _run_graph(
            named,
            prompt,
            storage=storage,
            shared_inference=shared_inference,
            graph_checkpoint_store=graph_checkpoint_store,
            graph_resume_id=graph_resume_id,
        )
    if kind == "group_chat":
        return await _run_group_chat(
            named, prompt, storage=storage, shared_inference=shared_inference
        )
    return await _run_multi_agent(
        named, prompt, storage=storage, shared_inference=shared_inference
    )


__all__ = ["OrchestrationOutcome", "run_orchestration"]
