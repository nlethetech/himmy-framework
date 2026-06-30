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
    #: HITL: a member paused on an approval-gated tool — the run goes AWAITING_APPROVAL.
    awaiting_approval: bool = False
    #: The paused MEMBER :class:`AgentCheckpoint` id (so the unchanged pending-approvals
    #: path can read it under ``metadata['checkpoint_id']``).
    member_checkpoint_id: str | None = None
    #: The durable GRAPH checkpoint id to resume the orchestration from.
    orchestration_checkpoint_id: str | None = None
    #: The name of the member (graph node) that paused.
    awaiting_member: str | None = None


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
                # ``tools_module`` survived sanitization only for an operator-provisioned
                # spec (it is a privileged field), so passing it through is safe: a tenant
                # spec reached here with it already stripped. Without it an operator
                # member's declared tools would not resolve in the team registry.
                tools_module=spec.tools_module,
                handoffs=handoffs,
            )
        )
    return TeamSpec(members=members, entry=names[0])


def _build_team_runtime(
    team_spec: Any,
    *,
    storage: Any,
    shared_inference: Any,
    checkpoint_store: Any = None,
    tool_authorizer: Any = None,
    owner_workspace_id: str | None = None,
    owner_subject_scope: str | None = None,
) -> Any:
    """Build ``(team, registry, runtime)`` sharing the run service's storage.

    The team registry is pre-loaded with every member's toolkit packs (by ``build_team``);
    the runtime is wired over the SAME ``storage`` the run record lives in so the
    orchestration's thread + events land in the one canonical store. The shared inference
    is reused unless a member pins its own provider (then a multi-provider manager is built
    by ``build_team_inference``), preserving the offline-stub / configured-gateway default.

    ``checkpoint_store`` (HITL) is the surface-owned :class:`AgentCheckpoint` store; it is
    threaded onto the team runtime so a graph member turn driven with ``hitl=True`` can
    pause into a durable member checkpoint. It is DISTINCT from the graph checkpoint store
    (which persists the orchestration position). With no requires_approval tool present the
    path is byte-identical to before (the store is simply never written).

    ``tool_authorizer`` (P0 confused-deputy fix) is the LAUNCHING principal's
    tool-capability gate, rebuilt from the run's persisted actor. Threaded into the member
    runtime's ToolService so every member tool dispatch is gated by the launching
    principal's grants — closing the confused-deputy hole where a team/workflow could
    invoke a side-effecting member tool its launcher's own role was never granted
    ``tool:<name>:invoke``/``:write`` for. ``None`` (offline / zero-config / no RBAC policy
    wired) leaves member tool dispatch byte-unchanged, exactly like the single-agent path.
    """
    from himmy.config.team_spec import build_team, build_team_inference
    from himmy.runtime.builder import build_runtime
    from himmy.runtime.from_spec import resolve_tools_module

    # P1 tenancy (orchestration path): scope the members' memory/KB (and tasks/notes) packs to
    # THIS run's tenant + (within-tenant) subject so two tenants' — or two users of one tenant's
    # — team/group-chat/graph/workflow runs never share the durable memory/KB namespace. Without
    # this, build_team falls back to ToolkitConfig.from_env() (static scope) and every member run
    # pools onto ONE shared memory subject / KB scope — a cross-tenant confused-deputy read/write.
    # Mirrors the Studio team path (himmy/api/studio_service.py). ``None``/``None`` leaves the
    # historical static scope — byte-unchanged offline / all_tenants.
    team_cfg = None
    if owner_workspace_id is not None or owner_subject_scope is not None:
        from himmy.toolkit import ToolkitConfig

        team_cfg = ToolkitConfig.from_env()
        team_cfg.tenant_scope = owner_workspace_id
        team_cfg.subject_scope = owner_subject_scope
    # An operator member may declare a ``tools_module`` (privileged; tenant specs had it
    # stripped at sanitize). Wire the SAME dotted-path resolver the CLI/from_spec use so
    # the member's declared tools resolve in the team registry — including an
    # approval-gated tool a graph member can pause on.
    team, registry = build_team(
        team_spec,
        toolkit_config=team_cfg,
        resolve_tools_module=resolve_tools_module,
    )
    member_pins_provider = any(m.provider for m in team_spec.members)
    inference = (
        build_team_inference(team_spec)
        if member_pins_provider or shared_inference is None
        else shared_inference
    )
    # A sub-agent / member runtime inherits the launcher's authorizer ATTENUATED (never
    # wider than the launcher) — capability can only narrow down the orchestration, matching
    # the spawn-chain contract the capability module documents.
    member_authorizer = (
        tool_authorizer.attenuate() if tool_authorizer is not None else None
    )
    runtime, _inference, _tools = build_runtime(
        inference=inference,
        tool_registry=registry,
        storage=storage,
        checkpoint_store=checkpoint_store,
        tool_authorizer=member_authorizer,
    )
    return team, registry, runtime


async def _run_multi_agent(
    named: list[tuple[str, AgentSpec]],
    prompt: str,
    *,
    storage: Any,
    shared_inference: Any,
    tool_authorizer: Any = None,
    owner_workspace_id: str | None = None,
    owner_subject_scope: str | None = None,
) -> OrchestrationOutcome:
    """Drive the handoff/delegation orchestrator over the member team.

    HITL is UNSUPPORTED for this kind (it drives an in-memory loop over one evolving
    thread whose live local state is not checkpointed). A member calling an
    approval-gated tool therefore FAILS CLOSED: the tool gate denies the call (it never
    runs), the orchestrator raises :class:`HitlUnsupportedError`, and that is projected
    here into a clean FAILED outcome naming the tool — never a silent continue.
    """
    from himmy.orchestrators import MultiAgentOrchestrator
    from himmy.orchestrators.multi_agent import HitlUnsupportedError

    team_spec = _build_team_spec(named, kind="multi_agent")
    team, registry, runtime = await asyncio.to_thread(
        _build_team_runtime,
        team_spec,
        storage=storage,
        shared_inference=shared_inference,
        tool_authorizer=tool_authorizer,
        owner_workspace_id=owner_workspace_id,
        owner_subject_scope=owner_subject_scope,
    )
    orch = MultiAgentOrchestrator(
        runtime, team, registry, max_turns=_MULTI_AGENT_MAX_TURNS
    )
    try:
        result = await orch.run(prompt)
    except HitlUnsupportedError as exc:
        return OrchestrationOutcome(
            stopped_reason="error", failed=True, error=str(exc)
        )
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
    tool_authorizer: Any = None,
    owner_workspace_id: str | None = None,
    owner_subject_scope: str | None = None,
) -> OrchestrationOutcome:
    """Drive the selector-driven group chat over the member team.

    HITL is UNSUPPORTED for this kind (the shared evolving thread + speaker order are
    not checkpointed). A speaker calling an approval-gated tool FAILS CLOSED: the tool is
    denied (never runs) and the run is projected to a clean FAILED naming the tool.
    """
    from himmy.orchestrators import GroupChatOrchestrator
    from himmy.orchestrators.multi_agent import HitlUnsupportedError

    team_spec = _build_team_spec(named, kind="group_chat")
    team, registry, runtime = await asyncio.to_thread(
        _build_team_runtime,
        team_spec,
        storage=storage,
        shared_inference=shared_inference,
        tool_authorizer=tool_authorizer,
        owner_workspace_id=owner_workspace_id,
        owner_subject_scope=owner_subject_scope,
    )
    orch = GroupChatOrchestrator(
        runtime, team, registry, max_rounds=_GROUP_CHAT_MAX_ROUNDS
    )
    try:
        result = await orch.run(prompt)
    except HitlUnsupportedError as exc:
        return OrchestrationOutcome(
            stopped_reason="error", failed=True, error=str(exc)
        )
    return OrchestrationOutcome(
        thread_id=result.thread.thread_id,
        output_text=result.output_text,
        stopped_reason=result.stopped_reason,
        route=list(result.speaker_order),
    )


def _member_llm_config(spec: AgentSpec) -> Any:
    """The member's LLMConfig keyed by the team inference dispatch key.

    Routes the turn through the SAME ``<provider>:<model>`` dispatch key the team
    inference manager registered each backend under (``build_team_inference``);
    ``spec.to_llm_config()`` carries only the plain model name, which misses that key
    and silently falls through to the stub default — so a graph/workflow run would
    execute on the stub regardless of the pinned provider. Forces ``AUTO_TOOLS`` when
    the member has bound tools so the model actually calls them (and so an
    approval-gated tool can pause).
    """
    from himmy.config.team_spec import dispatch_key_for
    from himmy.services.inference.models import ResponseFormat

    update: dict[str, Any] = {"model_key": dispatch_key_for(spec.provider, spec.model)}
    if spec.tools:
        update["response_format"] = ResponseFormat.AUTO_TOOLS
    return spec.to_llm_config().model_copy(update=update)


def _node_prompt(node_name: str, state: dict[str, Any]) -> str:
    """The pipeline prompt for a node: the base prompt + the prior step's output."""
    prior = state.get("last_output") or ""
    base_prompt = state.get("prompt") or ""
    return f"{base_prompt}\n\nPrevious step output:\n{prior}" if prior else base_prompt


def _build_pipeline_graph(named: list[tuple[str, AgentSpec]], make_node: Any) -> Any:
    """Compose the linear one-node-per-member pipeline graph (shared by run + resume)."""
    from himmy.orchestrators import END, StateGraph

    graph = StateGraph(name="workflow")
    node_names = [name for name, _ in named]
    for index, (name, spec) in enumerate(named):
        graph.add_node(name, make_node(name, spec))
        if index == 0:
            graph.set_entry_point(name)
        else:
            graph.add_edge(node_names[index - 1], name)
    graph.add_edge(node_names[-1], END)
    return graph


async def _run_graph(
    named: list[tuple[str, AgentSpec]],
    prompt: str,
    *,
    storage: Any,
    shared_inference: Any,
    graph_checkpoint_store: Any,
    graph_resume_id: str | None,
    checkpoint_store: Any = None,
    approve_member: bool | None = None,
    actor: str = "human",
    tool_authorizer: Any = None,
    owner_workspace_id: str | None = None,
    owner_subject_scope: str | None = None,
) -> OrchestrationOutcome:
    """Drive a durable linear state-graph: one node per member, output threaded forward.

    Each node runs its agent with ``hitl=True`` on the SHARED runtime, reading the prior
    node's text from the graph state (``last_output``) and writing its own back, so the
    members form a pipeline. When a member calls an approval-gated tool the node raises a
    :class:`~himmy.orchestrators.state_graph.NodeInterrupt`, which the graph converts into a
    durable HITL interrupt — the orchestration goes AWAITING_APPROVAL with the paused member
    checkpoint id.

    On a HITL RESUME (``approve_member`` is not None) the paused member's
    ``resume_agent_loop`` is driven exactly-once via the ``resume_member`` splice; the graph
    advances and continues (possibly pausing again at a later member). The graph compiles
    against the caller's :class:`SqliteGraphCheckpointStore`, so an interrupted long run
    resumes from its last completed superstep via ``graph_resume_id``.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.orchestrators import MemberResumeOutcome
    from himmy.orchestrators.state_graph import NodeInterrupt
    from himmy.runtime.checkpoint import InMemoryGraphCheckpointStore

    team_spec = _build_team_spec(named, kind="group_chat")
    _team, _registry, runtime = await asyncio.to_thread(
        _build_team_runtime,
        team_spec,
        storage=storage,
        shared_inference=shared_inference,
        checkpoint_store=checkpoint_store,
        tool_authorizer=tool_authorizer,
        owner_workspace_id=owner_workspace_id,
        owner_subject_scope=owner_subject_scope,
    )

    def _make_node(node_name: str, spec: AgentSpec) -> Any:
        async def _node(state: dict[str, Any]) -> dict[str, Any]:
            task = Task(
                title=f"{node_name}-step",
                prompt=_node_prompt(node_name, state),
                context=({"tool_names": list(spec.tools)} if spec.tools else {}),
            )
            # hitl=True only when the runtime actually carries a checkpoint store AND the
            # member has tools that could be gated; otherwise stay byte-identical to the
            # plain pipeline turn (a tool-less member can never pause).
            use_hitl = checkpoint_store is not None and bool(spec.tools)
            if use_hitl:
                loop = await runtime.run_agent_loop(
                    spec.to_persona(),
                    task,
                    llm_config=_member_llm_config(spec),
                    hitl=True,
                )
                if loop.stopped_reason == "awaiting_approval":
                    # A member paused on an approval-gated tool — escape the node runner
                    # as a typed interrupt (NOT a failure) so the graph persists a durable
                    # HITL pause and the orchestration goes AWAITING_APPROVAL.
                    raise NodeInterrupt(node_name, loop.checkpoint_id or "")
                text = loop.final.output_text or ""
            else:
                result = await runtime.run_task_detailed(
                    spec.to_persona(), task, llm_config=_member_llm_config(spec)
                )
                text = result.output_text or ""
            outputs = dict(state.get("outputs") or {})
            outputs[node_name] = text
            return {"last_output": text, "outputs": outputs}

        return _node

    graph = _build_pipeline_graph(named, _make_node)
    compiled = graph.compile(
        checkpoint_store=graph_checkpoint_store or InMemoryGraphCheckpointStore(),
        memory_store=storage,
    )

    async def _resume_member(
        node: str, member_cp: str, state: dict[str, Any]
    ) -> MemberResumeOutcome:
        """Drive the paused member's resume_agent_loop, returning its spliced delta.

        Exactly-once is the member checkpoint's own ``claim()`` + idempotency ledger
        (inside ``resume_agent_loop``); a crash-retry replays the recorded tool result.
        If the SAME member pauses again on a second gated tool, the outcome flags
        ``paused_again`` with the new member checkpoint so the graph re-stamps the pause.

        CRASH-SAFE ADVANCE: if the member checkpoint is ALREADY resolved (a crash between
        the member resolving and the graph persisting its advance) ``resume_agent_loop``
        loses the claim and raises. Rather than stranding the graph node pending forever,
        recover the recorded member output and advance ANYWAY — the gated tool already ran
        exactly once (the ledger), so this only finishes a half-applied advance. The split
        between "advance" and "no-op" is owned by the SERVICE layer (a fresh double-approve
        of an already-terminal run is refused at the inbox before reaching here).
        """
        from himmy.core.errors import HimmyError

        try:
            member_loop = await runtime.resume_agent_loop(
                member_cp, approved=bool(approve_member), actor=actor, hitl=True
            )
        except HimmyError as exc:
            if "already resolved" not in str(exc):
                raise
            # The member is already resolved. Two cases:
            #  * a CONCURRENT double-approve loser: a sibling already advanced the GRAPH
            #    too (it is no longer interrupted-at-this-member) — re-raise so the
            #    service treats it as a clean exactly-once no-op (never a double advance);
            #  * a genuine CRASH-RETRY: the member resolved but the graph never advanced
            #    (still interrupted at this node) — recover the member's REAL final output
            #    from its durable checkpoint and advance ANYWAY so the run is never
            #    stranded AND downstream sees the real text. The gated tool already ran
            #    once (the ledger); this only re-threads the lost text.
            recovered = _recovered_member_advance(
                graph_checkpoint_store,
                graph_resume_id,
                node,
                member_cp,
                checkpoint_store,
            )
            if recovered is None:
                raise
            outputs = dict(state.get("outputs") or {})
            outputs[node] = recovered
            return MemberResumeOutcome(
                delta={"last_output": recovered, "outputs": outputs}
            )
        if member_loop.stopped_reason == "awaiting_approval":
            return MemberResumeOutcome(
                paused_again=True,
                member_checkpoint_id=member_loop.checkpoint_id or "",
            )
        text = member_loop.final.output_text or ""
        outputs = dict(state.get("outputs") or {})
        outputs[node] = text
        return MemberResumeOutcome(delta={"last_output": text, "outputs": outputs})

    if approve_member is not None:
        result = await compiled.invoke(
            resume=graph_resume_id, resume_member=_resume_member
        )
    else:
        result = await compiled.invoke(
            {"prompt": prompt, "last_output": "", "outputs": {}},
            resume=graph_resume_id,
        )

    return _project_graph_result(result, graph_checkpoint_store)


def _recovered_member_advance(
    graph_checkpoint_store: Any,
    graph_resume_id: str | None,
    node: str,
    member_cp: str,
    checkpoint_store: Any = None,
) -> str | None:
    """Recover a resolved member's output for a crash-retry advance, or None (no-op).

    Returns the member's recorded final text ONLY when the durable graph checkpoint is
    STILL interrupted at exactly this node + member (a genuine crash between the member
    resolving and the graph advancing). When a sibling has already advanced the graph (a
    concurrent double-approve loser) this returns ``None`` so the caller re-raises and the
    advance happens exactly once.

    The recovered text is the member's REAL final answer, read from the ``final_output``
    the resolved member checkpoint recorded AT RESOLVE TIME (closing #2): the graph state
    never saw it (the crash struck before the advance persisted), so threading it from the
    durable member record is what lets downstream see the true output rather than an empty
    string. The gated tool's side effect already ran exactly once via the ledger, so this
    only re-threads the lost text. A member checkpoint that never recorded a final answer
    (``final_output is None`` — an older/partial record, or no member store wired) falls
    back to the empty string so recovery still ADVANCES (never strands the run).
    """
    if graph_checkpoint_store is None or not graph_resume_id:
        return None
    chk = graph_checkpoint_store.load(graph_resume_id)
    if (
        chk is None
        or chk.paused_node != node
        or chk.member_checkpoint_id != member_cp
    ):
        # The graph already advanced past this member — a concurrent loser; no-op.
        return None
    if checkpoint_store is not None:
        member_chk = checkpoint_store.load(member_cp)
        if member_chk is not None and member_chk.final_output is not None:
            return str(member_chk.final_output)
    return ""


def _project_graph_result(result: Any, graph_checkpoint_store: Any) -> OrchestrationOutcome:
    """Map a :class:`GraphRunResult` onto an :class:`OrchestrationOutcome`.

    An ``interrupted`` result that carries a paused MEMBER reference is an AWAITING_APPROVAL
    pause; a bare ``interrupted`` (timeout/crash, no member) keeps the existing projection
    so the ordinary durable resume path still works.
    """
    final_state = result.final_state or {}
    failed = result.status == "failed"
    outcome = OrchestrationOutcome(
        thread_id=f"graph:{result.checkpoint_id}",
        output_text=str(final_state.get("last_output") or "") or None,
        stopped_reason=result.status,
        route=list(result.node_sequence),
        failed=failed,
        error=result.error if failed else None,
        graph_checkpoint_id=result.checkpoint_id,
    )
    if result.status == "interrupted" and graph_checkpoint_store is not None:
        loaded = graph_checkpoint_store.load(result.checkpoint_id)
        if loaded is not None and loaded.member_checkpoint_id:
            outcome.awaiting_approval = True
            outcome.member_checkpoint_id = loaded.member_checkpoint_id
            outcome.orchestration_checkpoint_id = result.checkpoint_id
            outcome.awaiting_member = loaded.paused_node or None
    return outcome


def _is_graph_kind(kind: str, resource_kind: str) -> bool:
    """True when this run drives the durable linear graph (a workflow, or a graph team).

    A WORKFLOW always runs as the durable linear graph pipeline regardless of ``kind`` (a
    workflow IS an ordered pipeline); a TEAM runs the graph only when its ``kind`` is
    ``graph``. These are exactly the kinds with robust nested HITL.
    """
    return resource_kind == "workflow" or kind == "graph"


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
    checkpoint_store: Any = None,
    approve_member: bool | None = None,
    actor: str = "human",
    tool_authorizer: Any = None,
    owner_workspace_id: str | None = None,
    owner_subject_scope: str | None = None,
) -> OrchestrationOutcome:
    """Resolve the member specs and drive the orchestrator for ``kind`` (T3b entry point).

    A WORKFLOW (``resource_kind == 'workflow'``) always runs as the durable linear graph
    pipeline regardless of ``kind`` (a workflow IS an ordered pipeline); a TEAM runs the
    orchestrator its ``kind`` names. Returns the :class:`OrchestrationOutcome` the run
    service projects onto the canonical run record.

    ``checkpoint_store`` (HITL) is the surface-owned :class:`AgentCheckpoint` store; threaded
    into the graph member runtime so a member calling an approval-gated tool pauses to
    AWAITING_APPROVAL (graph + workflow kinds only). ``approve_member`` (not None) routes to
    the HITL RESUME splice instead of a fresh run — ``True`` approves, ``False`` rejects.

    ``tool_authorizer`` (P0 confused-deputy fix) is the LAUNCHING principal's
    tool-capability gate, rebuilt by the run service from the run's persisted actor. It is
    threaded into every member runtime (attenuated) so a team/workflow can never invoke a
    side-effecting tool the launching principal's own role was not granted — closing the
    same confused-deputy hole the single-agent path closes. ``None`` (offline / zero-config
    / no RBAC policy wired) leaves member tool dispatch byte-unchanged.

    ``owner_workspace_id`` / ``owner_subject_scope`` (P1 tenancy) are the LAUNCHING run's tenant
    and (within-tenant) subject. They are threaded into every member runtime's toolkit config so
    the members' memory/KB (and tasks/notes) packs are namespaced to this run's owner — closing
    the cross-tenant (and cross-user) confused-deputy DATA leak where two tenants' team/group-
    chat/graph/workflow runs would otherwise share ONE durable memory subject / KB scope. Mirrors
    the single-agent and Studio team paths. ``None``/``None`` (offline / all_tenants / no tenant
    binding) leaves the historical static scope — byte-unchanged.
    """
    named = _member_specs(members, operator_provisioned=operator_provisioned)

    if _is_graph_kind(kind, resource_kind):
        return await _run_graph(
            named,
            prompt,
            storage=storage,
            shared_inference=shared_inference,
            graph_checkpoint_store=graph_checkpoint_store,
            graph_resume_id=graph_resume_id,
            checkpoint_store=checkpoint_store,
            approve_member=approve_member,
            actor=actor,
            tool_authorizer=tool_authorizer,
            owner_workspace_id=owner_workspace_id,
            owner_subject_scope=owner_subject_scope,
        )
    if kind == "group_chat":
        return await _run_group_chat(
            named,
            prompt,
            storage=storage,
            shared_inference=shared_inference,
            tool_authorizer=tool_authorizer,
            owner_workspace_id=owner_workspace_id,
            owner_subject_scope=owner_subject_scope,
        )
    return await _run_multi_agent(
        named,
        prompt,
        storage=storage,
        shared_inference=shared_inference,
        tool_authorizer=tool_authorizer,
        owner_workspace_id=owner_workspace_id,
        owner_subject_scope=owner_subject_scope,
    )


__all__ = ["OrchestrationOutcome", "run_orchestration", "_is_graph_kind"]
