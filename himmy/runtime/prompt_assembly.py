"""Prompt / tools / request assembly for :class:`SingleAgentRuntime`.

Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
step ``request_builder``). :class:`RequestBuilder` owns the "turn a persona +
task + thread + ctx into a typed :class:`InferenceRequest`" behavior:

* rendering the operator-authored prompts and the projected snapshot blocks
  SEPARATELY (``render_prompt_parts``) and routing the INJECTED context through
  the input guardrail before it enters the model (``render_guarded_prompts``);
* resolving the effective model key (``effective_model_key``);
* deriving the per-turn :class:`CachePolicy` (``prompt_cache_policy``); and
* building the final :class:`InferenceRequest` — messages, generation params,
  tool binding, cache policy, tool executor (``build_request``).

The runtime constructs one of these in ``__init__`` and its former methods
become thin delegating shims. Behavior is byte-for-byte identical to the
pre-extraction inline code — in particular the **byte-stable system+tools
prefix** and the tenant/conversation cache-KEY derivation are unchanged, so a
recorded replay cassette and every provider prompt-cache prefix stay identical.

The four cache-key module functions (:func:`_cache_scope_metadata`,
:func:`_prompt_cache_key_for_scope`, :func:`_openai_conversation_cache_key_enabled`,
:func:`_prompt_cache_key_for_conversation`) also live here and are re-exported
from ``single_agent`` so their existing import paths and live env reads are
unchanged.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError
from himmy.services.inference.models import (
    BoundTool,
    CachePolicy,
    InferenceMessage,
    InferenceRequest,
    ResponseFormat,
)
from himmy.services.inference.prompt_cache import CacheCapability

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.inference.models import LLMConfig


def _cache_scope_metadata(ctx: dict[str, Any]) -> dict[str, Any]:
    """Derive tenant-isolation metadata for the inference cache from ``ctx``.

    Stamps the :data:`~himmy.services.inference.cache.CACHE_SCOPE_METADATA_KEYS`
    that the runtime knows about (``subject_id`` from ``context_subject_id`` and
    any ``tenant_id``/``workspace_id`` carried on ``context_metadata``) onto
    ``InferenceRequest.metadata`` so the response cache partitions per principal.
    Only non-empty values are emitted; an unscoped run yields ``{}`` so the cache
    key — and any recorded replay cassette — is byte-for-byte unchanged.
    """
    meta: dict[str, Any] = {}
    subject_id = ctx.get("context_subject_id")
    if subject_id:
        meta["subject_id"] = subject_id
    context_metadata = ctx.get("context_metadata")
    if isinstance(context_metadata, dict):
        for key in ("tenant_id", "workspace_id"):
            value = context_metadata.get(key)
            if value:
                meta[key] = value
    return meta


def _prompt_cache_key_for_scope(scope_metadata: dict[str, Any]) -> str | None:
    """Derive a stable, per-principal PROVIDER prompt-cache partition key (sec-r2).

    Folds the same tenant-isolation metadata :func:`_cache_scope_metadata` stamps for the
    internal response cache into one deterministic ``key=value|…`` string, so the
    OpenAI-family adapter can set ``prompt_cache_key`` and never serve one principal a
    cache-read of another's prefix on a shared provider API key. Returns ``None`` for an
    unscoped run (offline / single-tenant / CLI — no principal metadata) so the request
    payload stays byte-identical to the pre-sec-r2 no-cache-key contract.
    """
    from himmy.services.inference.cache import CACHE_SCOPE_METADATA_KEYS

    parts = [
        f"{key}={scope_metadata[key]}"
        for key in CACHE_SCOPE_METADATA_KEYS
        if scope_metadata.get(key) not in (None, "")
    ]
    return "|".join(parts) if parts else None


def _openai_conversation_cache_key_enabled() -> bool:
    """Whether to stamp a per-conversation OpenAI ``prompt_cache_key`` (default ON).

    ``HIMMY_OPENAI_CONVERSATION_CACHE_KEY=0``/``false``/``no``/``off`` opts out so the
    request payload is byte-identical to the pre-P2.3 no-cache-key contract. Read per call
    (not cached) so a test/host can flip it without reconstructing the runtime.
    """
    return os.environ.get(
        "HIMMY_OPENAI_CONVERSATION_CACHE_KEY", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


def _prompt_cache_key_for_conversation(thread_id: str | None) -> str | None:
    """A stable, per-conversation OpenAI ``prompt_cache_key`` routing hint (P2.3).

    OpenAI's ``prompt_cache_key`` is a pure ROUTING hint: it steers a conversation's turns
    to the same backend so its automatic prefix cache is more likely to hit, and has ZERO
    effect on the generated output. This folds the run's ``thread_id`` (a fresh UUID per
    :class:`ChatThread`, so stable across the turns of one run but distinct across runs)
    into a short opaque digest so the hint is stable within a run and distinct across runs
    without leaking the raw id. Returns ``None`` when no ``thread_id`` is available (keeping
    the payload byte-identical). Applied by the runtime for OpenAI-family managers only; the
    OpenAI adapter additionally gates on :func:`is_openai_family_model`, so no other provider
    is ever affected.
    """
    if not thread_id:
        return None
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:32]
    return f"himmy-conv-{digest}"


class RequestBuilder:
    """Assembles prompts + tools + the typed request for one runtime.

    Holds a back-reference to the owning :class:`SingleAgentRuntime` and reads its
    live wiring (``prompt_manager``, ``context_prompt_mapper``, ``default_model_key``,
    ``inference_service``, ``tool_service``, ``_enable_prompt_cache``, ``_guard_input``,
    ``_wrap_executor_with_retry``) at call time, so runtime reconfiguration between
    runs is honored exactly as when the logic lived inline on the runtime.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    # --------------------------------------------------------------- prompts
    async def render_guarded_prompts(
        self,
        persona: Persona,
        task: Task,
        ctx: dict[str, Any],
        snapshot: Any,
        *,
        trace_id: str | None = None,
        thread_id: str | None = None,
    ) -> tuple[str, str, list[str]]:
        """Render prompts, routing INJECTED context through the guardrail first.

        Recalled long-term memory (``MemoryContextAdapter``) and retrieved KB docs
        reach the model as projected snapshot blocks — content the agent did NOT
        author and that an attacker may have planted in a remembered fact or an
        ingested document (indirect prompt injection / data exfiltration). The
        base persona/task prompts are the operator's own text and are left alone;
        only the injected blocks are passed through :meth:`_guard_input` (the
        configured injection/DLP/blocklist guards), so a poisoned memory or KB chunk
        is redacted/blocked/flagged BEFORE it enters the model. No guardrail
        configured ⇒ ``_guard_input`` is a passthrough, so the merged prompts are
        byte-identical to the unguarded render.
        """
        rt = self._rt
        system_prompt, task_prompt, missing, sys_block, task_block = (
            self.render_prompt_parts(persona, task, ctx, snapshot)
        )
        if sys_block:
            guarded_sys = await rt._guard_input(
                sys_block,
                agent_id=persona.agent_id,
                trace_id=trace_id,
                thread_id=thread_id,
            )
            system_prompt = f"{system_prompt}\n\n{guarded_sys}".strip()
        if task_block:
            guarded_task = await rt._guard_input(
                task_block,
                agent_id=persona.agent_id,
                trace_id=trace_id,
                thread_id=thread_id,
            )
            task_prompt = f"{task_prompt}\n\n{guarded_task}".strip()
        return system_prompt, task_prompt, missing

    def render_prompt_parts(
        self,
        persona: Persona,
        task: Task,
        ctx: dict[str, Any],
        snapshot: Any,
    ) -> tuple[str, str, list[str], str, str]:
        """Render the base prompts and the projected snapshot blocks SEPARATELY.

        Returns ``(system_prompt, task_prompt, missing, sys_block, task_block)`` where
        the two prompts are the operator-authored text (persona/task/system_prefix)
        and ``sys_block``/``task_block`` are the projected snapshot content kept apart
        so a caller can guard the injected context independently of the base prompt.
        """
        rt = self._rt
        system_prompt = ""
        task_prompt = task.prompt
        missing: list[str] = []

        if rt.prompt_manager is not None:
            from himmy.services.prompts.manager import (
                SystemPromptVariables,
                TaskPromptVariables,
            )

            # The persona's instructions are its directives — render them as
            # objectives so they reach the model EVEN WHEN a description is set.
            # (Previously the background used `description or instructions`, which
            # silently dropped every instruction whenever a description existed.)
            objectives = list(persona.instructions or [])
            objectives += list(getattr(persona, "objectives", []) or [])
            objectives += list(ctx.get("objectives", []) or [])
            # Skills: ctx override wins, else persona.metadata.skills/required_skills.
            skills = ctx.get("skills")
            if skills is None:
                skills = persona.metadata.get("skills") or list(
                    getattr(persona, "required_skills", []) or []
                )
            skills = list(skills or [])

            system_vars = SystemPromptVariables(
                role=ctx.get("role") or persona.role,
                persona=persona.description,
                objectives=objectives,
                skills=skills,
                datetime=ctx.get("datetime", ""),
            )
            system_prompt = rt.prompt_manager.get_system_prompt(system_vars)

            task_vars = TaskPromptVariables(
                task=task.prompt,
                output_format=ctx.get("output_format", ""),
                output_schema=ctx.get("output_schema"),
            )
            task_prompt = rt.prompt_manager.get_task_prompt(task_vars) or task.prompt

        # Prepend any system_prefix.
        prefix = ctx.get("system_prefix")
        if prefix:
            system_prompt = f"{prefix}\n\n{system_prompt}".strip()

        # Project snapshot keys into system/task blocks (kept SEPARATE from the base
        # prompts so the injected context can be guarded independently — see
        # render_guarded_prompts).
        sys_block = ""
        task_block = ""
        map_spec = ctx.get("context_prompt_map_spec")
        if (
            rt.context_prompt_mapper is not None
            and map_spec is not None
            and snapshot is not None
        ):
            try:
                sys_block, task_block, missing = rt.context_prompt_mapper.project(
                    snapshot, map_spec
                )
            except Exception:  # pragma: no cover - defensive
                missing = []
                sys_block = ""
                task_block = ""
        return system_prompt, task_prompt, missing, sys_block, task_block

    # ----------------------------------------------------------- inference
    def effective_model_key(
        self, ctx: dict[str, Any], llm_config: LLMConfig | None
    ) -> str:
        """Resolve the effective model key (llm_config > task.context > default)."""
        if llm_config is not None and llm_config.model_key:
            return llm_config.model_key
        return ctx.get("model_key") or self._rt.default_model_key

    def prompt_cache_policy(
        self,
        model_key: str,
        *,
        cache_busted: bool,
        scope_metadata: dict[str, Any],
        thread_id: str | None = None,
    ) -> CachePolicy | None:
        """The per-turn :class:`CachePolicy` (or ``None`` to leave the request unmarked).

        Returns a default ``CachePolicy()`` only when ALL hold:

        * prompt caching is enabled on this runtime (``enable_prompt_cache`` /
          ``HIMMY_PROMPT_CACHE``);
        * compaction did NOT just rewrite the system prefix this turn
          (``cache_busted`` — skip the breakpoint so we don't mark a stale prefix);
        * the underlying manager for ``model_key`` declares a non-NONE cache capability
          (resolved at the call site via the inference service, ``getattr`` default
          NONE) — so every local/offline backend keeps ``None`` and a byte-identical
          payload.

        Returning ``None`` is the byte-identical no-cache path; the system prefix being
        stable WITHIN a run is guaranteed by the runtime (the SYSTEM message — with its
        baked datetime/recalled-memory/KB snapshot — is appended once on the first turn
        and never re-rendered on continuation turns), so the only intra-run buster is
        compaction, which ``cache_busted`` handles.

        sec-r2 + sec-r3 #5: when the run is tenant/principal-scoped (``scope_metadata``
        non-empty), the policy carries a ``cache_key`` derived from that scope so BOTH
        provider families partition the prompt cache per principal:

        * OpenAI-family sets the native ``prompt_cache_key`` routing hint;
        * Anthropic (which has NO out-of-band partition key) folds the scope key into the
          cached prefix as a leading salt block (``anthropic_scope_salt_block``), so two
          principals never share cacheable prefix BYTES — closing the shared-key
          cross-tenant cache-read side-channel on a byte-identical prefix (which the new
          history-cache breakpoint would otherwise widen to tenant/tool-result content).

        P2.3: an UNSCOPED run against an OpenAI-family manager
        (:attr:`CacheCapability.OPENAI_AUTOMATIC`) instead carries a per-conversation
        ``cache_key`` derived from ``thread_id`` (:func:`_prompt_cache_key_for_conversation`).
        OpenAI's ``prompt_cache_key`` is a pure routing hint (steers a conversation's turns to
        one backend for better automatic-cache affinity) with ZERO output effect, so this is a
        speed-only win; it is opt-outable via ``HIMMY_OPENAI_CONVERSATION_CACHE_KEY=0``. A
        scope key (when present) always WINS so tenant isolation is never diluted, and no
        non-OpenAI family is touched: the conversation key is only minted for
        ``OPENAI_AUTOMATIC`` (Anthropic's ``ANTHROPIC_EXPLICIT`` never enters this branch, so
        its cacheable prefix bytes stay unchanged), and the OpenAI adapter itself additionally
        gates ``prompt_cache_key`` on :func:`is_openai_family_model`.

        An unscoped non-OpenAI run keeps ``cache_key=None`` so the payload is byte-identical to
        the pre-change no-cache-key contract.
        """
        rt = self._rt
        if cache_busted or not rt._enable_prompt_cache:
            return None
        capability = rt.inference_service.cache_capability_for(model_key)
        if capability is CacheCapability.NONE:
            return None
        cache_key = _prompt_cache_key_for_scope(scope_metadata)
        if (
            cache_key is None
            and capability is CacheCapability.OPENAI_AUTOMATIC
            and _openai_conversation_cache_key_enabled()
        ):
            cache_key = _prompt_cache_key_for_conversation(thread_id)
        return CachePolicy(cache_key=cache_key)

    def build_request(
        self,
        thread: Any,
        ctx: dict[str, Any],
        llm_config: LLMConfig | None,
        *,
        trace_id: str | None = None,
        cache_busted: bool = False,
    ) -> tuple[InferenceRequest, list[str] | None]:
        """Build the typed InferenceRequest with llm_config-over-context precedence.

        ``trace_id`` (optional) threads onto the transient-retry events the
        wrapped tool executor emits, so retries link to the run like every
        other emission. ``cache_busted`` (C5) suppresses the prompt-cache opt-in for
        this one turn — set when compaction just rewrote the system prefix, so the
        adapter doesn't mark a now-stale prefix and pay a write premium on the miss.
        """
        from himmy.agents.base_agent.thread import MessageRole

        rt = self._rt
        messages = [
            InferenceMessage(
                role=m.role.value if isinstance(m.role, MessageRole) else str(m.role),
                content=m.content,
                metadata=dict(m.metadata),
                tool_call_id=m.metadata.get("tool_call_id"),
                name=m.metadata.get("tool_name"),
            )
            for m in thread.messages
        ]

        model_key = self.effective_model_key(ctx, llm_config)
        generation_params: dict[str, Any] = {}
        response_format: ResponseFormat | None = None
        output_json_schema: dict[str, Any] | None = None
        workflow = None
        route_override = None
        timeout_seconds: float | None = None
        seed: int | None = None
        tool_names = ctx.get("tool_names")
        # Default ON; a caller running its own richer structured-output validation
        # (e.g. TypedAgent's pydantic + repair loop) opts out via this context flag so
        # the inference service does not pre-empt it at the boundary.
        validate_structured_output = ctx.get("validate_structured_output", True)
        if not isinstance(validate_structured_output, bool):
            validate_structured_output = True

        if llm_config is not None:
            response_format = llm_config.response_format
            output_json_schema = llm_config.output_json_schema
            workflow = llm_config.workflow
            route_override = llm_config.route_override
            timeout_seconds = llm_config.timeout_seconds
            seed = llm_config.seed
            if llm_config.temperature is not None:
                generation_params["temperature"] = llm_config.temperature
            if llm_config.max_tokens is not None:
                generation_params["max_tokens"] = llm_config.max_tokens
            if llm_config.top_p is not None:
                generation_params["top_p"] = llm_config.top_p
            if llm_config.use_cache is not None:
                # Forward the cache lever so InferenceService's response cache
                # (honored via generation_params['use_cache']) actually engages.
                generation_params["use_cache"] = llm_config.use_cache
            generation_params.update(llm_config.extra_params or {})
        else:
            # Fall back to task.context for the schema / format hints.
            output_json_schema = ctx.get("output_schema")
            fmt = ctx.get("response_format")
            if isinstance(fmt, ResponseFormat):
                response_format = fmt
            elif isinstance(fmt, str):
                try:
                    response_format = ResponseFormat(fmt)
                except ValueError:
                    response_format = None

        # Bind tools when a tool service is present.
        # RO-9: compute the WORKFLOW single-tool override OUTSIDE the tool_service
        # guard so the event payload always reflects the intended single tool,
        # and fail fast with a clear message when WORKFLOW can't actually bind it.
        bound_tools: list[BoundTool] = []
        tool_names_override: list[str] | None = None
        is_forced_workflow = (
            response_format == ResponseFormat.WORKFLOW
            and workflow is not None
            and workflow.current_tool_name is not None
        )
        if is_forced_workflow:
            tool_names_override = [workflow.current_tool_name]  # type: ignore[union-attr,list-item]

        if rt.tool_service is not None:
            if is_forced_workflow:
                bound_tools = rt.tool_service.bound_tools(tool_names_override)
                bound_names = {bt.name for bt in bound_tools}
                step_tool = tool_names_override[0]  # type: ignore[index]
                if step_tool not in bound_names:
                    raise HimmyError(
                        f"WORKFLOW response_format requires the step tool "
                        f"{step_tool!r} to be bound, but it is not registered "
                        f"with the tool_service"
                    )
            else:
                bound_tools = rt.tool_service.bound_tools(tool_names)
        elif is_forced_workflow:
            # A WORKFLOW run with no tool_service can never bind the step tool;
            # surface the real cause instead of a generic INFERENCE_FAILED later.
            raise HimmyError(
                "WORKFLOW response_format requires a tool_service with the "
                f"named step tool {tool_names_override[0]!r} bound; none is wired"  # type: ignore[index]
            )

        request = InferenceRequest(
            model_key=model_key,
            messages=messages,
            response_format=response_format,
            output_json_schema=output_json_schema,
            workflow=workflow,
            generation_params=generation_params,
            seed=seed,
            validate_structured_output=validate_structured_output,
            route_override=route_override,
            metadata=(scope_metadata := _cache_scope_metadata(ctx)),
            cache_policy=self.prompt_cache_policy(
                model_key,
                cache_busted=cache_busted,
                scope_metadata=scope_metadata,
                thread_id=getattr(thread, "thread_id", None),
            ),
            bound_tools=bound_tools,
            # The single execution seam for the bound tools (see ToolExecutor),
            # wrapped with bounded turn-level retry for transient failures.
            tool_executor=(
                rt._wrap_executor_with_retry(
                    rt.tool_service.tool_executor(),
                    ctx,
                    thread_id=getattr(thread, "thread_id", None),
                    agent_id=getattr(thread, "agent_id", None),
                    trace_id=trace_id,
                )
                if rt.tool_service is not None
                else None
            ),
            tool_names_override=tool_names_override,
        )
        if timeout_seconds is not None:
            request.timeout_seconds = timeout_seconds
        return request, tool_names_override or tool_names
