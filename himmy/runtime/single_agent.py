"""Runtime kernel: SingleAgentRuntime — the per-task conductor of one agent run.

``SingleAgentRuntime.run_task`` is the single public entry point that turns a
persona + task into an answered :class:`~himmy.agents.base_agent.thread.ChatThread`
plus a complete audit trail. It resolves/builds a context snapshot, renders the
system + task prompts, calls inference, replays tool exchanges onto the thread,
appends the assistant message, registers entities, and emits the full RunEvent
sequence. Every dependency except ``inference_service`` is optional and the
runtime degrades cleanly when one is absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import queue
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, ValidationError

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.core.metadata import AssistantMessageMetadata
from himmy.runtime.checkpoint import (
    APPROVED,
    REJECTED,
    AgentCheckpoint,
    CheckpointStore,
    PendingToolCall,
)
from himmy.runtime.termination import final_answer_text, is_no_progress
from himmy.services.inference.models import (
    BoundTool,
    CachePolicy,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    LLMConfig,
    ResponseFormat,
    ToolCallRecord,
    ToolExecutor,
    ToolReturnRecord,
)
from himmy.services.inference.prompt_cache import (
    CacheCapability,
    cache_metrics_payload,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.services.context.service import ContextService
    from himmy.services.governance.consent import Decision, Purpose
    from himmy.services.guardrails.base import GuardrailPipeline
    from himmy.services.inference.service import InferenceService, StreamDelta
    from himmy.services.prompts.manager import PromptManager
    from himmy.services.prompts.mapper import ContextPromptMapper
    from himmy.services.storage.service import ThreadEventStore
    from himmy.services.tools.models import ToolExecutionResult

    # The runtime TRAIN gate (WS4.6). Resolves a Decision for (subject, purpose);
    # ConsentLedger.decision conforms. ``None`` ⇒ ungoverned, gate inert.
    ConsentDecider = Callable[[str, Purpose], Decision]


# An optional caller-facing event callback (RO-6). Invoked best-effort inside
# ``_emit`` alongside the storage/registry/observability sinks so a UI driving a
# long run can receive incremental progress without polling storage.
OnEvent = Callable[[RunEvent], Awaitable[None]]


# The framework-enforced ceiling on agent-loop turns. ``max_turns`` is caller-
# supplied, but the runtime's drive loops are ``while True`` underneath — a
# typo'd or hostile value (say ``10**9``) must not turn them into unbounded
# spend. Every loop entry point validates against this hard cap (the runtime
# counterpart of the state graph's ``recursion_limit``); raise it deliberately
# here if a workload genuinely needs more turns. The default ``max_turns=6``
# is unaffected.
HARD_MAX_TURNS = 100

# How many characters of a tool's result text ride on the TOOL_COMPLETED event payload (for
# observers / live UIs). Default 2000 keeps event logs small; a consumer that wants to reconstruct
# the FULL structured result from the event (e.g. to render a rich card) can raise it via
# ``HIMMY_TOOL_RESULT_EVENT_MAX``. The model context is unaffected — this only bounds the event.
try:
    _TOOL_RESULT_EVENT_MAX = max(200, int(os.environ.get("HIMMY_TOOL_RESULT_EVENT_MAX", "2000")))
except ValueError:
    _TOOL_RESULT_EVENT_MAX = 2000



def _validate_max_turns(max_turns: int, entry_point: str) -> None:
    """Reject an out-of-range ``max_turns`` before any loop work starts.

    Raises :class:`HimmyError` when ``max_turns`` is below 1 or above the
    framework ceiling :data:`HARD_MAX_TURNS`. Shared by every public loop
    entry point (``run_agent_loop`` / ``stream_agent_loop`` /
    ``resume_agent_loop``) so the bound is enforced uniformly.
    """
    if max_turns < 1:
        raise HimmyError(f"{entry_point} requires max_turns >= 1.")
    if max_turns > HARD_MAX_TURNS:
        raise HimmyError(
            f"{entry_point} requires max_turns <= {HARD_MAX_TURNS} "
            f"(got {max_turns}): the runtime enforces a hard turn ceiling."
        )


# Turn-level transient tool retry. The tool service marks a transiently-failed
# call (timeout / rate-limited / provider-unavailable — its own
# ``RETRYABLE_TOOL_CODES`` taxonomy) as a failed ``ToolReturnRecord``, and unless
# the tool declares ``retry_hints`` that failure flows straight into the turn —
# the loop just carries on with the failure in the transcript. The runtime
# therefore retries the affected TOOL execution (never the whole inference) a
# bounded number of extra times with a short exponential backoff, emitting a
# trace event per retry. Both knobs are per-task overridable via the
# ``tool_retry_attempts`` / ``tool_retry_backoff_seconds`` context keys.
DEFAULT_TOOL_RETRY_ATTEMPTS = 2
DEFAULT_TOOL_RETRY_BACKOFF_SECONDS = 0.2

#: What a BLOCKED user prompt is replaced with before it reaches the model. A
#: blocking input guardrail (DLP ``…:block``, blocklist, injection) must NOT let the
#: offending text through — the model never sees the blocked prompt; it sees this.
_INPUT_BLOCK_PLACEHOLDER = (
    "[blocked: this message was withheld by an input guardrail and not delivered]"
)
#: What a BLOCKED assistant answer is replaced with before it reaches the user / the
#: persisted thread. Used when the firing guardrail did not already substitute its own
#: safe text (e.g. GroundingGuardrail supplies a tailored refusal; a DLP ``…:block`` /
#: blocklist returns the original, so the runtime substitutes this).
_OUTPUT_BLOCK_PLACEHOLDER = (
    "I can't share that response: it was withheld by an output guardrail because it "
    "contained blocked content."
)


def _transient_tool_codes() -> frozenset[str]:
    """The tool error codes the runtime treats as transient (turn-level retry).

    Sourced from the tools kernel's own ``RETRYABLE_TOOL_CODES`` taxonomy
    (timeout / rate-limited / provider-unavailable) so the two layers can never
    drift; imported lazily because the tools kernel is not on this kernel's
    import path.
    """
    from himmy.services.tools.service import RETRYABLE_TOOL_CODES

    return frozenset(code.value for code in RETRYABLE_TOOL_CODES)


def _tool_timeout_code() -> str:
    """The tools kernel's TIMEOUT error-code value (the side-effect-unsafe retry).

    Unlike rate-limit / provider-unavailable (the call never reached the tool), a
    TIMEOUT means the tool MAY have run and committed a side effect, so retrying it
    is only safe for read-only tools. Imported lazily — the tools kernel is not on
    this kernel's import path.
    """
    from himmy.services.tools.models import ToolErrorCode

    return ToolErrorCode.TIMEOUT.value


class TaskContext(BaseModel):
    """The typed contract for every ``task.context`` key the runtime recognizes.

    ``task.context`` threads through the whole runtime as a plain ``dict``, so a
    malformed value for a recognized key (say ``tool_names`` as a bare string, or
    ``compaction_spec`` as a non-mapping) used to surface as a confusing mid-run
    failure — or worse, mis-run silently (a string iterates as characters). Every
    public entry point now validates its incoming context against this model via
    :func:`_validated_ctx` and fails fast with a clear error naming the offending
    field(s). Unknown extra keys pass through untouched (``extra='allow'``) so
    application-level context and forward/backward compatibility are preserved.
    Internally the runtime keeps threading the plain dict — this model is the
    boundary contract, not a new in-band type.
    """

    model_config = ConfigDict(extra="allow")

    # --- inference / tools ----------------------------------------------------
    model_key: str | None = None
    tool_names: list[str] | None = None
    skill_routing_hints: list[str] | None = None
    response_format: ResponseFormat | str | None = None
    output_schema: dict[str, Any] | None = None
    #: Validate a structured reply against ``output_schema`` at the inference service
    #: boundary. ``None`` -> the default (on); a caller with its own richer validation
    #: (TypedAgent) sets ``False`` so the service does not pre-empt its repair loop.
    validate_structured_output: bool | None = None

    # --- prompt rendering -------------------------------------------------------
    role: str | None = None
    objectives: list[str] | None = None
    skills: list[str] | None = None
    datetime: str = ""
    output_format: str = ""
    system_prefix: str | None = None

    # --- context snapshot ---------------------------------------------------
    snapshot_id: str | None = None
    context_subject_id: str | None = None
    #: ``ContextBuildSpec | dict`` — validated by ``ContextService.build_snapshot``.
    context_build_spec: Any = None
    context_metadata: dict[str, Any] | None = None
    #: ``ContextPromptMapSpec | dict`` — validated by ``ContextPromptMapper.project``.
    context_prompt_map_spec: Any = None

    # --- loop behavior --------------------------------------------------------
    compaction_spec: dict[str, Any] | None = None
    #: Extra turn-level retries for a TRANSIENT tool failure (timeout /
    #: rate-limited / provider-unavailable). ``None`` -> the default
    #: (:data:`DEFAULT_TOOL_RETRY_ATTEMPTS`); ``0`` disables the retry.
    tool_retry_attempts: int | None = None
    #: Base backoff (seconds) between those retries; doubles per attempt.
    tool_retry_backoff_seconds: float | None = None


def _validated_ctx(context: dict[str, Any] | None, entry_point: str) -> dict[str, Any]:
    """Validate the runtime-recognized context keys at a public entry point.

    Returns a plain ``dict`` copy of ``context`` (the runtime keeps threading the
    dict internally — the :class:`TaskContext` model is only the boundary
    contract, so a valid context's runtime behavior is byte-identical). A
    malformed value for a KNOWN key raises a :class:`HimmyError` naming the
    entry point and the offending field(s) instead of a confusing mid-run
    failure; unknown extra keys pass through untouched.
    """
    raw = dict(context or {})
    try:
        TaskContext.model_validate(raw)
    except ValidationError as exc:
        raise HimmyError(f"{entry_point}: invalid task context: {exc}") from exc
    return raw


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


# WS4.6 — the human data subject participating in the *current* run, set ONLY when a
# ``consent_decider`` is wired (governed deployments) and the task carries a
# ``context_subject_id``. It is published by ``_subject_scope`` around EVERY public entry
# point that emits subject-bearing spine records — all FIVE of them: ``run_task`` (and its
# ``run_task_detailed`` alias), ``run_agent_loop``, ``continue_turn``, ``stream_task``, and
# ``resume_agent_loop`` (the HITL-resume path, which rebuilds ``ctx`` from the checkpoint
# and then emits resumed run events / tool messages / a bumped thread version) — so the
# multi-turn, streaming AND resume paths are governed too (not just ``run_task``).
# Otherwise their records would be subject-less and the ConsentAwareRegistry would fail
# closed and silently drop them even for a consented subject.
# ``_emit`` / ``_register_message`` / ``_register_thread_version`` stamp this onto
# the ``run_event`` / ``message`` / ``chat_thread`` record metadata so a
# ``ConsentAwareRegistry`` can resolve (and gate / crypto-shred) the subject behind those
# otherwise subject-less spine records. A :class:`contextvars.ContextVar` keeps this
# correct under concurrent runs and the default ``None`` keeps the ungoverned path
# byte-identical — no metadata is ever stamped when no decider is configured.
_CURRENT_SUBJECT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "himmy_run_subject", default=None
)


@runtime_checkable
class ToolServiceProtocol(Protocol):
    """The minimal tool-service surface the runtime depends on (RO-12).

    Replaces the previous ``Any`` typing of ``tool_service`` with a structural
    contract: anything exposing ``bound_tools(names) -> list[BoundTool]`` (the
    pure-data binding the runtime feeds to ``InferenceRequest.bound_tools``) and
    ``tool_executor() -> ToolExecutor`` (the execution seam attached to the request)
    satisfies the runtime. ``ToolService`` conforms to this without inheritance.
    """

    def bound_tools(
        self, names: list[str] | None = None
    ) -> list[BoundTool]:  # pragma: no cover - structural typing
        ...

    def tool_executor(self) -> ToolExecutor:  # pragma: no cover - structural typing
        """Return the callback that executes the bound tools by name."""
        ...

    async def execute(
        self, invocation: Any, *, idempotency_store: Any = None
    ) -> Any:  # pragma: no cover - structural typing
        """Execute one tool invocation (used by HITL resume to run an approved tool).

        ``idempotency_store`` (a ``ToolIdempotencyStore``, optional) lets resume
        paths replay an already-executed call instead of running it twice;
        implementations that never dedup may accept and ignore it.
        """
        ...


class _CheckpointToolIdempotencyStore:
    """Adapts a checkpoint's ``executed_tool_results`` to ``ToolIdempotencyStore``.

    The HITL-resume idempotency record: ``put`` writes the serialized execution
    result onto the checkpoint AND persists it IMMEDIATELY — before the resume
    proceeds to transcript writes, further pending calls, or the status flip to
    ``approved``. So once a state-mutating tool has run, a crash (or any failure)
    anywhere before the checkpoint is resolved cannot re-execute it: the next
    ``resume_agent_loop`` finds the record and the tool service replays the
    recorded result instead.
    """

    def __init__(self, checkpoint: AgentCheckpoint, store: CheckpointStore) -> None:
        self._checkpoint = checkpoint
        self._store = store

    def get(self, key: str) -> ToolExecutionResult | None:
        """Return the recorded result for ``key``, or None if never executed.

        Reads the DURABLE ledger (re-loading the checkpoint from the store), not
        just this call's in-memory copy: a resume that re-enters after a crash —
        or that loses the atomic claim race to a sibling that already executed and
        recorded this key — sees the persisted result and replays it instead of
        running a side-effecting tool a second time. Falls back to the in-memory
        copy when the row is gone (e.g. an in-test store that drops it).
        """
        latest = self._store.load(self._checkpoint.checkpoint_id)
        ledger = (
            latest.executed_tool_results
            if latest is not None
            else self._checkpoint.executed_tool_results
        )
        raw = ledger.get(key) or self._checkpoint.executed_tool_results.get(key)
        if raw is None:
            return None
        # Imported lazily: the tools kernel is not on this kernel's import path.
        from himmy.services.tools.models import ToolExecutionResult

        # Keep the working copy consistent so the durable record survives the
        # checkpoint's later status-flip save (which writes this in-memory copy).
        self._checkpoint.executed_tool_results.setdefault(key, raw)
        return ToolExecutionResult.model_validate(raw)

    def put(self, key: str, result: ToolExecutionResult) -> None:
        """Record ``result`` on the checkpoint and persist it durably, NOW.

        Merges any concurrently-recorded keys from the durable ledger first so a
        sibling resume's record is not clobbered by this copy's status-flip save.
        """
        latest = self._store.load(self._checkpoint.checkpoint_id)
        if latest is not None:
            for k, v in latest.executed_tool_results.items():
                self._checkpoint.executed_tool_results.setdefault(k, v)
        self._checkpoint.executed_tool_results[key] = result.model_dump(mode="json")
        self._store.save(self._checkpoint)


@dataclass
class RunResult:
    """A typed view of one ``run_task`` outcome for the application layer (RO-5).

    ``run_task`` still returns the :class:`ChatThread` for back-compat;
    :meth:`SingleAgentRuntime.run_task_detailed` returns this richer object so a
    caller can read status/cost/structured output and the real typed tool
    exchanges (RO-4) without scraping thread rows or catching exceptions.
    """

    thread: ChatThread
    status: str
    output_text: str | None = None
    output_structured: Any = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_returns: list[ToolReturnRecord] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    model_path: str = ""
    provider_name: str = ""
    request_id: str | None = None
    trace_id: str | None = None
    workflow: Any = None
    workflow_complete: bool | None = None
    # True when the provider ran the FULL tool round-trip internally (e.g. pydantic-ai),
    # so this turn already holds the final answer and the loop must not continue.
    round_trip_complete: bool = False

    @property
    def succeeded(self) -> bool:
        """True when the terminal inference status is SUCCESS."""
        return self.status == InferenceStatus.SUCCESS.value


@dataclass
class AgentLoopResult:
    """The outcome of a runtime-owned multi-turn agent loop.

    ``turns`` is every :class:`RunResult` in order; ``final`` is the last.
    ``stopped_reason`` is one of ``final`` (model answered with no tool calls),
    ``max_turns``, ``budget``, or ``error``. Token/cost totals are summed across
    turns so a caller can see the whole run's spend.
    """

    thread: ChatThread
    turns: list[RunResult] = field(default_factory=list)
    stopped_reason: str = "final"
    checkpoint_id: str | None = None

    @property
    def final(self) -> RunResult:
        """The last turn's result."""
        return self.turns[-1]

    @property
    def turn_count(self) -> int:
        """How many model turns the loop ran."""
        return len(self.turns)

    @property
    def total_cost(self) -> float:
        """Summed provider cost across all turns."""
        return sum(t.cost for t in self.turns)

    @property
    def total_input_tokens(self) -> int:
        """Summed input tokens across all turns."""
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        """Summed output tokens across all turns."""
        return sum(t.output_tokens for t in self.turns)


#: The forced final-turn instruction when a tool-using loop ends with no answer.
_SYNTHESIS_NUDGE = (
    "You have already gathered the information you need from the tools above. "
    "Now answer the user's original question directly and completely, using only "
    "those results. Do not call any tools."
)

# Raw-I/O capture (debug inspector): per-field and per-message size caps so an
# opt-in capture can never bloat the event log unboundedly.
_IO_FIELD_CAP = 4000
_IO_MSG_CAP = 1500
_IO_MAX_MESSAGES = 16


def _truncate(text: str, cap: int) -> str:
    text = text or ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _snapshot_grounding(snapshot: Any) -> list[dict[str, Any]]:
    """Knowledge citations a snapshot pulled into the prompt (one entry per KB field).

    Reads each ``knowledge_base``-sourced :class:`ContextField`: the query it ran and
    the chunks it retrieved, each with a snippet, similarity, and source URI. Returns
    ``[]`` when no KB field was resolved, so non-RAG agents add nothing.
    """
    out: list[dict[str, Any]] = []
    fields = getattr(snapshot, "fields", None) or {}
    for key, fld in fields.items():
        if getattr(fld, "source", None) != "knowledge_base":
            continue
        value = getattr(fld, "value", None) or {}
        chunks = value.get("chunks") if isinstance(value, dict) else None
        meta = getattr(fld, "metadata", None) or {}
        citations = []
        for c in chunks or []:
            snippet = c.get("text") or c.get("context_window") or ""
            citations.append(
                {
                    "text": _truncate(str(snippet), 400),
                    "similarity": c.get("similarity"),
                    "source_uri": c.get("source_uri"),
                }
            )
        out.append(
            {
                "source": "knowledge",
                "key": key,
                "query": meta.get("query"),
                "kb_name": meta.get("kb_name"),
                "citations": citations,
            }
        )
    return out


def build_io_capture(request: Any, response: Any) -> dict[str, Any]:
    """A bounded snapshot of one inference's raw I/O (for the trace inspector).

    Captures the messages sent to the model, the bound tool names, the raw response
    text, and the parsed tool calls — each size-capped. Opt-in (see the runtime's
    ``capture_io`` flag); never on by default, so there's zero cost or exposure unless
    a developer explicitly turns it on.
    """
    messages = []
    for m in (request.messages or [])[-_IO_MAX_MESSAGES:]:
        messages.append(
            {
                "role": getattr(m, "role", "?"),
                "content": _truncate(getattr(m, "content", "") or "", _IO_MSG_CAP),
            }
        )
    tool_calls = [
        {"tool": c.tool_name, "args": c.args} for c in (response.tool_calls or [])
    ]
    return {
        "model": getattr(response, "model_path", None),
        "messages": messages,
        "tools": [t.name for t in (request.bound_tools or [])],
        "response_text": _truncate(
            getattr(response, "output_text", "") or "", _IO_FIELD_CAP
        ),
        "tool_calls": tool_calls,
    }


class SingleAgentRuntime:
    """Conducts one agent run end-to-end: persona + task in, answered thread out.

    The runtime is stateless and per-task; multi-agent orchestrations compose
    multiple ``run_task`` calls. Pass only ``inference_service`` for a minimal,
    offline run; wire memory/context/tools/registry to gain persistence,
    evidenced context, tool calling, and lineage respectively.
    """

    def __init__(
        self,
        *,
        inference_service: InferenceService,
        memory_store: ThreadEventStore | None = None,
        tool_service: ToolServiceProtocol | None = None,
        context_service: ContextService | None = None,
        prompt_manager: PromptManager | None = None,
        context_prompt_mapper: ContextPromptMapper | None = None,
        entity_registry: EntityRegistryProtocol | None = None,
        default_model_key: str = "default",
        save_threads: bool = True,
        default_deadline_seconds: float | None = None,
        strict_snapshot: bool = False,
        on_event: OnEvent | list[OnEvent] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        input_guardrail: GuardrailPipeline | None = None,
        output_guardrail: GuardrailPipeline | None = None,
        capture_io: bool | None = None,
        consent_decider: ConsentDecider | None = None,
        enable_prompt_cache: bool | None = None,
    ) -> None:
        """Wire the runtime; auto-create prompt manager/mapper when omitted.

        ``default_deadline_seconds`` (RO-1) is an optional wall-clock budget for
        the whole run (snapshot build + render + inference + persistence),
        overridable per call via ``run_task(..., deadline_seconds=...)``. A
        cancelled or timed-out run still emits a terminal ``AGENT_RUN_FINISHED``
        (``error='cancelled'``) and saves the partial thread before re-raising.
        ``strict_snapshot`` (RO-11) makes an explicitly-requested-but-failed
        snapshot raise instead of degrading silently. ``on_event`` (RO-6) is an
        optional caller-facing callback (or list) for streaming/progress.
        ``consent_decider`` (WS4.6) is the opt-in TRAIN gate: when set (governed
        deployments only) a participating human subject lacking TRAIN consent has
        raw-I/O capture forced off and ``rendered_prompt`` stripped from the
        ``INFERENCE_REQUESTED`` event; ``None`` (the default) leaves capture
        byte-identical to a pre-WS4.6 runtime. ``enable_prompt_cache`` (C5) toggles
        universal provider prompt caching: ``None`` (default) reads the env
        ``HIMMY_PROMPT_CACHE`` (default ON, ``0``/``false`` to opt out); an explicit
        bool wins. When ON, a per-turn request opting into caching only marks the
        prefix on a manager that supports it — a no-op for every local/offline backend.
        """
        self.inference_service = inference_service
        self.memory_store = memory_store
        self.tool_service = tool_service
        self.context_service = context_service
        self.entity_registry = entity_registry
        self.default_model_key = default_model_key
        self.save_threads = save_threads
        self.default_deadline_seconds = default_deadline_seconds
        self.strict_snapshot = strict_snapshot
        self._checkpoint_store = checkpoint_store
        # Per-checkpoint resume serialization (HITL exactly-once). The store-level
        # atomic claim is the cross-process gate; this in-process lock keeps two
        # concurrent resumes of the SAME checkpoint on one event loop (a
        # double-clicked Approve, two tabs, an automation retry) from interleaving
        # between the claim and the gated tool's execution — so the approved action
        # runs exactly once. Keyed by checkpoint_id, created on demand.
        self._resume_locks: dict[str, asyncio.Lock] = {}
        self._input_guardrail = input_guardrail
        self._output_guardrail = output_guardrail
        # Opt-in raw-I/O capture for the trace inspector (off unless asked, or the
        # HIMMY_CAPTURE_IO env is truthy).
        self._capture_io = (
            capture_io
            if capture_io is not None
            else os.environ.get("HIMMY_CAPTURE_IO", "").lower() in ("1", "true", "yes")
        )
        # WS4.6 TRAIN gate. When wired (governed deployments only), a participating human
        # subject (ctx['context_subject_id']) lacking TRAIN consent suppresses raw-I/O
        # capture AND the persisted ``rendered_prompt`` for that run. ``None`` ⇒ unchanged.
        self._consent_decider = consent_decider
        # C5: universal prompt caching is default-ON in the agent loop. When enabled AND
        # the underlying manager declares a non-NONE cache capability, the per-turn request
        # carries a CachePolicy() so the adapter can mark the stable system+tools prefix.
        # Opt out per-runtime (enable_prompt_cache=False) or process-wide (HIMMY_PROMPT_CACHE=0).
        # A NONE-capability manager makes the policy a harmless no-op, so this never changes
        # the offline/local default's payloads. Explicit constructor arg wins over the env.
        self._enable_prompt_cache = (
            enable_prompt_cache
            if enable_prompt_cache is not None
            else os.environ.get("HIMMY_PROMPT_CACHE", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._on_event: list[OnEvent] = self._coerce_callbacks(on_event)

        # Auto-create the prompt primitives when available; they have no required
        # dependencies and the framework expects rendered prompts by default.
        if prompt_manager is None:
            try:
                from himmy.services.prompts.manager import PromptManager

                prompt_manager = PromptManager()
            except Exception:  # pragma: no cover - defensive
                prompt_manager = None
        if context_prompt_mapper is None:
            try:
                from himmy.services.prompts.mapper import ContextPromptMapper

                context_prompt_mapper = ContextPromptMapper()
            except Exception:  # pragma: no cover - defensive
                context_prompt_mapper = None
        self.prompt_manager = prompt_manager
        self.context_prompt_mapper = context_prompt_mapper

    def _train_suppressed(self, ctx: dict[str, Any]) -> bool:
        """Whether this run's raw I/O / rendered prompt must be suppressed (WS4.6).

        The TRAIN gate fires only when a ``consent_decider`` is wired (governed
        deployments) AND the participating **human** subject — ``ctx['context_subject_id']``,
        never ``persona.agent_id`` (an agent is not a data subject) — lacks an ``ALLOW``
        decision for :attr:`~himmy.services.governance.consent.Purpose.TRAIN`. With no
        decider the result is always ``False`` so capture behaviour is byte-identical to a
        pre-WS4.6 runtime.
        """
        if self._consent_decider is None:
            return False
        subject = ctx.get("context_subject_id")
        if not subject:
            return False
        from himmy.services.governance.consent import Effect, Purpose

        return self._consent_decider(subject, Purpose.TRAIN).effect is not Effect.ALLOW

    @staticmethod
    def _coerce_callbacks(
        on_event: OnEvent | list[OnEvent] | None,
    ) -> list[OnEvent]:
        """Normalize the ``on_event`` argument to a list of callables."""
        if on_event is None:
            return []
        if isinstance(on_event, list):
            return [cb for cb in on_event if cb is not None]
        return [on_event]

    def add_event_listener(self, callback: OnEvent) -> None:
        """Register an additional caller-facing event callback (RO-6)."""
        self._on_event.append(callback)

    # ------------------------------------------------------------------ public
    async def run_task(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        llm_config: LLMConfig | None = None,
        snapshot_id: str | None = None,
        deadline_seconds: float | None = None,
    ) -> ChatThread:
        """Run one task for a persona and return the (appended) chat thread.

        Mirrors the documented sequence: snapshot resolve/build, prompt render +
        context projection, message appends (SYSTEM on first turn, USER, TOOL
        rows, ASSISTANT), entity registration/links, and the full event series.
        ``llm_config`` takes precedence over ``task.context`` for model knobs.

        Back-compat surface (returns the thread). The terminal assistant message
        metadata carries the full status/cost/token/error/structured contract for
        callers that cannot accept a new return type; :meth:`run_task_detailed`
        exposes the same as a typed :class:`RunResult`.
        """
        result = await self.run_task_detailed(
            persona,
            task,
            thread,
            llm_config=llm_config,
            snapshot_id=snapshot_id,
            deadline_seconds=deadline_seconds,
        )
        return result.thread

    async def run_task_detailed(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        llm_config: LLMConfig | None = None,
        snapshot_id: str | None = None,
        deadline_seconds: float | None = None,
    ) -> RunResult:
        """Run one task and return a typed :class:`RunResult` (RO-5).

        Identical pipeline to :meth:`run_task` but returns status/cost/structured
        output and the real typed ``tool_calls``/``tool_returns`` records, so the
        application layer can detect FAILED runs (invariant #4) without scraping
        thread rows or catching exceptions. An optional ``deadline_seconds``
        (RO-1) bounds the whole run; on timeout/cancellation a terminal
        ``AGENT_RUN_FINISHED(error='cancelled')`` is emitted and the partial
        thread saved before the :class:`asyncio.CancelledError` re-raises.
        """
        from himmy.agents.base_agent.thread import ChatThread

        ctx = _validated_ctx(task.context, "run_task")
        is_new_thread = thread is None
        if thread is None:
            thread = ChatThread(agent_id=persona.agent_id)
        trace_id = f"{thread.thread_id}:{task.task_id}"

        deadline = (
            deadline_seconds
            if deadline_seconds is not None
            else self.default_deadline_seconds
        )

        # WS4.6: wrap the whole try/except (not just _run_task_body's own inner scope) in
        # the subject scope so that the deadline-expiry / cancellation terminal
        # AGENT_RUN_FINISHED event and the final _maybe_save_thread — emitted in the except
        # block, OUTSIDE _run_task_body's scope — are still subject-tagged for a consented
        # subject; otherwise the fail-closed ConsentAwareRegistry would silently drop the
        # cancellation event and partial-thread save in governed mode. Governed only; a
        # no-op when no consent_decider is wired. The inner _run_task_body scope nests.
        with self._subject_scope(ctx):
            try:
                if deadline is not None and deadline > 0:
                    async with _timeout(deadline):
                        return await self._run_task_body(
                            persona,
                            task,
                            thread,
                            ctx,
                            trace_id,
                            is_new_thread=is_new_thread,
                            llm_config=llm_config,
                            snapshot_id=snapshot_id,
                        )
                return await self._run_task_body(
                    persona,
                    task,
                    thread,
                    ctx,
                    trace_id,
                    is_new_thread=is_new_thread,
                    llm_config=llm_config,
                    snapshot_id=snapshot_id,
                )
            except (TimeoutError, asyncio.CancelledError):
                # RO-1: a cancelled run (external cancellation -> CancelledError) or a
                # deadline expiry (asyncio.timeout surfaces TimeoutError on exit) still
                # records a terminal event and persists the partial thread. We always
                # re-raise CancelledError so the run unwinds as a cancellation.
                await self._emit(
                    RunEvent(
                        event_type=EventType.AGENT_RUN_FINISHED,
                        trace_id=trace_id,
                        thread_id=thread.thread_id,
                        agent_id=persona.agent_id,
                        error="cancelled",
                        payload={"status": "CANCELLED"},
                    )
                )
                await self._maybe_save_thread(thread)
                raise asyncio.CancelledError() from None

    async def run_agent_loop(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        max_turns: int = 6,
        cost_budget: float | None = None,
        llm_config: LLMConfig | None = None,
        hitl: bool = False,
        stop_on_no_progress: bool = False,
        synthesize_empty: bool = True,
        route_tools: bool | None = None,
        route_max_tools: int = 4,
        steer_queue: queue.Queue[str] | None = None,
    ) -> AgentLoopResult:
        """Run a bounded, runtime-owned agentic loop: act -> observe -> re-invoke.

        The first turn is a normal run; while a turn calls tools (so the model
        likely wants to act on their results), the runtime feeds the updated thread
        back for another model turn — until the model answers WITHOUT tool calls
        (``final``), or ``max_turns`` / ``cost_budget`` is reached, or a turn FAILS.
        Unlike delegating to a provider's opaque loop, the runtime *bounds* the
        turns, accrues spend, and emits an ``AGENT_TURN_COMPLETED`` event per turn.

        With ``hitl=True`` (requires a ``checkpoint_store``) the loop PAUSES when a
        turn calls a tool that requires approval: it persists an
        :class:`~himmy.runtime.checkpoint.AgentCheckpoint`, emits
        ``APPROVAL_REQUIRED``, and returns with ``stopped_reason='awaiting_approval'``
        and a ``checkpoint_id`` for :meth:`resume_agent_loop`.

        ``steer_queue`` (optional) is the between-turns steering seam: a
        thread-safe queue of user guidance texts that the drive loop drains at the
        top of EACH continuation turn, appending every queued text as a USER
        message before the next request is built — so the model reacts to live
        human steering mid-mission. ``None`` (the default) leaves loop behavior
        byte-identical. Turn bounds (``max_turns`` / :data:`HARD_MAX_TURNS`) are
        unchanged by steering.
        """
        _validate_max_turns(max_turns, "run_agent_loop")
        # Reject a malformed context BEFORE the tool router / first turn runs.
        _validated_ctx(task.context, "run_agent_loop")
        if hitl and self._checkpoint_store is None:
            raise HimmyError("hitl=True requires a checkpoint_store on the runtime.")

        if self._should_route(route_tools):
            task = await self._route_tools(task, route_max_tools)

        # WS4.6: publish the subject for the WHOLE loop so the turn-completed events,
        # the per-turn continuation messages, and any synthesis turn — all of which emit
        # subject-bearing spine records OUTSIDE the first turn's own scope — resolve to
        # the subject (governed only; a no-op when no consent_decider is wired). The
        # inner run_task_detailed scope nests cleanly inside this one.
        ctx = dict(task.context or {})
        with self._subject_scope(ctx):
            first = await self.run_task_detailed(
                persona, task, thread, llm_config=llm_config
            )
            thread = first.thread
            trace_id = f"{thread.thread_id}:{task.task_id}"
            await self._emit_turn_completed(trace_id, thread, persona, 1, first)

            result = await self._drive_loop(
                persona,
                task,
                thread,
                ctx,
                trace_id,
                turns=[first],
                max_turns=max_turns,
                cost_budget=cost_budget,
                llm_config=llm_config,
                hitl=hitl,
                stop_on_no_progress=stop_on_no_progress,
                turns_offset=0,
                cost_offset=0.0,
                steer_queue=steer_queue,
            )
            if synthesize_empty:
                result = await self._maybe_synthesize(
                    result, persona, trace_id, llm_config
                )
            return result

    #: Adaptive routing threshold: with MORE bound tools than this, an unset
    #: ``route_tools`` (None) routes automatically — the schema block for a big
    #: toolset costs more per turn than the one small routing call, and small
    #: local models pick badly from large catalogs. At or below it, no routing.
    AUTO_ROUTE_OVER_TOOLS = 8

    def _should_route(self, route_tools: bool | None) -> bool:
        """Resolve the tri-state routing flag (explicit wins; None = adaptive)."""
        if route_tools is not None:
            return route_tools
        registry = getattr(self.tool_service, "registry", None)
        if registry is None:
            return False
        try:
            return len(registry.list()) > self.AUTO_ROUTE_OVER_TOOLS
        except Exception:  # noqa: BLE001 - routing is an optimization, never a crash
            return False

    async def _route_tools(self, task: Task, max_tools: int) -> Task:
        """Narrow the bound tools to the relevant few for this task (Tier 1.3).

        A no-op unless a tool service is wired, the task hasn't already pinned
        ``tool_names``, and there are more candidate tools than ``max_tools``. Returns
        a copy of the task with ``context['tool_names']`` set to the routed subset.
        """
        if self.tool_service is None or max_tools < 1:
            return task
        ctx = task.context or {}
        if ctx.get("tool_names") is not None:
            return task  # caller already chose the tools — respect that
        registry = getattr(self.tool_service, "registry", None)
        if registry is None:
            return task
        candidates = [(d.name, d.description) for d in registry.list()]
        if len(candidates) <= max_tools:
            return task

        from himmy.runtime.tool_router import select_tools

        # Skills contribute "use this when …" hints so the router knows which tools a
        # capability implies for this request, beyond the bare prompt.
        query = task.prompt
        hints = ctx.get("skill_routing_hints") or []
        if hints:
            query = f"{query}\n\nRelevant capabilities:\n" + "\n".join(
                f"- {h}" for h in hints
            )

        selected = await select_tools(
            self.inference_service,
            query,
            candidates,
            max_tools=max_tools,
            model_key=str(ctx.get("model_key") or self.default_model_key),
        )
        return task.model_copy(update={"context": {**ctx, "tool_names": selected}})

    async def _maybe_synthesize(
        self,
        result: AgentLoopResult,
        persona: Persona,
        trace_id: str,
        llm_config: LLMConfig | None,
    ) -> AgentLoopResult:
        """One forced final turn when a tool-using loop ended with no answer (Tier 1.1).

        Small models often call a tool, get the result, then fail to write the final
        answer (an empty reply). When the loop stops with an empty answer but tools
        WERE used, run one more turn with tools unbound and an explicit instruction to
        answer from the results already gathered — converting an empty into an answer.
        """
        # Only rescue the genuine "model fell silent" stops. ``no_progress`` is an
        # opt-in deliberate halt whose stop reason callers rely on, so leave it.
        if result.stopped_reason not in ("final", "max_turns"):
            return result
        if (result.final.output_text or "").strip():
            return result  # already answered — nothing to nudge
        if not any(t.tool_calls for t in result.turns):
            return result  # no tools were used — synthesis has nothing to work from

        from himmy.agents.base_agent.task import Task

        nudge = Task(
            title="synthesis",
            prompt=_SYNTHESIS_NUDGE,
            context={"tool_names": []},  # unbind tools: force a text answer
        )
        synth = await self.run_task_detailed(
            persona, nudge, thread=result.thread, llm_config=llm_config
        )
        index = result.turn_count + 1
        await self._emit_turn_completed(trace_id, synth.thread, persona, index, synth)
        return AgentLoopResult(
            thread=synth.thread,
            turns=[*result.turns, synth],
            stopped_reason="synthesized",
        )

    async def continue_turn(
        self,
        persona: Persona,
        thread: ChatThread,
        *,
        task_context: dict[str, Any] | None = None,
        llm_config: LLMConfig | None = None,
    ) -> RunResult:
        """Run ONE more inference turn on an existing thread (no new user prompt).

        The model sees the thread as-is (including any prior tool results) and either
        calls more tools or produces a final answer. ``task_context`` carries the
        recognized run knobs (``tool_names``, ``model_key``, ``output_schema``), so a
        multi-agent orchestrator can switch the bound tool set / model per turn. This
        is the public seam over the runtime's own continuation step (used by
        :meth:`run_agent_loop`).
        """
        ctx = _validated_ctx(task_context, "continue_turn")
        trace_id = f"{thread.thread_id}:continue"
        # WS4.6: publish the subject so this turn's message/thread/event records resolve
        # to it (governed only; a no-op when no consent_decider is wired).
        with self._subject_scope(ctx):
            return await self._continue_turn(
                persona, thread, ctx, trace_id, llm_config=llm_config
            )

    async def reinject_system_prompt(
        self,
        persona: Persona,
        thread: ChatThread,
        *,
        task_context: dict[str, Any] | None = None,
    ) -> str:
        """Re-render and (re)inject ``persona``'s system prompt onto ``thread``.

        When control transfers between personas on a SHARED thread (a multi-agent
        handoff), the thread still carries the PREVIOUS persona's SYSTEM message, so a
        plain :meth:`continue_turn` would run the new persona under the old persona's
        instructions. This re-renders the system prompt for ``persona`` (honoring the
        same ``task_context`` knobs as a fresh run — ``role``/``objectives``/``skills``/
        ``system_prefix``) and REPLACES the leading SYSTEM message in place (appending
        one when none exists). The replacement Message is registered into the audit
        spine and the thread version is bumped, so the persona switch is provenance-
        native and replayable. Returns the rendered system prompt (``""`` when no
        prompt manager is wired). No-op for a persona whose prompt already matches.
        """
        from himmy.agents.base_agent.task import Task
        from himmy.agents.base_agent.thread import Message, MessageRole

        ctx = _validated_ctx(task_context, "reinject_system_prompt")
        # A synthetic, promptless task so the renderer can build the system block
        # (the task prompt is irrelevant here — we only swap the SYSTEM message).
        task = Task(title=f"{persona.name}-handoff", prompt="", context=ctx)
        snapshot, _snapshot_id, _err = await self._resolve_snapshot(
            persona, task, ctx, None
        )
        # Guard injected context (recalled memory / KB docs projected into the system
        # block) before it lands in the persona-switch SYSTEM message.
        system_prompt, _task_prompt, _missing = await self._render_guarded_prompts(
            persona, task, ctx, snapshot
        )

        existing = next(
            (m for m in thread.messages if m.role == MessageRole.SYSTEM), None
        )
        if existing is not None and existing.content == system_prompt:
            return system_prompt  # already the right persona — nothing to do

        new_system = Message(
            role=MessageRole.SYSTEM,
            content=system_prompt,
            metadata={"persona": persona.name, "agent_id": persona.agent_id},
        )
        if existing is not None:
            index = thread.messages.index(existing)
            thread.messages[index] = new_system
        else:
            # No SYSTEM yet: it must lead the thread so the model reads it first.
            thread.messages.insert(0, new_system)
        thread.version += 1
        self._register_message(new_system)
        self._register_thread_version(thread)
        return system_prompt

    async def stream_task(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        llm_config: LLMConfig | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Stream one task's assistant reply as :class:`StreamDelta` chunks.

        Mirrors :meth:`run_task_detailed`'s pre-inference setup (snapshot, prompt
        render, system/user message appends) but delegates to
        :meth:`InferenceService.run_stream`, yielding incremental text. The final
        ``done`` delta carries the materialized response; the assistant message is
        appended to the thread before that delta is yielded. Single-turn (no tool
        loop) — intended for streaming a chat reply to a UI/stdout. Closing the
        generator early (the client dropping the stream) closes the underlying
        provider stream deterministically rather than leaving it to the event
        loop's lazy async-generator finalizer.
        """
        from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
        from himmy.services.inference.service import StreamDelta

        if thread is None:
            thread = ChatThread(agent_id=persona.agent_id)
        ctx = _validated_ctx(task.context, "stream_task")
        # WS4.6: publish the subject so the streamed turn's system/user/assistant
        # messages and the bumped thread version resolve to it (governed only; a no-op
        # when no consent_decider is wired).
        with self._subject_scope(ctx):
            snapshot, _snapshot_id, _err = await self._resolve_snapshot(
                persona, task, ctx, None
            )
            # Injected context (recalled memory / retrieved KB docs) is guarded here
            # so a poisoned memory/KB chunk is redacted/blocked before it reaches the
            # model (indirect prompt-injection seam) — parity with run_task_detailed.
            system_prompt, task_prompt, _missing = await self._render_guarded_prompts(
                persona, task, ctx, snapshot, thread_id=thread.thread_id
            )
            if not any(m.role == MessageRole.SYSTEM for m in thread.messages):
                sys_msg = Message(role=MessageRole.SYSTEM, content=system_prompt)
                thread.append_message(sys_msg)
                self._register_message(sys_msg)
            user_msg = Message(
                role=MessageRole.USER,
                content=await self._guard_input(
                    task_prompt,
                    agent_id=persona.agent_id,
                    thread_id=thread.thread_id,
                ),
            )
            thread.append_message(user_msg)
            self._register_message(user_msg)

            trace_id = f"{thread.thread_id}:{task.task_id}"
            # Audit parity with run_task_detailed: a streamed run is a run — it
            # opens with AGENT_RUN_STARTED and ALWAYS closes with a terminal
            # AGENT_RUN_FINISHED (success, failure, or 'cancelled' when the client
            # drops the stream / the consuming task is cancelled mid-run).
            await self._emit(
                RunEvent(
                    event_type=EventType.AGENT_RUN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={
                        "model_key": self._effective_model_key(ctx, llm_config),
                        "persona_name": persona.name,
                        "streamed": True,
                    },
                )
            )

            request, _tool_names = self._build_request(
                thread, ctx, llm_config, trace_id=trace_id
            )
            # The output guardrail must be enforced on a streamed answer too (audit
            # parity with the non-streaming paths). Two regimes:
            #  * BUFFER: the output guard can WITHHOLD blocked content (DLP ``…:block``
            #    / blocklist / blocking injection). An already-streamed secret can't be
            #    recalled, so we must NOT stream — we drain the provider stream silently,
            #    guard the full answer, then emit it as one ``done`` delta.
            #  * GUARD-AFTER: a redact-only / grounding-only output guard. Stream the
            #    tokens, then guard the final text; if it changed, the PERSISTED message
            #    and the ``done`` payload carry the guarded text (the already-streamed
            #    deltas can't be recalled, but the durable copy must be clean) and a
            #    correction delta is emitted so a consumer that materializes from
            #    ``delta`` rather than ``response`` also lands on the guarded text.
            buffer_output = (
                self._output_guardrail is not None
                and self._output_guardrail.suppresses_output_content()
            )
            # ``run_stream`` is an async generator; own the reference so an early
            # close/cancellation of THIS generator can close it in the finally.
            stream = self.inference_service.run_stream(request)
            run_finished = False
            try:
                async for delta in stream:
                    if delta.done and delta.response is not None:
                        raw_text = delta.response.output_text or ""
                        guarded_text = (
                            await self._guard_output(
                                raw_text,
                                agent_id=persona.agent_id,
                                trace_id=trace_id,
                                thread_id=thread.thread_id,
                            )
                            or ""
                        )
                        corrected = guarded_text != raw_text
                        assistant = Message(
                            role=MessageRole.ASSISTANT,
                            content=guarded_text,
                            metadata={
                                "request_id": request.request_id,
                                "streamed": True,
                                **({"guarded": True} if corrected else {}),
                            },
                        )
                        thread.append_message(assistant)
                        self._register_message(assistant)
                        self._register_thread_version(thread)
                        await self._emit(
                            RunEvent(
                                event_type=EventType.AGENT_RUN_FINISHED,
                                trace_id=trace_id,
                                thread_id=thread.thread_id,
                                agent_id=persona.agent_id,
                                latency_ms=delta.response.latency_ms,
                                cost=delta.response.cost,
                                error=(
                                    delta.response.error.message
                                    if delta.response.error is not None
                                    and delta.response.status != InferenceStatus.SUCCESS
                                    else None
                                ),
                                payload={
                                    "status": delta.response.status.value,
                                    "streamed": True,
                                },
                            )
                        )
                        run_finished = True
                        # The ``done`` delta must never carry the pre-guard text. Rewrite
                        # its response (and the textual ``delta``) to the guarded answer.
                        guarded_response = delta.response.model_copy(
                            update={"output_text": guarded_text}
                        )
                        if buffer_output:
                            # Nothing was streamed; deliver the whole guarded answer now.
                            yield StreamDelta(
                                request_id=request.request_id,
                                delta=guarded_text,
                                index=delta.index,
                                done=True,
                                response=guarded_response,
                            )
                        else:
                            if corrected:
                                # The already-streamed tokens were the raw answer; emit a
                                # correction so a delta-materializing consumer ends clean.
                                yield StreamDelta(
                                    request_id=request.request_id,
                                    delta="",
                                    index=delta.index,
                                    event_type="guarded_output",
                                    event_payload={"output_text": guarded_text},
                                )
                            yield delta.model_copy(update={"response": guarded_response})
                        continue
                    if buffer_output and not delta.done:
                        # Suppress intermediate text in BUFFER mode — the answer is only
                        # safe to surface after the guard runs on the ``done`` delta.
                        continue
                    yield delta
            except (GeneratorExit, asyncio.CancelledError):
                # Early termination at a yield point: record the terminal event
                # (mirrors run_task_detailed's cancellation leg) before unwinding.
                if not run_finished:
                    await self._emit(
                        RunEvent(
                            event_type=EventType.AGENT_RUN_FINISHED,
                            trace_id=trace_id,
                            thread_id=thread.thread_id,
                            agent_id=persona.agent_id,
                            error="cancelled",
                            payload={"status": "CANCELLED", "streamed": True},
                        )
                    )
                raise
            finally:
                # GeneratorExit (client closed the stream) / CancelledError land
                # here at a yield point; close the provider stream NOW so its own
                # cleanup runs before we unwind, then the exception re-raises.
                await stream.aclose()

    async def stream_agent_loop(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread | None = None,
        *,
        max_turns: int = 6,
        cost_budget: float | None = None,
        llm_config: LLMConfig | None = None,
        hitl: bool = False,
        stop_on_no_progress: bool = False,
        synthesize_empty: bool = True,
        route_tools: bool | None = None,
        route_max_tools: int = 4,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Stream tokens THROUGH the whole multi-turn tool loop (opt-in).

        :meth:`stream_task` only streams a single turn, so a tool-using run surfaces
        nothing until the final answer. This generator mirrors
        :meth:`run_agent_loop`'s EXACT bounding — the same stop-condition /
        no-progress / cost-budget / HITL logic — but interleaves the work as
        :class:`~himmy.services.inference.service.StreamDelta` chunks: text deltas
        (``event_type=None``), ``"tool_call"`` and ``"tool_result"`` events (one per
        tool exchange, carrying the tool name + args / result), and a ``"turn_end"``
        marker between turns. The terminal ``done=True`` delta carries the
        materialized :class:`AgentLoopResult` in ``event_payload['result']`` so a
        caller can read the typed outcome exactly as :meth:`run_agent_loop` returns.

        The first turn streams its tokens via :meth:`stream_task` (so a UI sees the
        provider's real incremental output); each continuation turn buffers then
        re-chunks at 24 chars for deterministic offline replay. ``run_agent_loop``
        (non-streaming) is unchanged — this is a parallel, additive surface that
        reuses the same checkpoint / pending-approval helpers, so the bounding,
        spend accrual, and ``AGENT_TURN_COMPLETED`` emission are identical.

        With ``hitl=True`` (requires a ``checkpoint_store``) the loop PAUSES exactly
        as the non-streaming loop does: it persists a checkpoint, emits
        ``APPROVAL_REQUIRED`` and yields a final ``done`` delta whose
        ``AgentLoopResult`` has ``stopped_reason='awaiting_approval'`` and a
        ``checkpoint_id`` for :meth:`resume_agent_loop`.

        Early termination is clean: a client closing the stream mid-run
        (``GeneratorExit`` — e.g. an SSE consumer disconnecting) or a task
        cancellation (``CancelledError``) closes the in-flight inner turn
        generators deterministically in a ``finally`` before the exception
        re-raises, so no suspended generators or provider streams are left
        dangling for the event loop's lazy async-generator finalizer.
        """
        from himmy.services.inference.service import StreamDelta

        _validate_max_turns(max_turns, "stream_agent_loop")
        # Reject a malformed context BEFORE the tool router / first turn runs.
        _validated_ctx(task.context, "stream_agent_loop")
        if hitl and self._checkpoint_store is None:
            raise HimmyError("hitl=True requires a checkpoint_store on the runtime.")

        if self._should_route(route_tools):
            task = await self._route_tools(task, route_max_tools)

        from himmy.agents.base_agent.thread import ChatThread as _ChatThread

        # Own the thread reference up front so we keep operating on the SAME thread
        # stream_task mutates (it creates one internally when None — we pre-create it
        # here instead so the continuation turns and tool replay land on it).
        if thread is None:
            thread = _ChatThread(agent_id=persona.agent_id)

        ctx = dict(task.context or {})
        # WS4.6: publish the subject for the WHOLE streamed loop (governed only; a
        # no-op when no consent_decider is wired) so the per-turn continuation
        # messages, turn-completed events, and any synthesis turn — all emitted
        # OUTSIDE the first turn's own scope — resolve to the subject. The inner
        # stream_task / run_task_detailed scopes nest cleanly inside this one.
        with self._subject_scope(ctx):
            # Own the inner turn generators so EARLY termination — the client
            # closing this generator (GeneratorExit) or the consuming task being
            # cancelled (CancelledError) at any yield point — closes the in-flight
            # turn in the ``finally`` below instead of leaving the suspended inner
            # generators (and the provider stream beneath them) dangling until the
            # event loop's lazy async-generator finalizer runs.
            first_stream = self.stream_task(
                persona, task, thread, llm_config=llm_config
            )
            drive: AsyncGenerator[StreamDelta | AgentLoopResult, None] | None = None
            try:
                # --- first turn: stream tokens via stream_task, materialize it ---
                first_response: InferenceResponse | None = None
                async for delta in first_stream:
                    if delta.done:
                        # Swallow the single-turn ``done`` delta: this loop owns the
                        # one terminal ``done`` (carrying the AgentLoopResult).
                        # Capture the materialized response so we can reconstruct
                        # the turn result.
                        first_response = delta.response
                        continue
                    yield delta

                # run_stream always yields a done delta
                assert first_response is not None
                trace_id = f"{thread.thread_id}:{task.task_id}"

                first = self._result_from_response(first_response, trace_id=trace_id)
                first.thread = thread  # the real thread stream_task ran on
                # stream_task does NOT replay TOOL exchanges; replay them now (so a
                # continuation turn sees the tool results), surface them as events.
                await self._append_tool_messages(
                    thread,
                    first_response,
                    request_id=first.request_id or first_response.request_id,
                    trace_id=trace_id,
                    agent_id=persona.agent_id,
                )
                for tool_delta in self._tool_deltas(first):
                    yield tool_delta
                await self._emit_turn_completed(trace_id, thread, persona, 1, first)
                yield StreamDelta(
                    request_id=first.request_id or first_response.request_id,
                    event_type="turn_end",
                    event_payload={"turn": 1, "tool_calls": len(first.tool_calls)},
                )

                # --- continuation turns: mirror _drive_loop, streaming each turn ---
                result: AgentLoopResult | None = None
                drive = self._stream_drive_loop(
                    persona,
                    task,
                    thread,
                    ctx,
                    trace_id,
                    turns=[first],
                    max_turns=max_turns,
                    cost_budget=cost_budget,
                    llm_config=llm_config,
                    hitl=hitl,
                    stop_on_no_progress=stop_on_no_progress,
                )
                async for item in drive:
                    if isinstance(item, AgentLoopResult):
                        result = item
                        break
                    yield item

                assert result is not None
                if synthesize_empty:
                    # Reuse the exact non-streaming synthesis rescue so an empty
                    # tool-using answer is converted to a text answer identically.
                    result = await self._maybe_synthesize(
                        result, persona, trace_id, llm_config
                    )
                yield StreamDelta(
                    request_id=first.request_id or first_response.request_id,
                    done=True,
                    event_type="done",
                    event_payload={"result": result},
                )
            finally:
                # Runs on normal completion (both acloses are no-ops on exhausted
                # generators) AND on GeneratorExit/CancelledError, which then
                # re-raise naturally after the inner generators are closed.
                await first_stream.aclose()
                if drive is not None:
                    await drive.aclose()

    async def _stream_drive_loop(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        turns: list[RunResult],
        max_turns: int,
        cost_budget: float | None,
        llm_config: LLMConfig | None,
        hitl: bool,
        stop_on_no_progress: bool,
    ) -> AsyncGenerator[StreamDelta | AgentLoopResult, None]:
        """Drive streamed continuation turns until a stop condition.

        Mirrors :meth:`_drive_loop`'s EXACT stop-condition / no-progress /
        cost-budget / HITL logic (run from a continuation perspective, so the
        ``turns_offset`` / ``cost_offset`` are zero), but per continuation turn it
        runs the turn and yields its text + ``tool_call`` / ``tool_result`` /
        ``turn_end`` :class:`StreamDelta`s. The final yielded item is the terminal
        :class:`AgentLoopResult` (so the caller stops iterating and owns the single
        ``done`` delta). The checkpoint / pending-approval helpers are shared with
        the non-streaming loop — nothing is duplicated.
        """
        from himmy.services.inference.service import StreamDelta

        while True:
            last = turns[-1]
            if not last.succeeded:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="error"
                )
                return
            if hitl:
                pending = self._pending_approvals(last)
                if pending:
                    checkpoint = self._save_checkpoint(
                        persona,
                        task,
                        thread,
                        ctx,
                        llm_config,
                        max_turns,
                        cost_budget,
                        len(turns),
                        sum(t.cost for t in turns),
                        pending,
                    )
                    await self._emit(
                        RunEvent(
                            event_type=EventType.APPROVAL_REQUIRED,
                            trace_id=trace_id,
                            thread_id=thread.thread_id,
                            agent_id=persona.agent_id,
                            payload={
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "tools": [p.tool_name for p in pending],
                            },
                        )
                    )
                    yield AgentLoopResult(
                        thread=thread,
                        turns=turns,
                        stopped_reason="awaiting_approval",
                        checkpoint_id=checkpoint.checkpoint_id,
                    )
                    return
            if not last.tool_calls:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
                return
            if last.round_trip_complete:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
                return
            if final_answer_text(last) is not None:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final_answer"
                )
                return
            if stop_on_no_progress and is_no_progress(turns):
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="no_progress"
                )
                return
            if len(turns) >= max_turns:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="max_turns"
                )
                return
            if cost_budget is not None and sum(t.cost for t in turns) >= cost_budget:
                yield AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="budget"
                )
                return
            index = len(turns) + 1
            await self._emit(
                RunEvent(
                    event_type=EventType.AGENT_TURN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={"turn": index},
                )
            )
            # Continuation turns buffer then re-chunk for deterministic offline
            # replay (the stub streams in 24-char chunks; matching that keeps the
            # reassembled text identical across turns).
            result = await self._continue_turn(
                persona, thread, ctx, trace_id, llm_config=llm_config
            )
            turns.append(result)
            for text_delta in self._text_deltas(result):
                yield text_delta
            for tool_delta in self._tool_deltas(result):
                yield tool_delta
            await self._emit_turn_completed(trace_id, thread, persona, index, result)
            yield StreamDelta(
                request_id=result.request_id or "",
                event_type="turn_end",
                event_payload={"turn": index, "tool_calls": len(result.tool_calls)},
            )

    @staticmethod
    def _result_from_response(
        response: InferenceResponse, *, trace_id: str
    ) -> RunResult:
        """Reconstruct a :class:`RunResult` from a streamed first turn's response.

        :meth:`stream_task` yields its terminal ``done`` delta carrying the
        materialized :class:`InferenceResponse` but no typed result; this rebuilds
        the same :class:`RunResult` shape :meth:`_continue_turn` produces so the
        streamed first turn drives the loop identically to a non-streamed one.
        """
        assistant_text = response.output_text
        if assistant_text is None and response.output_structured is not None:
            assistant_text = json.dumps(response.output_structured, default=str)
        error_message = response.error.message if response.error else None
        error_code = response.error.code.value if response.error else None
        return RunResult(
            thread=cast("ChatThread", None),  # not used by the loop's stop logic
            status=response.status.value,
            output_text=assistant_text,
            output_structured=response.output_structured,
            tool_calls=list(response.tool_calls),
            tool_returns=list(response.tool_returns),
            error=error_message,
            error_code=error_code,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            latency_ms=response.latency_ms,
            model_path=response.model_path,
            provider_name=response.provider_name,
            request_id=response.request_id,
            trace_id=trace_id,
            workflow=response.workflow,
            workflow_complete=(
                response.workflow.is_complete if response.workflow is not None else None
            ),
            round_trip_complete=bool(response.metadata.get("round_trip_complete")),
        )

    @staticmethod
    def _text_deltas(result: RunResult) -> Iterator[StreamDelta]:
        """Re-chunk a continuation turn's text at 24 chars (stub-faithful)."""
        from himmy.services.inference.service import StreamDelta

        text = result.output_text or ""
        request_id = result.request_id or ""
        index = 0
        for start in range(0, len(text), 24):
            yield StreamDelta(
                request_id=request_id, delta=text[start : start + 24], index=index
            )
            index += 1

    @staticmethod
    def _tool_deltas(result: RunResult) -> Iterator[StreamDelta]:
        """Yield a ``tool_call`` + paired ``tool_result`` delta per tool exchange."""
        from himmy.services.inference.service import StreamDelta

        returns_by_id = {r.tool_call_id: r for r in result.tool_returns}
        request_id = result.request_id or ""
        for call in result.tool_calls:
            yield StreamDelta(
                request_id=request_id,
                event_type="tool_call",
                event_payload={
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "tool_args": dict(call.args),
                },
            )
            ret = returns_by_id.get(call.tool_call_id)
            yield StreamDelta(
                request_id=request_id,
                event_type="tool_result",
                event_payload={
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "outcome": ret.outcome if ret is not None else "unknown",
                    "content": ret.content if ret is not None else None,
                },
            )

    def _resume_lock_for(self, checkpoint_id: str) -> asyncio.Lock:
        """The per-checkpoint resume lock (created on first use), serializing resumes.

        Two concurrent resumes of the SAME checkpoint on one event loop must not
        interleave between the atomic claim and the gated tool's execution, or both
        could run the approved action. Different checkpoints get different locks, so
        unrelated resumes still proceed in parallel.
        """
        lock = self._resume_locks.get(checkpoint_id)
        if lock is None:
            lock = asyncio.Lock()
            self._resume_locks[checkpoint_id] = lock
        return lock

    async def resume_agent_loop(
        self,
        checkpoint_id: str,
        *,
        approved: bool,
        llm_config: LLMConfig | None = None,
        hitl: bool = True,
        actor: str = "human",
    ) -> AgentLoopResult:
        """Resume a paused agent run after a human approves or rejects the action.

        Rehydrates the checkpoint, applies the decision to each pending tool call
        — executing it (``approved=True``) and recording the real result, or
        recording a rejection — then drives one more model turn (so the model sees
        the outcome) and continues the loop. Exactly-once is enforced at three
        layers: a per-checkpoint in-process lock serializes concurrent resumes on one
        event loop; an ATOMIC store claim (``awaiting_approval``/``resolving`` ->
        ``resolving``) gates cross-process resumes and the loser is refused; and each
        approved execution is recorded on the checkpoint the moment it completes
        (``executed_tool_results``, read from the DURABLE ledger), so a resume retried
        after a crash replays recorded results instead of re-executing a
        state-mutating tool.
        """
        if self._checkpoint_store is None:
            raise HimmyError("resume_agent_loop requires a checkpoint_store.")
        if self._checkpoint_store.load(checkpoint_id) is None:
            raise HimmyError(f"unknown checkpoint {checkpoint_id!r}.")
        # Serialize concurrent in-process resumes of this checkpoint end-to-end: the
        # claim, the gated tool's execution, and the status flip all happen under the
        # lock, so a second concurrent resume only enters AFTER the first has resolved
        # (and is then refused or replays via the ledger). The cross-process gate is
        # the store-level atomic claim below; the lock closes the much more common
        # single-process race (double-clicked Approve, two browser tabs, an
        # automation retry hitting one server).
        async with self._resume_lock_for(checkpoint_id):
            return await self._resume_agent_loop_locked(
                checkpoint_id,
                approved=approved,
                llm_config=llm_config,
                hitl=hitl,
                actor=actor,
            )

    async def _resume_agent_loop_locked(
        self,
        checkpoint_id: str,
        *,
        approved: bool,
        llm_config: LLMConfig | None,
        hitl: bool,
        actor: str = "human",
    ) -> AgentLoopResult:
        """The resume body, run under the per-checkpoint lock (see resume_agent_loop)."""
        assert self._checkpoint_store is not None  # guaranteed by the caller
        # Atomic claim: compare-and-set the status to ``resolving`` as a single store
        # operation. Concurrent resumes of the same checkpoint (two tabs, an
        # automation retry, two workers on the shared SQLite file) race here, and only
        # the winner proceeds to execute the approval-gated tool — the loser's claim
        # returns False and it is refused outright, so the gated action runs EXACTLY
        # once. (The old plain status check was a TOCTOU: both callers read
        # awaiting_approval, both executed, then both flipped the status.) An
        # already-resolved (approved/rejected) checkpoint also loses here.
        if not self._checkpoint_store.claim(checkpoint_id):
            current = self._checkpoint_store.load(checkpoint_id)
            status = current.status if current is not None else "unknown"
            raise HimmyError(
                f"checkpoint {checkpoint_id!r} already resolved ({status})."
            )
        # We hold the claim: re-load so the executed_tool_results ledger (and status)
        # reflect the just-claimed row rather than the pre-claim snapshot.
        checkpoint = self._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:  # pragma: no cover - the claim just observed it
            raise HimmyError(f"unknown checkpoint {checkpoint_id!r}.")
        # Checkpoints written by the validated entry points always pass; this guards
        # the resumed drive loop against a hand-crafted/tampered checkpoint row.
        _validate_max_turns(checkpoint.max_turns, "resume_agent_loop")
        if approved and self.tool_service is None:
            raise HimmyError(
                "cannot resume approved: no tool_service to execute the action."
            )

        from himmy.agents.base_agent.task import Task as _Task
        from himmy.agents.base_agent.thread import ChatThread as _ChatThread
        from himmy.agents.personas.persona import Persona as _Persona
        from himmy.services.tools.models import ToolInvocation

        persona = _Persona.model_validate(checkpoint.persona)
        task = _Task.model_validate(checkpoint.task)
        thread = _ChatThread.model_validate(checkpoint.thread)
        # Like the max_turns guard above: contexts checkpointed by the validated
        # entry points always pass; this catches a hand-crafted/tampered ctx row.
        ctx = _validated_ctx(checkpoint.ctx, "resume_agent_loop")
        trace_id = f"{thread.thread_id}:{task.task_id}"
        resume_llm = llm_config
        if resume_llm is None and checkpoint.llm_config is not None:
            resume_llm = LLMConfig.model_validate(checkpoint.llm_config)

        # WS4.6: publish the subject (rebuilt from the checkpoint's ``ctx``, which carries
        # ``context_subject_id``) for the WHOLE resume path so the resumed tool messages,
        # the APPROVAL_* / turn events, the bumped thread version, the inner _continue_turn
        # (which does NOT self-wrap — only the public continue_turn does), and _drive_loop
        # all emit subject-tagged spine records — otherwise a HITL-resumed run for a fully
        # consented subject would emit subject-less records the fail-closed
        # ConsentAwareRegistry silently drops. Governed only; a no-op when no
        # consent_decider is wired. Inner scopes nest cleanly.
        # The per-checkpoint idempotency record: each approved execution is recorded
        # (and persisted) the moment it completes, so a repeated resume — including a
        # retry after a crash between executing a state-mutating tool and resolving
        # the checkpoint — replays the recorded result instead of running it twice.
        idempotency = _CheckpointToolIdempotencyStore(
            checkpoint, self._checkpoint_store
        )

        with self._subject_scope(ctx):
            # Apply the human decision to each pending tool call, recording the outcome
            # on the thread as a TOOL message (so the next model turn sees it).
            for call in checkpoint.pending_tool_calls:
                if approved:
                    assert self.tool_service is not None
                    execution = await self.tool_service.execute(
                        ToolInvocation(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            args=dict(call.args),
                            metadata={
                                "approved": True,
                                "idempotency_key": call.tool_call_id,
                            },
                        ),
                        idempotency_store=idempotency,
                    )
                    tool_returns = [
                        ToolReturnRecord(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            content=execution.result,
                            outcome=execution.outcome,
                            metadata={
                                "approved_by": actor,
                                "error_code": execution.error_code.value
                                if execution.error_code
                                else None,
                            },
                        )
                    ]
                    event_type = EventType.APPROVAL_GRANTED
                else:
                    tool_returns = [
                        ToolReturnRecord(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            content={"rejected": True, "reason": "rejected by human"},
                            outcome="rejected",
                            metadata={"approved_by": actor},
                        )
                    ]
                    event_type = EventType.APPROVAL_REJECTED
                synthetic = InferenceResponse(
                    request_id=f"resume:{checkpoint_id}",
                    status=InferenceStatus.SUCCESS,
                    tool_calls=[
                        ToolCallRecord(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            args=dict(call.args),
                        )
                    ],
                    tool_returns=tool_returns,
                )
                await self._append_tool_messages(
                    thread,
                    synthetic,
                    request_id=synthetic.request_id,
                    trace_id=trace_id,
                    agent_id=persona.agent_id,
                )
                await self._emit(
                    RunEvent(
                        event_type=event_type,
                        trace_id=trace_id,
                        thread_id=thread.thread_id,
                        agent_id=persona.agent_id,
                        tool_call_id=call.tool_call_id,
                        payload={
                            "checkpoint_id": checkpoint_id,
                            "tool_name": call.tool_name,
                            # Enriched (P0-B) so approvals are mineable per agent /
                            # tool / decision / latency without re-joining tables.
                            "decision": "granted" if approved else "rejected",
                            "agent_name": persona.name,
                            "actor": actor,
                            "time_to_decision_ms": _decision_latency_ms(
                                checkpoint.created_at
                            ),
                        },
                    )
                )

            # Resolve the checkpoint exactly once (idempotency guard above).
            self._checkpoint_store.save(
                checkpoint.model_copy(
                    update={"status": APPROVED if approved else REJECTED}
                )
            )
            thread.version += 1
            self._register_thread_version(thread)

            # One continuation turn so the model reacts to the decision, then drive on.
            index = checkpoint.turns_completed + 1
            await self._emit(
                RunEvent(
                    event_type=EventType.AGENT_TURN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={"turn": index},
                )
            )
            result = await self._continue_turn(
                persona, thread, ctx, trace_id, llm_config=resume_llm
            )
            await self._emit_turn_completed(trace_id, thread, persona, index, result)
            loop = await self._drive_loop(
                persona,
                task,
                thread,
                ctx,
                trace_id,
                turns=[result],
                max_turns=checkpoint.max_turns,
                cost_budget=checkpoint.cost_budget,
                llm_config=resume_llm,
                hitl=hitl,
                stop_on_no_progress=False,
                turns_offset=checkpoint.turns_completed,
                cost_offset=checkpoint.cost_completed,
            )
            self._record_resume_final_output(checkpoint_id, loop)
            return loop

    def _record_resume_final_output(
        self, checkpoint_id: str, loop: AgentLoopResult
    ) -> None:
        """Persist the resolved member's FINAL output onto its terminal checkpoint (#2).

        The crash-recovery anchor for orchestration HITL. By the time this runs the
        gated tool has fired exactly once and the checkpoint is already resolved
        (``approved``/``rejected``); the member's final answer is only produced by the
        drive loop AFTER that terminal save, so it is written back here — DURABLY, onto
        the SAME store row that holds the claim + idempotency ledger, BEFORE the caller
        returns (and therefore before the orchestration graph advance persists). If a
        crash then strikes between the member resolving and the graph persisting its
        advance, the graph recovery reads this text back and threads the REAL member
        output downstream instead of an empty string.

        A member that paused AGAIN on a second gated tool has NOT produced a final
        answer (the original checkpoint stays terminal but re-pauses into a fresh
        checkpoint), so nothing is recorded — ``final_output`` stays ``None``. The
        write merges onto a freshly LOADED row so the ledger written during execution
        is preserved, and never resurrects a since-pruned checkpoint.
        """
        assert self._checkpoint_store is not None  # guaranteed by the caller
        if loop.stopped_reason == "awaiting_approval":
            return
        latest = self._checkpoint_store.load(checkpoint_id)
        if latest is None:  # pragma: no cover - pruned mid-resume; nothing to anchor
            return
        final_text = (loop.final.output_text or "") if loop.turns else ""
        self._checkpoint_store.save(
            latest.model_copy(update={"final_output": final_text})
        )

    async def _drive_loop(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        turns: list[RunResult],
        max_turns: int,
        cost_budget: float | None,
        llm_config: LLMConfig | None,
        hitl: bool,
        stop_on_no_progress: bool,
        turns_offset: int,
        cost_offset: float,
        steer_queue: queue.Queue[str] | None = None,
    ) -> AgentLoopResult:
        """Drive continuation turns until a stop condition (shared by run/resume)."""
        while True:
            last = turns[-1]
            if not last.succeeded:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="error"
                )
            if hitl:
                pending = self._pending_approvals(last)
                if pending:
                    checkpoint = self._save_checkpoint(
                        persona,
                        task,
                        thread,
                        ctx,
                        llm_config,
                        max_turns,
                        cost_budget,
                        turns_offset + len(turns),
                        cost_offset + sum(t.cost for t in turns),
                        pending,
                    )
                    await self._emit(
                        RunEvent(
                            event_type=EventType.APPROVAL_REQUIRED,
                            trace_id=trace_id,
                            thread_id=thread.thread_id,
                            agent_id=persona.agent_id,
                            payload={
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "tools": [p.tool_name for p in pending],
                            },
                        )
                    )
                    return AgentLoopResult(
                        thread=thread,
                        turns=turns,
                        stopped_reason="awaiting_approval",
                        checkpoint_id=checkpoint.checkpoint_id,
                    )
            if not last.tool_calls:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
            if last.round_trip_complete:
                # The provider already ran the tool round-trip and produced the final
                # answer this turn (e.g. pydantic-ai/OpenAI); continuing would re-send a
                # history the strict API rejects. Stop here.
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final"
                )
            if final_answer_text(last) is not None:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="final_answer"
                )
            if stop_on_no_progress and is_no_progress(turns):
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="no_progress"
                )
            if turns_offset + len(turns) >= max_turns:
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="max_turns"
                )
            if (
                cost_budget is not None
                and cost_offset + sum(t.cost for t in turns) >= cost_budget
            ):
                return AgentLoopResult(
                    thread=thread, turns=turns, stopped_reason="budget"
                )
            index = turns_offset + len(turns) + 1
            # Between-turns steering (opt-in): drain queued user guidance at the
            # top of this continuation turn so the next request includes it.
            if steer_queue is not None:
                self._drain_steer_queue(steer_queue, thread)
            await self._emit(
                RunEvent(
                    event_type=EventType.AGENT_TURN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={"turn": index},
                )
            )
            result = await self._continue_turn(
                persona, thread, ctx, trace_id, llm_config=llm_config
            )
            turns.append(result)
            await self._emit_turn_completed(trace_id, thread, persona, index, result)

    def _drain_steer_queue(
        self, steer_queue: queue.Queue[str], thread: ChatThread
    ) -> None:
        """Append every queued steering text as a USER message on the thread.

        Each non-empty text becomes one USER message (``metadata={'steer': True}``)
        in arrival order, so the very next ``_continue_turn`` request — built from
        the thread as-is — carries the guidance. Thread-safe by construction:
        ``queue.Queue`` may be fed from any thread (an HTTP handler steering a
        background mission) while the loop drains it here on the event loop.
        """
        from himmy.agents.base_agent.thread import Message, MessageRole

        injected = False
        while True:
            try:
                text = steer_queue.get_nowait()
            except queue.Empty:
                break
            content = str(text).strip()
            if not content:
                continue
            message = Message(
                role=MessageRole.USER, content=content, metadata={"steer": True}
            )
            thread.append_message(message)
            self._register_message(message)
            injected = True
        if injected:
            thread.version += 1
            self._register_thread_version(thread)

    @staticmethod
    def _pending_approvals(result: RunResult) -> list[PendingToolCall]:
        """The tool calls in a turn that were denied for lack of approval."""
        denied = {
            r.tool_call_id
            for r in result.tool_returns
            if r.outcome == "denied"
            and (r.metadata or {}).get("error_code") == "POLICY_BLOCKED"
        }
        return [
            PendingToolCall(
                tool_call_id=c.tool_call_id, tool_name=c.tool_name, args=dict(c.args)
            )
            for c in result.tool_calls
            if c.tool_call_id in denied
        ]

    def _save_checkpoint(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        llm_config: LLMConfig | None,
        max_turns: int,
        cost_budget: float | None,
        turns_completed: int,
        cost_completed: float,
        pending: list[PendingToolCall],
    ) -> AgentCheckpoint:
        """Persist a paused run as a durable checkpoint and return it."""
        assert self._checkpoint_store is not None
        checkpoint = AgentCheckpoint(
            persona=persona.model_dump(mode="json"),
            task=task.model_dump(mode="json"),
            thread=thread.model_dump(mode="json"),
            ctx=ctx,
            llm_config=llm_config.model_dump(mode="json")
            if llm_config is not None
            else None,
            max_turns=max_turns,
            cost_budget=cost_budget,
            turns_completed=turns_completed,
            cost_completed=cost_completed,
            pending_tool_calls=pending,
        )
        self._checkpoint_store.save(checkpoint)
        return checkpoint

    async def _emit_turn_completed(
        self,
        trace_id: str,
        thread: ChatThread,
        persona: Persona,
        index: int,
        result: RunResult,
    ) -> None:
        await self._emit(
            RunEvent(
                event_type=EventType.AGENT_TURN_COMPLETED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                cost=result.cost,
                payload={
                    "turn": index,
                    "tool_calls": len(result.tool_calls),
                    "status": result.status,
                },
            )
        )

    async def _maybe_compact(
        self,
        persona: Persona,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        llm_config: LLMConfig | None,
    ) -> bool:
        """Summarize old turns in-place when the thread outgrows its token budget.

        Opt-in via ``ctx['compaction_spec']``. Keeps the system head + recent tail,
        replaces the middle with one model-written summary message, and emits a
        ``CONTEXT_COMPACTED`` event (the audit trail of what was condensed). A no-op
        when not configured or under budget.

        Returns ``True`` iff compaction actually rewrote the thread this turn. The
        caller (C5) uses that to BUST the prompt cache for the very next request:
        compaction inserts a new ``[Summary …]`` SYSTEM message, which changes the
        joined system prefix and would otherwise pay a write premium on a stale-cache
        miss. Skipping the breakpoint that one turn lets the prefix re-stabilize.
        """
        spec = ctx.get("compaction_spec")
        if not spec:
            return False
        from himmy.agents.base_agent.thread import Message, MessageRole
        from himmy.runtime.compaction import (
            SUMMARY_INSTRUCTION,
            ContextCompactor,
            estimate_tokens,
        )

        compactor = ContextCompactor(
            max_tokens=int(spec.get("max_tokens", 3000)),
            keep_recent=int(spec.get("keep_recent", 6)),
        )
        plan = compactor.plan(thread.messages)
        if not plan.should_compact:
            return False

        span_text = compactor.render_span(plan.summarize)
        model_key = str(ctx.get("model_key") or self.default_model_key)
        summary_req = InferenceRequest(
            model_key=model_key,
            response_format=ResponseFormat.TEXT,
            messages=[
                InferenceMessage(role="system", content=SUMMARY_INSTRUCTION),
                InferenceMessage(role="user", content=span_text),
            ],
            metadata=_cache_scope_metadata(ctx),
        )
        summary_resp = await self.inference_service.run(summary_req)
        summary_text = (summary_resp.output_text or "").strip()
        if not summary_text:
            return False  # summarization failed/empty — leave history intact (safe)

        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[Summary of earlier conversation]\n{summary_text}",
            metadata={"compacted": True},
        )
        # Only apply if the summary is actually smaller than what it replaces — a verbose
        # summary of a tiny span would otherwise grow the context, not shrink it.
        if estimate_tokens(summary_msg.content) >= compactor.estimate(plan.summarize):
            return False
        head = list(thread.messages[: plan.head_count])
        tail = list(thread.messages[plan.tail_start :])
        compacted_count = len(plan.summarize)
        thread.messages[:] = [*head, summary_msg, *tail]
        thread.version += 1

        await self._emit(
            RunEvent(
                event_type=EventType.CONTEXT_COMPACTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload={
                    "summarized_messages": compacted_count,
                    "before_tokens": plan.before_tokens,
                    "after_tokens": compactor.estimate(thread.messages),
                    "kept_recent": len(tail),
                },
            )
        )

        # P0-C: the compaction summary is the run's distilled experience ("every
        # fact, decision, tool result"). Instead of discarding it when the thread
        # middle is replaced, persist it as an episodic trace so learning loops have
        # a corpus of "what happened" from day one. Best-effort: a persistence
        # failure must never break the run (the in-context summary already applied).
        save_episodic = getattr(self.memory_store, "save_episodic_memory", None)
        if save_episodic is not None:
            from himmy.services.storage.models import EpisodicMemoryObject

            subject_raw = ctx.get("subject_id")
            with contextlib.suppress(Exception):
                await save_episodic(
                    EpisodicMemoryObject(
                        subject_id=str(subject_raw) if subject_raw else None,
                        agent_id=persona.agent_id,
                        payload={
                            "summary": summary_text,
                            "source": "compaction",
                            "tier": "archival",
                        },
                        metadata={
                            "trace_id": trace_id,
                            "thread_id": thread.thread_id,
                            "summarized_messages": compacted_count,
                        },
                    )
                )
        return True

    async def _continue_turn(
        self,
        persona: Persona,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        llm_config: LLMConfig | None,
    ) -> RunResult:
        """One more inference turn on an existing thread (no new user prompt).

        Builds the request from the thread as-is (so the model sees prior tool
        results), runs inference, replays tool exchanges, and appends the assistant
        turn. Persona/prompt lineage was linked by the first turn, so this only
        registers the new message + the bumped thread version.
        """
        from himmy.agents.base_agent.thread import Message, MessageRole

        # WS4.6: a TRAIN-denied subject forces raw-I/O capture OFF for this turn.
        capture_io = self._capture_io and not self._train_suppressed(ctx)
        # C5: compaction rewrites the system prefix; bust the prompt cache for THIS turn
        # so the adapter doesn't mark a now-stale prefix and pay a write-premium miss.
        compacted = await self._maybe_compact(persona, thread, ctx, trace_id, llm_config)
        request, tool_names = self._build_request(
            thread, ctx, llm_config, trace_id=trace_id, cache_busted=compacted
        )
        await self._emit(
            RunEvent(
                event_type=EventType.INFERENCE_REQUESTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                request_id=request.request_id,
                payload={"model_key": request.model_key, "tool_names": tool_names},
            )
        )
        response = await self.inference_service.run(request)
        await self._emit(
            RunEvent(
                event_type=(
                    EventType.INFERENCE_SUCCEEDED
                    if response.status == InferenceStatus.SUCCESS
                    else EventType.INFERENCE_FAILED
                ),
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                request_id=request.request_id,
                latency_ms=response.latency_ms,
                cost=response.cost,
                error=(response.error.message if response.error else None),
                payload={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    **(
                        cache_metrics_payload(request, response)
                        if response.status == InferenceStatus.SUCCESS
                        else {}
                    ),
                    **(
                        {"io": build_io_capture(request, response)}
                        if capture_io
                        else {}
                    ),
                },
            )
        )
        await self._append_tool_messages(
            thread,
            response,
            request_id=request.request_id,
            trace_id=trace_id,
            agent_id=persona.agent_id,
        )

        assistant_text = response.output_text
        if assistant_text is None and response.output_structured is not None:
            assistant_text = json.dumps(response.output_structured, default=str)
        assistant_text = await self._guard_output(
            assistant_text,
            agent_id=persona.agent_id,
            trace_id=trace_id,
            thread_id=thread.thread_id,
        )
        error_message = response.error.message if response.error else None
        error_code = response.error.code.value if response.error else None
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_text or "",
            metadata={
                "request_id": request.request_id,
                "trace_id": trace_id,
                "status": response.status.value,
                "error": error_message,
                "error_code": error_code,
                "cost": response.cost,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "output_structured": response.output_structured,
            },
        )
        thread.append_message(assistant_message)
        thread.version += 1
        self._register_message(assistant_message)
        self._register_thread_version(thread)

        return RunResult(
            thread=thread,
            status=response.status.value,
            output_text=assistant_text,
            output_structured=response.output_structured,
            tool_calls=list(response.tool_calls),
            tool_returns=list(response.tool_returns),
            error=error_message,
            error_code=error_code,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            latency_ms=response.latency_ms,
            model_path=response.model_path,
            provider_name=response.provider_name,
            request_id=request.request_id,
            trace_id=trace_id,
            round_trip_complete=bool(response.metadata.get("round_trip_complete")),
        )

    async def _run_task_body(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        is_new_thread: bool,
        llm_config: LLMConfig | None,
        snapshot_id: str | None,
    ) -> RunResult:
        """Publish the run's subject (WS4.6) then run the pipeline; reset on exit.

        Splitting the subject-contextvar set/reset from the pipeline keeps the spine-record
        stampers (``_emit`` / ``_register_*``) correct under concurrent runs while leaving
        the pipeline body unchanged. The subject is ``None`` (no stamping) unless a
        governed ``consent_decider`` is wired and the task names a ``context_subject_id``.
        """
        with self._subject_scope(ctx):
            return await self._run_task_pipeline(
                persona,
                task,
                thread,
                ctx,
                trace_id,
                is_new_thread=is_new_thread,
                llm_config=llm_config,
                snapshot_id=snapshot_id,
            )

    def _run_subject(self, ctx: dict[str, Any]) -> str | None:
        """The governed run's human data subject (``None`` unless a decider is wired)."""
        if self._consent_decider is None:
            return None
        subject = ctx.get("context_subject_id")
        return str(subject) if subject else None

    @contextlib.contextmanager
    def _subject_scope(self, ctx: dict[str, Any]) -> Iterator[None]:
        """Publish the run's human subject on ``_CURRENT_SUBJECT`` for the scope's life.

        Every public runtime entry point that emits subject-bearing spine records
        (``run_task``/``run_agent_loop``/``continue_turn``/``stream_task``/
        ``resume_agent_loop``) wraps its body in this so a governed
        :class:`ConsentAwareRegistry` can resolve (and gate / crypto-shred) the subject
        behind the otherwise subject-less ``run_event`` / ``message`` / ``chat_thread``
        records — preventing silent fail-closed data loss for a fully-consented subject
        on the multi-turn / streaming / HITL-resume paths.

        The published subject is ``None`` (no stamping) unless a governed
        ``consent_decider`` is wired AND the task names a ``context_subject_id``, so the
        ungoverned path is byte-identical (``_subject_metadata`` returns ``None`` and
        ``to_record(metadata=None)`` is unchanged). The :class:`contextvars.ContextVar`
        keeps this correct under concurrent runs.
        """
        token = _CURRENT_SUBJECT.set(self._run_subject(ctx))
        try:
            yield
        finally:
            _CURRENT_SUBJECT.reset(token)

    async def _run_task_pipeline(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        trace_id: str,
        *,
        is_new_thread: bool,
        llm_config: LLMConfig | None,
        snapshot_id: str | None,
    ) -> RunResult:
        """The run pipeline proper (subject contextvar set by ``_run_task_body``)."""
        from himmy.agents.base_agent.thread import Message, MessageRole

        # WS4.6 TRAIN gate: a participating human subject lacking TRAIN consent forces
        # raw-I/O capture OFF and strips the persisted ``rendered_prompt`` for this run.
        # Inert (False) whenever no ``consent_decider`` is wired (offline path unchanged).
        train_suppressed = self._train_suppressed(ctx)
        capture_io = self._capture_io and not train_suppressed

        # --- 1. snapshot resolve/build -------------------------------------
        # Thread the run reference (trace/thread id) into the context-build metadata so a
        # context adapter can scope to THIS run's own event sequence (e.g. the P2
        # trajectory-aware learned-hints advisor mining the run's own tool-call order).
        # Additive + idempotent: only fills the keys when absent, so a caller that already
        # set them — and every adapter that ignores them — is unaffected.
        cb_meta = ctx.get("context_metadata")
        if isinstance(cb_meta, dict):
            cb_meta.setdefault("run_trace_id", trace_id)
            cb_meta.setdefault("run_thread_id", thread.thread_id)
        elif cb_meta is None and ctx.get("context_build_spec") is not None:
            ctx["context_metadata"] = {
                "run_trace_id": trace_id,
                "run_thread_id": thread.thread_id,
            }
        snapshot, snapshot_id, snapshot_error = await self._resolve_snapshot(
            persona, task, ctx, snapshot_id
        )

        # --- 2. render prompts (+ project snapshot keys) -------------------
        # Injected context (recalled memory / retrieved KB docs) is routed through
        # the guardrail here so a poisoned memory or KB chunk is redacted/blocked
        # before it reaches the model (indirect prompt-injection seam).
        system_prompt, task_prompt, _missing = await self._render_guarded_prompts(
            persona, task, ctx, snapshot, thread_id=thread.thread_id
        )

        # --- 3. append SYSTEM (first turn) + USER --------------------------
        first_turn = not any(m.role == MessageRole.SYSTEM for m in thread.messages)
        if first_turn and system_prompt:
            system_message = Message(role=MessageRole.SYSTEM, content=system_prompt)
            thread.append_message(system_message)
            self._register_message(system_message)
        user_message = Message(
            role=MessageRole.USER,
            content=await self._guard_input(
                task_prompt,
                agent_id=persona.agent_id,
                thread_id=thread.thread_id,
            ),
        )
        thread.append_message(user_message)
        self._register_message(user_message)

        # --- 4. register persona/agent/prompt entities --------------------
        persona_record = self._register_entity(persona)
        prompt_record = self._register_entity(task)

        # --- 5. AGENT_RUN_STARTED -----------------------------------------
        started_payload: dict[str, Any] = {
            "model_key": self._effective_model_key(ctx, llm_config),
            "snapshot_id": snapshot_id,
            "persona_name": persona.name,
        }
        if snapshot_error is not None:
            started_payload["snapshot_error"] = snapshot_error
        await self._emit(
            RunEvent(
                event_type=EventType.AGENT_RUN_STARTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload=started_payload,
            )
        )

        # --- 6. build the inference request -------------------------------
        request, tool_names = self._build_request(
            thread, ctx, llm_config, trace_id=trace_id
        )

        requested_payload: dict[str, Any] = {
            "model_key": request.model_key,
            "route_override": request.route_override,
            "response_format": request.response_format.value
            if request.response_format
            else None,
            "retrieval_ctx": list((snapshot.fields or {}).keys())
            if snapshot is not None
            else [],
            "tool_names": tool_names,
        }
        # WS4.6: omit the full user prompt for a TRAIN-denied subject so no cleartext
        # prompt lands on the event log / spine. Present verbatim otherwise.
        if not train_suppressed:
            requested_payload["rendered_prompt"] = task_prompt
        await self._emit(
            RunEvent(
                event_type=EventType.INFERENCE_REQUESTED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                request_id=request.request_id,
                payload=requested_payload,
            )
        )

        # --- 7. call inference --------------------------------------------
        # InferenceService.run never raises for provider/manager errors
        # (invariant #3); CancelledError still propagates and is handled by the
        # deadline wrapper above.
        response = await self.inference_service.run(request)

        if response.status == InferenceStatus.SUCCESS:
            await self._emit(
                RunEvent(
                    event_type=EventType.INFERENCE_SUCCEEDED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    request_id=request.request_id,
                    latency_ms=response.latency_ms,
                    cost=response.cost,
                    payload={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "model_path": response.model_path,
                        "provider_name": response.provider_name,
                        **cache_metrics_payload(request, response),
                        **(
                            {"io": build_io_capture(request, response)}
                            if capture_io
                            else {}
                        ),
                    },
                )
            )
        else:
            await self._emit(
                RunEvent(
                    event_type=EventType.INFERENCE_FAILED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    request_id=request.request_id,
                    latency_ms=response.latency_ms,
                    error=response.error.message
                    if response.error
                    else "inference failed",
                    payload={
                        "error_code": response.error.code.value
                        if response.error
                        else None
                    },
                )
            )

        # --- 8. replay TOOL exchanges onto the thread + emit tool events ---
        await self._append_tool_messages(
            thread,
            response,
            request_id=request.request_id,
            trace_id=trace_id,
            agent_id=persona.agent_id,
        )

        # --- 9. append ASSISTANT message ----------------------------------
        assistant_text = response.output_text
        if assistant_text is None and response.output_structured is not None:
            assistant_text = json.dumps(response.output_structured, default=str)
        assistant_text = await self._guard_output(
            assistant_text,
            agent_id=persona.agent_id,
            trace_id=trace_id,
            thread_id=thread.thread_id,
        )
        error_message = response.error.message if response.error else None
        error_code = response.error.code.value if response.error else None
        assistant_metadata: AssistantMessageMetadata = {
            "request_id": request.request_id,
            "trace_id": trace_id,
            "latency_ms": response.latency_ms,
            "model_path": response.model_path,
            "provider_name": response.provider_name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": response.cost,
            "status": response.status.value,
            # Invariant #4: stamp error + structured output so the application
            # layer can detect FAILED runs without exceptions.
            "error": error_message,
            "error_code": error_code,
            "output_structured": response.output_structured,
        }
        if response.workflow is not None:
            assistant_metadata["workflow_complete"] = response.workflow.is_complete
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_text or "",
            # dict() widens the TypedDict to the open dict[str, Any] the model field
            # expects (metadata stays extensible; the TypedDict only types the write).
            metadata=dict(assistant_metadata),
        )
        thread.append_message(assistant_message)

        # --- 10. register message + chat_thread version + links -----------
        # RO-8: bump the thread version on the 2nd+ turn regardless of whether a
        # registry is wired, so persisted thread versions are correct even
        # without lineage.
        if not is_new_thread:
            thread.version += 1
        self._register_message(assistant_message)
        thread_record = self._register_thread_version(thread)
        self._link_lineage(
            persona_record=persona_record,
            prompt_record=prompt_record,
            thread_record=thread_record,
            snapshot=snapshot,
            persona=persona,
            thread=thread,
        )

        # --- 11. AGENT_RUN_FINISHED + save thread -------------------------
        await self._emit(
            RunEvent(
                event_type=EventType.AGENT_RUN_FINISHED,
                trace_id=trace_id,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                latency_ms=response.latency_ms,
                cost=response.cost,
                error=error_message
                if response.status != InferenceStatus.SUCCESS
                else None,
                payload={"status": response.status.value},
            )
        )
        await self._maybe_save_thread(thread)

        return RunResult(
            thread=thread,
            status=response.status.value,
            output_text=assistant_text,
            output_structured=response.output_structured,
            tool_calls=list(response.tool_calls),
            tool_returns=list(response.tool_returns),
            error=error_message,
            error_code=error_code,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            latency_ms=response.latency_ms,
            model_path=response.model_path,
            provider_name=response.provider_name,
            request_id=request.request_id,
            trace_id=trace_id,
            workflow=response.workflow,
            workflow_complete=(
                response.workflow.is_complete if response.workflow is not None else None
            ),
            round_trip_complete=bool(response.metadata.get("round_trip_complete")),
        )

    # ------------------------------------------------------------- snapshot
    async def _guard_input(
        self,
        text: str,
        *,
        agent_id: str | None = None,
        trace_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """Apply the input guardrail to a user prompt (redact); ``None`` → passthrough."""
        return await self._apply_guardrail(
            self._input_guardrail,
            text,
            stage="input",
            agent_id=agent_id,
            trace_id=trace_id,
            thread_id=thread_id,
        )

    async def _guard_output(
        self,
        text: str | None,
        *,
        agent_id: str | None = None,
        trace_id: str | None = None,
        thread_id: str | None = None,
    ) -> str | None:
        """Apply the output guardrail to an assistant reply (redact); passthrough None."""
        if text is None:
            return None
        return await self._apply_guardrail(
            self._output_guardrail,
            text,
            stage="output",
            agent_id=agent_id,
            trace_id=trace_id,
            thread_id=thread_id,
        )

    async def _apply_guardrail(
        self,
        guardrail: Any,
        text: str,
        *,
        stage: str,
        agent_id: str | None,
        trace_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """Run a guardrail and emit GUARDRAIL_APPLIED when it redacts or blocks.

        A clean pass (no flags, no block, text unchanged) emits nothing, so the
        event stream only carries the safety layer when it actually did something —
        which is exactly what the audit panel surfaces.

        Enforcement is real, not advisory: when a guardrail BLOCKS (``allowed`` is
        False) the offending text NEVER propagates. Blocking guardrails (DLP
        ``…:block``, blocklist, injection) return the original text on a block, so
        the runtime substitutes a safe placeholder here instead of letting it
        through — on the INPUT stage the model never sees the blocked prompt, and on
        the OUTPUT stage the user and the persisted thread never see the blocked
        answer. A guardrail that already supplied its own safe replacement (e.g.
        GroundingGuardrail's tailored refusal) is honored; otherwise the stage's
        generic placeholder is used. The GUARDRAIL_APPLIED event always reflects
        reality (``blocked``/``redacted`` describe the text actually returned).

        Whether a safe replacement was supplied is read from the verdict's
        ``block_replacement`` flag (set by :meth:`GuardrailPipeline.inspect` per
        blocking guardrail), NOT by comparing ``verdict.text`` to the original. In a
        MIXED pipeline an upstream redactor changes the text before a downstream
        blocker returns its (already-redacted) input unchanged, so ``verdict.text``
        no longer equals the original even though the blocker supplied nothing — the
        old equality check fell open and leaked the offending content through.
        """
        if guardrail is None:
            return text
        # Run inspection in a worker thread. NOTE: for a regex/CPU-bound guardrail
        # this does NOT move the work off the event loop in any real sense — Python's
        # ``re`` engine holds the GIL while matching, so the loop is still stalled for
        # the duration of the scan (a large sub-cap input pinned the loop ~900 ms even
        # through to_thread). The actual loop protection for regex guardrails is the
        # bounded (linear, non-backtracking) pattern + the small per-scan input cap in
        # builtins (``_MAX_PII_SCAN_LEN``), which bound worst-case CPU to ~tens of ms.
        # to_thread is kept because it DOES help I/O-bound guardrails (e.g. ones that
        # call out to a service) which release the GIL; it is harmless for regex ones.
        # Semantics are unchanged; only the thread the work runs on differs.
        verdict = await asyncio.to_thread(
            guardrail.inspect, text, context={"stage": stage}
        )
        result = cast(str, verdict.text)
        if not verdict.allowed:
            # The block is enforced: never return the offending text. Honor a safe
            # replacement the BLOCKING guardrail itself produced; otherwise fall back
            # to the stage's generic placeholder. ``block_replacement`` is authoritative
            # — a bare ``result != text`` would be fooled by an upstream redaction in a
            # mixed redact+block pipeline (finding #3's fail-open).
            if not getattr(verdict, "block_replacement", False):
                result = (
                    _INPUT_BLOCK_PLACEHOLDER
                    if stage == "input"
                    else _OUTPUT_BLOCK_PLACEHOLDER
                )
        redacted = result != text
        if verdict.flags or verdict.reasons or not verdict.allowed or redacted:
            await self._emit(
                RunEvent(
                    event_type=EventType.GUARDRAIL_APPLIED,
                    trace_id=trace_id,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    payload={
                        "stage": stage,
                        "blocked": not verdict.allowed,
                        "redacted": redacted,
                        "flags": list(verdict.flags),
                        "reasons": list(verdict.reasons),
                    },
                )
            )
        return result

    async def _resolve_snapshot(
        self,
        persona: Persona,
        task: Task,
        ctx: dict[str, Any],
        snapshot_id: str | None,
    ) -> tuple[Any, str | None, str | None]:
        """Resolve a snapshot from arg/context, or build one.

        Returns ``(snapshot, resolved_id, snapshot_error)``. A snapshot was
        explicitly *requested* when a ``snapshot_id`` was supplied or a
        ``context_build_spec`` is present; in that case a load/build failure is
        diagnosed (RO-11): the error is captured for the AGENT_RUN_STARTED /
        CONTEXT_SNAPSHOT_BUILT payload, and — when ``strict_snapshot`` is on —
        re-raised as an :class:`HimmyError` so the caller knows the requested
        evidence was unavailable instead of silently running without it.
        """
        snapshot: Any = None
        snapshot_error: str | None = None
        resolved_id = snapshot_id or ctx.get("snapshot_id")
        requested = bool(
            snapshot_id
            or ctx.get("snapshot_id")
            or ctx.get("context_build_spec") is not None
        )

        if self.context_service is None:
            if requested and self.strict_snapshot:
                raise HimmyError("snapshot requested but no context_service is wired")
            return (
                None,
                resolved_id,
                ("no context_service wired" if requested else None),
            )

        # Load an existing snapshot when an id was supplied and storage is present.
        if resolved_id and self.memory_store is not None:
            loader = getattr(self.memory_store, "load_snapshot", None)
            if loader is not None:
                try:
                    snapshot = await loader(resolved_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - diagnose, don't crash
                    snapshot_error = f"snapshot load failed: {exc}"
                if snapshot is None and snapshot_error is None:
                    snapshot_error = f"snapshot {resolved_id!r} not found"

        # Otherwise build one from a declared build spec.
        if snapshot is None and ctx.get("context_build_spec") is not None:
            subject_id = ctx.get("context_subject_id") or persona.agent_id
            try:
                snapshot = await self.context_service.build_snapshot(
                    subject_id=subject_id,
                    task_id=task.task_id,
                    build_spec=ctx["context_build_spec"],
                    metadata=ctx.get("context_metadata"),
                )
                resolved_id = snapshot.snapshot_id
                snapshot_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - diagnose, don't crash
                snapshot = None
                snapshot_error = f"snapshot build failed: {exc}"

        if snapshot is not None:
            await self._emit(
                RunEvent(
                    event_type=EventType.CONTEXT_SNAPSHOT_BUILT,
                    thread_id=None,
                    agent_id=persona.agent_id,
                    payload={
                        "snapshot_id": getattr(snapshot, "snapshot_id", None),
                        "subject_id": getattr(snapshot, "subject_id", None),
                        "missing_required_keys": list(
                            getattr(snapshot, "missing_required_keys", []) or []
                        ),
                        # Knowledge/RAG grounding — which chunks were retrieved into
                        # the prompt, with citations (so the GUI can show "why it
                        # said that"). Empty when no KB-sourced field was resolved.
                        "grounding": _snapshot_grounding(snapshot),
                    },
                )
            )
            resolved_id = getattr(snapshot, "snapshot_id", resolved_id)
        elif snapshot_error is not None:
            # RO-11: surface the failure on the audit trail so 'requested but
            # unavailable' is distinguishable from 'no snapshot requested'.
            await self._emit(
                RunEvent(
                    event_type=EventType.CONTEXT_SNAPSHOT_BUILT,
                    thread_id=None,
                    agent_id=persona.agent_id,
                    error=snapshot_error,
                    payload={
                        "snapshot_id": resolved_id,
                        "snapshot_error": snapshot_error,
                    },
                )
            )
            if requested and self.strict_snapshot:
                raise HimmyError(snapshot_error)
        return snapshot, resolved_id, snapshot_error

    # --------------------------------------------------------------- prompts
    async def _render_guarded_prompts(
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
        system_prompt, task_prompt, missing, sys_block, task_block = (
            self._render_prompt_parts(persona, task, ctx, snapshot)
        )
        if sys_block:
            guarded_sys = await self._guard_input(
                sys_block,
                agent_id=persona.agent_id,
                trace_id=trace_id,
                thread_id=thread_id,
            )
            system_prompt = f"{system_prompt}\n\n{guarded_sys}".strip()
        if task_block:
            guarded_task = await self._guard_input(
                task_block,
                agent_id=persona.agent_id,
                trace_id=trace_id,
                thread_id=thread_id,
            )
            task_prompt = f"{task_prompt}\n\n{guarded_task}".strip()
        return system_prompt, task_prompt, missing

    def _render_prompt_parts(
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
        system_prompt = ""
        task_prompt = task.prompt
        missing: list[str] = []

        if self.prompt_manager is not None:
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
            system_prompt = self.prompt_manager.get_system_prompt(system_vars)

            task_vars = TaskPromptVariables(
                task=task.prompt,
                output_format=ctx.get("output_format", ""),
                output_schema=ctx.get("output_schema"),
            )
            task_prompt = self.prompt_manager.get_task_prompt(task_vars) or task.prompt

        # Prepend any system_prefix.
        prefix = ctx.get("system_prefix")
        if prefix:
            system_prompt = f"{prefix}\n\n{system_prompt}".strip()

        # Project snapshot keys into system/task blocks (kept SEPARATE from the base
        # prompts so the injected context can be guarded independently — see
        # _render_guarded_prompts).
        sys_block = ""
        task_block = ""
        map_spec = ctx.get("context_prompt_map_spec")
        if (
            self.context_prompt_mapper is not None
            and map_spec is not None
            and snapshot is not None
        ):
            try:
                sys_block, task_block, missing = self.context_prompt_mapper.project(
                    snapshot, map_spec
                )
            except Exception:  # pragma: no cover - defensive
                missing = []
                sys_block = ""
                task_block = ""
        return system_prompt, task_prompt, missing, sys_block, task_block

    # ----------------------------------------------------------- inference
    def _effective_model_key(
        self, ctx: dict[str, Any], llm_config: LLMConfig | None
    ) -> str:
        """Resolve the effective model key (llm_config > task.context > default)."""
        if llm_config is not None and llm_config.model_key:
            return llm_config.model_key
        return ctx.get("model_key") or self.default_model_key

    def _prompt_cache_policy(
        self, model_key: str, *, cache_busted: bool
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
        """
        if cache_busted or not self._enable_prompt_cache:
            return None
        capability = self.inference_service.cache_capability_for(model_key)
        if capability is CacheCapability.NONE:
            return None
        return CachePolicy()

    def _build_request(
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

        model_key = self._effective_model_key(ctx, llm_config)
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

        if self.tool_service is not None:
            if is_forced_workflow:
                bound_tools = self.tool_service.bound_tools(tool_names_override)
                bound_names = {bt.name for bt in bound_tools}
                step_tool = tool_names_override[0]  # type: ignore[index]
                if step_tool not in bound_names:
                    raise HimmyError(
                        f"WORKFLOW response_format requires the step tool "
                        f"{step_tool!r} to be bound, but it is not registered "
                        f"with the tool_service"
                    )
            else:
                bound_tools = self.tool_service.bound_tools(tool_names)
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
            metadata=_cache_scope_metadata(ctx),
            cache_policy=self._prompt_cache_policy(model_key, cache_busted=cache_busted),
            bound_tools=bound_tools,
            # The single execution seam for the bound tools (see ToolExecutor),
            # wrapped with bounded turn-level retry for transient failures.
            tool_executor=(
                self._wrap_executor_with_retry(
                    self.tool_service.tool_executor(),
                    ctx,
                    thread_id=getattr(thread, "thread_id", None),
                    agent_id=getattr(thread, "agent_id", None),
                    trace_id=trace_id,
                )
                if self.tool_service is not None
                else None
            ),
            tool_names_override=tool_names_override,
        )
        if timeout_seconds is not None:
            request.timeout_seconds = timeout_seconds
        return request, tool_names_override or tool_names

    def _tool_is_read_only(self, tool_name: str) -> bool:
        """True only when ``tool_name`` is provably read-only (no side effect).

        Used to decide whether a TIMEOUT may be retried: a read-only tool has no
        side effect, so re-running it after a timeout is safe; anything else (a write
        tool, or an ambiguously-named one whose intent can't be inferred) is treated
        as side-effecting and NOT retried on timeout. Resolution: the definition's
        explicit ``read_only`` flag wins; otherwise the name is classified
        (``classify_read_only`` — ``None``/ambiguous ⇒ not read-only); an unknown
        tool (no definition) is conservatively treated as side-effecting.
        """
        if self.tool_service is None:
            return False
        registry = getattr(self.tool_service, "registry", None)
        if registry is None:
            return False
        definition = registry.get(tool_name)
        if definition is None:
            return False
        if definition.read_only is not None:
            return bool(definition.read_only)
        from himmy.services.tools.access import classify_read_only

        return classify_read_only(tool_name) is True

    def _wrap_executor_with_retry(
        self,
        executor: ToolExecutor,
        ctx: dict[str, Any],
        *,
        thread_id: str | None,
        agent_id: str | None,
        trace_id: str | None,
    ) -> ToolExecutor:
        """Bound turn-level retry for TRANSIENT tool failures (timeout / rate-limit).

        Provider-level errors are retried inside the inference service, but a
        transiently-failed TOOL call used to land in the turn as a failed
        ``ToolReturnRecord`` the loop just carried on with. This wraps the tool
        executor so a failure whose ``error_code`` is in the tools kernel's own
        retryable taxonomy (:func:`_transient_tool_codes`) re-executes the
        affected tool — never the whole inference — up to
        ``ctx['tool_retry_attempts']`` extra times (default
        :data:`DEFAULT_TOOL_RETRY_ATTEMPTS`; ``0`` disables and returns the
        executor unwrapped) with a short exponential backoff
        (``ctx['tool_retry_backoff_seconds']`` base, default
        :data:`DEFAULT_TOOL_RETRY_BACKOFF_SECONDS`). Each retry emits a
        ``TOOL_CALLED`` event tagged ``transient_retry`` so the trace shows it.
        Non-transient failures (and exhausted retries) keep the current
        behavior exactly: the last failed record flows into the turn.

        SIDE-EFFECT SAFETY: a ``TIMEOUT`` does NOT mean the tool didn't run — an
        HTTP POST / write may have committed server-side after the client gave up,
        so re-firing it would duplicate the write/send/charge. The retry therefore
        skips ``TIMEOUT`` for any tool that is not provably read-only (a write or an
        ambiguously-named tool); read-only tools (no side effect) still retry on
        timeout. The other transient codes (``RATE_LIMITED`` /
        ``PROVIDER_UNAVAILABLE``) mean the call never reached the tool, so they stay
        retryable for every tool.
        """
        raw_attempts = ctx.get("tool_retry_attempts")
        retries = (
            DEFAULT_TOOL_RETRY_ATTEMPTS
            if raw_attempts is None
            else max(0, int(raw_attempts))
        )
        if retries == 0:
            return executor
        raw_backoff = ctx.get("tool_retry_backoff_seconds")
        backoff = (
            DEFAULT_TOOL_RETRY_BACKOFF_SECONDS
            if raw_backoff is None
            else max(0.0, float(raw_backoff))
        )
        transient = _transient_tool_codes()
        timeout_code = _tool_timeout_code()

        async def _execute(tool_name: str, args: dict[str, Any]) -> ToolReturnRecord:
            record = await executor(tool_name, args)
            for attempt in range(1, retries + 1):
                code = (record.metadata or {}).get("error_code")
                if record.outcome != "failed" or code not in transient:
                    return record
                # A timed-out non-read-only tool may already have committed its
                # side effect — do not re-fire it (idempotency would be violated).
                if code == timeout_code and not self._tool_is_read_only(tool_name):
                    return record
                await self._emit(
                    RunEvent(
                        event_type=EventType.TOOL_CALLED,
                        trace_id=trace_id,
                        thread_id=thread_id,
                        agent_id=agent_id,
                        tool_call_id=record.tool_call_id,
                        payload={
                            "tool_name": tool_name,
                            "transient_retry": attempt,
                            "max_retries": retries,
                            "error_code": code,
                        },
                    )
                )
                if backoff > 0.0:
                    await asyncio.sleep(backoff * (2 ** (attempt - 1)))
                record = await executor(tool_name, args)
            return record

        return _execute

    # ------------------------------------------------------- tool messages
    async def _append_tool_messages(
        self,
        thread: Any,
        response: InferenceResponse,
        *,
        request_id: str,
        trace_id: str,
        agent_id: str | None,
    ) -> None:
        """Replay each tool call/return pair as a TOOL Message (full metadata).

        RO-2: per tool exchange this also emits a ``TOOL_CALLED`` event for the
        call and a ``TOOL_COMPLETED`` / ``TOOL_FAILED`` event for the paired
        return (keyed on ``ret.outcome``), threading tool_call_id / tool_name /
        tool_args / request_id / trace_id so the events link to the run like the
        other emissions and power the ai_call_log / lineage view.
        """
        from himmy.agents.base_agent.thread import Message, MessageRole

        returns_by_id: dict[str, ToolReturnRecord] = {
            r.tool_call_id: r for r in response.tool_returns
        }
        for call in response.tool_calls:
            ret = returns_by_id.get(call.tool_call_id)

            # TOOL_CALLED: emitted before the return is recorded.
            await self._emit(
                RunEvent(
                    event_type=EventType.TOOL_CALLED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=agent_id,
                    request_id=request_id,
                    tool_call_id=call.tool_call_id,
                    payload={
                        "tool_name": call.tool_name,
                        "tool_args": dict(call.args),
                    },
                )
            )

            content = ret.content if ret is not None else None
            try:
                content_text = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, default=str)
                )
            except TypeError:  # pragma: no cover - defensive
                content_text = str(content)
            # Surface tool failures as a clear ERROR line so the model can adapt
            # instead of seeing a bare ``null`` content.
            if ret is not None and ret.outcome in ("failed", "denied"):
                meta = ret.metadata or {}
                code = meta.get("error_code", ret.outcome.upper())
                detail = meta.get("error_message") or content_text or ""
                content_text = f"ERROR: {code}: {detail}".strip().rstrip(":")
            message = Message(
                role=MessageRole.TOOL,
                content=content_text,
                metadata={
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "tool_outcome": ret.outcome if ret is not None else "unknown",
                    "tool_args": dict(call.args),
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "timestamp": message_timestamp(),
                    "tool_return_metadata": dict(ret.metadata)
                    if ret is not None
                    else {},
                },
            )
            thread.append_message(message)
            self._register_message(message)

            # TOOL_COMPLETED / TOOL_FAILED keyed on the return's outcome. The result
            # text + latency ride on the payload so an observer (e.g. Studio's live
            # cognition/ledger view) can show what each tool returned and how long it
            # took without re-reading the thread.
            outcome = ret.outcome if ret is not None else "unknown"
            completed = outcome == "success"
            ret_meta = (ret.metadata or {}) if ret is not None else {}
            result_text = (
                content_text
                if len(content_text) <= _TOOL_RESULT_EVENT_MAX
                else (content_text[: _TOOL_RESULT_EVENT_MAX - 1] + "…")
            )
            await self._emit(
                RunEvent(
                    event_type=(
                        EventType.TOOL_COMPLETED if completed else EventType.TOOL_FAILED
                    ),
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=agent_id,
                    request_id=request_id,
                    tool_call_id=call.tool_call_id,
                    error=None if completed else f"tool outcome: {outcome}",
                    payload={
                        "tool_name": call.tool_name,
                        "tool_args": dict(call.args),
                        "tool_outcome": outcome,
                        "result": result_text,
                        "latency_ms": ret_meta.get("latency_ms"),
                    },
                )
            )

    # --------------------------------------------------------------- entities
    @staticmethod
    def _subject_metadata() -> dict[str, Any] | None:
        """The ``{subject_id: ...}`` stamp for the current run, or ``None`` (ungoverned).

        Returns ``None`` whenever no governed run subject is published (the offline /
        ungoverned path), so ``to_record(metadata=None)`` keeps the projected record
        byte-identical to a pre-WS4.6 runtime. When a governed run names a subject, the
        stamp lets a :class:`ConsentAwareRegistry` resolve + gate the otherwise
        subject-less spine records.
        """
        subject = _CURRENT_SUBJECT.get()
        return {"subject_id": subject} if subject else None

    def _register_entity(self, obj: Any, *, stamp_subject: bool = False) -> Any:
        """Register a domain object's record when a registry is wired; else None.

        ``stamp_subject`` is only set for the genuinely subject-bearing run artefacts
        (messages, threads); infrastructure records (persona/prompt/agent) are never
        stamped so a governed ``registry.query(metadata={'subject_id': ...})`` returns
        only real subject data.
        """
        if self.entity_registry is None:
            return None
        to_record = getattr(obj, "to_record", None)
        if to_record is None:
            return None
        try:
            metadata = self._subject_metadata() if stamp_subject else None
            record = to_record(metadata=metadata)
            return self.entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def _register_message(self, message: Any) -> Any:
        """Register a Message entity (kind="message") when a registry is present."""
        return self._register_entity(message, stamp_subject=True)

    def _register_thread_version(self, thread: Any) -> Any:
        """Project the current chat_thread version into a record (when a registry).

        RO-8: the version bump now happens in the run body (regardless of
        registry), so this helper only projects the record at the already-bumped
        version. Returns ``None`` when no registry is wired.
        """
        if self.entity_registry is None:
            return None
        try:
            record = thread.to_record(metadata=self._subject_metadata())
            return self.entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def _link_lineage(
        self,
        *,
        persona_record: Any,
        prompt_record: Any,
        thread_record: Any,
        snapshot: Any,
        persona: Persona,
        thread: Any,
    ) -> None:
        """Wire the documented lineage relations between run artefacts."""
        if self.entity_registry is None or thread_record is None:
            return
        link = self.entity_registry.link
        try:
            if persona_record is not None:
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=persona_record.record_id,
                    relation="uses_persona",
                )
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=persona_record.record_id,
                    relation="thread_for_agent",
                )
            if prompt_record is not None:
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=prompt_record.record_id,
                    relation="in_thread",
                )
            if snapshot is not None:
                snapshot_record = getattr(snapshot, "to_record", None)
                if snapshot_record is not None:
                    sr = self.entity_registry.register(snapshot.to_record())
                    link(
                        from_record_id=thread_record.record_id,
                        to_record_id=sr.record_id,
                        relation="built_from",
                    )
                    link(
                        from_record_id=sr.record_id,
                        to_record_id=thread_record.record_id,
                        relation="observed_in_run",
                    )
        except Exception:  # pragma: no cover - defensive
            pass

    # --------------------------------------------------------------- helpers
    async def _emit(self, event: RunEvent) -> None:
        """Best-effort fan-out of one event to all configured sinks.

        Order: storage (the durable spine) -> entity registry -> observability
        span (invariant #6) -> caller-facing ``on_event`` callbacks (RO-6). Every
        sink is isolated so one failing sink can never break the run or starve
        the others, and ``CancelledError`` is honored so a cancelled run unwinds.
        """
        if self.memory_store is not None:
            appender = getattr(self.memory_store, "append_event", None)
            if appender is not None:
                try:
                    await appender(event)
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - defensive
                    pass
        if self.entity_registry is not None:
            try:
                self.entity_registry.register(
                    event.to_record(metadata=self._subject_metadata())
                )
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            from himmy.services.observability import emit_event_span

            emit_event_span(event)
        except Exception:  # pragma: no cover - defensive
            pass
        # Prometheus metrics: translate the event into bounded-cardinality counters.
        try:
            from himmy.services.observability.metrics import get_metrics_sink

            await get_metrics_sink().append_event(event)
        except Exception:  # pragma: no cover - defensive
            pass
        # RO-6: stream incremental progress to caller-supplied callbacks.
        for callback in self._on_event:
            try:
                await callback(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - never let a listener break the run
                pass

    async def _maybe_save_thread(self, thread: Any) -> None:
        """Persist the thread when ``save_threads`` and a memory store are present."""
        if not self.save_threads or self.memory_store is None:
            return
        saver = getattr(self.memory_store, "save_thread", None)
        if saver is None:
            return
        try:
            await saver(thread)
        except Exception:  # pragma: no cover - defensive
            pass


def message_timestamp() -> str:
    """Return an ISO timestamp for a tool message (delegates to the core helper)."""
    from himmy.core.ids import utc_now_iso

    return utc_now_iso()


def _decision_latency_ms(created_at: str | None) -> float | None:
    """Milliseconds from a checkpoint's creation to now (None if unparseable).

    Powers ``time_to_decision_ms`` on approval events — a judge-free signal of how
    long a human deliberated before approving/rejecting a gated tool.
    """
    if not created_at:
        return None
    from datetime import UTC, datetime

    try:
        start = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - start
    return max(0.0, delta.total_seconds() * 1000.0)


def _timeout(seconds: float) -> asyncio.Timeout:
    """Return an ``asyncio.timeout(seconds)`` context manager (RO-1).

    ``asyncio.timeout`` exists on Python 3.11+ (the project targets 3.12). It
    raises :class:`asyncio.CancelledError` inside the block on expiry, which the
    runtime catches to emit a terminal cancelled event before re-raising.
    """
    # ``asyncio.timeout`` is always present on 3.11+; keep the lookup defensive
    # so the module still imports on any interpreter.
    timeout_cm = getattr(asyncio, "timeout", None)
    if timeout_cm is None:  # pragma: no cover - 3.10 fallback only
        raise HimmyError("deadline_seconds requires Python 3.11+ (asyncio.timeout)")
    return cast(asyncio.Timeout, timeout_cm(seconds))


__all__ = [
    "SingleAgentRuntime",
    "RunResult",
    "AgentLoopResult",
    "ToolServiceProtocol",
    "OnEvent",
]
