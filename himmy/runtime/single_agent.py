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
import json
import logging
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
from himmy.runtime.audit import _CURRENT_SUBJECT, AuditEmitter
from himmy.runtime.checkpoint import (
    AgentCheckpoint,
    CheckpointStore,
    PendingToolCall,
)
from himmy.runtime.compaction import (
    _AUTO_COMPACT_KEEP_DEFAULT,  # noqa: F401 - re-exported for back-compat test import paths
    _AUTO_COMPACT_TOKENS_DEFAULT,  # noqa: F401 - re-exported for back-compat test import paths
    CompactionRunner,
    _auto_compact_default_spec,  # noqa: F401 - re-exported for back-compat test import paths
    _scrub_compaction_summary,  # noqa: F401 - re-exported for back-compat test import paths
)
from himmy.runtime.loop import LoopDriver
from himmy.runtime.prompt_assembly import (
    RequestBuilder,
    _prompt_cache_key_for_conversation,
    _prompt_cache_key_for_scope,
)
from himmy.runtime.resume import ResumeCoordinator
from himmy.runtime.snapshot import SnapshotResolver
from himmy.runtime.streaming import StreamDriver
from himmy.runtime.tool_exchange import ToolExchange
from himmy.services.inference.models import (
    BoundTool,
    CachePolicy,
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

# Module logger. The audit-spine sinks in ``_emit`` swallow their exceptions so a
# failing sink can never break a run, but a silent swallow hides a durable-write
# loss; log a warning so operators get a signal without the run itself failing.
log = logging.getLogger(__name__)


# The audit-spine event fan-out + entity projection (``_emit`` / ``_register_*`` /
# ``_link_lineage`` / ``_subject_metadata`` / ``_maybe_save_thread``) plus the
# ``_count_sink_drop`` metric helper and the ``_CURRENT_SUBJECT`` subject contextvar
# were extracted verbatim to :mod:`himmy.runtime.audit` (P3 decomposition step
# ``audit``). ``_count_sink_drop`` and ``_CURRENT_SUBJECT`` are re-imported at module
# scope (see the imports block) so their historical ``himmy.runtime.single_agent``
# import paths — pinned by tests and ``_subject_scope`` — stay valid.


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

# How many characters of a tool's result text are written to the MODEL-FACING thread
# (the TOOL Message in ``_append_tool_messages``). Unlike ``_TOOL_RESULT_EVENT_MAX``
# (which only bounds the observability event), this bounds what the model actually
# sees — and because the whole thread is re-sent on every subsequent turn, an
# uncapped multi-KB tool blob (a scraped page, a file dump, a wide DB result) is
# re-billed on every later turn of the run. Capping the model copy is a pure
# efficiency win: it shrinks the re-sent prefix while the FULL result stays intact
# on the tool-return record + the TOOL_COMPLETED event for observers.
#
# Default 6000 chars (~1.5k tokens) keeps a single tool result from dominating the
# window while still leaving room for a substantial page/table. Raise it, or set
# ``HIMMY_TOOL_RESULT_MODEL_MAX=0`` to DISABLE the cap entirely (restores the
# pre-C5 uncapped behaviour). A per-tool opt-out (for tools whose full output is
# essential — the model must see every byte) is available via the tool definition
# metadata flag ``model_result_uncapped=True`` (see ``_tool_result_uncapped``).
_TOOL_RESULT_MODEL_MAX_DEFAULT = 6000
try:
    _TOOL_RESULT_MODEL_MAX = max(0, int(os.environ.get("HIMMY_TOOL_RESULT_MODEL_MAX", str(_TOOL_RESULT_MODEL_MAX_DEFAULT))))
except ValueError:
    _TOOL_RESULT_MODEL_MAX = _TOOL_RESULT_MODEL_MAX_DEFAULT


# Default-on auto-compaction policy (``_auto_compact_default_spec`` /
# ``_AUTO_COMPACT_*_DEFAULT``) and the mandatory compaction-summary scrub
# (``_scrub_compaction_summary`` + ``_SUMMARY_*``) were extracted verbatim to
# :mod:`himmy.runtime.compaction` (P3 decomposition step ``compaction``). They are
# re-imported at module scope (see the imports block) so their existing
# ``himmy.runtime.single_agent`` import paths — pinned by tests — stay valid.


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


# The prompt/tools/request-assembly cache-key helpers
# (``_cache_scope_metadata`` / ``_prompt_cache_key_for_scope`` /
# ``_openai_conversation_cache_key_enabled`` / ``_prompt_cache_key_for_conversation``)
# live in ``himmy.runtime.prompt_assembly`` and are imported above. They are re-exported
# here so their existing import paths (tests import them from ``single_agent``) and the
# module-level ``_cache_scope_metadata`` call in this file resolve unchanged.


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
#
# ``_CURRENT_SUBJECT`` is DEFINED in :mod:`himmy.runtime.audit` (the audit-spine's
# subject source) and re-imported here (see the imports block) so ``_subject_scope``
# and its historical ``from himmy.runtime.single_agent import _CURRENT_SUBJECT`` import
# path resolve to the one canonical contextvar object.


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
        # The whole audit-spine concern — event fan-out across all sinks (``_emit``),
        # entity-registry projection + lineage (``_register_*`` / ``_link_lineage``),
        # the governed-run subject stamp (``_subject_metadata``) and thread persistence
        # (``_maybe_save_thread``) — lives in a cohesive collaborator; those methods
        # delegate. Reads live sink wiring (memory_store, entity_registry, _on_event,
        # save_threads) off ``self``. Behaviour is byte-identical; the runtime delegates.
        self._audit_emitter = AuditEmitter(self)
        self.default_model_key = default_model_key
        self.save_threads = save_threads
        self.default_deadline_seconds = default_deadline_seconds
        self.strict_snapshot = strict_snapshot
        # Snapshot resolution (load-or-build a context snapshot, emit
        # CONTEXT_SNAPSHOT_BUILT, honor strict_snapshot) lives in a collaborator;
        # ``_resolve_snapshot`` delegates. Reads live wiring off ``self``.
        self._snapshot_resolver = SnapshotResolver(self)
        # Tool-loop replay / retry / result-capping lives in a collaborator; the
        # ``_append_tool_messages`` / ``_wrap_executor_with_retry`` / ``_tool_*`` shims
        # delegate. Reads live wiring (tool_service, _emit, _register_message) off ``self``.
        self._tool_exchange = ToolExchange(self)
        # Prompt / tools / request assembly (render prompts, guard injected context,
        # resolve the model key + per-turn cache policy, build the typed
        # InferenceRequest) lives in a collaborator; the ``_render_*`` /
        # ``_effective_model_key`` / ``_prompt_cache_policy`` / ``_build_request`` shims
        # delegate. Reads live wiring (prompt_manager, context_prompt_mapper,
        # inference_service, tool_service, _enable_prompt_cache) off ``self``.
        self._request_builder = RequestBuilder(self)
        # Runtime-side auto-compaction (pick spec, plan, summarize, scrub/guard, splice
        # the thread, emit CONTEXT_COMPACTED, persist episodic) lives in a collaborator;
        # ``_maybe_compact`` delegates. Reads live wiring (default_model_key,
        # inference_service, _guard_input, _emit, memory_store) off ``self``.
        self._compaction_runner = CompactionRunner(self)
        # Multi-turn drive (stop-condition ladder, between-turns steering, forced
        # synthesis, adaptive tool routing) lives in a collaborator; the
        # ``_drive_loop`` / ``_drain_steer_queue`` / ``_maybe_synthesize`` /
        # ``_should_route`` / ``_route_tools`` shims delegate. Reads live wiring
        # (tool_service, inference_service, _emit, _continue_turn, _save_checkpoint,
        # _pending_approvals, run_task_detailed, _emit_turn_completed) off ``self``.
        self._loop_driver = LoopDriver(self)
        # The streaming surfaces (single-turn ``stream_task``, the whole-loop
        # ``stream_agent_loop``, the streamed continuation-drive ladder, and the
        # ``RunResult``/text/tool delta reconstruction) live in a collaborator; the
        # public ``stream_task`` / ``stream_agent_loop`` async generators delegate
        # into it via ``async for d in inner: yield d`` (wrapped in a
        # ``finally: await inner.aclose()`` so an early client close / cancellation
        # closes the inner generators — and the provider stream beneath them —
        # deterministically). Reads live wiring (inference_service, _output_guardrail,
        # _checkpoint_store, _subject_scope, _resolve_snapshot, _render_guarded_prompts,
        # _guard_input, _guard_output, _emit, _build_request, _effective_model_key,
        # _register_message, _register_thread_version, _should_route, _route_tools,
        # stream_task, _append_tool_messages, _emit_turn_completed, _maybe_synthesize,
        # _pending_approvals, _save_checkpoint, _continue_turn) off ``self``.
        self._stream_driver = StreamDriver(self)
        self._checkpoint_store = checkpoint_store
        # Per-checkpoint resume serialization (HITL exactly-once). The store-level
        # atomic claim is the cross-process gate; this in-process lock keeps two
        # concurrent resumes of the SAME checkpoint on one event loop (a
        # double-clicked Approve, two tabs, an automation retry) from interleaving
        # between the claim and the gated tool's execution — so the approved action
        # runs exactly once. Keyed by checkpoint_id, created on demand.
        self._resume_locks: dict[str, asyncio.Lock] = {}
        # HITL resume/checkpoint machinery (per-checkpoint locked resume body, the
        # orchestration final-output anchor, the per-checkpoint lock lookup, and the
        # paused-run checkpoint save) lives in a collaborator; the
        # ``_resume_agent_loop_locked`` / ``_record_resume_final_output`` /
        # ``_resume_lock_for`` / ``_save_checkpoint`` shims delegate. The public
        # ``resume_agent_loop`` and the ``_resume_locks`` state stay here; the
        # coordinator back-references them (and the rest of the live wiring:
        # _checkpoint_store, tool_service, _subject_scope, _append_tool_messages,
        # _emit, _register_thread_version, _continue_turn, _emit_turn_completed,
        # _drive_loop) off ``self`` at call time. Behaviour is byte-identical.
        self._resume_coordinator = ResumeCoordinator(self)
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
                    result, persona, trace_id, llm_config, ctx
                )
            return result

    #: Adaptive routing threshold: with MORE bound tools than this, an unset
    #: ``route_tools`` (None) routes automatically — the schema block for a big
    #: toolset costs more per turn than the one small routing call, and small
    #: local models pick badly from large catalogs. At or below it, no routing.
    AUTO_ROUTE_OVER_TOOLS = 8

    def _should_route(self, route_tools: bool | None) -> bool:
        """Delegating shim → :meth:`LoopDriver.should_route`."""
        return self._loop_driver.should_route(route_tools)

    async def _route_tools(self, task: Task, max_tools: int) -> Task:
        """Delegating shim → :meth:`LoopDriver.route_tools`."""
        return await self._loop_driver.route_tools(task, max_tools)

    async def _maybe_synthesize(
        self,
        result: AgentLoopResult,
        persona: Persona,
        trace_id: str,
        llm_config: LLMConfig | None,
        ctx: dict[str, Any] | None = None,
    ) -> AgentLoopResult:
        """Delegating shim → :meth:`LoopDriver.maybe_synthesize`."""
        return await self._loop_driver.maybe_synthesize(
            result, persona, trace_id, llm_config, ctx
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
        # Delegating shim → :meth:`StreamDriver.stream_task`. The ``async for … yield``
        # + ``finally: await inner.aclose()`` guarantees an early client close
        # (GeneratorExit) / cancellation propagates into the driver generator's own
        # ``finally`` (which closes the provider stream) deterministically — parity
        # with the pre-extraction inline finalizer.
        inner = self._stream_driver.stream_task(
            persona, task, thread, llm_config=llm_config
        )
        try:
            async for delta in inner:
                yield delta
        finally:
            await inner.aclose()

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
        # Delegating shim → :meth:`StreamDriver.stream_agent_loop`. Same ``async for
        # … yield`` + ``finally: await inner.aclose()`` wrapping as ``stream_task`` so
        # a client close / cancellation closes the driver's owned inner turn
        # generators (and the provider stream beneath them) deterministically.
        inner = self._stream_driver.stream_agent_loop(
            persona,
            task,
            thread,
            max_turns=max_turns,
            cost_budget=cost_budget,
            llm_config=llm_config,
            hitl=hitl,
            stop_on_no_progress=stop_on_no_progress,
            synthesize_empty=synthesize_empty,
            route_tools=route_tools,
            route_max_tools=route_max_tools,
        )
        try:
            async for delta in inner:
                yield delta
        finally:
            await inner.aclose()

    def _stream_drive_loop(
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
        """Delegating shim → :meth:`StreamDriver._stream_drive_loop`."""
        return self._stream_driver._stream_drive_loop(
            persona,
            task,
            thread,
            ctx,
            trace_id,
            turns=turns,
            max_turns=max_turns,
            cost_budget=cost_budget,
            llm_config=llm_config,
            hitl=hitl,
            stop_on_no_progress=stop_on_no_progress,
        )

    @staticmethod
    def _result_from_response(
        response: InferenceResponse, *, trace_id: str
    ) -> RunResult:
        """Delegating shim → :meth:`StreamDriver._result_from_response`."""
        return StreamDriver._result_from_response(response, trace_id=trace_id)

    @staticmethod
    def _text_deltas(result: RunResult) -> Iterator[StreamDelta]:
        """Delegating shim → :meth:`StreamDriver._text_deltas`."""
        return StreamDriver._text_deltas(result)

    @staticmethod
    def _tool_deltas(result: RunResult) -> Iterator[StreamDelta]:
        """Delegating shim → :meth:`StreamDriver._tool_deltas`."""
        return StreamDriver._tool_deltas(result)

    def _resume_lock_for(self, checkpoint_id: str) -> asyncio.Lock:
        """Delegating shim → :meth:`ResumeCoordinator.resume_lock_for`."""
        return self._resume_coordinator.resume_lock_for(checkpoint_id)

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
        """Delegating shim → :meth:`ResumeCoordinator.resume_agent_loop_locked`."""
        return await self._resume_coordinator.resume_agent_loop_locked(
            checkpoint_id,
            approved=approved,
            llm_config=llm_config,
            hitl=hitl,
            actor=actor,
        )

    def _record_resume_final_output(
        self, checkpoint_id: str, loop: AgentLoopResult
    ) -> None:
        """Delegating shim → :meth:`ResumeCoordinator.record_resume_final_output`."""
        self._resume_coordinator.record_resume_final_output(checkpoint_id, loop)

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
        """Delegating shim → :meth:`LoopDriver.drive_loop`."""
        return await self._loop_driver.drive_loop(
            persona,
            task,
            thread,
            ctx,
            trace_id,
            turns=turns,
            max_turns=max_turns,
            cost_budget=cost_budget,
            llm_config=llm_config,
            hitl=hitl,
            stop_on_no_progress=stop_on_no_progress,
            turns_offset=turns_offset,
            cost_offset=cost_offset,
            steer_queue=steer_queue,
        )

    def _drain_steer_queue(
        self, steer_queue: queue.Queue[str], thread: ChatThread
    ) -> None:
        """Delegating shim → :meth:`LoopDriver.drain_steer_queue`."""
        self._loop_driver.drain_steer_queue(steer_queue, thread)

    @staticmethod
    def _pending_approvals(result: RunResult) -> list[PendingToolCall]:
        """The tool calls in a turn that were denied for lack of HUMAN APPROVAL.

        Keys on the ``requires_approval`` gate's signature ONLY — ``outcome == 'denied'``
        AND ``error_code == 'POLICY_BLOCKED'``. A CAPABILITY/RBAC denial carries the distinct
        ``CAPABILITY_DENIED`` code (red-team r6) and is deliberately EXCLUDED: re-running it
        after an "approval" would deny it again (the run principal's roles are unchanged),
        wedging the run and misleading the operator — so a missing-capability call is a hard
        failure the model sees, never a resumable approval checkpoint.
        """
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
        """Delegating shim → :meth:`ResumeCoordinator.save_checkpoint`."""
        return self._resume_coordinator.save_checkpoint(
            persona,
            task,
            thread,
            ctx,
            llm_config,
            max_turns,
            cost_budget,
            turns_completed,
            cost_completed,
            pending,
        )

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

        DEFAULT-ON (eff-p0 #3) with a deliberately conservative high budget: an explicit
        ``ctx['compaction_spec']`` wins, otherwise :func:`_auto_compact_default_spec`
        supplies a default budget so long runs don't re-send their whole (O(turns^2))
        history uncompressed. The default is disable-able via ``HIMMY_AUTO_COMPACT=0``
        and tunable via ``HIMMY_AUTO_COMPACT_TOKENS`` / ``HIMMY_AUTO_COMPACT_KEEP``.

        Keeps the system head + recent tail, replaces the middle with one model-written
        summary message, and emits a ``CONTEXT_COMPACTED`` event (the audit trail of what
        was condensed). A no-op when under budget, when the default is disabled and no
        explicit spec was given, or when the summary would not actually shrink the span
        (the summary-only-if-smaller guard below).

        Returns ``True`` iff compaction actually rewrote the thread this turn. The
        caller (C5) uses that to BUST the prompt cache for the very next request:
        compaction inserts a new ``[Summary …]`` message, which changes the joined
        message prefix and would otherwise pay a write premium on a stale-cache miss.
        Skipping the breakpoint that one turn lets the prefix re-stabilize.

        SECURITY (sec-r1 + sec-r3): the summary is distilled from untrusted USER/TOOL
        content, so it is (1) run through the configured input guardrail (injection/DLP/
        blocklist) when one is wired AND — regardless of any guardrail — through a
        MANDATORY credential + injection scrub (:func:`_scrub_compaction_summary`) so the
        offline / default deployment (which wires no input guardrail) still cannot leak a
        secret at rest or launder a directive, and (2) inserted at USER — never SYSTEM —
        trust so a planted "instruction" can't be laundered into a standing directive.
        Operator steer/control messages are pinned out of the summarize span upstream
        (see :mod:`himmy.runtime.compaction`) so they always ride verbatim.
        """
        return await self._compaction_runner.maybe_compact(
            persona, thread, ctx, trace_id, llm_config
        )

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
        # sec-r3 #2: capture the pre-guard text so a guardrail CORRECTION (rewrite /
        # redaction / refusal substitution) can be marked ``guarded`` on the assistant
        # turn. The compaction planner pins guarded turns verbatim (sec-r2); previously
        # only the streaming path stamped this flag, so on the DEFAULT non-streaming
        # tool-loop continuation a corrected refusal was left unmarked and could be
        # summarized away, defeating the sec-r2 protection on the primary code path.
        raw_assistant_text = assistant_text
        assistant_text = await self._guard_output(
            assistant_text,
            agent_id=persona.agent_id,
            trace_id=trace_id,
            thread_id=thread.thread_id,
        )
        guarded_corrected = assistant_text != raw_assistant_text
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
                **({"guarded": True} if guarded_corrected else {}),
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
            try:
                _CURRENT_SUBJECT.reset(token)
            except ValueError:
                # The streamed paths (``stream_task`` / ``stream_agent_loop``) yield to
                # their consumer while suspended INSIDE this scope. In CPython an async
                # generator runs each step in the CALLER's context, so a consumer that
                # closes the stream from a different context than the first ``anext``
                # ran in — an early ``break`` under ``aclosing(...)``, or the event
                # loop's lazy async-generator finalizer, which closes from a fresh task
                # whose context is a COPY — runs this ``finally`` during ``aclose()``
                # holding a foreign token. ``reset(token)`` then raises "Token was
                # created in a different Context" and aborts the turn, so a downstream
                # SSE/NDJSON consumer sees an empty / failed chat turn.
                #
                # Swallowing is right for THIS context: it never observed our ``set()``
                # (the var is context-local, ``default=None``), so there is nothing to
                # undo, and stamping ``None`` here instead would clobber a subject a
                # nesting scope legitimately owns. The ENTERING context does keep the
                # stamp — ``contextvars`` offers no way to reset a context you are not
                # currently in — but that residue predates this guard (the raising
                # ``reset`` left it too, and additionally killed the run) and every
                # emit path re-``set``s the subject through this same scope.
                #
                # Narrow by construction: a re-used token raises ``RuntimeError``, not
                # ``ValueError``, so a genuine double-exit still surfaces here.
                log.debug(
                    "subject-scope reset skipped: token was created in a different "
                    "context (streamed close); the entering context keeps its stamp",
                    exc_info=True,
                )

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
        # sec-r3 #2: capture the pre-guard text so a guardrail CORRECTION marks the
        # assistant turn ``guarded`` (compaction pins guarded turns verbatim, sec-r2).
        raw_assistant_text = assistant_text
        assistant_text = await self._guard_output(
            assistant_text,
            agent_id=persona.agent_id,
            trace_id=trace_id,
            thread_id=thread.thread_id,
        )
        guarded_corrected = assistant_text != raw_assistant_text
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
        if guarded_corrected:
            assistant_metadata["guarded"] = True
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
        return await self._snapshot_resolver.resolve(
            persona, task, ctx, snapshot_id
        )

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
        """Delegating shim → :meth:`RequestBuilder.render_guarded_prompts`."""
        return await self._request_builder.render_guarded_prompts(
            persona,
            task,
            ctx,
            snapshot,
            trace_id=trace_id,
            thread_id=thread_id,
        )

    def _render_prompt_parts(
        self,
        persona: Persona,
        task: Task,
        ctx: dict[str, Any],
        snapshot: Any,
    ) -> tuple[str, str, list[str], str, str]:
        """Delegating shim → :meth:`RequestBuilder.render_prompt_parts`."""
        return self._request_builder.render_prompt_parts(persona, task, ctx, snapshot)

    # ----------------------------------------------------------- inference
    def _effective_model_key(
        self, ctx: dict[str, Any], llm_config: LLMConfig | None
    ) -> str:
        """Delegating shim → :meth:`RequestBuilder.effective_model_key`."""
        return self._request_builder.effective_model_key(ctx, llm_config)

    def _prompt_cache_policy(
        self,
        model_key: str,
        *,
        cache_busted: bool,
        scope_metadata: dict[str, Any],
        thread_id: str | None = None,
    ) -> CachePolicy | None:
        """Delegating shim → :meth:`RequestBuilder.prompt_cache_policy` (test-poked)."""
        return self._request_builder.prompt_cache_policy(
            model_key,
            cache_busted=cache_busted,
            scope_metadata=scope_metadata,
            thread_id=thread_id,
        )

    def _build_request(
        self,
        thread: Any,
        ctx: dict[str, Any],
        llm_config: LLMConfig | None,
        *,
        trace_id: str | None = None,
        cache_busted: bool = False,
    ) -> tuple[InferenceRequest, list[str] | None]:
        """Delegating shim → :meth:`RequestBuilder.build_request` (test-poked)."""
        return self._request_builder.build_request(
            thread,
            ctx,
            llm_config,
            trace_id=trace_id,
            cache_busted=cache_busted,
        )

    def _tool_is_read_only(self, tool_name: str) -> bool:
        """Delegating shim → :meth:`ToolExchange.tool_is_read_only`."""
        return self._tool_exchange.tool_is_read_only(tool_name)

    def _wrap_executor_with_retry(
        self,
        executor: ToolExecutor,
        ctx: dict[str, Any],
        *,
        thread_id: str | None,
        agent_id: str | None,
        trace_id: str | None,
    ) -> ToolExecutor:
        """Delegating shim → :meth:`ToolExchange.wrap_executor_with_retry`."""
        return self._tool_exchange.wrap_executor_with_retry(
            executor,
            ctx,
            thread_id=thread_id,
            agent_id=agent_id,
            trace_id=trace_id,
        )

    # ------------------------------------------------------- tool messages
    def _tool_result_uncapped(self, tool_name: str) -> bool:
        """Delegating shim → :meth:`ToolExchange.tool_result_uncapped`."""
        return self._tool_exchange.tool_result_uncapped(tool_name)

    async def _append_tool_messages(
        self,
        thread: Any,
        response: InferenceResponse,
        *,
        request_id: str,
        trace_id: str,
        agent_id: str | None,
    ) -> None:
        """Delegating shim → :meth:`ToolExchange.append_tool_messages` (1 test poke)."""
        await self._tool_exchange.append_tool_messages(
            thread,
            response,
            request_id=request_id,
            trace_id=trace_id,
            agent_id=agent_id,
        )

    # --------------------------------------------------------------- entities
    @staticmethod
    def _subject_metadata() -> dict[str, Any] | None:
        """Delegating shim → :meth:`AuditEmitter.subject_metadata`.

        Kept as a class-level method (its callers ``_emit`` / ``_register_*`` and
        tests read ``runtime._subject_metadata``); the ``{subject_id: ...}`` logic
        lives in :class:`~himmy.runtime.audit.AuditEmitter`.
        """
        return AuditEmitter.subject_metadata()

    def _register_entity(self, obj: Any, *, stamp_subject: bool = False) -> Any:
        """Delegating shim → :meth:`AuditEmitter.register_entity`."""
        return self._audit_emitter.register_entity(obj, stamp_subject=stamp_subject)

    def _register_message(self, message: Any) -> Any:
        """Delegating shim → :meth:`AuditEmitter.register_message`."""
        return self._audit_emitter.register_message(message)

    def _register_thread_version(self, thread: Any) -> Any:
        """Delegating shim → :meth:`AuditEmitter.register_thread_version`."""
        return self._audit_emitter.register_thread_version(thread)

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
        """Delegating shim → :meth:`AuditEmitter.link_lineage`."""
        self._audit_emitter.link_lineage(
            persona_record=persona_record,
            prompt_record=prompt_record,
            thread_record=thread_record,
            snapshot=snapshot,
            persona=persona,
            thread=thread,
        )

    # --------------------------------------------------------------- helpers
    async def _emit(self, event: RunEvent) -> None:
        """Delegating shim → :meth:`AuditEmitter.emit` (a stub test needs this method)."""
        await self._audit_emitter.emit(event)

    async def _maybe_save_thread(self, thread: Any) -> None:
        """Delegating shim → :meth:`AuditEmitter.maybe_save_thread`."""
        await self._audit_emitter.maybe_save_thread(thread)


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
    # Re-exported from ``prompt_assembly`` so their historical import path
    # (``from himmy.runtime.single_agent import _prompt_cache_key_*``) is preserved.
    "_prompt_cache_key_for_scope",
    "_prompt_cache_key_for_conversation",
]
