"""Tool-exchange replay / retry / result-capping for :class:`SingleAgentRuntime`.

Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
step ``tool_exchange``). :class:`ToolExchange` owns the tool-loop plumbing:

* replaying each tool call/return pair onto the thread as a ``TOOL`` message with
  full metadata, emitting the paired ``TOOL_CALLED`` + ``TOOL_COMPLETED``/
  ``TOOL_FAILED`` events in the exact same order (``append_tool_messages``);
* capping the MODEL-FACING copy of a tool result (``_cap_tool_result_for_model``,
  ``tool_result_uncapped``); and
* bounded turn-level retry of TRANSIENT tool failures with side-effect safety
  (``wrap_executor_with_retry``, ``tool_is_read_only``).

The runtime constructs one of these in ``__init__`` and its former methods become
thin delegating shims. Behavior — event ORDER, result-cap bytes, retry taxonomy,
exception types — is byte-for-byte identical to the pre-extraction inline code.

The module-facing result-cap knobs (``_TOOL_RESULT_MODEL_MAX`` /
``_TOOL_RESULT_EVENT_MAX``) and the ``message_timestamp`` / ``_tool_timeout_code``
helpers deliberately stay in ``single_agent`` and are read LIVE off that module at
call time, so a test that monkeypatches ``single_agent._TOOL_RESULT_MODEL_MAX``
still steers this code exactly as before.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

from himmy.core.events import EventType, RunEvent
from himmy.services.inference.models import (
    InferenceResponse,
    ToolExecutor,
    ToolReturnRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.runtime.single_agent import SingleAgentRuntime


def _cap_tool_result_for_model(text: str, cap: int) -> str:
    """Bound a tool result's model-facing text at ``cap`` characters.

    Mirrors :func:`_truncate` but appends an explicit, self-describing marker so the
    model knows the content was shortened for context economy (and by how much),
    rather than silently seeing a mid-sentence cut. Returns ``text`` unchanged when
    ``cap`` is ``0`` (cap disabled) or the text already fits.
    """
    text = text or ""
    if cap <= 0 or len(text) <= cap:
        return text
    dropped = len(text) - cap
    return text[:cap] + f"\n…[truncated {dropped} chars]"


def _transient_tool_codes() -> frozenset[str]:
    """The tool error codes the runtime treats as transient (turn-level retry).

    Sourced from the tools kernel's own ``RETRYABLE_TOOL_CODES`` taxonomy
    (timeout / rate-limited / provider-unavailable) so the two layers can never
    drift; imported lazily because the tools kernel is not on this kernel's
    import path.
    """
    from himmy.services.tools.service import RETRYABLE_TOOL_CODES

    return frozenset(code.value for code in RETRYABLE_TOOL_CODES)


class ToolExchange:
    """Replays / retries / caps tool exchanges for one runtime.

    Holds a back-reference to the owning :class:`SingleAgentRuntime` and reads its
    live wiring (``tool_service``, ``_emit``, ``_register_message``) at call time,
    so runtime reconfiguration between runs is honored exactly as when the logic
    lived inline on the runtime.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    def tool_is_read_only(self, tool_name: str) -> bool:
        """True only when ``tool_name`` is provably safe to RE-FIRE after a TIMEOUT.

        A TIMEOUT does not mean the tool didn't run — the effect may have committed
        server-side after the client gave up — so re-firing is only safe for a tool
        with NO side effect. This is deliberately strict:

        * an AUTHORITATIVE ``read_only=True`` (the author explicitly asserted no side
          effect via the ``read_only`` argument) is trusted; but
        * a merely-INFERRED read-only — a GET/HEAD HTTP method, or a name the strict
          classifier reads as safe — is NOT, because a GET endpoint can still have a
          server-side side effect (an analytics beacon, ``GET /trigger``) and a name is
          only a guess. For those, timeout-retry falls back to the strict NAME check
          only when there is no method signal; a method-derived read-only never
          re-fires on its own.

        Anything else (a write tool, an ambiguous name, an unknown tool with no
        definition) is conservatively treated as side-effecting and NOT retried on
        timeout.
        """
        if self._rt.tool_service is None:
            return False
        registry = getattr(self._rt.tool_service, "registry", None)
        if registry is None:
            return False
        definition = registry.get(tool_name)
        if definition is None:
            return False
        # An author's explicit read_only=True is authoritative — safe to re-fire.
        if definition.read_only_authoritative:
            return bool(definition.read_only)
        # A method-DERIVED read_only (HTTP GET/HEAD/OPTIONS) is a parallelism hint only;
        # the remote server may violate the convention, so never re-fire on it.
        if definition.http_config is not None:
            return False
        # An author's explicit read_only=False is honoured; an inferred/absent value on
        # a local tool falls back to the strict name gate.
        if definition.read_only is False:
            return False
        from himmy.services.tools.access import classify_parallel_safe

        return classify_parallel_safe(tool_name)

    def wrap_executor_with_retry(
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
        import himmy.runtime.single_agent as _single_agent

        raw_attempts = ctx.get("tool_retry_attempts")
        retries = (
            _single_agent.DEFAULT_TOOL_RETRY_ATTEMPTS
            if raw_attempts is None
            else max(0, int(raw_attempts))
        )
        if retries == 0:
            return executor
        raw_backoff = ctx.get("tool_retry_backoff_seconds")
        backoff = (
            _single_agent.DEFAULT_TOOL_RETRY_BACKOFF_SECONDS
            if raw_backoff is None
            else max(0.0, float(raw_backoff))
        )
        transient = _transient_tool_codes()
        timeout_code = _single_agent._tool_timeout_code()

        async def _execute(tool_name: str, args: dict[str, Any]) -> ToolReturnRecord:
            record = await executor(tool_name, args)
            for attempt in range(1, retries + 1):
                code = (record.metadata or {}).get("error_code")
                if record.outcome != "failed" or code not in transient:
                    return record
                # A timed-out non-read-only tool may already have committed its
                # side effect — do not re-fire it (idempotency would be violated).
                if code == timeout_code and not self.tool_is_read_only(tool_name):
                    return record
                await self._rt._emit(
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
    def tool_result_uncapped(self, tool_name: str) -> bool:
        """Whether ``tool_name`` opts OUT of the model-facing result cap.

        A tool whose full output is essential (the model must see every byte — e.g. a
        tool that returns a signed artifact, a full document the next step must quote
        verbatim) can preserve its complete result two ways:

        * declaratively — its :class:`ToolDefinition` metadata carries
          ``model_result_uncapped=True``; or
        * by deployment — its name appears in the comma-separated env allowlist
          ``HIMMY_TOOL_RESULT_UNCAPPED`` (whitespace-trimmed, case-sensitive).

        Returns ``False`` (cap applies) for any unknown tool or when neither opt-out
        is set. Resolution is best-effort: a missing/odd tool service or metadata
        never raises here — the cap simply applies.
        """
        raw = os.environ.get("HIMMY_TOOL_RESULT_UNCAPPED", "")
        allowlist = {n.strip() for n in raw.split(",") if n.strip()}
        if tool_name in allowlist:
            return True
        registry = getattr(self._rt.tool_service, "registry", None)
        if registry is None:
            return False
        try:
            definition = registry.get(tool_name)
        except Exception:  # noqa: BLE001 - opt-out lookup is best-effort, never fatal
            return False
        if definition is None:
            return False
        return bool((getattr(definition, "metadata", None) or {}).get("model_result_uncapped"))

    async def append_tool_messages(
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
        import himmy.runtime.single_agent as _single_agent
        from himmy.agents.base_agent.thread import Message, MessageRole

        returns_by_id: dict[str, ToolReturnRecord] = {
            r.tool_call_id: r for r in response.tool_returns
        }
        for call in response.tool_calls:
            ret = returns_by_id.get(call.tool_call_id)

            # TOOL_CALLED: emitted before the return is recorded.
            await self._rt._emit(
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
            # C5: cap the MODEL-FACING copy of the result. ``content_text`` (the FULL
            # result) is preserved untouched for the tool-return record + the
            # TOOL_COMPLETED event below; only ``model_content_text`` — what rides on
            # the re-sent thread every subsequent turn — is bounded. A per-tool
            # opt-out (metadata flag / env allowlist) or ``HIMMY_TOOL_RESULT_MODEL_MAX=0``
            # keeps the full result on the thread.
            if _single_agent._TOOL_RESULT_MODEL_MAX and not self.tool_result_uncapped(
                call.tool_name
            ):
                model_content_text = _cap_tool_result_for_model(
                    content_text, _single_agent._TOOL_RESULT_MODEL_MAX
                )
            else:
                model_content_text = content_text
            # sec-r3 #1: the message ``content`` is the capped MODEL-facing copy (what
            # rides the re-sent thread), but the entity spine projects Message.content
            # into the canonical kind='message'/'chat_thread' audit records. If those
            # only ever saw the truncated copy, an auditor reconstructing "what the tool
            # actually returned" from the spine would get a lossy, marker-suffixed answer
            # while the full bytes lived only on a transient event/DTO. When (and only
            # when) the model copy was actually shortened, stash the FULL untruncated
            # result on ``metadata['full_content']`` so the audit trail stays faithful.
            tool_metadata: dict[str, Any] = {
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "tool_outcome": ret.outcome if ret is not None else "unknown",
                "tool_args": dict(call.args),
                "request_id": request_id,
                "trace_id": trace_id,
                "timestamp": _single_agent.message_timestamp(),
                "tool_return_metadata": dict(ret.metadata)
                if ret is not None
                else {},
            }
            if model_content_text != content_text:
                tool_metadata["full_content"] = content_text
            message = Message(
                role=MessageRole.TOOL,
                content=model_content_text,
                metadata=tool_metadata,
            )
            thread.append_message(message)
            self._rt._register_message(message)

            # TOOL_COMPLETED / TOOL_FAILED keyed on the return's outcome. The result
            # text + latency ride on the payload so an observer (e.g. Studio's live
            # cognition/ledger view) can show what each tool returned and how long it
            # took without re-reading the thread.
            outcome = ret.outcome if ret is not None else "unknown"
            completed = outcome == "success"
            ret_meta = (ret.metadata or {}) if ret is not None else {}
            result_text = (
                content_text
                if len(content_text) <= _single_agent._TOOL_RESULT_EVENT_MAX
                else (content_text[: _single_agent._TOOL_RESULT_EVENT_MAX - 1] + "…")
            )
            await self._rt._emit(
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
