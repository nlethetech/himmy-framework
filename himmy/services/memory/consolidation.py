"""Audited memory consolidation: ADD / UPDATE / DELETE / NOOP on the spine.

This is the Mem0 insight expressed on himmy's audit spine: a new candidate fact is not
blindly appended. The consolidator recalls the most semantically-similar EXISTING facts
for the subject and decides one of four actions, then applies it *bi-temporally* —
superseding facts are invalidated (``valid_to`` stamped + ``supersedes`` /
``invalidated_by`` links), never physically deleted (Graphiti semantics) — and the
WHOLE decision is emitted as a ``MEMORY_CONSOLIDATED`` :class:`RunEvent`, projected to
the registry. That makes the extract -> decide -> apply loop replayable and auditable,
which none of Mem0/Zep/Letta/Cognee offer.

Two decision paths, both fully offline:

* **Offline default** (no client manager, or a stub one): a deterministic similarity
  rule — a near-duplicate of an existing fact is ``NOOP``; a high-similarity-but-changed
  fact ``UPDATE``s the closest match; anything else is ``ADD``.
* **LLM path** (opt-in, ``use_llm=True`` with a real :class:`ClientManager`): the same
  decision is made by a ``STRUCTURED_OUTPUT`` inference whose schema pins the action.
  The offline :class:`StubClientManager` satisfies that schema, so even this path runs
  with no keys or network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from himmy.core.events import EventType, RunEvent
from himmy.core.ids import utc_now_iso
from himmy.services.inference.models import InferenceMessage, InferenceRequest
from himmy.services.memory.store import MemoryRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.core.events import EventSink
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.services.inference.client_manager import ClientManager
    from himmy.services.memory.service import MemoryService

ConsolidationAction = Literal["ADD", "UPDATE", "DELETE", "NOOP"]

#: The JSON schema the LLM consolidation path constrains the model to. Enum order is
#: deliberate: the offline stub picks the first enum value, so an opt-in LLM run with
#: the stub yields a safe ``NOOP`` rather than a fabricated mutation.
_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["NOOP", "ADD", "UPDATE", "DELETE"]},
        "target_memory_id": {"type": ["string", "null"]},
        "new_text": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}


@dataclass
class ConsolidationResult:
    """The outcome of consolidating one candidate fact."""

    action: ConsolidationAction
    record: MemoryRecord | None
    target_id: str | None
    reason: str
    similarity: float = 0.0


class MemoryConsolidator:
    """Decide and apply ADD/UPDATE/DELETE/NOOP for a candidate fact, audited.

    ``dup_threshold`` is the similarity at/above which a candidate is treated as a
    duplicate (``NOOP``); ``update_threshold`` is the lower bar at/above which a
    different-but-related candidate ``UPDATE``s (supersedes) the closest existing fact.
    A registry and/or event sink (both optional) project the decision onto the spine.
    """

    def __init__(
        self,
        memory: MemoryService,
        *,
        client_manager: ClientManager | None = None,
        registry: EntityRegistryProtocol | None = None,
        event_sink: EventSink | None = None,
        dup_threshold: float = 0.95,
        update_threshold: float = 0.80,
        top_k: int = 5,
        use_llm: bool = False,
    ) -> None:
        self._memory = memory
        self._client_manager = client_manager
        self._registry = registry
        self._event_sink = event_sink
        self._dup_threshold = dup_threshold
        self._update_threshold = update_threshold
        self._top_k = top_k
        self._use_llm = use_llm

    async def consolidate(
        self,
        candidate_text: str,
        *,
        subject_id: str = "default",
        kind: str = "semantic",
        source: str = "llm_extracted",
        tier: str = "recall",
    ) -> ConsolidationResult:
        """Decide and apply the right mutation for ``candidate_text`` and audit it."""
        hits = await self._memory.recall(
            candidate_text,
            subject_id=subject_id,
            top_k=self._top_k,
            active_only=True,
        )
        top = hits[0] if hits else None
        top_sim = top.similarity if top is not None else 0.0

        if self._use_llm and self._client_manager is not None:
            action, target_id, new_text, reason = await self._decide_with_llm(
                candidate_text, hits
            )
        else:
            action, target_id, new_text, reason = self._decide_offline(
                candidate_text, top, top_sim
            )

        result = self._apply(
            action=action,
            candidate_text=new_text or candidate_text,
            target_id=target_id,
            reason=reason,
            similarity=top_sim,
            subject_id=subject_id,
            kind=kind,
            source=source,
            tier=tier,
        )
        await self._emit_consolidated(result, candidate_text=candidate_text)
        return result

    # -- decision paths --------------------------------------------------------

    def _decide_offline(
        self, candidate_text: str, top: Any, top_sim: float
    ) -> tuple[ConsolidationAction, str | None, str | None, str]:
        """Deterministic similarity-rule decision (no model, fully offline)."""
        if top is None:
            return "ADD", None, candidate_text, "no related memory exists"
        if top_sim >= self._dup_threshold:
            return (
                "NOOP",
                top.record.memory_id,
                None,
                f"near-duplicate of existing fact (sim={top_sim:.3f})",
            )
        if top_sim >= self._update_threshold:
            return (
                "UPDATE",
                top.record.memory_id,
                candidate_text,
                f"supersedes a closely-related fact (sim={top_sim:.3f})",
            )
        return "ADD", None, candidate_text, f"distinct fact (sim={top_sim:.3f})"

    async def _decide_with_llm(
        self, candidate_text: str, hits: list[Any]
    ) -> tuple[ConsolidationAction, str | None, str | None, str]:
        """Opt-in LLM decision via STRUCTURED_OUTPUT (stub-satisfiable offline)."""
        from himmy.services.inference.service import InferenceService

        existing = "\n".join(
            f"- id={h.record.memory_id} (sim={h.similarity:.3f}): {h.record.text}"
            for h in hits
        )
        prompt = (
            "You maintain a user's long-term memory. Decide whether the NEW fact "
            "should be added, should update/replace an existing fact, should delete "
            "an existing fact, or requires no change (it is already known).\n\n"
            f"NEW fact: {candidate_text}\n\n"
            f"EXISTING related facts:\n{existing or '(none)'}"
        )
        request = InferenceRequest(
            messages=[InferenceMessage(role="user", content=prompt)],
            output_json_schema=_DECISION_SCHEMA,
        )
        service = InferenceService(self._client_manager)  # type: ignore[arg-type]
        response = await service.run(request)
        data = response.output_structured
        if not isinstance(data, dict):
            return "NOOP", None, None, "model returned no structured decision"
        raw_action = str(data.get("action", "NOOP")).upper()
        action: ConsolidationAction = (
            raw_action  # type: ignore[assignment]
            if raw_action in ("ADD", "UPDATE", "DELETE", "NOOP")
            else "NOOP"
        )
        target_id = data.get("target_memory_id")
        new_text = data.get("new_text")
        reason = str(data.get("reason", "")) or "llm decision"
        return (
            action,
            str(target_id) if target_id else None,
            str(new_text) if new_text else None,
            reason,
        )

    # -- bi-temporal apply -----------------------------------------------------

    def _apply(
        self,
        *,
        action: ConsolidationAction,
        candidate_text: str,
        target_id: str | None,
        reason: str,
        similarity: float,
        subject_id: str,
        kind: str,
        source: str,
        tier: str,
    ) -> ConsolidationResult:
        """Apply the decided action to the store + spine, bi-temporally."""
        if action == "NOOP":
            return ConsolidationResult("NOOP", None, target_id, reason, similarity)

        if action == "DELETE":
            now = utc_now_iso()
            target = self._memory.get(target_id) if target_id else None
            if target is None:
                return ConsolidationResult("NOOP", None, None, "no target to delete")
            invalidated = self._memory.invalidate(target.memory_id, valid_to=now)
            self._link_invalidation(old_id=target.memory_id, new_id=None)
            return ConsolidationResult(
                "DELETE", invalidated, target.memory_id, reason, similarity
            )

        if action == "UPDATE" and target_id is not None:
            old = self._memory.get(target_id)
            if old is not None:
                now = utc_now_iso()
                # The new value shares the old fact's semantic identity (``stable_key``)
                # so its spine record continues the SAME version chain — v2 superseding
                # the old v1. That chain IS the fact's transaction-time history.
                stable_key = old.stable_key or old.memory_id
                new_record = self._memory.remember(
                    candidate_text,
                    subject_id=subject_id,
                    kind=kind,
                    source=source,
                    tier=tier,
                    stable_key=stable_key,
                    project_version=2,
                )
                self._memory.invalidate(
                    old.memory_id, valid_to=now, superseded_by=new_record.memory_id
                )
                self._link_supersede(new_id=new_record.memory_id, old_id=old.memory_id)
                return ConsolidationResult(
                    "UPDATE", new_record, old.memory_id, reason, similarity
                )

        # ADD (or an UPDATE whose target vanished) — mint a fresh fact.
        new_record = self._memory.remember(
            candidate_text,
            subject_id=subject_id,
            kind=kind,
            source=source,
            tier=tier,
        )
        return ConsolidationResult("ADD", new_record, None, reason, similarity)

    def _link_supersede(self, *, new_id: str, old_id: str) -> None:
        """Link new -[supersedes]-> old and old -[invalidated_by]-> new on the spine.

        The superseding fact shares the old fact's ``stable_key`` and was projected as
        version 2 of that chain; the old fact is version 1. The links join those two
        spine records so ``registry.trace(new, relations={'supersedes'})`` reaches the
        old value and the existing :class:`LineageGraph` renders the fact's evolution.
        """
        if self._registry is None:
            return
        new_record_id = self._record_id(new_id, version=2)
        old_record_id = self._record_id(old_id, version=1)
        if new_record_id is None or old_record_id is None:
            return
        self._registry.link(
            from_record_id=new_record_id,
            to_record_id=old_record_id,
            relation="supersedes",
        )
        self._registry.link(
            from_record_id=old_record_id,
            to_record_id=new_record_id,
            relation="invalidated_by",
        )

    def _link_invalidation(self, *, old_id: str, new_id: str | None) -> None:
        """Link old -[invalidated_by]-> new (or self) for a DELETE on the spine."""
        if self._registry is None:
            return
        old_record_id = self._record_id(old_id, version=1)
        if old_record_id is None:
            return
        target_record_id = (
            self._record_id(new_id, version=1) if new_id is not None else old_record_id
        )
        if target_record_id is None:
            return
        self._registry.link(
            from_record_id=old_record_id,
            to_record_id=target_record_id,
            relation="invalidated_by",
        )

    def _record_id(self, memory_id: str, *, version: int) -> str | None:
        """Resolve the spine ``record_id`` for a stored memory at ``version``."""
        record = self._memory.get(memory_id)
        if record is None:
            return None
        return record.to_record(version=version).record_id

    # -- audit -----------------------------------------------------------------

    async def _emit_consolidated(
        self, result: ConsolidationResult, *, candidate_text: str
    ) -> None:
        """Emit + project the MEMORY_CONSOLIDATED audit event.

        ``consolidate`` is async, so the sink is awaited directly here — guaranteeing
        the audit event is delivered before the decision is returned to the caller.
        """
        if self._registry is None and self._event_sink is None:
            return
        event = RunEvent(
            event_type=EventType.MEMORY_CONSOLIDATED,
            payload={
                "action": result.action,
                "candidate": candidate_text,
                "target": result.target_id,
                "reason": result.reason,
                "similarity": result.similarity,
                "memory_id": result.record.memory_id if result.record else None,
            },
        )
        if self._registry is not None:
            try:
                self._registry.register(event.to_record())
            except Exception:  # pragma: no cover - defensive
                pass
        if self._event_sink is not None:
            try:
                await self._event_sink.append_event(event)
            except Exception:  # pragma: no cover - a sink must not break consolidation
                pass


__all__ = [
    "ConsolidationAction",
    "ConsolidationResult",
    "MemoryConsolidator",
]
