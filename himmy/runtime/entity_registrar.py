"""Runtime collaborator: audit-spine entity registration + lineage linking.

Extracted verbatim from :class:`~himmy.runtime.single_agent.SingleAgentRuntime`
(the ``# --- entities`` method group) so the registration/lineage concern lives
in one cohesive object instead of being interleaved through the conductor. The
runtime constructs one :class:`_EntityRegistrar` in ``__init__`` and delegates
its ``_register_*`` / ``_link_lineage`` helpers to it; behaviour is byte-identical.

The registrar owns exactly one dependency — the optional
:class:`~himmy.entities.protocol.EntityRegistryProtocol` — and reads the current
run subject through the ``subject_metadata`` callable the runtime supplies (which
keeps a single source of truth for the WS4.6 governed-run subject stamp). Every
projection is best-effort: a ``None`` registry or a failing ``to_record`` yields
``None`` and never breaks the run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.agents.personas.persona import Persona
    from himmy.entities.protocol import EntityRegistryProtocol


class _EntityRegistrar:
    """Projects run artefacts into the entity registry and wires their lineage.

    A thin collaborator over an optional ``entity_registry``. ``subject_metadata``
    is the runtime's own ``{subject_id: ...}`` stamp function (``None`` on the
    ungoverned path), passed in so the governed-run subject tag has one definition
    shared with ``SingleAgentRuntime._emit``.
    """

    def __init__(
        self,
        entity_registry: EntityRegistryProtocol | None,
        subject_metadata: Callable[[], dict[str, Any] | None],
    ) -> None:
        self._entity_registry = entity_registry
        self._subject_metadata = subject_metadata

    def register_entity(self, obj: Any, *, stamp_subject: bool = False) -> Any:
        """Register a domain object's record when a registry is wired; else None.

        ``stamp_subject`` is only set for the genuinely subject-bearing run artefacts
        (messages, threads); infrastructure records (persona/prompt/agent) are never
        stamped so a governed ``registry.query(metadata={'subject_id': ...})`` returns
        only real subject data.
        """
        if self._entity_registry is None:
            return None
        to_record = getattr(obj, "to_record", None)
        if to_record is None:
            return None
        try:
            metadata = self._subject_metadata() if stamp_subject else None
            record = to_record(metadata=metadata)
            return self._entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def register_message(self, message: Any) -> Any:
        """Register a Message entity (kind="message") when a registry is present."""
        return self.register_entity(message, stamp_subject=True)

    def register_thread_version(self, thread: Any) -> Any:
        """Project the current chat_thread version into a record (when a registry).

        RO-8: the version bump now happens in the run body (regardless of
        registry), so this helper only projects the record at the already-bumped
        version. Returns ``None`` when no registry is wired.
        """
        if self._entity_registry is None:
            return None
        try:
            record = thread.to_record(metadata=self._subject_metadata())
            return self._entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def link_lineage(
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
        if self._entity_registry is None or thread_record is None:
            return
        link = self._entity_registry.link
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
                    sr = self._entity_registry.register(snapshot.to_record())
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
