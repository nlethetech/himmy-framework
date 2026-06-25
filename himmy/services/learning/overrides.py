"""Operator overrides (Sprint 2): teach & correct what self-learning concluded.

Sprint 1 mined a per-tool :class:`~himmy.services.learning.service.ToolReputation`
from recorded events and let it reorder / flag tools automatically. This module adds
the *human gate*: an operator can PIN a tool (force-trust it — never demoted, never
flagged flaky), MUTE / ignore it (exclude it from auto-learning), RESET its history
(forget accumulated reputation so it rebuilds from now), or CLEAR an override (back to
pure auto-learning). It also carries the per-workspace ``trust_feedback`` flag that
activates Sprint-1's recorded-but-inert 👍/👎 outcome blend.

THE SHARED APPLY PATH. :func:`apply_override` is the SOLE place an override becomes a
:class:`ToolReputation` change. It is applied at ONE point —
:meth:`LearningService._reputation_for_tool` — so every downstream reader (the runtime
:class:`ToolReputationProvider` snapshot that ``bound_tools`` reorders against, the
learned-hints adapter, AND :func:`build_learning_report`'s visible table) inherits the
same semantics. The display and the behaviour can therefore never diverge. RESET is the
one exception: it changes which EVENTS feed the score (an upstream timestamp cutoff in
``_recent_counts`` / ``_outcome_signal``), not the post-read reputation, so
:func:`apply_override` leaves a reset reputation untouched.

PERSISTENCE. Overrides are mutable operator INTENT (config), not append-only telemetry,
so they live on the durable, versioned EntityRecord spine —
:class:`~himmy.entities.spine_factory.SpineFactory` — reached identically by every front
end via :meth:`OverrideStore.for_context` / :meth:`OverrideStore.for_cli`. This mirrors
``studio_feedback.record_feedback``'s pattern (a ``new_version`` per change, the version
chain IS the audit log) and adds ZERO new event-type schema, so the Sprint-1 miner read
path and the feature-off invariant are untouched.

  TWO-SPINE NOTE (deliberate divergence from ``studio_feedback``): that module persists
  run feedback to ``.himmy/entities.db`` via ``himmy.api.studio_entities.get_entity_registry``.
  Overrides MUST instead use :class:`SpineFactory` (``.himmy/spine.db``) because the
  non-API runtime (``himmy.runtime.from_spec``) CANNOT import ``himmy.api`` and therefore
  cannot reach ``entities.db``. ``spine.db`` is the only durable registry both the runtime
  and the front ends can reach, so it is the single shared store that makes an override an
  operator sets the override the runtime honours. Override audit consequently lives on a
  different DB than run-feedback audit — intentional.

Everything here is 100% best-effort: any registry error in a load or a mutation degrades
to an empty :class:`OverrideSet` / trust-off — learning never breaks or slows a run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from himmy.core.ids import utc_now_iso
from himmy.entities.records import EntityQuery, stable_id_for
from himmy.services.learning.service import NEUTRAL_SCORE, ToolReputation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.protocol import EntityRegistryProtocol

logger = logging.getLogger("himmy.services.learning.overrides")

#: Spine record kind for a per-tool operator override (pin / mute / reset / clear).
TOOL_OVERRIDE_KIND = "tool_override"

#: Spine record kind for the per-workspace ``trust_feedback`` gate flag.
TRUST_FEEDBACK_KIND = "trust_feedback"

#: The effective ``outcome_weight`` the runtime adopts when ``trust_feedback`` is ON and
#: the operator has NOT set an explicit ``spec.outcome_weight > 0``. A sensible default
#: blend that lifts the recorded 👍/👎 outcomes from inert (0.0) without overriding an
#: intentional P2 value. Re-exported from / kept in sync with the service module.
TRUST_FEEDBACK_DEFAULT_WEIGHT = 0.3

#: The subject (``stable_id`` for the trust flag, ``default`` fallback for a None
#: workspace) overrides are scoped to, mirroring ``studio_feedback._feedback_subject``.
_DEFAULT_SUBJECT = "default"


class OverrideKind(str, Enum):
    """What an operator did to a tool's auto-learned reputation."""

    #: Force-trust: treated reliable (score 1.0), never flagged flaky, never demoted.
    PIN = "pin"
    #: Exclude from auto-learning: never demoted / flagged / hinted, shown as "ignored".
    MUTE = "mute"
    #: Forget accumulated reputation: ignore events at/before a cutoff so it rebuilds.
    RESET = "reset"
    #: Remove any override — back to pure auto-learning (a tombstone in the chain).
    CLEAR = "clear"


@dataclass(frozen=True)
class ToolOverride:
    """One operator override for one tool, as recorded on the spine."""

    tool_name: str
    kind: OverrideKind
    #: For RESET: the ISO-microsecond UTC cutoff; events at/before it are forgotten.
    #: None for every other kind.
    reset_cutoff: str | None = None
    #: Who set it (``cli`` / ``human``), for the audit trail.
    actor: str = "operator"
    #: When it was set (ISO-microsecond UTC).
    updated_at: str | None = None


@dataclass(frozen=True)
class OverrideSet:
    """The current overrides for one workspace plus the ``trust_feedback`` gate.

    Loaded ONCE at runtime-build time (one best-effort spine read alongside the
    reputation snapshot) and once per report / CLI / Studio call. An override set
    mid-session takes effect on the next build — the same staleness model as the
    reputation snapshot. An empty set with ``trust_feedback`` off is the identity:
    every path is byte-identical to Sprint 1.
    """

    by_tool: dict[str, ToolOverride] = field(default_factory=dict)
    trust_feedback: bool = False

    def for_tool(self, tool_name: str) -> ToolOverride | None:
        """The override for a tool, or None (pure auto-learning)."""
        return self.by_tool.get(tool_name)

    def reset_cutoff_for(self, tool_name: str) -> str | None:
        """The RESET cutoff for a tool, or None when it has no active reset."""
        ov = self.by_tool.get(tool_name)
        if ov is not None and ov.kind is OverrideKind.RESET:
            return ov.reset_cutoff
        return None


def apply_override(rep: ToolReputation, ov: ToolOverride | None) -> ToolReputation:
    """Fold an override into a reputation — THE shared apply function.

    The SOLE place override semantics become a :class:`ToolReputation` change, so the
    runtime provider, the hint adapter, and the visible report inherit identical
    behaviour from one call site (:meth:`LearningService._reputation_for_tool`).

    * **PIN / MUTE** -> ``replace(rep, score=NEUTRAL_SCORE, has_min_samples=False)``.
      Score 1.0 sorts first under the stable sort ``bound_tools`` applies, so the tool is
      never demoted; ``has_min_samples=False`` makes ``is_unreliable()`` and ``flaky``
      False UNCONDITIONALLY regardless of failures, so it is never flagged / hinted. PIN
      and MUTE share this numeric shape — the verb is carried separately (via
      :meth:`OverrideSet.for_tool`) so the DISPLAY can label "pinned" vs "ignored" and the
      outcome blend can exclude a muted tool. NOTE: forcing 1.0 means a muted-but-genuinely-
      bad tool sorts among the reliable tools rather than staying in place; that is accepted
      — mute is an explicit operator act, and this is the minimal change that guarantees
      "never demoted" through the EXISTING ``score_for`` / ``is_unreliable`` path with zero
      new branches in ``tools/service.py``. Mute is NOT a boost; it is an opt-out.
    * **RESET / CLEAR / None** -> ``rep`` unchanged. RESET is handled upstream (an event
      cutoff in ``_recent_counts`` / ``_outcome_signal``); CLEAR / None mean no override,
      so the reputation is byte-identical to Sprint 1.

    The ``operational_score`` and the raw ``completed`` / ``failed`` counts are preserved
    so the report can still SHOW the underlying numbers next to the "pinned" / "ignored"
    verdict.
    """
    if ov is None:
        return rep
    if ov.kind in (OverrideKind.PIN, OverrideKind.MUTE):
        return replace(rep, score=NEUTRAL_SCORE, has_min_samples=False)
    # RESET / CLEAR are not post-read reputation changes — identity here.
    return rep


class OverrideStore:
    """Durable, versioned, workspace-scoped operator-override store on the spine.

    Wraps an :class:`~himmy.entities.protocol.EntityRegistryProtocol` (the
    ``.himmy/spine.db`` spine via :class:`SpineFactory`). Each mutation is a
    ``new_version`` of a deterministic ``stable_id``, so the version chain IS the
    who-changed-what-when audit trail and a read is an O(1) ``get_latest`` per tool. A
    registry can be injected for tests (an in-memory / temp SQLite registry); the
    classmethods :meth:`for_cli` / :meth:`for_context` resolve the canonical spine for
    the CLI / runtime / Studio so all three share one store.

    Every method is best-effort: any registry error degrades to an empty
    :class:`OverrideSet` (load) or a swallowed no-op (mutation).
    """

    def __init__(self, registry: EntityRegistryProtocol) -> None:
        self._registry = registry

    # ----------------------------------------------------------------- factories
    @classmethod
    def for_cli(cls) -> OverrideStore:
        """An override store on the durable CLI spine (``.himmy/spine.db``)."""
        from himmy.entities.spine_factory import SpineFactory

        return cls(SpineFactory.for_cli())

    @classmethod
    def for_context(cls, server: bool | None = None) -> OverrideStore:
        """An override store on the spine chosen by the server-context flag.

        Both branches resolve to the SAME canonical ``.himmy/spine.db`` by default, so
        the runtime (``for_context``) and the CLI (``for_cli``) reach one store — the
        override an operator sets is the override the runtime honours.
        """
        from himmy.entities.spine_factory import SpineFactory

        return cls(SpineFactory.for_context(server=server))

    # ----------------------------------------------------------------- ids/scope
    @staticmethod
    def _subject(workspace_id: str | None) -> str:
        """The subject an override is scoped to (``default`` for a None workspace)."""
        return workspace_id or _DEFAULT_SUBJECT

    @classmethod
    def _override_stable_id(cls, workspace_id: str | None, tool_name: str) -> str:
        """Deterministic version-chain id for a (workspace, tool) override."""
        return stable_id_for(
            f"{cls._subject(workspace_id)}:{tool_name}", namespace=TOOL_OVERRIDE_KIND
        )

    @classmethod
    def _trust_stable_id(cls, workspace_id: str | None) -> str:
        """Deterministic version-chain id for a workspace's trust_feedback flag."""
        return stable_id_for(cls._subject(workspace_id), namespace=TRUST_FEEDBACK_KIND)

    @classmethod
    def _metadata(
        cls, workspace_id: str | None, *, actor: str, source: str
    ) -> dict[str, str]:
        """Spine metadata carrying tenant + audit context (queried on ``workspace_id``)."""
        return {
            "workspace_id": cls._subject(workspace_id),
            "subject_id": cls._subject(workspace_id),
            "actor": actor,
            "source": source,
        }

    # ----------------------------------------------------------------- mutations
    def set_override(
        self,
        tool_name: str,
        kind: OverrideKind,
        *,
        workspace_id: str | None = None,
        actor: str = "operator",
        source: str = "cli",
        reset_cutoff: str | None = None,
    ) -> ToolOverride | None:
        """Pin / mute / reset a tool (a new spine version). Best-effort -> None on error.

        For :attr:`OverrideKind.RESET` a ``reset_cutoff`` is stamped (defaulting to now)
        so events at/before it are forgotten. :attr:`OverrideKind.CLEAR` is routed through
        :meth:`clear_override` instead.
        """
        if kind is OverrideKind.CLEAR:
            return self.clear_override(
                tool_name, workspace_id=workspace_id, actor=actor, source=source
            )
        at = utc_now_iso()
        cutoff = (reset_cutoff or at) if kind is OverrideKind.RESET else None
        payload = {
            "tool_name": tool_name,
            "kind": kind.value,
            "reset_cutoff": cutoff,
            "actor": actor,
            "updated_at": at,
        }
        try:
            self._registry.new_version(
                stable_id=self._override_stable_id(workspace_id, tool_name),
                kind=TOOL_OVERRIDE_KIND,
                payload=payload,
                metadata=self._metadata(workspace_id, actor=actor, source=source),
            )
        except Exception:  # pragma: no cover - defensive: overrides never break a run
            logger.debug("set_override failed for %r", tool_name, exc_info=True)
            return None
        return ToolOverride(
            tool_name=tool_name,
            kind=kind,
            reset_cutoff=cutoff,
            actor=actor,
            updated_at=at,
        )

    def clear_override(
        self,
        tool_name: str,
        *,
        workspace_id: str | None = None,
        actor: str = "operator",
        source: str = "cli",
    ) -> ToolOverride | None:
        """Remove a tool's override — write a CLEAR tombstone version (audited).

        ``load`` treats the latest version being a CLEAR as "no override", so the tool
        returns to pure auto-learning while the un-pin stays in the version chain.
        """
        at = utc_now_iso()
        payload = {
            "tool_name": tool_name,
            "kind": OverrideKind.CLEAR.value,
            "reset_cutoff": None,
            "actor": actor,
            "updated_at": at,
        }
        try:
            self._registry.new_version(
                stable_id=self._override_stable_id(workspace_id, tool_name),
                kind=TOOL_OVERRIDE_KIND,
                payload=payload,
                metadata=self._metadata(workspace_id, actor=actor, source=source),
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("clear_override failed for %r", tool_name, exc_info=True)
            return None
        return ToolOverride(
            tool_name=tool_name,
            kind=OverrideKind.CLEAR,
            actor=actor,
            updated_at=at,
        )

    def set_trust_feedback(
        self,
        enabled: bool,
        *,
        workspace_id: str | None = None,
        actor: str = "operator",
        source: str = "cli",
    ) -> bool | None:
        """Flip the per-workspace ``trust_feedback`` gate (a new spine version).

        Returns the persisted value, or None on a swallowed error.
        """
        payload = {"enabled": bool(enabled), "updated_at": utc_now_iso()}
        try:
            self._registry.new_version(
                stable_id=self._trust_stable_id(workspace_id),
                kind=TRUST_FEEDBACK_KIND,
                payload=payload,
                metadata=self._metadata(workspace_id, actor=actor, source=source),
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("set_trust_feedback failed", exc_info=True)
            return None
        return bool(enabled)

    # --------------------------------------------------------------------- load
    def load(self, workspace_id: str | None = None) -> OverrideSet:
        """Load the current overrides + trust flag for a workspace (best-effort).

        Reads every ``tool_override`` record for the workspace (a JSONB-containment
        metadata query, at parity across the sqlite / postgres registry backends), keeps
        the latest version per tool, drops CLEARed tools, and resolves the latest
        ``trust_feedback`` flag. Any registry error -> an empty :class:`OverrideSet`
        (trust off) so the caller degrades to pure Sprint-1 auto-learning.
        """
        subject = self._subject(workspace_id)
        by_tool: dict[str, ToolOverride] = {}
        try:
            records = self._registry.query(
                EntityQuery(
                    kind=TOOL_OVERRIDE_KIND,
                    metadata_filters={"workspace_id": subject},
                )
            )
        except Exception:  # pragma: no cover - defensive: load never breaks a run
            logger.debug("override load failed for %r", subject, exc_info=True)
            records = []
        # Keep the highest-version record per stable_id (the current state).
        latest_by_stable: dict[str, object] = {}
        for rec in records:
            prev = latest_by_stable.get(rec.stable_id)
            if prev is None or rec.version > prev.version:  # type: ignore[attr-defined]
                latest_by_stable[rec.stable_id] = rec
        for rec in latest_by_stable.values():
            payload = getattr(rec, "payload", {}) or {}
            tool_name = payload.get("tool_name")
            kind_raw = payload.get("kind")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            try:
                kind = OverrideKind(kind_raw)
            except (ValueError, TypeError):
                continue
            if kind is OverrideKind.CLEAR:
                continue  # tombstone -> no active override for this tool
            by_tool[tool_name] = ToolOverride(
                tool_name=tool_name,
                kind=kind,
                reset_cutoff=payload.get("reset_cutoff"),
                actor=str(payload.get("actor", "operator")),
                updated_at=payload.get("updated_at"),
            )

        trust = False
        try:
            latest = self._registry.get_latest(self._trust_stable_id(workspace_id))
            if latest is not None and latest.kind == TRUST_FEEDBACK_KIND:
                trust = bool((latest.payload or {}).get("enabled", False))
        except Exception:  # pragma: no cover - defensive
            logger.debug("trust_feedback load failed for %r", subject, exc_info=True)
            trust = False

        return OverrideSet(by_tool=by_tool, trust_feedback=trust)


__all__ = [
    "TOOL_OVERRIDE_KIND",
    "TRUST_FEEDBACK_KIND",
    "TRUST_FEEDBACK_DEFAULT_WEIGHT",
    "OverrideKind",
    "OverrideSet",
    "OverrideStore",
    "ToolOverride",
    "apply_override",
]
