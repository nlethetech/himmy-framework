"""WS4.6 — consent & purpose-limitation: the policy-decision-point (PDP) core.

A single pure decision function — :meth:`ConsentPolicy.decide` — that mirrors
:class:`himmy.api.auth.rbac.AccessPolicy`: frozen, data-driven, built from the environment
by :func:`build_consent_policy`, and **OFF by default**. When ``governed`` is ``False``
(the offline / zero-config path) ``decide`` returns :attr:`Effect.ALLOW` unconditionally —
and the enforcement gates are never even constructed — exactly how RBAC/encryption stay off
until a deployment opts in. When governed, an unknown or denied ``(subject, purpose)``
defaults to :attr:`Effect.DENY` for ``RETAIN``/``TRAIN`` and :attr:`Effect.EPHEMERAL` for
``INFER`` (secure-by-default *where it counts*).

Consent itself is recorded by
:class:`himmy.services.governance.consent_ledger.ConsentLedger` as immutable,
content-addressed ``EntityRecord``s; this module is the pure decision layer over the latest
recorded state and has **no storage/runtime coupling**, so ``decide`` is unit-testable with
zero I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from himmy.core.ids import utc_now_iso
from himmy.entities.records import stable_id_for

#: EntityRecord kind for a consent decision (one version per assertion).
CONSENT_KIND = "consent"


class Purpose(str, Enum):
    """A purpose a subject's data may be used for (purpose-limitation axis)."""

    RETAIN = "retain"  # write subject-identifying data to storage / event-log / memory / spine
    TRAIN = "train"  # capture raw I/O / include in a dataset / replay / downstream-derived use
    INFER = "infer"  # use subject data in a live model call (no persistence implied)
    ANALYTICS = "analytics"
    SHARE = "share"
    IMPROVE = "improve"


class ConsentState(str, Enum):
    """The latest recorded state of a subject's consent for one purpose."""

    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"  # no record on file


class Effect(str, Enum):
    """What a gate must do for a ``(subject, purpose)`` given the decision."""

    ALLOW = "allow"  # persist / use freely
    EPHEMERAL = "ephemeral"  # use for the live request only — never written
    DENY = "deny"  # refuse


@dataclass(frozen=True)
class Decision:
    """The PDP verdict for one ``(subject, purpose)``."""

    subject_id: str
    purpose: Purpose
    effect: Effect
    reason: str = ""

    @property
    def allowed(self) -> bool:
        """Whether the data may be **persisted/retained** for this purpose.

        Only :attr:`Effect.ALLOW` permits persistence; :attr:`Effect.EPHEMERAL` permits
        live use but never a write, and :attr:`Effect.DENY` refuses both.
        """
        return self.effect is Effect.ALLOW


def consent_stable_id(
    subject_id: str, purpose: Purpose | str, *, workspace_id: str | None = None
) -> str:
    """Deterministic stable id for a ``(workspace, subject, purpose)`` version-chain.

    ``workspace_id`` scopes the identity to one tenant so two tenants that share a
    ``subject_id`` never collide. It is kept optional and, when ``None``, the id is the
    legacy ``(subject, purpose)`` value so pre-existing records and ungoverned callers
    still resolve unchanged; governed callers must always pass a concrete workspace.
    """
    value = purpose.value if isinstance(purpose, Purpose) else str(purpose)
    if workspace_id is None:
        key = f"{subject_id}|{value}"
    else:
        key = f"{workspace_id}|{subject_id}|{value}"
    return stable_id_for(key, namespace=CONSENT_KIND)


class ConsentRecord(BaseModel):
    """One version in a subject's consent chain for a single purpose."""

    subject_id: str
    purpose: Purpose
    state: ConsentState
    workspace_id: str | None = None
    actor: str = ""  # who recorded this decision
    source: str = ""  # e.g. "api" | "cli" | "import"
    basis: str | None = None  # legal basis / free note
    expires_at: str | None = None
    recorded_at: str = Field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class ConsentDefaults:
    """Per-purpose effect for an **UNKNOWN** (no record) subject when governed."""

    per_purpose: dict[Purpose, Effect] = field(default_factory=dict)
    fallback: Effect = Effect.DENY

    def effect_for(self, purpose: Purpose) -> Effect:
        """The default effect for ``purpose`` (or :attr:`fallback`)."""
        return self.per_purpose.get(purpose, self.fallback)


#: Secure-by-default rule for a governed deployment: no opt-in ⇒ no retention/training,
#: but a live inference may still run ephemerally (the data is used, never written).
DEFAULT_CONSENT_DEFAULTS = ConsentDefaults(
    per_purpose={
        Purpose.RETAIN: Effect.DENY,
        Purpose.TRAIN: Effect.DENY,
        Purpose.INFER: Effect.EPHEMERAL,
        Purpose.ANALYTICS: Effect.DENY,
        Purpose.SHARE: Effect.DENY,
        Purpose.IMPROVE: Effect.DENY,
    },
    fallback=Effect.DENY,
)


def _effect_for_state(
    state: ConsentState, purpose: Purpose, defaults: ConsentDefaults
) -> Effect:
    """Map a recorded state to an effect (UNKNOWN falls to the per-purpose default)."""
    if state is ConsentState.GRANTED:
        return Effect.ALLOW
    if state in (ConsentState.DENIED, ConsentState.WITHDRAWN):
        return Effect.DENY
    return defaults.effect_for(purpose)


@dataclass(frozen=True)
class ConsentPolicy:
    """Pure, frozen decision function over a recorded consent state.

    ``governed`` is the offline-first switch (mirrors the RBAC authenticator bypass): when
    ``False`` every decision is :attr:`Effect.ALLOW` so the zero-config path is unchanged.
    """

    governed: bool = False
    defaults: ConsentDefaults = DEFAULT_CONSENT_DEFAULTS

    def decide(
        self,
        subject_id: str,
        purpose: Purpose,
        state: ConsentState,
        *,
        surface: str | None = None,
    ) -> Decision:
        """Return the :class:`Decision` for ``(subject_id, purpose)`` at ``state``."""
        if not self.governed:
            return Decision(subject_id, purpose, Effect.ALLOW, reason="ungoverned")
        effect = _effect_for_state(state, purpose, self.defaults)
        where = f" @{surface}" if surface else ""
        return Decision(
            subject_id, purpose, effect, reason=f"{state.value}->{effect.value}{where}"
        )


def _load_defaults(path: str | None) -> ConsentDefaults:
    """Load ``ConsentDefaults`` from a JSON ``{purpose: effect}`` file, or the built-in."""
    if not path:
        return DEFAULT_CONSENT_DEFAULTS
    raw = json.loads(Path(path).expanduser().read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"HIMMY_CONSENT_FILE {path} must be a JSON object")
    per_purpose = dict(DEFAULT_CONSENT_DEFAULTS.per_purpose)
    fallback = DEFAULT_CONSENT_DEFAULTS.fallback
    for key, value in raw.items():
        try:
            effect = Effect(str(value).lower())
        except ValueError:
            continue
        if key in ("*", "default"):
            fallback = effect
        else:
            try:
                per_purpose[Purpose(str(key).lower())] = effect
            except ValueError:
                continue
    return ConsentDefaults(per_purpose=per_purpose, fallback=fallback)


def _is_on(value: str) -> bool:
    """Whether an env value means 'on'."""
    return value.strip().lower() in ("1", "on", "true", "yes")


def build_consent_policy() -> ConsentPolicy:
    """Select the policy from env: ``HIMMY_CONSENT`` (governed) + ``HIMMY_CONSENT_FILE``."""
    if not _is_on(os.environ.get("HIMMY_CONSENT", "")):
        return ConsentPolicy(governed=False)
    return ConsentPolicy(
        governed=True, defaults=_load_defaults(os.environ.get("HIMMY_CONSENT_FILE"))
    )


__all__ = [
    "CONSENT_KIND",
    "Purpose",
    "ConsentState",
    "Effect",
    "Decision",
    "ConsentRecord",
    "ConsentDefaults",
    "ConsentPolicy",
    "DEFAULT_CONSENT_DEFAULTS",
    "consent_stable_id",
    "build_consent_policy",
]
