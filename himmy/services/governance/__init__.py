"""Data-governance kernel: retention/erasure (WS4.2) and consent (WS4.6).

Offline-first: importing this package wires nothing. Consent enforcement is opt-in via
``HIMMY_CONSENT`` and is constructed only in the governed branch of the API container.
"""

from __future__ import annotations

from himmy.services.governance.consent import (
    CONSENT_KIND,
    DEFAULT_CONSENT_DEFAULTS,
    ConsentDefaults,
    ConsentPolicy,
    ConsentRecord,
    ConsentState,
    Decision,
    Effect,
    Purpose,
    build_consent_policy,
    consent_stable_id,
)
from himmy.services.governance.consent_ledger import ConsentLedger
from himmy.services.governance.consent_registry import ConsentAwareRegistry
from himmy.services.governance.consent_resolver import SubjectResolver
from himmy.services.governance.consent_storage import (
    GATED_SAVE_METHODS,
    ConsentGatedStorage,
)
from himmy.services.governance.retention import (
    ERASURE_KIND,
    RetentionService,
    SubjectKeyVault,
)
from himmy.services.governance.training_export import ConsentFilteredExporter

__all__ = [
    # consent (WS4.6)
    "CONSENT_KIND",
    "Purpose",
    "ConsentState",
    "Effect",
    "Decision",
    "ConsentRecord",
    "ConsentDefaults",
    "ConsentPolicy",
    "DEFAULT_CONSENT_DEFAULTS",
    "ConsentLedger",
    "consent_stable_id",
    "build_consent_policy",
    # consent enforcement (WS4.6)
    "SubjectResolver",
    "ConsentGatedStorage",
    "GATED_SAVE_METHODS",
    "ConsentAwareRegistry",
    "ConsentFilteredExporter",
    # retention / erasure (WS4.2)
    "ERASURE_KIND",
    "RetentionService",
    "SubjectKeyVault",
]
