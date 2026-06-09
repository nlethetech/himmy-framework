"""WS4.6 — resolve the data subject (and its encryptable fields) for a record.

The consent gates need two things from any persisted object: *which subject does this
belong to?* and *which string fields carry that subject's content* (so they can be
encrypted under the subject's shreddable key). This module centralises both so a new
subject-bearing record type is handled in exactly one place — and so the gates can
**fail closed**: a record whose type is subject-bearing but whose ``subject_id`` cannot be
resolved is treated as a denial in a governed deployment, never silently persisted.
"""

from __future__ import annotations

from dataclasses import dataclass

from himmy.services.context.models import ContextSnapshot
from himmy.services.storage.models import (
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RunRecord,
)


@dataclass(frozen=True)
class _Spec:
    """How to read a subject id and the content fields for one record type."""

    subject_attr: str
    encrypted_fields: tuple[str, ...]


#: Subject-bearing typed records → (subject attribute, encryptable content fields).
#: Types absent here are treated as infrastructure (not subject data) and pass through.
_SPECS: dict[type, _Spec] = {
    RunRecord: _Spec("subject_id", ("output_text",)),
    RecommendationItem: _Spec("subject_id", ("title", "summary", "rationale")),
    ContextSnapshot: _Spec("subject_id", ()),
    MemoryObject: _Spec("subject_id", ()),
    EpisodicMemoryObject: _Spec("subject_id", ()),
}


class SubjectResolver:
    """Resolve the data subject + encryptable fields for persisted records."""

    def is_subject_bearing(self, obj: object) -> bool:
        """Whether ``obj``'s type carries subject data the gates must govern."""
        return type(obj) in _SPECS

    def subject_of(self, obj: object) -> str | None:
        """Return the record's ``subject_id`` (``None`` when absent/blank/not-bearing)."""
        spec = _SPECS.get(type(obj))
        if spec is None:
            return None
        value = getattr(obj, spec.subject_attr, None)
        return value or None

    def encrypted_fields(self, obj: object) -> tuple[str, ...]:
        """The string fields on ``obj`` that should be encrypted under the subject key."""
        spec = _SPECS.get(type(obj))
        return spec.encrypted_fields if spec is not None else ()


__all__ = ["SubjectResolver"]
