"""Tests for the framework metadata-key vocabulary (core.metadata).

These confirm the documented TypedDicts behave as a typed *view* over the open
``dict[str, Any]`` metadata fields: a constructed TypedDict round-trips as a plain
dict, merges transparently, and never constrains extra keys (extensibility intact).
"""

from __future__ import annotations

from himmy.core.metadata import (
    AssistantMessageMetadata,
    RouteMetadata,
)
from himmy.services.inference.models import InferenceMessage


def test_assistant_metadata_round_trips_as_plain_dict() -> None:
    """A TypedDict instance is a plain dict and assigns to an open metadata field."""
    meta: AssistantMessageMetadata = {
        "request_id": "r1",
        "status": "SUCCESS",
        "output_tokens": 7,
    }
    # Widens cleanly to the model's open dict[str, Any] field.
    msg = InferenceMessage(role="assistant", content="hi", metadata=dict(meta))
    assert msg.metadata["request_id"] == "r1"
    assert msg.metadata["output_tokens"] == 7


def test_metadata_stays_extensible_with_unknown_keys() -> None:
    """Unknown keys remain valid — the vocabulary documents, it does not constrain."""
    meta = {"cache_hit": True, "custom_extension_key": {"any": "value"}}
    merged = {**RouteMetadata(route_index=0), **meta}
    assert merged["custom_extension_key"] == {"any": "value"}
    assert merged["cache_hit"] is True
    assert merged["route_index"] == 0
