"""Q2: provider-lane keying — how ``model_key`` maps to a claim lane.

The lane discriminator is the mechanical signal the claim predicate keys on so a stalled
LOCAL-model probe cannot block claiming CLOUD runs. These pin the mapping (the plan acceptance
"model_key -> provider/probe mapping is specified").
"""

from __future__ import annotations

import pytest

from himmy.services.storage.run_lane import (
    LANE_CLOUD,
    LANE_DEFAULT,
    LANE_LOCAL,
    lane_for_model_key,
)


@pytest.mark.parametrize(
    ("model_key", "expected"),
    [
        ("ollama:llama3", LANE_LOCAL),
        ("OLLAMA:Llama3", LANE_LOCAL),  # case-insensitive provider
        ("claude-cli:haiku", LANE_LOCAL),
        ("claude_cli:haiku", LANE_LOCAL),
        ("himalaya:0.5b", LANE_LOCAL),
        ("local:whatever", LANE_LOCAL),
        ("anthropic:claude-3-5-sonnet", LANE_CLOUD),
        ("openai:gpt-4o", LANE_CLOUD),
        ("openrouter:anthropic/claude", LANE_CLOUD),
        ("gateway:x", LANE_CLOUD),
        ("gpt-4o", LANE_CLOUD),  # bare model, no prefix -> cloud (safe default)
        ("unknownvendor:model", LANE_CLOUD),  # unknown prefix -> cloud (fail-open claim)
        (None, LANE_DEFAULT),
        ("", LANE_DEFAULT),
        ("   ", LANE_DEFAULT),
    ],
)
def test_lane_for_model_key(model_key: str | None, expected: str) -> None:
    assert lane_for_model_key(model_key) == expected


def test_unknown_provider_is_cloud_not_local() -> None:
    """An unknown provider must NOT land in the local lane (it would be wrongly probe-gated)."""
    assert lane_for_model_key("mystery:m") != LANE_LOCAL
