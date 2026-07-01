"""Tests for the universal prompt-cache helper (C1) + cross-provider accounting (C2).

These cover the single source of truth in ``himmy.services.inference.prompt_cache``:

* the ``CachePolicy`` request field defaults to ``None`` (byte-identical payloads);
* ``CacheCapability`` / thresholds / ``compute_cached_cost`` are importable and behave;
* ``compute_cached_cost`` takes PER-PROVIDER ``read_rate``/``write_mult`` (not one 0.1);
* the typed usage breakdown splits Anthropic and OpenAI ``usage`` correctly;
* the OpenAI manager now reads ``prompt_tokens_details.cached_tokens`` and bills the
  discount while keeping ``input_tokens`` a FULL total (the Anthropic side is pinned in
  ``tests/contracts/test_anthropic_shapes.py``).

All offline: fake usage objects and injected fake clients, no provider key, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference import pricing
from himmy.services.inference.models import (
    CachePolicy,
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ModelPrice,
)
from himmy.services.inference.openai_manager import OpenAIClientManager
from himmy.services.inference.prompt_cache import (
    DEFAULT_MIN_CACHEABLE_TOKENS,
    CacheCapability,
    UsageBreakdown,
    anthropic_cache_control,
    apply_history_cache_breakpoint,
    cache_savings_usd,
    compute_cached_cost,
    min_cacheable_tokens,
    read_anthropic_usage,
    read_openai_usage,
    resolve_cache_rates,
    stable_prefix_estimate_tokens,
)
from tests.conftest import run_async


class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


# ----------------------------------------------------------- CachePolicy field
def test_cache_policy_defaults_to_none_on_request() -> None:
    """A bare InferenceRequest has cache_policy None (no behavior change opt-in)."""
    req = InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])
    assert req.cache_policy is None


def test_cache_policy_defaults_are_conservative() -> None:
    """CachePolicy() defaults: enabled, system_and_tools, 5m, no key, model-default min."""
    policy = CachePolicy()
    assert policy.enabled is True
    assert policy.scope == "system_and_tools"
    assert policy.ttl == "5m"
    assert policy.cache_key is None
    assert policy.min_prefix_tokens is None


# ------------------------------------------------------------ threshold table
def test_unknown_model_falls_back_to_conservative_min() -> None:
    """An unknown model family gets the conservative fallback floor, never a low magic."""
    assert min_cacheable_tokens("some-future-model-x") == DEFAULT_MIN_CACHEABLE_TOKENS
    assert min_cacheable_tokens("anthropic:claude-3-5-sonnet-latest") == 1024
    assert min_cacheable_tokens("claude-3-5-haiku-latest") == 2048
    assert min_cacheable_tokens("openai:gpt-4o") == 1024


def test_policy_override_wins_over_table() -> None:
    """An explicit CachePolicy.min_prefix_tokens overrides the per-family floor."""
    assert min_cacheable_tokens("claude-3-5-haiku-latest", policy_override=10) == 10


def test_stable_prefix_estimate_counts_system_and_tools_only() -> None:
    """The estimate covers system text + bound tools, ignoring volatile user turns."""
    from himmy.services.inference.models import BoundTool

    req = InferenceRequest(
        messages=[
            InferenceMessage(role="system", content="x" * 400),
            InferenceMessage(role="user", content="y" * 4000),  # volatile, excluded
        ],
        bound_tools=[BoundTool(name="lookup", description="d" * 40)],
    )
    est = stable_prefix_estimate_tokens(req)
    # ~100 tokens from the 400-char system block, plus a small tools contribution;
    # the 4000-char user turn must NOT inflate it.
    assert 100 <= est < 200


# ----------------------------------------------- history/tail breakpoint helper
def test_anthropic_cache_control_ttl_gating() -> None:
    """The marker builder: plain 5m ephemeral, 1h only when the SDK supports it."""
    assert anthropic_cache_control(ttl="5m", ttl_supported=True) == {
        "type": "ephemeral"
    }
    assert anthropic_cache_control(ttl="1h", ttl_supported=True) == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    # 1h requested but unsupported → degrade to 5m (no ttl key).
    assert anthropic_cache_control(ttl="1h", ttl_supported=False) == {
        "type": "ephemeral"
    }


def test_apply_history_breakpoint_wraps_string_content() -> None:
    """A string tail is lifted into a text block carrying the marker."""
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    apply_history_cache_breakpoint(messages, ttl="5m", ttl_supported=False)
    assert messages[0] == {"role": "user", "content": "first"}  # earlier turn untouched
    assert messages[-1]["content"] == [
        {"type": "text", "text": "reply", "cache_control": {"type": "ephemeral"}}
    ]


def test_apply_history_breakpoint_marks_last_block_only() -> None:
    """Block-form content gets the marker on its LAST block; earlier blocks untouched."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
            ],
        }
    ]
    apply_history_cache_breakpoint(messages, ttl="5m", ttl_supported=False)
    content = messages[-1]["content"]
    assert "cache_control" not in content[0]
    assert content[-1]["cache_control"] == {"type": "ephemeral"}


def test_apply_history_breakpoint_empty_string_not_marked() -> None:
    """An empty-string tail is not a cacheable prefix — left intact, never wrapped."""
    messages = [{"role": "assistant", "content": ""}]
    apply_history_cache_breakpoint(messages, ttl="5m", ttl_supported=False)
    assert messages[-1] == {"role": "assistant", "content": ""}


def test_apply_history_breakpoint_no_duplicate_on_already_marked_tail() -> None:
    """A tail block that ALREADY carries cache_control is left as-is (budget respected)."""
    marked = {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    messages = [{"role": "system", "content": [marked]}]
    apply_history_cache_breakpoint(messages, ttl="1h", ttl_supported=True)
    # Not overwritten to a 1h marker, not duplicated — still the single 5m marker.
    assert messages[-1]["content"] == [marked]


def test_apply_history_breakpoint_empty_messages_noop() -> None:
    """No messages → no-op (nothing to mark, never raises)."""
    messages: list[dict[str, Any]] = []
    apply_history_cache_breakpoint(messages, ttl="5m", ttl_supported=False)
    assert messages == []


# -------------------------------------------------------- usage breakdown split
def test_read_anthropic_usage_splits_uncached_read_creation() -> None:
    """Anthropic input_tokens is the UNCACHED count; reads/writes are separate lines."""
    usage = _Obj(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=2,
    )
    b = read_anthropic_usage(usage)
    assert b.uncached_input == 10
    assert b.cache_read == 30
    assert b.cache_creation == 2
    assert b.output == 5
    assert b.total_input == 42  # full total reconstructed for back-compat


def test_read_openai_usage_subtracts_cached_from_prompt_total() -> None:
    """OpenAI prompt_tokens is the FULL total; cached_tokens is nested + subtracted."""
    usage = _Obj(
        prompt_tokens=100,
        completion_tokens=12,
        prompt_tokens_details=_Obj(cached_tokens=40),
    )
    b = read_openai_usage(usage)
    assert b.uncached_input == 60  # 100 total - 40 cached
    assert b.cache_read == 40
    assert b.cache_creation == 0  # OpenAI has no separate write line
    assert b.output == 12
    assert b.total_input == 100  # full total preserved


def test_read_openai_usage_handles_dict_details_and_absence() -> None:
    """cached_tokens reads from a dict detail and defaults to 0 when absent."""
    dict_details = read_openai_usage(
        _Obj(prompt_tokens=50, completion_tokens=3, prompt_tokens_details={"cached_tokens": 20})
    )
    assert dict_details.cache_read == 20 and dict_details.uncached_input == 30
    no_details = read_openai_usage(_Obj(prompt_tokens=50, completion_tokens=3))
    assert no_details.cache_read == 0 and no_details.uncached_input == 50
    assert read_openai_usage(None).total_input == 0


# ----------------------------------------------------- per-provider cache rates
def test_resolve_cache_rates_are_provider_specific_not_one_constant() -> None:
    """Anthropic and OpenAI resolve DIFFERENT rates — the C2 must-fix."""
    price = ModelPrice(input_per_1k=1.0, output_per_1k=2.0)
    a_read, a_write_5m = resolve_cache_rates(
        price, capability=CacheCapability.ANTHROPIC_EXPLICIT, ttl="5m"
    )
    assert (a_read, a_write_5m) == (0.1, 1.25)
    _, a_write_1h = resolve_cache_rates(
        price, capability=CacheCapability.ANTHROPIC_EXPLICIT, ttl="1h"
    )
    assert a_write_1h == 2.0
    o_read, o_write = resolve_cache_rates(
        price, capability=CacheCapability.OPENAI_AUTOMATIC
    )
    assert (o_read, o_write) == (0.5, 0.0)
    # A non-supporting manager gets full-rate (no discount, no premium).
    n_read, n_write = resolve_cache_rates(price, capability=CacheCapability.NONE)
    assert (n_read, n_write) == (1.0, 1.0)


def test_per_model_multiplier_overrides_family_default() -> None:
    """An explicit per-model cache tier overrides the provider-family constant."""
    price = ModelPrice(input_per_1k=1.0, output_per_1k=2.0)
    read, write = resolve_cache_rates(
        price,
        capability=CacheCapability.OPENAI_AUTOMATIC,
        cache_read_multiplier=0.25,
        cache_write_multiplier=0.0,
    )
    assert (read, write) == (0.25, 0.0)


# ----------------------------------------------------------- compute_cached_cost
def test_compute_cached_cost_all_uncached_matches_plain_cost() -> None:
    """With no cache tokens, the cache-aware cost equals the plain non-cached cost."""
    price = ModelPrice(input_per_1k=3.0, output_per_1k=15.0)
    b = UsageBreakdown(uncached_input=100, output=20)
    cost = compute_cached_cost(price, b, read_rate=0.1, write_mult=1.25)
    assert cost == pytest.approx(price.cost(input_tokens=100, output_tokens=20))


def test_compute_cached_cost_bills_reads_cheap_writes_premium() -> None:
    """Reads bill at read_rate, writes at write_mult of the input rate — not 1.0x."""
    price = ModelPrice(input_per_1k=3.0, output_per_1k=15.0)
    b = UsageBreakdown(uncached_input=10, cache_read=1000, cache_creation=100, output=5)
    cost = compute_cached_cost(price, b, read_rate=0.1, write_mult=1.25)
    expected = price.cost(input_tokens=10, output_tokens=5) + (3.0 / 1000.0) * (
        1000 * 0.1 + 100 * 1.25
    )
    assert cost == pytest.approx(expected)
    # And strictly cheaper than billing every cache token at the full input rate.
    full = price.cost(input_tokens=10 + 1000 + 100, output_tokens=5)
    assert cost < full


def test_cache_savings_is_net_of_write_premium() -> None:
    """Savings = full-rate cost of cached tokens minus the discounted cost (net)."""
    price = ModelPrice(input_per_1k=3.0, output_per_1k=15.0)
    b = UsageBreakdown(cache_read=1000, cache_creation=100)
    saved = cache_savings_usd(price, b, read_rate=0.1, write_mult=1.25)
    expected = (3.0 / 1000.0) * ((1000 + 100) - (1000 * 0.1 + 100 * 1.25))
    assert saved == pytest.approx(expected)


# ------------------------------------------- OpenAI manager end-to-end (C2 fix)
class _FakeCompletions:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.seen.update(kwargs)
        return self._result


class _FakeClient:
    def __init__(self, result: Any) -> None:
        self.chat = _Obj(completions=_FakeCompletions(result))


def _completion_with_cache(
    *, prompt: int, completion: int, cached: int
) -> _Obj:
    message = _Obj(content="cached reply", tool_calls=None)
    usage = _Obj(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=_Obj(cached_tokens=cached),
    )
    return _Obj(choices=[_Obj(message=message)], usage=usage)


def test_openai_manager_bills_cached_tokens_at_discount() -> None:
    """The OpenAI manager no longer bills cached input at full rate (was ignored)."""
    client = _FakeClient(
        _completion_with_cache(prompt=1000, completion=20, cached=800)
    )
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    req = InferenceRequest(
        model_key="gpt-4o-mini",
        messages=[
            InferenceMessage(role="system", content="be brief"),
            InferenceMessage(role="user", content="hello"),
        ],
    )
    resp = run_async(mgr.generate(req))

    assert resp.status == InferenceStatus.SUCCESS
    # Full input total preserved for downstream consumers.
    assert resp.input_tokens == 1000
    assert resp.output_tokens == 20

    price = pricing.price_for("openai:gpt-4o-mini")
    # 200 uncached + 20 output at full rate; 800 cached reads at 0.5x of input rate.
    expected = price.cost(input_tokens=200, output_tokens=20) + (
        price.input_per_1k / 1000.0
    ) * (800 * 0.5)
    assert resp.cost == pytest.approx(expected)
    # Strictly cheaper than the old full-rate-on-everything billing.
    assert resp.cost < price.cost(input_tokens=1000, output_tokens=20)
    # Metadata carries the breakdown + savings.
    assert resp.metadata["cache_read_tokens"] == 800
    assert resp.metadata["cache_creation_tokens"] == 0
    assert resp.metadata["cache_savings_usd"] > 0.0


def test_openai_manager_no_cache_leaves_cost_and_metadata_unchanged() -> None:
    """A response with no cached tokens bills the plain cost and stamps no cache meta."""
    client = _FakeClient(
        _completion_with_cache(prompt=100, completion=12, cached=0)
    )
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    req = InferenceRequest(
        model_key="gpt-4o-mini",
        messages=[InferenceMessage(role="user", content="hi")],
    )
    resp = run_async(mgr.generate(req))
    price = pricing.price_for("openai:gpt-4o-mini")
    assert resp.cost == pytest.approx(
        price.cost(input_tokens=100, output_tokens=12)
    )
    assert "cache_read_tokens" not in resp.metadata
