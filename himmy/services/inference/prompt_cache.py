"""Universal prompt-cache helper: the single source of truth for prefix caching.

This module is the shared contract every provider adapter and the cost accounting
agree on. It is DISTINCT from the *response* cache in
:mod:`himmy.services.inference.cache` (which skips the whole call): here we model the
provider-side PROMPT/prefix cache — the stable ``system + bound_tools`` prefix a
provider can serve cheaply on a repeat render. The two coexist.

It deliberately has NO network and NO provider-SDK imports, so it stays import-safe
and offline-first. It provides:

* :class:`CacheCapability` — how a concrete manager participates (or no-ops).
* :data:`MIN_CACHEABLE_TOKENS` — a per-model-FAMILY minimum-prefix table with a
  conservative unknown fallback. The exact minima are EXTERNAL, provider-versioned
  facts that drift; this table is best-effort, not authoritative, and any miss only
  produces a harmless no-op marker (we never mark a sub-threshold prefix).
* :func:`stable_prefix_estimate_tokens` — a deterministic, length-based estimate of
  the cacheable prefix size (system text + bound-tools schemas), reusing the
  ``~4 chars/token`` convention from ``client_manager._estimate_tokens``.
* :class:`UsageBreakdown` + :func:`read_anthropic_usage` / :func:`read_openai_usage`
  — typed split of a provider ``usage`` object into
  ``{uncached_input, cache_read, cache_creation, output}``.
* :func:`compute_cached_cost` — the ONE cost function both adapters call, taking a
  PER-PROVIDER ``read_rate`` / ``write_mult`` (never a single shared constant).
* :func:`resolve_cache_rates` — resolves those rates from an optional per-model
  cache-rate tier (``cache_read_multiplier`` / ``cache_write_multiplier`` on a price
  entry) with provider-family constants as the fallback.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from himmy.services.inference.models import (
    BoundTool,
    InferenceRequest,
    ModelPrice,
)

#: The lowest ``anthropic`` SDK version known to accept ``cache_control`` with a
#: ``ttl`` field and ``extra_headers`` on ``messages.create`` (the extended-cache-ttl
#: surface). Below this — or when the SDK can't be introspected — the adapter degrades
#: to the default 5m ephemeral cache, which every cache-capable version accepts.
_ANTHROPIC_TTL_FLOOR: tuple[int, int, int] = (0, 49, 0)


class CacheCapability(str, Enum):
    """How a concrete client manager participates in prompt caching.

    Resolved at the call site via ``getattr(manager, 'cache_capability',
    CacheCapability.NONE)`` so a manager that never sets it (or a custom one) fails
    safe to a no-op rather than raising.

    * ``NONE`` — the manager ignores ``cache_policy`` entirely (all local managers,
      Gemini-native, stub, replaying/recording).
    * ``ANTHROPIC_EXPLICIT`` — Anthropic, which takes an explicit ``cache_control``
      breakpoint on the last system block.
    * ``OPENAI_AUTOMATIC`` — OpenAI-family (OpenAI/Groq/Together and OpenRouter with
      an openai-backed model), which caches automatically on a stable prefix; the
      adapter's job is to guarantee prefix stability and optionally pass a routing key.
    * ``OPENROUTER_PASSTHROUGH`` — OpenRouter forwarding to an Anthropic/Gemini
      backend, which requires block-form ``cache_control`` passthrough.
    """

    NONE = "NONE"
    ANTHROPIC_EXPLICIT = "ANTHROPIC_EXPLICIT"
    OPENAI_AUTOMATIC = "OPENAI_AUTOMATIC"
    OPENROUTER_PASSTHROUGH = "OPENROUTER_PASSTHROUGH"


#: Conservative fallback minimum when the model family is unknown. Chosen high so a
#: sub-threshold prefix is NEVER marked cacheable (a no-op marker would just waste a
#: breakpoint; a missed cache is cheaper than an incorrect one).
DEFAULT_MIN_CACHEABLE_TOKENS = 4096

#: Per-model-FAMILY minimum cacheable-prefix token floors. Keyed on a lowercase
#: family prefix matched against the resolved model id (longest match wins). These
#: are EXTERNAL provider facts that drift over time — the source numbers were already
#: stale once — so this is intentionally a best-effort table with a conservative
#: family fallback (:data:`DEFAULT_MIN_CACHEABLE_TOKENS`), NOT an authoritative spec.
#: Update via the same review that touches the provider adapters.
MIN_CACHEABLE_TOKENS: dict[str, int] = {
    # Anthropic explicit cache minimums: 1024 for most, 2048 for the small Haikus.
    "claude-3-5-haiku": 2048,
    "claude-3-haiku": 2048,
    "claude-haiku": 2048,
    "claude": 1024,
    "anthropic/claude": 1024,
    # OpenAI-family automatic caching kicks in at ~1024 prompt tokens.
    "gpt-4o": 1024,
    "gpt-4.1": 1024,
    "gpt-4": 1024,
    "gpt-3.5": 1024,
    "o1": 1024,
    "o3": 1024,
    "o4": 1024,
    "openai/": 1024,
}


#: Provider-family default cache multipliers, applied to ``input_per_1k`` when a
#: per-model cache tier is absent in the price table. CRITICAL: these are NOT one
#: shared constant — Anthropic reads at 0.1x but writes at a 5m/1h premium, while the
#: OpenAI family reads at ~0.5x with no write premium. Using one rate would
#: systematically mis-bill an entire provider cluster.
_ANTHROPIC_READ_RATE = 0.1
_ANTHROPIC_WRITE_MULT_5M = 1.25
_ANTHROPIC_WRITE_MULT_1H = 2.0
_OPENAI_READ_RATE = 0.5
_OPENAI_WRITE_MULT = 0.0  # OpenAI-family has no separate cache-write line.


def _family_min_tokens(model: str) -> int:
    """Resolve the min cacheable-prefix floor for a model id (longest-prefix match)."""
    lowered = (model or "").lower()
    best_key = ""
    for key in MIN_CACHEABLE_TOKENS:
        if lowered.startswith(key) and len(key) > len(best_key):
            best_key = key
        # Also match a family that appears after a provider prefix (e.g.
        # "anthropic:claude-..." or "openai:gpt-..."): compare the tail too.
        elif key in lowered and len(key) > len(best_key) and key not in ("o1",):
            best_key = key
    return MIN_CACHEABLE_TOKENS.get(best_key, DEFAULT_MIN_CACHEABLE_TOKENS)


def min_cacheable_tokens(model: str, policy_override: int | None = None) -> int:
    """The minimum stable-prefix token count for ``model`` to be worth caching.

    ``policy_override`` (from ``CachePolicy.min_prefix_tokens``) wins when set; else
    the per-family table is consulted with the conservative unknown fallback.
    """
    if policy_override is not None:
        return int(policy_override)
    return _family_min_tokens(model)


def _estimate_tokens(text: str) -> int:
    """Deterministic, length-based token estimate (~4 chars per token, min 0).

    Mirrors ``client_manager._estimate_tokens`` but floors at 0 for empty text so an
    empty prefix estimates to 0 (a missing system message is not "1 token cacheable").
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _tools_text(bound_tools: list[BoundTool]) -> str:
    """A stable text rendering of the bound-tools array for prefix estimation.

    Tool names are sorted so the estimate is order-independent (matching the
    name-sorted stability guarantee the adapters provide). Only name/description/
    schema keys contribute — enough to approximate the serialized tool size.
    """
    parts: list[str] = []
    for tool in sorted(bound_tools, key=lambda t: t.name):
        parts.append(tool.name)
        parts.append(tool.description or "")
        parts.append(str(tool.args_json_schema or ""))
        if tool.output_json_schema:
            parts.append(str(tool.output_json_schema))
    return "\n".join(parts)


def stable_prefix_estimate_tokens(request: InferenceRequest) -> int:
    """Estimate the stable-prefix (system + bound_tools) token count, deterministically.

    NO network ``count_tokens`` call — a borderline misgate only yields a harmless
    no-op marker. The estimate covers every system message plus the bound-tools
    schemas, which together form the provider-agnostic cacheable prefix.
    """
    system_parts = [
        m.content for m in request.messages if m.role == "system" and m.content
    ]
    system_text = "\n\n".join(system_parts)
    tools_text = _tools_text(request.bound_tools)
    return _estimate_tokens(system_text) + _estimate_tokens(tools_text)


def cache_policy_active(request: InferenceRequest) -> bool:
    """True only when the request carries a non-``None`` policy with ``enabled=True``.

    This is the SINGLE branch every adapter keys on. When it is ``False`` (the default,
    ``cache_policy is None``), the adapter must emit a byte-identical payload to the
    no-cache path — that invariant is what the provider contract tests pin.
    """
    policy = request.cache_policy
    return policy is not None and policy.enabled


def should_cache_prefix(
    request: InferenceRequest, model: str, capability: CacheCapability
) -> bool:
    """Decide whether to mark the stable system+tools prefix as cacheable.

    True iff the policy is active, the manager supports caching, the requested scope
    includes the system prefix, and the deterministic prefix-size estimate clears the
    per-model floor. A borderline misgate only produces a harmless no-op marker, so this
    is intentionally conservative (unknown families fall back to a high floor).
    """
    if capability is CacheCapability.NONE:
        return False
    if not cache_policy_active(request):
        return False
    policy = request.cache_policy
    assert policy is not None  # narrowed by cache_policy_active
    if policy.scope == "none":
        return False
    estimate = stable_prefix_estimate_tokens(request)
    floor = min_cacheable_tokens(model, policy.min_prefix_tokens)
    return estimate >= floor


def resolve_cache_capability(
    manager: Any, model_key: str = "default"
) -> CacheCapability:
    """Resolve a client manager's prompt-cache capability at the call site.

    This is the single resolution rule the runtime uses to decide whether attaching a
    :class:`~himmy.services.inference.models.CachePolicy` to a request could ever take
    effect. It is deliberately ``getattr``-based (a :class:`Protocol` cannot carry a
    runtime-default attribute) so EVERY manager — including custom/legacy ones and the
    local no-op managers (Stub/Ollama/ClaudeCli/Himalaya/PydanticAI/Gateway/Replaying)
    that never set the attribute — fails safe to :attr:`CacheCapability.NONE`.

    Wrapping managers (multi-provider/routing) dispatch a given ``model_key`` to a
    concrete sub-manager whose capability is the one that matters; they expose a
    ``cache_capability_for(model_key)`` method which, when present, is consulted first
    so the runtime sees the capability of the backend that will actually serve the key.
    A wrapper without that method (or any error during resolution) degrades to ``NONE``.
    """
    resolver = getattr(manager, "cache_capability_for", None)
    if callable(resolver):
        try:
            resolved = resolver(model_key)
        except Exception:  # noqa: BLE001 - capability resolution never breaks a run
            return CacheCapability.NONE
        if isinstance(resolved, CacheCapability):
            return resolved
        return CacheCapability.NONE
    capability = getattr(manager, "cache_capability", CacheCapability.NONE)
    if isinstance(capability, CacheCapability):
        return capability
    return CacheCapability.NONE


def _parse_version(version: str | None) -> tuple[int, int, int] | None:
    """Parse a ``major.minor.patch`` SDK version string, tolerating extra suffixes."""
    if not version:
        return None
    head = version.strip().split("+", 1)[0].split("-", 1)[0]
    parts = head.split(".")
    nums: list[int] = []
    for part in parts[:3]:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        nums.append(int(digits))
    if not nums:
        return None
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def anthropic_ttl_supported(sdk_version: str | None = None) -> bool:
    """True when the installed ``anthropic`` SDK is known to accept ``ttl`` + headers.

    Degrades safe: if the SDK isn't importable, carries no ``__version__``, or the string
    can't be parsed, this returns ``False`` and the adapter falls back to the 5m cache
    (no ``ttl`` field, no beta header) — never breaking the request on uncertainty.
    """
    version = sdk_version
    if version is None:
        try:
            import anthropic  # type: ignore

            version = getattr(anthropic, "__version__", None)
        except Exception:  # noqa: BLE001 - any import failure → degrade to 5m
            return False
    parsed = _parse_version(version)
    if parsed is None:
        return False
    return parsed >= _ANTHROPIC_TTL_FLOOR


#: Lowercased model-id prefixes that name an OpenAI-family model (so ``prompt_cache_key``
#: is accepted). Used directly and as OpenRouter ``openai/...`` underlying ids.
_OPENAI_FAMILY_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "chatgpt",
    "text-",
    "davinci",
)


def is_openai_family_model(model: str) -> bool:
    """True when ``model`` names an OpenAI-family model (accepts ``prompt_cache_key``).

    Matches both bare OpenAI ids (``gpt-4o``, ``o3-mini``) and OpenRouter's
    ``openai/...`` slash form. OpenRouter ids for other backends (``anthropic/...``,
    ``google/...``) are NOT OpenAI-family — those route through block-form passthrough.
    """
    lowered = (model or "").lower()
    if lowered.startswith("openai/"):
        return True
    return lowered.startswith(_OPENAI_FAMILY_PREFIXES)


def openrouter_passthrough_backend(model: str) -> str | None:
    """For an OpenRouter ``vendor/model`` id, the backend that takes block-form caching.

    Returns ``"anthropic"`` for ``anthropic/...`` and ``"google"`` for ``google/...``
    ids (which accept a ``cache_control`` block on the system message via OpenRouter's
    passthrough), else ``None`` (openai-backed and everything else rely on automatic
    prefix caching).
    """
    lowered = (model or "").lower()
    if lowered.startswith("anthropic/"):
        return "anthropic"
    if lowered.startswith("google/"):
        return "google"
    return None


def anthropic_system_blocks(
    system: str, *, ttl: str, ttl_supported: bool
) -> list[dict[str, Any]]:
    """Build the block-form ``system`` with one ``cache_control`` breakpoint.

    Anthropic renders ``tools -> system -> messages``, so a single breakpoint on the
    last (here only) system block caches BOTH tools and system — the whole stable prefix.
    A ``1h`` TTL is emitted only when the SDK is known to support it; otherwise the block
    carries a plain 5m ``ephemeral`` marker (no ``ttl`` key).
    """
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl == "1h" and ttl_supported:
        cache_control["ttl"] = "1h"
    return [{"type": "text", "text": system, "cache_control": cache_control}]


class UsageBreakdown(BaseModel):
    """A typed split of a provider ``usage`` object into cache-aware token buckets.

    * ``uncached_input`` — fresh (non-cached) input tokens billed at the full input rate.
    * ``cache_read`` — input tokens served from the prompt cache (billed at the
      provider read rate, a discount).
    * ``cache_creation`` — input tokens written INTO the cache (Anthropic only; billed
      at a write premium). OpenAI-family has no separate write line, so this is 0.
    * ``output`` — generated output tokens, always billed at the full output rate.

    ``total_input`` reconstructs the FULL input total (uncached + reads + writes) so
    ``InferenceResponse.input_tokens`` stays a faithful full count for back-compat.
    """

    uncached_input: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    output: int = 0

    @property
    def total_input(self) -> int:
        """The full input total (uncached + cache reads + cache creations)."""
        return self.uncached_input + self.cache_read + self.cache_creation


def read_anthropic_usage(usage: Any) -> UsageBreakdown:
    """Split an Anthropic ``usage`` object into a :class:`UsageBreakdown`.

    Anthropic returns ``input_tokens`` as the UNCACHED input only, with
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` reported separately
    (NOT folded). We surface them split so reads bill the discount and writes the
    premium; the response's ``input_tokens`` is reconstructed as the full total.
    """
    if usage is None:
        return UsageBreakdown()
    uncached = int(getattr(usage, "input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    output = int(getattr(usage, "output_tokens", 0) or 0)
    return UsageBreakdown(
        uncached_input=uncached,
        cache_read=cache_read,
        cache_creation=cache_creation,
        output=output,
    )


def read_openai_usage(usage: Any) -> UsageBreakdown:
    """Split an OpenAI-family ``usage`` object into a :class:`UsageBreakdown`.

    OpenAI reports ``prompt_tokens`` as the FULL input total with the cached portion
    nested under ``prompt_tokens_details.cached_tokens`` — so the uncached input is
    ``prompt_tokens - cached_tokens``. There is no separate cache-creation line.
    """
    if usage is None:
        return UsageBreakdown()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = 0
    if details is not None:
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens", 0) or 0)
        else:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
    cached = max(0, min(cached, prompt))
    return UsageBreakdown(
        uncached_input=prompt - cached,
        cache_read=cached,
        cache_creation=0,
        output=completion,
    )


def resolve_cache_rates(
    price: ModelPrice,
    *,
    capability: CacheCapability,
    ttl: str = "5m",
    cache_read_multiplier: float | None = None,
    cache_write_multiplier: float | None = None,
) -> tuple[float, float]:
    """Resolve ``(read_rate, write_mult)`` for a model, per provider family.

    An explicit per-model tier (``cache_read_multiplier`` / ``cache_write_multiplier``
    pulled from the price table) wins; otherwise the provider-family constant applies.
    Anthropic's write multiplier depends on the TTL (1.25x at 5m, 2.0x at 1h);
    OpenAI-family caching has no write premium.
    """
    if (
        capability is CacheCapability.ANTHROPIC_EXPLICIT
        or capability is CacheCapability.OPENROUTER_PASSTHROUGH
    ):
        read_rate = (
            cache_read_multiplier
            if cache_read_multiplier is not None
            else _ANTHROPIC_READ_RATE
        )
        if cache_write_multiplier is not None:
            write_mult = cache_write_multiplier
        else:
            write_mult = (
                _ANTHROPIC_WRITE_MULT_1H if ttl == "1h" else _ANTHROPIC_WRITE_MULT_5M
            )
        return float(read_rate), float(write_mult)
    if capability is CacheCapability.OPENAI_AUTOMATIC:
        read_rate = (
            cache_read_multiplier
            if cache_read_multiplier is not None
            else _OPENAI_READ_RATE
        )
        write_mult = (
            cache_write_multiplier
            if cache_write_multiplier is not None
            else _OPENAI_WRITE_MULT
        )
        return float(read_rate), float(write_mult)
    # NONE / unknown: no discount, no premium (full-rate everywhere).
    read = cache_read_multiplier if cache_read_multiplier is not None else 1.0
    write = cache_write_multiplier if cache_write_multiplier is not None else 1.0
    return float(read), float(write)


def compute_cached_cost(
    price: ModelPrice,
    breakdown: UsageBreakdown,
    *,
    read_rate: float,
    write_mult: float,
) -> float:
    """The SINGLE cache-aware cost function both adapters call.

    Full-rate for uncached input + output, then the cache portions at their
    provider-specific multipliers of the input rate::

        price.cost(uncached_input, output)
          + (input_per_1k / 1000) * (cache_read * read_rate + cache_creation * write_mult)

    With an all-uncached breakdown (no cache tokens) this is byte-identical to the
    pre-cache ``price.cost(input_tokens=input, output_tokens=output)``.
    """
    base = price.cost(
        input_tokens=breakdown.uncached_input, output_tokens=breakdown.output
    )
    cache_cost = (price.input_per_1k / 1000.0) * (
        breakdown.cache_read * read_rate + breakdown.cache_creation * write_mult
    )
    return base + cache_cost


def cache_metrics_payload(
    request: InferenceRequest, response: Any
) -> dict[str, Any]:
    """The shared cache observability payload for an ``INFERENCE_SUCCEEDED`` event.

    This is the SINGLE function both emit sites (``InferenceService`` and
    ``SingleAgentRuntime``) call so the event schema stays identical. It reads the
    additive cache keys an adapter stamped onto ``response.metadata``
    (``cache_read_tokens`` / ``cache_creation_tokens`` / ``cache_savings_usd``) and adds
    the standard caching AUDIT signal:

    * ``prefix_cache_requested`` — whether the request actually opted into caching
      (a non-``None``, ``enabled`` :class:`CachePolicy`). Present only when ``True``.
    * ``prefix_cache_miss`` — ``True`` when caching was requested, the stable prefix was
      over the per-model floor (so the adapter SHOULD have marked it), yet the provider
      reported zero cache activity (no read AND no creation). That is the standard
      busted-prefix audit signal — a silently-rewritten prefix or a marker the provider
      ignored — so it's caught instead of paying write premiums forever. A legitimately
      SUB-threshold prefix (never markable) is NOT flagged, and a warm response-cache
      hit (``cache_hit``) is never flagged because no real provider prefix call was made.

    When caching was NOT requested the payload is empty (byte-identical to the
    pre-cache event), so no-cache runs are unaffected.
    """
    metadata = getattr(response, "metadata", None) or {}
    payload: dict[str, Any] = {}
    cache_read = int(metadata.get("cache_read_tokens", 0) or 0)
    cache_creation = int(metadata.get("cache_creation_tokens", 0) or 0)
    if cache_read or cache_creation:
        payload["cache_read_tokens"] = cache_read
        payload["cache_creation_tokens"] = cache_creation
        savings = metadata.get("cache_savings_usd")
        if savings is not None:
            payload["cache_savings_usd"] = float(savings)
    if cache_policy_active(request):
        payload["prefix_cache_requested"] = True
        # Only flag a miss when the prefix was actually worth caching (over the
        # per-model floor) so a sub-threshold prefix isn't noise. A warm response-cache
        # hit (cache_hit) never touched a provider prefix cache, so it's not a bust.
        policy = request.cache_policy
        floor = min_cacheable_tokens(
            request.model_key, policy.min_prefix_tokens if policy else None
        )
        over_threshold = stable_prefix_estimate_tokens(request) >= floor
        if (
            over_threshold
            and not metadata.get("cache_hit")
            and cache_read == 0
            and cache_creation == 0
        ):
            payload["prefix_cache_miss"] = True
    return payload


def cache_savings_usd(
    price: ModelPrice,
    breakdown: UsageBreakdown,
    *,
    read_rate: float,
    write_mult: float,
) -> float:
    """USD saved versus billing every cache-read/creation token at the full input rate.

    Savings = (full-rate cost of the cached tokens) - (discounted cost). Cache
    *creation* at a write premium > 1.0 (Anthropic) is a NEGATIVE saving (an
    investment paid back on later reads); this returns the net so callers see the
    honest figure.
    """
    full_rate = (price.input_per_1k / 1000.0) * (
        breakdown.cache_read + breakdown.cache_creation
    )
    discounted = (price.input_per_1k / 1000.0) * (
        breakdown.cache_read * read_rate + breakdown.cache_creation * write_mult
    )
    return full_rate - discounted


__all__ = [
    "CacheCapability",
    "DEFAULT_MIN_CACHEABLE_TOKENS",
    "MIN_CACHEABLE_TOKENS",
    "UsageBreakdown",
    "anthropic_system_blocks",
    "anthropic_ttl_supported",
    "cache_metrics_payload",
    "cache_policy_active",
    "cache_savings_usd",
    "compute_cached_cost",
    "is_openai_family_model",
    "min_cacheable_tokens",
    "openrouter_passthrough_backend",
    "read_anthropic_usage",
    "read_openai_usage",
    "resolve_cache_capability",
    "resolve_cache_rates",
    "should_cache_prefix",
    "stable_prefix_estimate_tokens",
]
