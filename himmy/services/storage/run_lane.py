"""Provider-lane keying for the leased run work-queue (Q2).

A single "claim the oldest QUEUED run" CAS cannot honour the framework's "gate the
LOCAL model, never the cloud" health rule: if the only local model (Ollama / the
``claude`` CLI) is unreachable, a dispatcher that claims the globally-oldest run would
hand a worker a run it cannot execute, stall it, and — worse — starve cloud-provider
runs that ARE serviceable. The fix is to stamp each run with a coarse provider LANE
derived from its ``model_key`` and let the claim predicate filter on it, so the
dispatcher can temporarily exclude the ``local`` lane (when its liveness probe fails)
while still draining the ``cloud`` lane.

Two lanes are enough for the health gate the queue needs:

* :data:`LANE_LOCAL` — providers whose readiness is a LOCAL process/daemon the operator
  must have running (Ollama, the ``claude`` CLI / Claude Max, a self-hosted HimalayaGPT).
  These are the ones a liveness probe can find "down".
* :data:`LANE_CLOUD` — hosted HTTP providers (Anthropic, OpenAI, OpenRouter, the gateway).
  Reachability is the network's problem, not a local daemon's, so the local-probe gate
  must never block them.

``model_key`` follows the documented ``provider:model`` convention (e.g. ``ollama:llama3``,
``claude-cli:haiku``, ``anthropic:claude-3-5-sonnet``, ``openai:gpt-4o``); a bare model with
no prefix, or an unknown provider, maps to :data:`LANE_CLOUD` — the SAFE default, because a
cloud lane is never gated by the local probe, so a misclassified run is still claimable
rather than wedged behind a down local daemon. ``model_key=None`` (the offline / "use the
container default" case) maps to :data:`LANE_DEFAULT`, a neutral lane the dispatcher always
includes.
"""

from __future__ import annotations

#: The lane for runs targeting a LOCAL model daemon/CLI (probe-gated).
LANE_LOCAL = "local"
#: The lane for runs targeting a hosted cloud HTTP provider (never probe-gated).
LANE_CLOUD = "cloud"
#: The neutral lane for a run with no resolved ``model_key`` (always claimable).
LANE_DEFAULT = "default"

#: Provider tokens that resolve to :data:`LANE_LOCAL`. These mirror the ``provider_name``
#: values the local client managers stamp (himmy.services.inference.local) plus the prefix a
#: ``model_path`` like ``claude-cli:haiku`` carries. Matched case-insensitively against the
#: ``model_key`` prefix before the first ``:``.
_LOCAL_PROVIDERS = frozenset({"ollama", "claude-cli", "claude_cli", "himalaya", "local"})

#: Provider tokens that resolve to :data:`LANE_CLOUD`. Explicit so a typo in a known cloud
#: provider name still lands in the (safe) cloud lane via the unknown-default anyway, but the
#: set documents the canonical hosted providers.
_CLOUD_PROVIDERS = frozenset(
    {"anthropic", "openai", "openrouter", "gateway", "azure", "bedrock", "google"}
)


def lane_for_model_key(model_key: str | None) -> str:
    """Map a run's ``model_key`` to its coarse provider lane (Q2 lane keying).

    Returns :data:`LANE_LOCAL` for a local-daemon provider prefix, :data:`LANE_CLOUD` for a
    hosted provider (and for any UNKNOWN provider — the safe default, since the cloud lane is
    never gated by the local liveness probe), and :data:`LANE_DEFAULT` when no model_key was
    resolved. The provider is the token before the first ``:`` (case-insensitive); a bare
    model name with no prefix is treated as cloud.
    """
    if model_key is None:
        return LANE_DEFAULT
    key = model_key.strip()
    if not key:
        return LANE_DEFAULT
    provider = key.split(":", 1)[0].strip().lower() if ":" in key else ""
    if provider in _LOCAL_PROVIDERS:
        return LANE_LOCAL
    # Known cloud provider, an unprefixed bare model, or an unknown prefix: all cloud — the
    # cloud lane is never blocked by the local probe, so this default is fail-OPEN for claims.
    return LANE_CLOUD


__all__ = ["LANE_LOCAL", "LANE_CLOUD", "LANE_DEFAULT", "lane_for_model_key"]
