"""sec-r2 — the PROVIDER prompt cache is partitioned per principal.

The internal response cache is already scoped per tenant (``derive_cache_scope``), but
the provider prompt cache (OpenAI ``prompt_cache_key`` / Anthropic prefix bytes) was NOT:
``CachePolicy.cache_key`` was always ``None``, so ``_apply_prompt_cache_key`` never set a
partition hint. On a shared provider API key, a byte-identical system+tools prefix
(cold/empty-memory tenants, a shared default persona) could then serve tenant B a
cache-read of tenant A's prefix — a metering/cost side-channel.

Fix: the runtime derives a stable per-principal ``cache_key`` from the same tenant scope
metadata it already stamps, so the OpenAI-family adapter partitions the provider cache
per principal. An UNSCOPED run keeps ``cache_key=None`` (byte-identical offline contract).
"""

from __future__ import annotations

from himmy.runtime.single_agent import (
    SingleAgentRuntime,
    _prompt_cache_key_for_scope,
)
from himmy.services.inference.prompt_cache import CacheCapability
from himmy.services.inference.service import InferenceService


class _FakeAnthropic:
    """A caching-capable manager (so ``_prompt_cache_policy`` returns a policy)."""

    cache_capability = CacheCapability.ANTHROPIC_EXPLICIT
    provider_name = "anthropic"

    def resolve(self, model_key: str) -> str:
        return f"anthropic:{model_key}"


def _rt() -> SingleAgentRuntime:
    return SingleAgentRuntime(inference_service=InferenceService(_FakeAnthropic()))


# ------------------------------------------------------- pure scope-key derivation
def test_scope_key_is_none_for_an_unscoped_run() -> None:
    """No principal metadata → no cache_key (byte-identical to the pre-sec-r2 payload)."""
    assert _prompt_cache_key_for_scope({}) is None


def test_scope_key_folds_every_principal_dimension_stably() -> None:
    """subject/tenant/workspace fold into one deterministic, order-stable key."""
    key = _prompt_cache_key_for_scope(
        {"tenant_id": "acme", "workspace_id": "w1", "subject_id": "boss"}
    )
    # Order follows CACHE_SCOPE_METADATA_KEYS (tenant, workspace, subject) — deterministic.
    assert key == "tenant_id=acme|workspace_id=w1|subject_id=boss"


def test_distinct_principals_get_distinct_scope_keys() -> None:
    """Two tenants never share a provider-cache partition key."""
    a = _prompt_cache_key_for_scope({"tenant_id": "A", "subject_id": "s"})
    b = _prompt_cache_key_for_scope({"tenant_id": "B", "subject_id": "s"})
    assert a != b


def test_scope_key_ignores_empty_values() -> None:
    """Empty/None dimensions are dropped so they never create a spurious partition."""
    assert _prompt_cache_key_for_scope({"tenant_id": "A", "workspace_id": ""}) == (
        "tenant_id=A"
    )


# ------------------------------------------------------- runtime policy wiring
def test_runtime_policy_sets_per_principal_cache_key_when_scoped() -> None:
    """A tenant-scoped run stamps the policy with a per-principal provider-cache key."""
    rt = _rt()
    policy = rt._prompt_cache_policy(
        "default",
        cache_busted=False,
        scope_metadata={"tenant_id": "acme", "subject_id": "boss"},
    )
    assert policy is not None
    assert policy.cache_key == "tenant_id=acme|subject_id=boss"


def test_runtime_policy_leaves_cache_key_none_when_unscoped() -> None:
    """An unscoped run keeps cache_key=None: the offline/no-cache-key payload contract."""
    rt = _rt()
    policy = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}
    )
    assert policy is not None
    assert policy.cache_key is None


def test_two_tenants_get_policies_with_different_cache_keys() -> None:
    """The provider-cache partition differs per tenant — no cross-tenant prefix sharing."""
    rt = _rt()
    a = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={"tenant_id": "A"}
    )
    b = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={"tenant_id": "B"}
    )
    assert a is not None and b is not None
    assert a.cache_key != b.cache_key
