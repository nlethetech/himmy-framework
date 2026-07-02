"""P2.3 — an UNSCOPED OpenAI-family run carries a per-conversation ``prompt_cache_key``.

``_apply_prompt_cache_key`` is plumbed, but the runtime always built ``CachePolicy()`` with
``cache_key=None`` for an unscoped run, so OpenAI's ``prompt_cache_key`` routing-affinity hint
was never sent. This module proves the runtime now derives a STABLE per-conversation cache_key
(hashed from ``thread_id``) for OpenAI-family managers only:

* populated + stable WITHIN a run, distinct ACROSS runs (fresh ``thread_id`` per run);
* ``None`` / unaffected for non-OpenAI providers and when opted out;
* pure routing hint — it changes ONLY the ``prompt_cache_key`` payload field, nothing else, and
  is never emitted for a non-OpenAI model.
"""

from __future__ import annotations

import threading

from himmy.runtime.single_agent import (
    SingleAgentRuntime,
    _prompt_cache_key_for_conversation,
)
from himmy.services.inference.models import (
    CachePolicy,
    InferenceMessage,
    InferenceRequest,
)
from himmy.services.inference.openai_manager import OpenAIClientManager
from himmy.services.inference.prompt_cache import CacheCapability
from himmy.services.inference.service import InferenceService


class _FakeOpenAI:
    """A caching-capable OpenAI-family manager (``OPENAI_AUTOMATIC``)."""

    cache_capability = CacheCapability.OPENAI_AUTOMATIC
    provider_name = "openai"

    def resolve(self, model_key: str) -> str:
        return f"openai:{model_key}"


class _FakeAnthropic:
    """A caching-capable non-OpenAI manager (``ANTHROPIC_EXPLICIT``)."""

    cache_capability = CacheCapability.ANTHROPIC_EXPLICIT
    provider_name = "anthropic"

    def resolve(self, model_key: str) -> str:
        return f"anthropic:{model_key}"


def _rt(manager: object) -> SingleAgentRuntime:
    return SingleAgentRuntime(inference_service=InferenceService(manager))


# ------------------------------------------------------- pure conversation-key derivation
def test_conversation_key_is_none_without_a_thread_id() -> None:
    """No ``thread_id`` → no cache_key (byte-identical to the no-cache-key payload)."""
    assert _prompt_cache_key_for_conversation(None) is None
    assert _prompt_cache_key_for_conversation("") is None


def test_conversation_key_is_stable_for_the_same_thread_id() -> None:
    """Same ``thread_id`` folds to the same opaque digest every time (stable within a run)."""
    a = _prompt_cache_key_for_conversation("thread-xyz")
    b = _prompt_cache_key_for_conversation("thread-xyz")
    assert a == b
    assert a is not None and a.startswith("himmy-conv-")


def test_conversation_key_is_distinct_across_thread_ids() -> None:
    """Distinct threads (distinct runs) never collide onto one routing partition."""
    a = _prompt_cache_key_for_conversation("thread-a")
    b = _prompt_cache_key_for_conversation("thread-b")
    assert a != b


def test_conversation_key_does_not_leak_the_raw_thread_id() -> None:
    """The hint is an opaque digest, not the raw id."""
    key = _prompt_cache_key_for_conversation("secret-thread-id")
    assert key is not None
    assert "secret-thread-id" not in key


# ------------------------------------------------------- runtime policy wiring (OpenAI)
def test_openai_unscoped_run_gets_a_conversation_cache_key() -> None:
    """An unscoped OpenAI-family run stamps the per-conversation routing hint."""
    rt = _rt(_FakeOpenAI())
    policy = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id="run-1"
    )
    assert policy is not None
    assert policy.cache_key == _prompt_cache_key_for_conversation("run-1")


def test_openai_conversation_key_is_stable_within_a_run() -> None:
    """Every turn of one run (one ``thread_id``) resolves to the same routing hint."""
    rt = _rt(_FakeOpenAI())
    first = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id="run-1"
    )
    second = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id="run-1"
    )
    assert first is not None and second is not None
    assert first.cache_key == second.cache_key


def test_openai_conversation_key_differs_across_runs() -> None:
    """Two runs (fresh ``thread_id`` each) route to distinct partitions."""
    rt = _rt(_FakeOpenAI())
    a = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id="run-a"
    )
    b = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id="run-b"
    )
    assert a is not None and b is not None
    assert a.cache_key != b.cache_key


def test_scope_key_wins_over_conversation_key() -> None:
    """A tenant-scoped run keeps the per-principal key: tenant isolation is never diluted."""
    rt = _rt(_FakeOpenAI())
    policy = rt._prompt_cache_policy(
        "default",
        cache_busted=False,
        scope_metadata={"tenant_id": "acme", "subject_id": "boss"},
        thread_id="run-1",
    )
    assert policy is not None
    assert policy.cache_key == "tenant_id=acme|subject_id=boss"


def test_openai_conversation_key_absent_without_thread_id() -> None:
    """No ``thread_id`` (e.g. a threadless call site) → byte-identical no-cache-key payload."""
    rt = _rt(_FakeOpenAI())
    policy = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id=None
    )
    assert policy is not None
    assert policy.cache_key is None


# ------------------------------------------------------- non-OpenAI providers unaffected
def test_anthropic_unscoped_run_keeps_cache_key_none() -> None:
    """A non-OpenAI family never gets a conversation key (its prefix bytes stay unchanged)."""
    rt = _rt(_FakeAnthropic())
    policy = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id="run-1"
    )
    assert policy is not None
    assert policy.cache_key is None


def test_env_opt_out_disables_the_conversation_key(monkeypatch) -> None:
    """``HIMMY_OPENAI_CONVERSATION_CACHE_KEY=0`` restores the no-cache-key contract."""
    monkeypatch.setenv("HIMMY_OPENAI_CONVERSATION_CACHE_KEY", "0")
    rt = _rt(_FakeOpenAI())
    policy = rt._prompt_cache_policy(
        "default", cache_busted=False, scope_metadata={}, thread_id="run-1"
    )
    assert policy is not None
    assert policy.cache_key is None


# ------------------------------------------------------- output identity via the adapter
def _payload_for(model: str, cache_key: str | None) -> dict[str, object]:
    request = InferenceRequest(
        model_key=model,
        messages=[InferenceMessage(role="user", content="hi")],
        cache_policy=CachePolicy(cache_key=cache_key),
    )
    mgr = OpenAIClientManager.__new__(OpenAIClientManager)
    payload: dict[str, object] = {"model": model, "messages": []}
    mgr._apply_prompt_cache_key(request, model, payload)
    return payload


def test_conversation_key_only_adds_the_routing_field_for_openai() -> None:
    """The hint adds ONLY ``prompt_cache_key`` — no other payload field changes."""
    key = _prompt_cache_key_for_conversation("run-1")
    base = _payload_for("gpt-4o-mini", None)
    with_key = _payload_for("gpt-4o-mini", key)
    assert with_key.pop("prompt_cache_key") == key
    assert with_key == base  # everything else is byte-identical


def test_conversation_key_never_emitted_for_a_non_openai_model() -> None:
    """A non-OpenAI model id never receives ``prompt_cache_key`` even if the policy names one."""
    key = _prompt_cache_key_for_conversation("run-1")
    payload = _payload_for("anthropic/claude-3-5-sonnet", key)
    assert "prompt_cache_key" not in payload


# ------------------------------------------------------- thread-safety
def test_conversation_key_is_thread_safe_and_deterministic() -> None:
    """Concurrent derivation for the same thread_id yields one stable key (no shared state)."""
    results: list[str | None] = []
    lock = threading.Lock()

    def _derive() -> None:
        key = _prompt_cache_key_for_conversation("shared-run")
        with lock:
            results.append(key)

    threads = [threading.Thread(target=_derive) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = _prompt_cache_key_for_conversation("shared-run")
    assert results == [expected] * 32
