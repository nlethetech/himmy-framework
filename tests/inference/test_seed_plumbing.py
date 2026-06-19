"""Seed plumbing for reproducible generations (QUICK WIN 1).

``InferenceRequest.seed`` (and the ``LLMConfig.seed`` mirror) is an optional
sampling seed that:

* is OMITTED entirely (byte-identical no-op) when ``None`` — payloads, cache keys,
  and the INFERENCE_REQUESTED event are unchanged from the no-seed path;
* is forwarded to the providers that support it (OpenAI ``seed``, Ollama
  ``options.seed``) when set;
* partitions the response cache so two runs with different seeds never collide,
  while the SAME seed re-uses the cassette.
"""

from __future__ import annotations

from typing import Any

from himmy.services.inference import (
    InferenceMessage,
    InferenceRequest,
    StubClientManager,
    compute_cache_key,
)
from himmy.services.inference.local import OllamaClientManager
from himmy.services.inference.models import InferenceStatus
from himmy.services.inference.openai_manager import OpenAIClientManager
from tests.conftest import run_async


def _req(**kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": "default",
        "messages": [InferenceMessage(role="user", content="hello")],
    }
    base.update(kw)
    return InferenceRequest(**base)


# ----------------------------------------------------------------- cache key
def test_same_seed_same_cache_key() -> None:
    """Identical request + identical seed -> identical cache key (cassette reuse)."""
    a = compute_cache_key(_req(seed=7))
    b = compute_cache_key(_req(seed=7))
    assert a == b


def test_different_seed_different_cache_key() -> None:
    """Different seeds partition the cache so reproducible runs never collide."""
    a = compute_cache_key(_req(seed=7))
    b = compute_cache_key(_req(seed=8))
    assert a != b


def test_unset_seed_is_byte_identical_to_legacy_key() -> None:
    """``seed=None`` keeps the key byte-for-byte identical to the no-seed path."""
    assert compute_cache_key(_req()) == compute_cache_key(_req(seed=None))


def test_seeded_key_differs_from_unseeded() -> None:
    """A seeded request never aliases the unseeded cassette for the same prompt."""
    assert compute_cache_key(_req()) != compute_cache_key(_req(seed=0))


# ----------------------------------------------------------------- OpenAI egress
class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _FakeCompletions:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.seen.update(kwargs)
        return self._result


class _FakeChat:
    def __init__(self, result: Any) -> None:
        self.completions = _FakeCompletions(result)


class _FakeClient:
    def __init__(self, result: Any) -> None:
        self.chat = _FakeChat(result)


def _completion(text: str = "hi") -> _Obj:
    message = _Obj(content=text, tool_calls=None)
    usage = _Obj(prompt_tokens=5, completion_tokens=2)
    return _Obj(choices=[_Obj(message=message)], usage=usage)


def test_openai_forwards_seed_when_set() -> None:
    client = _FakeClient(_completion())
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    run_async(mgr.generate(_req(model_key="gpt-4o-mini", seed=42)))
    assert client.chat.completions.seen["seed"] == 42


def test_openai_omits_seed_when_unset() -> None:
    client = _FakeClient(_completion())
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    run_async(mgr.generate(_req(model_key="gpt-4o-mini")))
    assert "seed" not in client.chat.completions.seen


# ----------------------------------------------------------------- Ollama egress
def test_ollama_forwards_seed_in_options() -> None:
    captured: dict[str, Any] = {}

    async def _transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1}

    mgr = OllamaClientManager(model="llama3.2", transport=_transport)
    resp = run_async(mgr.generate(_req(seed=99)))
    assert resp.status == InferenceStatus.SUCCESS
    assert captured["options"]["seed"] == 99


def test_ollama_omits_seed_when_unset() -> None:
    captured: dict[str, Any] = {}

    async def _transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1}

    mgr = OllamaClientManager(model="llama3.2", transport=_transport)
    run_async(mgr.generate(_req()))
    # No options at all (no temperature/max_tokens/seed) -> no options key.
    assert "options" not in captured or "seed" not in captured.get("options", {})


# ----------------------------------------------------------------- event stamp
def test_seed_stamped_on_requested_event() -> None:
    """INFERENCE_REQUESTED carries ``seed`` when set, and omits it otherwise."""
    from himmy.core.events import EventType
    from himmy.services.inference import InferenceService

    class _Sink:
        def __init__(self) -> None:
            self.events: list[Any] = []

        async def append_event(self, event: Any) -> None:
            self.events.append(event)

    sink = _Sink()
    svc = InferenceService(StubClientManager(), event_sink=sink, max_retries=0)
    run_async(svc.run(_req(seed=5)))
    requested = [
        e for e in sink.events if e.event_type == EventType.INFERENCE_REQUESTED
    ]
    assert requested and requested[0].payload.get("seed") == 5

    sink.events.clear()
    run_async(svc.run(_req()))
    requested = [
        e for e in sink.events if e.event_type == EventType.INFERENCE_REQUESTED
    ]
    assert requested and "seed" not in requested[0].payload
