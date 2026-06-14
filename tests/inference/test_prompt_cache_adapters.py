"""Adapter-side prompt-cache translation tests (C3 Anthropic + C4 OpenAI/OpenRouter).

These pin the per-provider mechanism each adapter uses to mark the stable system+tools
prefix as cacheable, and — critically — that the ``cache_policy is None`` default emits a
byte-identical payload to the no-cache path (the invariant the provider shape contracts
rest on). All offline: injected fake clients capture the exact ``create`` kwargs; no
provider key, no network.

C3 (Anthropic, explicit ``cache_control``):
* enabled + over-threshold prefix → ``system`` becomes block form with a ``cache_control``
  breakpoint on the (single) system block — which caches BOTH tools and system;
* the 1h TTL is version-gated (degrades to 5m + no beta header when the SDK can't be
  introspected, which is the case in this checkout — ``anthropic`` is not installed);
* below threshold / disabled / scope=none / policy=None → plain string, byte-identical;
* both ``generate`` and ``generate_stream`` apply the same marking.

C4 (OpenAI-family automatic + OpenRouter passthrough):
* tools are name-sorted only when caching is active (stability for the automatic prefix);
* ``prompt_cache_key`` is sent only for an OpenAI-family model, omitted elsewhere;
* OpenRouter forwarding to an ``anthropic/``/``google/`` backend rewrites the system
  message to block-form ``cache_control`` passthrough; an openai-backed OpenRouter model
  relies on automatic caching (string content);
* the None path keeps string content + registry tool order.
"""

from __future__ import annotations

from typing import Any

from himmy.services.inference.anthropic_manager import AnthropicClientManager
from himmy.services.inference.models import (
    BoundTool,
    CachePolicy,
    InferenceMessage,
    InferenceRequest,
    ResponseFormat,
)
from himmy.services.inference.openai_manager import OpenAIClientManager
from himmy.services.inference.prompt_cache import (
    CacheCapability as Cap,
)
from himmy.services.inference.prompt_cache import (
    anthropic_system_blocks,
    anthropic_ttl_supported,
    is_openai_family_model,
    openrouter_passthrough_backend,
    should_cache_prefix,
)
from tests.conftest import run_async

#: A long system prompt that clears the per-model cacheable-prefix floor (~1024 tokens
#: for the claude family: ~4 chars/token, so well over 4096 chars).
BIG_SYSTEM = "You are a meticulous assistant. " * 200
#: A tiny system that is far below any floor.
SMALL_SYSTEM = "be brief"


# ============================================================ Anthropic (C3)
class _ATextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _AUsage:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _AMessage:
    def __init__(self, content: list[Any], usage: _AUsage) -> None:
        self.content = content
        self.usage = usage


class _AMessages:
    def __init__(self, message: _AMessage) -> None:
        self._message = message
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _AMessage:
        self.seen.update(kwargs)
        return self._message


class _AClient:
    def __init__(self, message: _AMessage) -> None:
        self.messages = _AMessages(message)


def _aclient() -> _AClient:
    return _AClient(_AMessage([_ATextBlock("ok")], _AUsage(11, 7)))


def _areq(system: str, **kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": "claude-3-5-sonnet-latest",
        "messages": [
            InferenceMessage(role="system", content=system),
            InferenceMessage(role="user", content="hello"),
        ],
    }
    base.update(kw)
    return InferenceRequest(**base)


def test_anthropic_enabled_over_threshold_marks_last_system_block() -> None:
    """Enabled + big prefix → block-form system with a cache_control breakpoint."""
    client = _aclient()
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)
    run_async(mgr.generate(_areq(BIG_SYSTEM, cache_policy=CachePolicy())))
    system = client.messages.seen["system"]
    assert isinstance(system, list) and len(system) == 1
    block = system[0]
    assert block["type"] == "text"
    assert block["text"] == BIG_SYSTEM
    assert block["cache_control"] == {"type": "ephemeral"}
    # 5m default → no extended-cache beta header.
    assert "extra_headers" not in client.messages.seen


def test_anthropic_default_none_is_byte_identical_plain_string() -> None:
    """cache_policy=None → system stays a plain string (byte-identical no-cache path)."""
    client = _aclient()
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)
    run_async(mgr.generate(_areq(BIG_SYSTEM)))  # no cache_policy
    assert client.messages.seen["system"] == BIG_SYSTEM
    assert "extra_headers" not in client.messages.seen


def test_anthropic_below_threshold_stays_plain_string() -> None:
    """A sub-threshold prefix is NOT marked even with caching enabled (no-op fallback)."""
    client = _aclient()
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)
    run_async(mgr.generate(_areq(SMALL_SYSTEM, cache_policy=CachePolicy())))
    assert client.messages.seen["system"] == SMALL_SYSTEM


def test_anthropic_disabled_policy_stays_plain_string() -> None:
    """A present-but-disabled policy behaves like None (plain string)."""
    client = _aclient()
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)
    run_async(mgr.generate(_areq(BIG_SYSTEM, cache_policy=CachePolicy(enabled=False))))
    assert client.messages.seen["system"] == BIG_SYSTEM


def test_anthropic_scope_none_stays_plain_string() -> None:
    """scope='none' opts out of the system breakpoint even when over threshold."""
    client = _aclient()
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)
    run_async(
        mgr.generate(_areq(BIG_SYSTEM, cache_policy=CachePolicy(scope="none")))
    )
    assert client.messages.seen["system"] == BIG_SYSTEM


def test_anthropic_1h_ttl_is_version_gated_degrades_to_5m() -> None:
    """1h TTL degrades to 5m + no beta header when the SDK can't be introspected.

    ``anthropic`` is not installed in this checkout, so ``anthropic_ttl_supported()`` is
    False and the adapter must NOT emit a ``ttl`` key or the extended-cache beta header.
    """
    assert anthropic_ttl_supported() is False
    client = _aclient()
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)
    run_async(
        mgr.generate(_areq(BIG_SYSTEM, cache_policy=CachePolicy(ttl="1h")))
    )
    block = client.messages.seen["system"][0]
    assert block["cache_control"] == {"type": "ephemeral"}  # no ttl key
    assert "extra_headers" not in client.messages.seen


def test_anthropic_1h_ttl_emits_block_and_header_when_supported() -> None:
    """When the SDK is known to support ttl, the 1h block + beta header are emitted."""
    block = anthropic_system_blocks(BIG_SYSTEM, ttl="1h", ttl_supported=True)[0]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # And the helper still degrades to 5m when unsupported.
    degraded = anthropic_system_blocks(BIG_SYSTEM, ttl="1h", ttl_supported=False)[0]
    assert degraded["cache_control"] == {"type": "ephemeral"}


def test_anthropic_structured_request_still_caches_prefix() -> None:
    """Caching marks the (instruction-augmented) system even on a structured request."""
    client = _aclient()
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    run_async(
        mgr.generate(
            _areq(
                BIG_SYSTEM,
                response_format=ResponseFormat.STRUCTURED_OUTPUT,
                output_json_schema=schema,
                cache_policy=CachePolicy(),
            )
        )
    )
    system = client.messages.seen["system"]
    assert isinstance(system, list)
    # The synthetic-tool instruction is appended INSIDE the cached block.
    assert system[0]["text"].startswith(BIG_SYSTEM)
    assert "__structured_output__" in system[0]["text"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_stream_path_also_marks_prefix() -> None:
    """generate_stream applies the same cache_control marking as generate."""

    class _Stream:
        def __init__(self, final: _AMessage) -> None:
            self._final = final

        async def __aenter__(self) -> _Stream:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        @property
        def text_stream(self) -> Any:
            async def _gen() -> Any:
                yield "ok"

            return _gen()

        async def get_final_message(self) -> _AMessage:
            return self._final

    class _StreamMessages:
        def __init__(self, final: _AMessage) -> None:
            self._final = final
            self.seen: dict[str, Any] = {}

        def stream(self, **kwargs: Any) -> _Stream:
            self.seen.update(kwargs)
            return _Stream(self._final)

    class _StreamClient:
        def __init__(self, final: _AMessage) -> None:
            self.messages = _StreamMessages(final)

    client = _StreamClient(_AMessage([_ATextBlock("ok")], _AUsage(3, 2)))
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)

    async def _drain() -> None:
        async for _ in mgr.generate_stream(_areq(BIG_SYSTEM, cache_policy=CachePolicy())):
            pass

    run_async(_drain())
    system = client.messages.seen["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_stream_none_path_byte_identical() -> None:
    """generate_stream with no policy keeps the plain-string system."""

    class _Stream:
        def __init__(self, final: _AMessage) -> None:
            self._final = final

        async def __aenter__(self) -> _Stream:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        @property
        def text_stream(self) -> Any:
            async def _gen() -> Any:
                yield "ok"

            return _gen()

        async def get_final_message(self) -> _AMessage:
            return self._final

    class _StreamMessages:
        def __init__(self, final: _AMessage) -> None:
            self._final = final
            self.seen: dict[str, Any] = {}

        def stream(self, **kwargs: Any) -> _Stream:
            self.seen.update(kwargs)
            return _Stream(self._final)

    class _StreamClient:
        def __init__(self, final: _AMessage) -> None:
            self.messages = _StreamMessages(final)

    client = _StreamClient(_AMessage([_ATextBlock("ok")], _AUsage(3, 2)))
    mgr = AnthropicClientManager(model="claude-3-5-sonnet-latest", client=client)

    async def _drain() -> None:
        async for _ in mgr.generate_stream(_areq(BIG_SYSTEM)):
            pass

    run_async(_drain())
    assert client.messages.seen["system"] == BIG_SYSTEM


# ============================================================ OpenAI / OpenRouter (C4)
class _OChatMessage:
    role = "assistant"

    def __init__(self, content: str | None) -> None:
        self.content = content
        self.tool_calls = None


class _OChoice:
    def __init__(self, message: _OChatMessage) -> None:
        self.index = 0
        self.finish_reason = "stop"
        self.message = message


class _OUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _OCompletion:
    def __init__(self, choices: list[_OChoice], usage: _OUsage) -> None:
        self.choices = choices
        self.usage = usage


class _OCompletions:
    def __init__(self, completion: _OCompletion) -> None:
        self._completion = completion
        self.seen: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _OCompletion:
        self.seen.update(kwargs)
        return self._completion


class _OChat:
    def __init__(self, completion: _OCompletion) -> None:
        self.completions = _OCompletions(completion)


class _OClient:
    def __init__(self, completion: _OCompletion) -> None:
        self.chat = _OChat(completion)


def _oclient() -> _OClient:
    return _OClient(_OCompletion([_OChoice(_OChatMessage("ok"))], _OUsage(11, 7)))


def _oreq(model: str, *, system: str = SMALL_SYSTEM, **kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": model,
        "messages": [
            InferenceMessage(role="system", content=system),
            InferenceMessage(role="user", content="hello"),
        ],
    }
    base.update(kw)
    return InferenceRequest(**base)


def test_openai_none_path_keeps_string_system_and_registry_tool_order() -> None:
    """No policy → string system content and registry tool order (byte-identical)."""
    client = _oclient()
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    req = _oreq(
        "gpt-4o-mini",
        bound_tools=[BoundTool(name="zebra"), BoundTool(name="alpha")],
    )
    run_async(mgr.generate(req))
    seen = client.chat.completions.seen
    # system content is a plain string.
    assert seen["messages"][0] == {"role": "system", "content": SMALL_SYSTEM}
    # registry order preserved (NOT name-sorted) on the None path.
    assert [t["function"]["name"] for t in seen["tools"]] == ["zebra", "alpha"]
    assert "prompt_cache_key" not in seen


def test_openai_enabled_name_sorts_tools() -> None:
    """An active policy name-sorts the tool array for prefix stability."""
    client = _oclient()
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    req = _oreq(
        "gpt-4o-mini",
        bound_tools=[BoundTool(name="zebra"), BoundTool(name="alpha")],
        cache_policy=CachePolicy(),
    )
    run_async(mgr.generate(req))
    seen = client.chat.completions.seen
    assert [t["function"]["name"] for t in seen["tools"]] == ["alpha", "zebra"]


def test_openai_family_gets_prompt_cache_key() -> None:
    """prompt_cache_key is sent for an OpenAI-family model when the policy names one."""
    client = _oclient()
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    run_async(
        mgr.generate(_oreq("gpt-4o-mini", cache_policy=CachePolicy(cache_key="agent-1")))
    )
    assert client.chat.completions.seen["prompt_cache_key"] == "agent-1"


def test_non_openai_model_omits_prompt_cache_key() -> None:
    """A non-OpenAI model never receives prompt_cache_key (no unsupported param)."""
    client = _oclient()
    # OpenRouter forwarding to an anthropic backend: NOT openai-family.
    mgr = OpenAIClientManager(
        model="anthropic/claude-3.5-sonnet",
        client=client,
        provider_name="openrouter",
    )
    run_async(
        mgr.generate(
            _oreq(
                "anthropic/claude-3.5-sonnet",
                cache_policy=CachePolicy(cache_key="agent-1"),
            )
        )
    )
    assert "prompt_cache_key" not in client.chat.completions.seen


def test_openrouter_anthropic_backend_rewrites_system_to_block_form() -> None:
    """OpenRouter→anthropic with caching → block-form cache_control on the system message."""
    client = _oclient()
    mgr = OpenAIClientManager(
        model="anthropic/claude-3.5-sonnet",
        client=client,
        provider_name="openrouter",
    )
    run_async(
        mgr.generate(
            _oreq("anthropic/claude-3.5-sonnet", cache_policy=CachePolicy())
        )
    )
    system_msg = client.chat.completions.seen["messages"][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == [
        {"type": "text", "text": SMALL_SYSTEM, "cache_control": {"type": "ephemeral"}}
    ]


def test_openrouter_google_backend_rewrites_system_to_block_form() -> None:
    """OpenRouter→google passthrough also rewrites the system to block form."""
    client = _oclient()
    mgr = OpenAIClientManager(
        model="google/gemini-2.0-flash",
        client=client,
        provider_name="openrouter",
    )
    run_async(
        mgr.generate(_oreq("google/gemini-2.0-flash", cache_policy=CachePolicy()))
    )
    content = client.chat.completions.seen["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_openai_backend_relies_on_automatic_string_content() -> None:
    """OpenRouter→openai-backed model keeps string content (automatic prefix caching)."""
    client = _oclient()
    mgr = OpenAIClientManager(
        model="openai/gpt-4o-mini",
        client=client,
        provider_name="openrouter",
    )
    run_async(mgr.generate(_oreq("openai/gpt-4o-mini", cache_policy=CachePolicy())))
    seen = client.chat.completions.seen
    assert seen["messages"][0] == {"role": "system", "content": SMALL_SYSTEM}
    # openai-family OpenRouter id still gets the routing key when one is set.
    assert "prompt_cache_key" not in seen  # none set here


def test_direct_openai_provider_never_does_block_passthrough() -> None:
    """A direct openai provider never rewrites system to block form (automatic only)."""
    client = _oclient()
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)  # provider 'openai'
    run_async(mgr.generate(_oreq("gpt-4o-mini", cache_policy=CachePolicy())))
    assert client.chat.completions.seen["messages"][0] == {
        "role": "system",
        "content": SMALL_SYSTEM,
    }


def test_openrouter_passthrough_disabled_policy_stays_string() -> None:
    """A disabled policy keeps string content even on an OpenRouter anthropic backend."""
    client = _oclient()
    mgr = OpenAIClientManager(
        model="anthropic/claude-3.5-sonnet",
        client=client,
        provider_name="openrouter",
    )
    run_async(
        mgr.generate(
            _oreq(
                "anthropic/claude-3.5-sonnet",
                cache_policy=CachePolicy(enabled=False),
            )
        )
    )
    assert client.chat.completions.seen["messages"][0] == {
        "role": "system",
        "content": SMALL_SYSTEM,
    }


def test_openai_stream_path_applies_cache_key_and_passthrough() -> None:
    """The stream path carries prompt_cache_key + block passthrough like generate."""

    class _Chunk:
        def __init__(self, content: str | None, usage: Any = None) -> None:
            delta = type("d", (), {"content": content})()
            self.choices = [type("c", (), {"delta": delta, "index": 0})()]
            self.usage = usage

    async def _aiter(chunks: list[Any]) -> Any:
        for c in chunks:
            yield c

    final_usage = type("u", (), {"prompt_tokens": 5, "completion_tokens": 2})()
    chunks = [_Chunk("hi"), _Chunk(None, usage=final_usage)]

    class _StreamCompletions:
        def __init__(self) -> None:
            self.seen: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.seen.update(kwargs)
            return _aiter(chunks)

    class _StreamClient:
        def __init__(self) -> None:
            self.chat = type("c", (), {"completions": _StreamCompletions()})()

    client = _StreamClient()
    mgr = OpenAIClientManager(
        model="anthropic/claude-3.5-sonnet",
        client=client,
        provider_name="openrouter",
    )

    async def _drain() -> None:
        async for _ in mgr.generate_stream(
            _oreq("anthropic/claude-3.5-sonnet", cache_policy=CachePolicy())
        ):
            pass

    run_async(_drain())
    seen = client.chat.completions.seen
    assert seen["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


# ===================================================== helper-level classification
def test_is_openai_family_model_matches_bare_and_slashed_ids() -> None:
    assert is_openai_family_model("gpt-4o-mini")
    assert is_openai_family_model("o3-mini")
    assert is_openai_family_model("openai/gpt-4o")
    assert not is_openai_family_model("anthropic/claude-3.5-sonnet")
    assert not is_openai_family_model("google/gemini-2.0-flash")


def test_openrouter_passthrough_backend_classification() -> None:
    assert openrouter_passthrough_backend("anthropic/claude-3.5-sonnet") == "anthropic"
    assert openrouter_passthrough_backend("google/gemini-2.0-flash") == "google"
    assert openrouter_passthrough_backend("openai/gpt-4o-mini") is None
    assert openrouter_passthrough_backend("gpt-4o") is None


def test_should_cache_prefix_gates_on_capability_policy_scope_and_size() -> None:
    big = _areq(BIG_SYSTEM, cache_policy=CachePolicy())
    assert should_cache_prefix(big, "claude-3-5-sonnet-latest", Cap.ANTHROPIC_EXPLICIT)
    # NONE capability never caches.
    assert not should_cache_prefix(big, "claude-3-5-sonnet-latest", Cap.NONE)
    # No policy → never caches.
    assert not should_cache_prefix(
        _areq(BIG_SYSTEM), "claude-3-5-sonnet-latest", Cap.ANTHROPIC_EXPLICIT
    )
    # Below threshold → never caches.
    assert not should_cache_prefix(
        _areq(SMALL_SYSTEM, cache_policy=CachePolicy()),
        "claude-3-5-sonnet-latest",
        Cap.ANTHROPIC_EXPLICIT,
    )
