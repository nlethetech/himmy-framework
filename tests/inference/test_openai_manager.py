"""Tests for the DIRECT OpenAI client manager (all offline, fake SDK client).

A fake async client (``client.chat.completions.create``) is injected so nothing touches
the network or needs a key. Covers text / json / structured / tool / stream paths,
usage+cost extraction, base_url override plumbing, and SDK-error → FAILED normalization.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.inference import pricing
from himmy.services.inference.models import (
    BoundTool,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ResponseFormat,
    ToolReturnRecord,
)
from himmy.services.inference.openai_manager import (
    OpenAIClientManager,
    _normalize_openai_error,
)
from tests.conftest import run_async

PRICED_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------- fake SDK types
class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 4,
) -> _Obj:
    message = _Obj(content=content, tool_calls=tool_calls)
    choice = _Obj(message=message)
    usage = _Obj(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return _Obj(choices=[choice], usage=usage)


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


def _req(**kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": PRICED_MODEL,
        "messages": [
            InferenceMessage(role="system", content="be brief"),
            InferenceMessage(role="user", content="hello"),
        ],
    }
    base.update(kw)
    return InferenceRequest(**base)


# ---------------------------------------------------------------------- text
def test_text_reply_maps_to_success_with_usage_and_cost() -> None:
    """A text completion maps to SUCCESS with token counts and a non-zero priced cost."""
    client = _FakeClient(
        _completion(content="hi from gpt", prompt_tokens=100, completion_tokens=12)
    )
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(
        mgr.generate(_req(generation_params={"temperature": 0.2, "max_tokens": 64}))
    )

    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "hi from gpt"
    assert resp.input_tokens == 100 and resp.output_tokens == 12
    assert resp.model_path == "openai:gpt-4o-mini"
    assert resp.provider_name == "openai"
    expected = pricing.price_for("openai:gpt-4o-mini").cost(
        input_tokens=100, output_tokens=12
    )
    assert resp.cost == pytest.approx(expected)
    assert resp.cost > 0.0
    seen = client.chat.completions.seen
    assert seen["model"] == "gpt-4o-mini"
    assert seen["temperature"] == 0.2
    assert seen["max_tokens"] == 64
    # roles are preserved straight through
    assert seen["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


# --------------------------------------------------------------------- json
def test_json_object_sets_response_format_and_parses() -> None:
    """JSON_OBJECT sets response_format and parses the returned JSON text."""
    client = _FakeClient(_completion(content='{"answer": 42}'))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(mgr.generate(_req(response_format=ResponseFormat.JSON_OBJECT)))

    assert resp.output_structured == {"answer": 42}
    assert client.chat.completions.seen["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------- structured
def test_structured_output_sets_json_schema_response_format() -> None:
    """STRUCTURED_OUTPUT sends the request schema via response_format=json_schema."""
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    client = _FakeClient(_completion(content='{"city": "Bardaghat"}'))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    resp = run_async(
        mgr.generate(
            _req(
                response_format=ResponseFormat.STRUCTURED_OUTPUT,
                output_json_schema=schema,
            )
        )
    )

    assert resp.output_structured == {"city": "Bardaghat"}
    rf = client.chat.completions.seen["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


# ------------------------------------------------------------------- tools
def test_tool_calls_are_extracted_and_executed() -> None:
    """message.tool_calls become ToolCallRecords and are executed via the executor."""
    ran: list[tuple[str, dict[str, Any]]] = []

    async def executor(name: str, args: dict[str, Any]) -> ToolReturnRecord:
        ran.append((name, args))
        return ToolReturnRecord(tool_call_id="x", tool_name=name, content="done")

    tc = _Obj(id="call_1", function=_Obj(name="lookup", arguments='{"q": "rates"}'))
    client = _FakeClient(_completion(content=None, tool_calls=[tc]))
    mgr = OpenAIClientManager(model=PRICED_MODEL, client=client)
    req = _req(
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[BoundTool(name="lookup", description="look up", read_only=True)],
        tool_executor=executor,
    )
    resp = run_async(mgr.generate(req))

    assert resp.status == InferenceStatus.SUCCESS
    assert [c.tool_name for c in resp.tool_calls] == ["lookup"]
    assert resp.tool_calls[0].args == {"q": "rates"}
    assert resp.tool_calls[0].tool_call_id == "call_1"
    assert resp.tool_returns[0].content == "done"
    assert resp.output_text is None
    assert ran == [("lookup", {"q": "rates"})]
    assert client.chat.completions.seen["tools"][0]["function"]["name"] == "lookup"


# --------------------------------------------------------------- base_url
def test_base_url_override_reaches_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """base_url + api_key are forwarded to AsyncOpenAI (OpenRouter-style endpoints)."""
    captured: dict[str, Any] = {}

    class _SpyClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = _FakeChat(_completion(content="routed"))

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _SpyClient)

    mgr = OpenAIClientManager(
        model="mistralai/mistral-small",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-unit",
    )
    resp = run_async(mgr.generate(_req(model_key="mistralai/mistral-small")))

    assert resp.output_text == "routed"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "sk-or-unit"


# ------------------------------------------------------------------ stream
def test_generate_stream_yields_deltas_then_response() -> None:
    """generate_stream yields content deltas, then a final InferenceResponse w/ usage."""

    chunks = [
        _Obj(choices=[_Obj(delta=_Obj(content="Hel"))], usage=None),
        _Obj(choices=[_Obj(delta=_Obj(content="lo!"))], usage=None),
        _Obj(choices=[], usage=_Obj(prompt_tokens=5, completion_tokens=2)),
    ]

    class _StreamCompletions:
        async def create(self, **kwargs: Any) -> Any:
            assert kwargs["stream"] is True

            async def _gen() -> Any:
                for c in chunks:
                    yield c

            return _gen()

    class _StreamChat:
        def __init__(self) -> None:
            self.completions = _StreamCompletions()

    class _StreamClient:
        def __init__(self) -> None:
            self.chat = _StreamChat()

    mgr = OpenAIClientManager(model=PRICED_MODEL, client=_StreamClient())

    async def _collect() -> tuple[list[str], Any]:
        deltas: list[str] = []
        final: Any = None
        async for piece in mgr.generate_stream(_req()):
            if isinstance(piece, str):
                deltas.append(piece)
            else:
                final = piece
        return deltas, final

    deltas, final = run_async(_collect())
    assert deltas == ["Hel", "lo!"]
    assert final is not None
    assert final.status == InferenceStatus.SUCCESS
    assert final.output_text == "Hello!"
    assert final.input_tokens == 5 and final.output_tokens == 2


# ------------------------------------------------------- error normalization
def test_sdk_error_is_normalized_to_failed_never_raises() -> None:
    """An exception from chat.completions.create becomes a FAILED response."""

    class _BoomCompletions:
        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    class _BoomClient:
        def __init__(self) -> None:
            self.chat = _Obj(completions=_BoomCompletions())

    mgr = OpenAIClientManager(model=PRICED_MODEL, client=_BoomClient())
    resp = run_async(mgr.generate(_req()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.UNKNOWN
    assert resp.model_path == "openai:gpt-4o-mini"


def test_rate_limit_error_is_retryable() -> None:
    """A RateLimitError-named exception maps to a retryable RATE_LIMITED code."""

    class RateLimitError(Exception):
        pass

    err = _normalize_openai_error(RateLimitError("429"))
    assert err.code == InferenceErrorCode.RATE_LIMITED
    assert err.retryable is True


def test_bad_request_error_is_invalid_and_not_retryable() -> None:
    """A BadRequestError-named exception maps to a non-retryable INVALID_REQUEST."""

    class BadRequestError(Exception):
        pass

    err = _normalize_openai_error(BadRequestError("nope"))
    assert err.code == InferenceErrorCode.INVALID_REQUEST
    assert err.retryable is False


def test_server_status_maps_to_provider_unavailable() -> None:
    """A 5xx status on a generic API error maps to retryable PROVIDER_UNAVAILABLE."""

    class APIStatusError(Exception):
        status_code = 503

    err = _normalize_openai_error(APIStatusError("down"))
    assert err.code == InferenceErrorCode.PROVIDER_UNAVAILABLE
    assert err.retryable is True
