"""Structured-reply validation at the service boundary (QUICK WIN 3).

Managers return ``output_structured`` UNVALIDATED. ``InferenceService.run`` now
validates a parsed structured reply against the requested ``output_json_schema`` at
the single service boundary (so EVERY provider benefits), turning a parseable but
schema-violating reply into a FAILED(``OUTPUT_VALIDATION``) response instead of
silently shipping SUCCESS. A valid reply is left byte-identical, and callers running
their own richer validation (TypedAgent) opt out via
``validate_structured_output=False``.
"""

from __future__ import annotations

from typing import Any

from himmy.services.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceService,
)
from himmy.services.inference.local import OllamaClientManager
from himmy.services.inference.models import (
    InferenceErrorCode,
    InferenceStatus,
    ResponseFormat,
)
from himmy.services.inference.openai_manager import OpenAIClientManager
from tests.conftest import run_async

# A strict schema: ``age`` must be an integer >= 0; ``name`` required string.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def _structured_req(**kw: Any) -> InferenceRequest:
    base: dict[str, Any] = {
        "model_key": "default",
        "messages": [InferenceMessage(role="user", content="profile")],
        "response_format": ResponseFormat.STRUCTURED_OUTPUT,
        "output_json_schema": _SCHEMA,
    }
    base.update(kw)
    return InferenceRequest(**base)


class _FixedManager:
    """A manager returning a fixed ``output_structured`` (provider-agnostic shim)."""

    def __init__(self, structured: Any) -> None:
        self._structured = structured

    def resolve(self, model_key: str) -> str:  # pragma: no cover - trivial
        return model_key

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=None,
            output_structured=self._structured,
            provider_name="fixed",
        )


def _svc(structured: Any) -> InferenceService:
    return InferenceService(_FixedManager(structured), max_retries=0)


# --------------------------------------------------------- violation is caught
def test_schema_violating_reply_is_failed() -> None:
    """A parseable-but-non-conforming reply becomes FAILED(OUTPUT_VALIDATION)."""
    resp = run_async(_svc({"name": "ana", "age": -3}).run(_structured_req()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.OUTPUT_VALIDATION
    assert resp.error.retryable is False


def test_missing_required_field_is_failed() -> None:
    resp = run_async(_svc({"name": "ana"}).run(_structured_req()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.OUTPUT_VALIDATION


def test_extra_property_is_failed_under_additional_false() -> None:
    resp = run_async(_svc({"name": "a", "age": 1, "x": 9}).run(_structured_req()))
    assert resp.status == InferenceStatus.FAILED


# --------------------------------------------------------- valid reply unchanged
def test_valid_reply_is_unchanged() -> None:
    payload = {"name": "ana", "age": 30}
    resp = run_async(_svc(payload).run(_structured_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_structured == payload


def test_text_request_is_never_validated() -> None:
    """A non-structured request is untouched even if it carries odd structured data."""
    svc = _svc({"anything": True})
    req = InferenceRequest(
        model_key="default",
        messages=[InferenceMessage(role="user", content="hi")],
        response_format=ResponseFormat.TEXT,
    )
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.SUCCESS


# --------------------------------------------------------- opt-out (TypedAgent)
def test_opt_out_skips_validation() -> None:
    """``validate_structured_output=False`` leaves a violating reply as SUCCESS."""
    resp = run_async(
        _svc({"name": "ana", "age": -3}).run(
            _structured_req(validate_structured_output=False)
        )
    )
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_structured == {"name": "ana", "age": -3}


# --------------------------------------------------------- real OpenAI provider
class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _FakeCompletions:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def create(self, **kwargs: Any) -> Any:
        return self._result


class _FakeChat:
    def __init__(self, result: Any) -> None:
        self.completions = _FakeCompletions(result)


class _FakeClient:
    def __init__(self, result: Any) -> None:
        self.chat = _FakeChat(result)


def _completion(text: str) -> _Obj:
    message = _Obj(content=text, tool_calls=None)
    usage = _Obj(prompt_tokens=5, completion_tokens=2)
    return _Obj(choices=[_Obj(message=message)], usage=usage)


def test_openai_violating_structured_reply_caught_at_boundary() -> None:
    """A real OpenAI manager parses the JSON; the service then fails the violation."""
    client = _FakeClient(_completion('{"name": "ana", "age": -5}'))
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    svc = InferenceService(mgr, max_retries=0)
    resp = run_async(svc.run(_structured_req(model_key="gpt-4o-mini")))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.OUTPUT_VALIDATION


def test_openai_valid_structured_reply_succeeds() -> None:
    client = _FakeClient(_completion('{"name": "ana", "age": 5}'))
    mgr = OpenAIClientManager(model="gpt-4o-mini", client=client)
    svc = InferenceService(mgr, max_retries=0)
    resp = run_async(svc.run(_structured_req(model_key="gpt-4o-mini")))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_structured == {"name": "ana", "age": 5}


# --------------------------------------------------------- real Ollama provider
def test_ollama_violating_structured_reply_caught_at_boundary() -> None:
    async def _transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "message": {"content": '{"name": "ana", "age": -1}'},
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

    mgr = OllamaClientManager(model="llama3.2", transport=_transport)
    svc = InferenceService(mgr, max_retries=0)
    resp = run_async(svc.run(_structured_req()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.OUTPUT_VALIDATION
