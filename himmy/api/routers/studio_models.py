"""API kernel: Studio models router — providers, API keys, and local Ollama models.

Mounted under ``/api/studio/models`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). The "pick your brain" surface:

  * ``GET  /providers``      — per-provider detection/configuration status (no key
    material is ever echoed), plus the active default + secrets writability.
  * ``POST /keys``           — save a provider API key via the secrets backend
    (same writable-check + 409 pattern as Connections).
  * ``POST /keys/validate``  — a cheap live check of a key (short-timeout models
    list / auth ping); the key is never logged and never echoed back.
  * ``GET  /ollama``         — local Ollama tags (graceful when Ollama is down).
  * ``POST /ollama/pull``    — SSE proxy of Ollama's ``/api/pull`` progress,
    bounded in duration.

Offline-first: every probe fails closed into a clear status string, never a 500.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from himmy.api.routers.studio_common import build_studio_router
from himmy.config.secrets import get_secret, get_writable_provider

router = build_studio_router("models", tag="studio-models")


# ---- provider catalog ----------------------------------------------------


@dataclass(frozen=True)
class _Provider:
    """One entry of the static provider catalog (what himmy can run on)."""

    name: str  # the --provider string build_manager_for() accepts
    label: str  # human name for the GUI
    kind: str  # "local" | "key"
    cost_hint: float  # relative cost; 0.0 = free local (routing order)
    key_name: str | None = None  # env/secret name holding the API key
    validate_url: str | None = None  # cheap live-check endpoint (None = unsupported)
    auth_style: str = "bearer"  # "bearer" | "anthropic"
    hint: str | None = None  # one-line setup hint for the GUI


_CATALOG: tuple[_Provider, ...] = (
    _Provider(
        name="ollama",
        label="Ollama",
        kind="local",
        cost_hint=0.0,
        hint="free local models — `ollama serve` then `ollama pull llama3.2`",
    ),
    _Provider(
        name="claude-cli",
        label="Claude CLI",
        kind="local",
        cost_hint=0.0,
        hint="uses your local `claude` sign-in (Max plan = no per-token cost)",
    ),
    _Provider(
        name="openrouter",
        label="OpenRouter",
        kind="key",
        cost_hint=0.5,
        key_name="OPENROUTER_API_KEY",
        validate_url="https://openrouter.ai/api/v1/auth/key",
        auth_style="bearer",
        hint="one key, many models — https://openrouter.ai/keys",
    ),
    _Provider(
        name="anthropic",
        label="Anthropic API",
        kind="key",
        cost_hint=1.0,
        key_name="ANTHROPIC_API_KEY",
        validate_url="https://api.anthropic.com/v1/models",
        auth_style="anthropic",
        hint="direct Claude API — https://console.anthropic.com",
    ),
    _Provider(
        name="openai",
        label="OpenAI API",
        kind="key",
        cost_hint=1.0,
        key_name="OPENAI_API_KEY",
        validate_url="https://api.openai.com/v1/models",
        auth_style="bearer",
        hint="direct GPT API — https://platform.openai.com/api-keys",
    ),
    _Provider(
        name="pydantic-ai",
        label="Pydantic AI Gateway",
        kind="key",
        cost_hint=1.0,
        key_name="PYDANTIC_AI_GATEWAY_API_KEY",
        validate_url=None,  # no cheap public ping endpoint
        auth_style="bearer",
        hint="gateway key for the pydantic-ai provider extra",
    ),
)

_BY_NAME: dict[str, _Provider] = {p.name: p for p in _CATALOG}

_READ_ONLY_DETAIL = (
    "the active secrets backend is read-only; set HIMMY_SECRETS=keychain or file"
)


# ---- response models -----------------------------------------------------


class ProviderInfo(BaseModel):
    """Detection/configuration status for one provider. Never carries key bytes."""

    name: str
    label: str
    kind: str  # "local" | "key"
    configured: bool
    detected_via: str | None  # "server" | "binary" | "env" | "secret" | None
    status: str  # human one-liner: "detected via env" / "key saved" / ...
    key_name: str | None = None
    cost_hint: float
    validate_supported: bool = False
    hint: str | None = None


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
    secrets_writable: bool
    default_provider: str
    default_detail: str


class SaveKeyRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=40)
    api_key: str = Field(..., min_length=1, max_length=512)


class ValidateKeyRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=40)
    api_key: str | None = Field(None, min_length=1, max_length=512)


class ValidateKeyResult(BaseModel):
    ok: bool
    provider: str
    detail: str
    status_code: int | None = None


class OllamaModel(BaseModel):
    name: str
    size: int | None = None
    modified_at: str | None = None
    parameter_size: str | None = None


class OllamaStatus(BaseModel):
    running: bool
    base_url: str
    models: list[OllamaModel]
    detail: str | None = None


class OllamaPullRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=200)


# ---- helpers ---------------------------------------------------------------


def _ollama_base_url() -> str:
    """The local Ollama endpoint (override with OLLAMA_BASE_URL for non-default)."""
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


class OllamaUnreachableError(RuntimeError):
    """Raised when the local Ollama server does not answer."""


async def _fetch_ollama_tags(base_url: str) -> list[dict[str, Any]]:
    """Raw ``/api/tags`` model dicts; raises :class:`OllamaUnreachableError` if down."""
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
    except (httpx.HTTPError, ValueError) as exc:
        raise OllamaUnreachableError(str(exc)) from exc
    return [m for m in models if isinstance(m, dict)]


def _saved_key_source(key_name: str) -> str | None:
    """Where a key currently comes from: "secret" (store) > "env" > None.

    Only presence is inspected — the value never leaves this function.
    """
    writable = get_writable_provider()
    if writable is not None and writable.get(key_name):
        return "secret"
    if get_secret(key_name):
        return "env"
    return None


def _apply_saved_keys_to_env() -> None:
    """Hydrate ``os.environ`` from the secret store for every catalog key.

    Provider managers (``build_manager_for``) read keys from the environment, so a
    key saved to the keychain/file backend must be reflected there to take effect.
    An explicitly-set env var is never clobbered.
    """
    writable = get_writable_provider()
    if writable is None:
        return
    for p in _CATALOG:
        if p.key_name and p.key_name not in os.environ:
            value = writable.get(p.key_name)
            if value:
                os.environ[p.key_name] = value


def _default_provider() -> tuple[str, str]:
    """Mirror :func:`himmy.runtime.builder.build_inference`'s auto-select, cheaply."""
    from himmy.runtime.builder import _provider_key_present, _pydantic_ai_available

    model = os.environ.get("HIMMY_EXAMPLES_MODEL")
    if model and _pydantic_ai_available() and _provider_key_present():
        return (
            "pydantic-ai",
            f"HIMMY_EXAMPLES_MODEL={model} with a provider key — real model by default",
        )
    return (
        "stub",
        "offline deterministic stub — runs pick a real provider per request "
        "(or set HIMMY_EXAMPLES_MODEL + a key)",
    )


async def _provider_info(
    p: _Provider, *, ollama_up: bool | None = None
) -> ProviderInfo:
    """Build one provider's status row. ``ollama_up`` lets callers share one probe."""
    configured = False
    detected_via: str | None = None
    status = "not configured"

    if p.name == "ollama":
        if ollama_up is None:
            try:
                await _fetch_ollama_tags(_ollama_base_url())
                ollama_up = True
            except OllamaUnreachableError:
                ollama_up = False
        if ollama_up:
            configured, detected_via, status = True, "server", "running"
        elif shutil.which("ollama"):
            detected_via, status = "binary", "installed · not running"
        else:
            status = "not installed"
    elif p.name == "claude-cli":
        if shutil.which("claude"):
            configured, detected_via, status = True, "binary", "detected via PATH"
        else:
            status = "not installed"
    elif p.key_name:
        source = _saved_key_source(p.key_name)
        if source == "secret":
            configured, detected_via, status = True, "secret", "key saved"
        elif source == "env":
            configured, detected_via, status = True, "env", "detected via env"

    return ProviderInfo(
        name=p.name,
        label=p.label,
        kind=p.kind,
        configured=configured,
        detected_via=detected_via,
        status=status,
        key_name=p.key_name,
        cost_hint=p.cost_hint,
        validate_supported=p.validate_url is not None,
        hint=p.hint,
    )


def _require_key_provider(name: str) -> _Provider:
    """Resolve a key-based provider or raise the right 4xx."""
    p = _BY_NAME.get(name)
    if p is None:
        raise HTTPException(status_code=404, detail=f"unknown provider {name!r}")
    if p.kind != "key" or p.key_name is None:
        raise HTTPException(
            status_code=400,
            detail=f"provider {name!r} is detected locally and doesn't use an API key",
        )
    return p


_VALIDATE_TIMEOUT = 8.0


async def _probe_key(p: _Provider, api_key: str) -> ValidateKeyResult:
    """One short-timeout authenticated GET against the provider; never logs the key."""
    assert p.validate_url is not None  # callers check validate_supported first
    if p.auth_style == "anthropic":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    else:
        headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=_VALIDATE_TIMEOUT) as client:
            resp = await client.get(p.validate_url, headers=headers)
    except httpx.TimeoutException:
        return ValidateKeyResult(
            ok=False,
            provider=p.name,
            detail=f"timed out after {_VALIDATE_TIMEOUT:.0f}s — network or provider "
            "issue",
        )
    except httpx.HTTPError as exc:
        return ValidateKeyResult(
            ok=False,
            provider=p.name,
            detail=f"network error reaching {p.label}: {exc.__class__.__name__}",
        )
    if resp.status_code == 200:
        detail = "key accepted"
        try:
            data = resp.json().get("data")
            if isinstance(data, list):
                detail = f"key accepted · {len(data)} models visible"
        except ValueError:
            pass
        return ValidateKeyResult(
            ok=True, provider=p.name, detail=detail, status_code=200
        )
    if resp.status_code in (401, 403):
        return ValidateKeyResult(
            ok=False,
            provider=p.name,
            detail=f"rejected (HTTP {resp.status_code}) — invalid or expired key",
            status_code=resp.status_code,
        )
    return ValidateKeyResult(
        ok=False,
        provider=p.name,
        detail=f"unexpected HTTP {resp.status_code} from {p.label}",
        status_code=resp.status_code,
    )


# ---- endpoints -------------------------------------------------------------


@router.get("/providers", response_model=ProvidersResponse)
async def providers() -> ProvidersResponse:
    """Every provider himmy can run on: detected/configured, how, and the default.

    Key material is never echoed — only presence + where it was found.
    """
    _apply_saved_keys_to_env()
    try:
        await _fetch_ollama_tags(_ollama_base_url())
        ollama_up = True
    except OllamaUnreachableError:
        ollama_up = False
    infos = [await _provider_info(p, ollama_up=ollama_up) for p in _CATALOG]
    default, default_detail = _default_provider()
    return ProvidersResponse(
        providers=infos,
        secrets_writable=get_writable_provider() is not None,
        default_provider=default,
        default_detail=default_detail,
    )


@router.post("/keys", response_model=ProviderInfo)
async def save_key(body: SaveKeyRequest) -> ProviderInfo:
    """Save a provider API key to the writable secrets backend (never echoed)."""
    p = _require_key_provider(body.provider)
    api_key = body.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=422, detail="api_key is blank")
    writable = get_writable_provider()
    if writable is None:
        raise HTTPException(status_code=409, detail=_READ_ONLY_DETAIL)
    assert p.key_name is not None
    writable.set(p.key_name, api_key)
    # Managers read keys from the environment — make the save effective immediately.
    os.environ[p.key_name] = api_key
    return await _provider_info(p, ollama_up=False)


@router.post("/keys/validate", response_model=ValidateKeyResult)
async def validate_key(body: ValidateKeyRequest) -> ValidateKeyResult:
    """Cheap live check of a key (the given one, else the saved one).

    A failed check is a 200 with ``ok=false`` + the reason — only a missing key or
    an unsupported provider is a 4xx. The key never appears in logs or responses.
    """
    p = _require_key_provider(body.provider)
    if p.validate_url is None:
        raise HTTPException(
            status_code=400,
            detail=f"live validation isn't supported for {p.label} — save the key "
            "and run an agent to test it",
        )
    assert p.key_name is not None
    api_key = (body.api_key or "").strip() or get_secret(p.key_name)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"no key to validate — include api_key or save one for {p.label}",
        )
    return await _probe_key(p, api_key)


@router.get("/ollama", response_model=OllamaStatus)
async def ollama_status() -> OllamaStatus:
    """Local Ollama tags (name/size/modified). Graceful when Ollama is down."""
    base_url = _ollama_base_url()
    try:
        raw = await _fetch_ollama_tags(base_url)
    except OllamaUnreachableError:
        return OllamaStatus(
            running=False,
            base_url=base_url,
            models=[],
            detail=f"Ollama isn't reachable at {base_url} — start it with "
            "`ollama serve`",
        )
    models: list[OllamaModel] = []
    for m in raw:
        name = m.get("name")
        if not isinstance(name, str) or not name:
            continue
        size = m.get("size")
        details = m.get("details") if isinstance(m.get("details"), dict) else {}
        param = details.get("parameter_size") if details else None
        models.append(
            OllamaModel(
                name=name,
                size=size if isinstance(size, int) else None,
                modified_at=(
                    m.get("modified_at")
                    if isinstance(m.get("modified_at"), str)
                    else None
                ),
                parameter_size=param if isinstance(param, str) else None,
            )
        )
    return OllamaStatus(running=True, base_url=base_url, models=models)


_PULL_NAME_RE = re.compile(r"^[A-Za-z0-9._\-:/]+$")
_PULL_DEFAULT_TIMEOUT = 1800.0  # 30 min — enough for a large tag on slow links
_PULL_MAX_TIMEOUT = 7200.0


def _pull_deadline_seconds() -> float:
    """Bound on a single pull stream (HIMMY_OLLAMA_PULL_TIMEOUT, clamped)."""
    raw = os.environ.get("HIMMY_OLLAMA_PULL_TIMEOUT", "")
    try:
        value = float(raw) if raw else _PULL_DEFAULT_TIMEOUT
    except ValueError:
        value = _PULL_DEFAULT_TIMEOUT
    return max(30.0, min(value, _PULL_MAX_TIMEOUT))


async def _pull_frames(base_url: str, model: str) -> AsyncIterator[dict[str, Any]]:
    """Proxy Ollama's streamed ``/api/pull`` progress as plain dict frames.

    Frames: ``{"type": "progress", status, completed?, total?}`` then a terminal
    ``{"type": "done"}`` or ``{"type": "error", message}``. Bounded by the pull
    deadline so an abandoned stream can't run forever.
    """
    deadline = time.monotonic() + _pull_deadline_seconds()
    timeout = httpx.Timeout(60.0, connect=5.0)
    try:
        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream(
                "POST", f"{base_url}/api/pull", json={"name": model, "stream": True}
            ) as resp,
        ):
            if resp.status_code != 200:
                await resp.aread()
                yield {
                    "type": "error",
                    "message": f"ollama answered HTTP {resp.status_code} for {model!r}",
                }
                return
            async for line in resp.aiter_lines():
                if time.monotonic() > deadline:
                    yield {
                        "type": "error",
                        "message": "pull timed out — try again or pull from a terminal",
                    }
                    return
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("error"):
                    yield {"type": "error", "message": str(data["error"])}
                    return
                if data.get("status") == "success":
                    yield {"type": "done", "model": model}
                    return
                frame: dict[str, Any] = {
                    "type": "progress",
                    "status": str(data.get("status") or ""),
                }
                if isinstance(data.get("completed"), int):
                    frame["completed"] = data["completed"]
                if isinstance(data.get("total"), int):
                    frame["total"] = data["total"]
                yield frame
        yield {"type": "done", "model": model}
    except httpx.HTTPError as exc:
        yield {
            "type": "error",
            "message": f"lost the connection to ollama: {exc.__class__.__name__}",
        }


def _sse(event: dict[str, Any]) -> str:
    """Encode one event dict as a Server-Sent-Events ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/ollama/pull")
async def ollama_pull(body: OllamaPullRequest) -> StreamingResponse:
    """Pull a model through the local Ollama server, streaming progress over SSE."""
    model = body.model.strip()
    if not model or not _PULL_NAME_RE.match(model):
        raise HTTPException(
            status_code=422,
            detail="model must look like `llama3.2` or `qwen2.5:7b` "
            "(letters, digits, . _ - : / only)",
        )
    base_url = _ollama_base_url()
    try:
        await _fetch_ollama_tags(base_url)
    except OllamaUnreachableError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Ollama isn't reachable at {base_url} — start it with "
            "`ollama serve`",
        ) from exc

    async def _stream() -> AsyncIterator[str]:
        try:
            async for frame in _pull_frames(base_url, model):
                yield _sse(frame)
        except Exception as exc:  # noqa: BLE001 - surface as a terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["router"]


# ---- OpenRouter catalog (public, no key required) -------------------------

#: Server-side cache for the OpenRouter model catalog: the list is ~300 models
#: and changes rarely, so one upstream fetch per hour is plenty. (data, fetched_at)
_OPENROUTER_CACHE: dict[str, Any] = {"data": None, "at": 0.0}
_OPENROUTER_TTL_S = 3600.0
_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"


def _per_million(price_per_token: Any) -> float | None:
    """OpenRouter prices are $/token strings ('0.0000005') — convert to $/1M."""
    try:
        value = float(price_per_token)
    except (TypeError, ValueError):
        return None
    if value < 0:  # OpenRouter marks dynamic pricing as -1
        return None
    return round(value * 1_000_000, 4)


def _map_openrouter_model(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Trim one upstream catalog entry to what the picker needs."""
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    pricing = raw.get("pricing") or {}
    prompt = _per_million(pricing.get("prompt"))
    completion = _per_million(pricing.get("completion"))
    return {
        "id": model_id,
        "name": raw.get("name") or model_id,
        "context_length": raw.get("context_length"),
        "prompt_per_million": prompt,
        "completion_per_million": completion,
        "free": prompt == 0 and completion == 0,
    }


@router.get("/openrouter")
async def openrouter_catalog(refresh: bool = False) -> dict[str, Any]:
    """The OpenRouter model catalog with per-million pricing, cached for 1h.

    The endpoint is public upstream (no key needed), so users can browse
    models and prices before saving a key. ``?refresh=1`` bypasses the cache.
    """
    now = time.monotonic()
    cached = _OPENROUTER_CACHE["data"]
    if cached is not None and not refresh and now - _OPENROUTER_CACHE["at"] < _OPENROUTER_TTL_S:
        return {"models": cached, "cached": True}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(_OPENROUTER_URL)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - degrade to the stale cache, then a clear 503
        if cached is not None:
            return {"models": cached, "cached": True, "stale": True}
        raise HTTPException(
            status_code=503,
            detail="couldn't reach openrouter.ai for the model catalog — "
            "check your connection and try again",
        ) from exc
    models = [
        mapped
        for raw in payload.get("data", [])
        if isinstance(raw, dict) and (mapped := _map_openrouter_model(raw)) is not None
    ]
    models.sort(key=lambda m: str(m["name"]).lower())
    _OPENROUTER_CACHE["data"] = models
    _OPENROUTER_CACHE["at"] = now
    return {"models": models, "cached": False}
