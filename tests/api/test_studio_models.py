"""Tests for the Studio models router — providers, keys, and Ollama (no network).

Drives the FastAPI app in-process via TestClient. Every outbound probe
(Ollama tags, provider key pings, the pull stream) is monkeypatched so the
suite is deterministic and offline.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api.routers import studio_models as sm
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)

_KEY_NAMES = [p.key_name for p in sm._CATALOG if p.key_name]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    # Deterministic environment: no stray provider keys / default-model override.
    for name in [*_KEY_NAMES, "HIMMY_EXAMPLES_MODEL", "OLLAMA_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)

    from himmy.api.app import create_app

    yield TestClient(create_app())
    configure_secrets(None)
    for name in _KEY_NAMES:
        os.environ.pop(name, None)


@pytest.fixture
def ollama_down(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise(base_url: str) -> list[dict[str, Any]]:
        raise sm.OllamaUnreachableError("connection refused")

    monkeypatch.setattr(sm, "_fetch_ollama_tags", _raise)


@pytest.fixture
def ollama_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _tags(base_url: str) -> list[dict[str, Any]]:
        return [
            {
                "name": "llama3.2:latest",
                "size": 2_019_393_189,
                "modified_at": "2026-06-01T10:00:00Z",
                "details": {"parameter_size": "3.2B"},
            },
            {"name": "qwen2.5:7b", "size": None},
            {"no_name": True},
        ]

    monkeypatch.setattr(sm, "_fetch_ollama_tags", _tags)


# ---- GET /providers -------------------------------------------------------


def test_providers_catalog_and_no_key_leak(
    client: TestClient, ollama_down: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    r = client.get("/api/studio/models/providers")
    assert r.status_code == 200
    body = r.json()
    by = {p["name"]: p for p in body["providers"]}
    assert set(by) == {
        "ollama",
        "claude-cli",
        "openrouter",
        "anthropic",
        "openai",
        "pydantic-ai",
    }
    assert "sk-ant-supersecret" not in r.text  # key material never echoed
    assert by["anthropic"]["configured"] is True
    assert by["anthropic"]["detected_via"] == "env"
    assert by["anthropic"]["status"] == "detected via env"
    assert by["openai"]["configured"] is False
    assert by["openai"]["status"] == "not configured"
    assert by["ollama"]["configured"] is False  # probe monkeypatched down
    assert by["ollama"]["cost_hint"] == 0.0
    assert body["secrets_writable"] is True
    assert body["default_provider"] == "stub"


def test_providers_reports_saved_key_and_hydrates_env(
    client: TestClient, ollama_down: None
) -> None:
    r = client.post(
        "/api/studio/models/keys",
        json={"provider": "openrouter", "api_key": "or-key-123"},
    )
    assert r.status_code == 200
    assert "or-key-123" not in r.text
    assert r.json()["status"] == "key saved"
    # Simulate a restart: the env copy is gone, only the secret store has it.
    os.environ.pop("OPENROUTER_API_KEY", None)
    r2 = client.get("/api/studio/models/providers")
    by = {p["name"]: p for p in r2.json()["providers"]}
    assert by["openrouter"]["configured"] is True
    assert by["openrouter"]["detected_via"] == "secret"
    # providers GET re-hydrates the env so managers can actually use the key
    assert os.environ.get("OPENROUTER_API_KEY") == "or-key-123"
    assert "or-key-123" not in r2.text


def test_providers_ollama_running(client: TestClient, ollama_up: None) -> None:
    by = {
        p["name"]: p
        for p in client.get("/api/studio/models/providers").json()["providers"]
    }
    assert by["ollama"]["configured"] is True
    assert by["ollama"]["status"] == "running"


# ---- POST /keys ------------------------------------------------------------


def test_save_key_round_trip(client: TestClient, ollama_down: None) -> None:
    r = client.post(
        "/api/studio/models/keys",
        json={"provider": "anthropic", "api_key": "  sk-ant-xyz  "},
    )
    assert r.status_code == 200
    info = r.json()
    assert info["configured"] is True and info["detected_via"] == "secret"
    assert "sk-ant-xyz" not in r.text
    # Stripped + effective immediately for env-reading managers.
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-xyz"
    from himmy.config.secrets import get_secret

    assert get_secret("ANTHROPIC_API_KEY") == "sk-ant-xyz"


def test_save_key_unknown_provider_404(client: TestClient) -> None:
    r = client.post(
        "/api/studio/models/keys", json={"provider": "nope", "api_key": "x"}
    )
    assert r.status_code == 404


def test_save_key_local_provider_400(client: TestClient) -> None:
    r = client.post(
        "/api/studio/models/keys", json={"provider": "ollama", "api_key": "x"}
    )
    assert r.status_code == 400
    assert "doesn't use an API key" in r.json()["detail"]


def test_save_key_blank_or_oversized_422(client: TestClient) -> None:
    r = client.post(
        "/api/studio/models/keys", json={"provider": "openai", "api_key": "   "}
    )
    assert r.status_code == 422
    r2 = client.post(
        "/api/studio/models/keys",
        json={"provider": "openai", "api_key": "x" * 513},
    )
    assert r2.status_code == 422


def test_save_key_read_only_backend_409(client: TestClient) -> None:
    configure_secrets(EnvSecrets())  # not writable
    try:
        r = client.post(
            "/api/studio/models/keys",
            json={"provider": "openai", "api_key": "sk-x"},
        )
        assert r.status_code == 409
        assert "read-only" in r.json()["detail"]
    finally:
        configure_secrets(None)


# ---- POST /keys/validate ----------------------------------------------------


def test_validate_uses_provided_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    async def fake_probe(p: sm._Provider, api_key: str) -> sm.ValidateKeyResult:
        seen["provider"], seen["key"] = p.name, api_key
        return sm.ValidateKeyResult(
            ok=True, provider=p.name, detail="key accepted", status_code=200
        )

    monkeypatch.setattr(sm, "_probe_key", fake_probe)
    r = client.post(
        "/api/studio/models/keys/validate",
        json={"provider": "openai", "api_key": "sk-live-1"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert seen == {"provider": "openai", "key": "sk-live-1"}
    assert "sk-live-1" not in r.text


def test_validate_falls_back_to_saved_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(
        "/api/studio/models/keys",
        json={"provider": "anthropic", "api_key": "sk-stored"},
    )

    async def fake_probe(p: sm._Provider, api_key: str) -> sm.ValidateKeyResult:
        return sm.ValidateKeyResult(
            ok=(api_key == "sk-stored"), provider=p.name, detail="checked"
        )

    monkeypatch.setattr(sm, "_probe_key", fake_probe)
    r = client.post("/api/studio/models/keys/validate", json={"provider": "anthropic"})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_validate_without_any_key_400(client: TestClient) -> None:
    r = client.post("/api/studio/models/keys/validate", json={"provider": "openai"})
    assert r.status_code == 400
    assert "no key to validate" in r.json()["detail"]


def test_validate_unsupported_provider_400(client: TestClient) -> None:
    r = client.post(
        "/api/studio/models/keys/validate",
        json={"provider": "pydantic-ai", "api_key": "gw-key"},
    )
    assert r.status_code == 400
    assert "isn't supported" in r.json()["detail"]


def test_probe_key_maps_http_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import run_async

    p = sm._BY_NAME["openai"]

    class _Resp:
        def __init__(self, status_code: int, data: Any = None) -> None:
            self.status_code = status_code
            self._data = data

        def json(self) -> Any:
            if self._data is None:
                raise ValueError("no body")
            return self._data

    class _Client:
        def __init__(self, resp: _Resp) -> None:
            self._resp = resp

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str]) -> _Resp:
            assert headers["Authorization"].startswith("Bearer ")
            return self._resp

    def with_resp(resp: _Resp) -> sm.ValidateKeyResult:
        monkeypatch.setattr(sm.httpx, "AsyncClient", lambda timeout: _Client(resp))
        return run_async(sm._probe_key(p, "sk-test"))

    ok = with_resp(_Resp(200, {"data": [{"id": "gpt"}, {"id": "o"}]}))
    assert ok.ok is True and "2 models" in ok.detail

    bad = with_resp(_Resp(401))
    assert bad.ok is False and "invalid or expired" in bad.detail

    odd = with_resp(_Resp(503))
    assert odd.ok is False and "503" in odd.detail
    # the key never appears in any verdict
    for v in (ok, bad, odd):
        assert "sk-test" not in v.detail


# ---- GET /ollama -------------------------------------------------------------


def test_ollama_status_running(client: TestClient, ollama_up: None) -> None:
    r = client.get("/api/studio/models/ollama")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True
    names = [m["name"] for m in body["models"]]
    assert names == ["llama3.2:latest", "qwen2.5:7b"]  # nameless entry dropped
    assert body["models"][0]["size"] == 2_019_393_189
    assert body["models"][0]["parameter_size"] == "3.2B"


def test_ollama_status_down_is_graceful(client: TestClient, ollama_down: None) -> None:
    r = client.get("/api/studio/models/ollama")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False and body["models"] == []
    assert "ollama serve" in body["detail"]


# ---- POST /ollama/pull --------------------------------------------------------


def _sse_events(text: str) -> list[dict[str, Any]]:
    events = []
    for frame in text.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_ollama_pull_streams_progress(
    client: TestClient, ollama_up: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_frames(base_url: str, model: str) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "progress", "status": "pulling manifest"}
        yield {
            "type": "progress",
            "status": "downloading",
            "completed": 512,
            "total": 1024,
        }
        yield {"type": "done", "model": model}

    monkeypatch.setattr(sm, "_pull_frames", fake_frames)
    r = client.post("/api/studio/models/ollama/pull", json={"model": "llama3.2"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert [e["type"] for e in events] == ["progress", "progress", "done"]
    assert events[1]["completed"] == 512 and events[1]["total"] == 1024


def test_ollama_pull_when_down_409(client: TestClient, ollama_down: None) -> None:
    r = client.post("/api/studio/models/ollama/pull", json={"model": "llama3.2"})
    assert r.status_code == 409
    assert "ollama serve" in r.json()["detail"]


def test_ollama_pull_bad_model_name_422(client: TestClient, ollama_up: None) -> None:
    r = client.post(
        "/api/studio/models/ollama/pull", json={"model": "bad name; rm -rf"}
    )
    assert r.status_code == 422
    r2 = client.post("/api/studio/models/ollama/pull", json={"model": "x" * 201})
    assert r2.status_code == 422


def test_ollama_pull_error_frame_terminates_stream(
    client: TestClient, ollama_up: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_frames(base_url: str, model: str) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "progress", "status": "pulling manifest"}
        yield {"type": "error", "message": "manifest not found"}

    monkeypatch.setattr(sm, "_pull_frames", fake_frames)
    r = client.post("/api/studio/models/ollama/pull", json={"model": "ghost:1b"})
    events = _sse_events(r.text)
    assert events[-1] == {"type": "error", "message": "manifest not found"}


def test_pull_frames_parses_ollama_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real proxy converts Ollama's JSON lines into typed frames."""
    from tests.conftest import run_async

    lines = [
        json.dumps({"status": "pulling manifest"}),
        "not-json",
        json.dumps({"status": "downloading", "completed": 10, "total": 100}),
        json.dumps({"status": "success"}),
    ]

    class _Resp:
        status_code = 200

        async def aread(self) -> bytes:
            return b""

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in lines:
                yield line

    class _StreamCtx:
        async def __aenter__(self) -> _Resp:
            return _Resp()

        async def __aexit__(self, *a: Any) -> None:
            return None

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        def stream(self, method: str, url: str, json: Any) -> _StreamCtx:
            assert url.endswith("/api/pull") and json["name"] == "llama3.2"
            return _StreamCtx()

    monkeypatch.setattr(sm.httpx, "AsyncClient", lambda timeout: _Client())

    async def collect() -> list[dict[str, Any]]:
        return [f async for f in sm._pull_frames("http://localhost:11434", "llama3.2")]

    frames = run_async(collect())
    assert [f["type"] for f in frames] == ["progress", "progress", "done"]
    assert frames[1] == {
        "type": "progress",
        "status": "downloading",
        "completed": 10,
        "total": 100,
    }
