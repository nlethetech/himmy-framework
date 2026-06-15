"""Model pricing: bundled snapshot, name normalization, precedence, and cost math."""

from __future__ import annotations

import json

import pytest

from himmy.services.inference import pricing
from himmy.services.inference.models import ModelPrice


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    # Point synced/env paths at a temp dir so tests never touch ~/.himmy, and reset cache.
    monkeypatch.setattr(pricing, "SYNCED_PRICES_PATH", tmp_path / "synced.json")
    monkeypatch.delenv("HIMMY_MODEL_PRICES", raising=False)
    pricing.reload()
    yield
    pricing.reload()


def test_bundled_snapshot_prices_known_models() -> None:
    p = pricing.price_for("gpt-4o-mini")
    assert p.input_per_1k == pytest.approx(1.5e-4)  # $0.15 / 1M = $0.00015 / 1K
    assert p.output_per_1k == pytest.approx(6.0e-4)
    assert pricing.is_priced("claude-3-5-sonnet")


def test_unknown_model_is_zero_not_guessed() -> None:
    p = pricing.price_for("totally-made-up-model-9000")
    assert p.input_per_1k == 0.0 and p.output_per_1k == 0.0
    assert not pricing.is_priced("totally-made-up-model-9000")


def test_name_normalization_strips_provider_prefix() -> None:
    assert pricing.price_for("openai:gpt-4o").input_per_1k > 0
    assert pricing.price_for("anthropic/claude-3-5-haiku").input_per_1k > 0


def test_name_normalization_strips_date_suffix() -> None:
    # A dated/aliased model resolves to its base price.
    assert pricing.price_for("gpt-4o-2024-08-06").input_per_1k == pytest.approx(
        pricing.price_for("gpt-4o").input_per_1k
    )
    assert pricing.price_for("claude-3-5-sonnet-latest").input_per_1k > 0


def test_cost_math() -> None:
    p = ModelPrice(input_per_1k=0.003, output_per_1k=0.015)  # $3 / $15 per 1M
    # 1000 in + 500 out → 1000/1000*0.003 + 500/1000*0.015 = 0.003 + 0.0075
    assert p.cost(input_tokens=1000, output_tokens=500) == pytest.approx(0.0105)


def test_synced_file_overrides_bundled(tmp_path, monkeypatch) -> None:
    synced = tmp_path / "synced.json"
    # LiteLLM flat shape, drop-in compatible.
    synced.write_text(
        json.dumps({"gpt-4o-mini": {"input_cost_per_token": 9.9e-7}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pricing, "SYNCED_PRICES_PATH", synced)
    pricing.reload()
    assert pricing.price_for("gpt-4o-mini").input_per_1k == pytest.approx(9.9e-4)


def test_env_file_has_highest_precedence(tmp_path, monkeypatch) -> None:
    override = tmp_path / "mine.json"
    override.write_text(
        json.dumps({"prices": {"gpt-4o": {"input_cost_per_token": 1.0e-9}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HIMMY_MODEL_PRICES", str(override))
    pricing.reload()
    assert pricing.price_for("gpt-4o").input_per_1k == pytest.approx(1.0e-6)


def test_litellm_per_token_is_converted_to_per_1k() -> None:
    table = pricing._prices_from_raw(
        {"m": {"input_cost_per_token": 2.0e-6, "output_cost_per_token": 8.0e-6}}
    )
    assert table["m"].input_per_1k == pytest.approx(2.0e-3)
    assert table["m"].output_per_1k == pytest.approx(8.0e-3)


# ---------------------------------------------------- live OpenRouter pricing (no net)

#: A fake OpenRouter ``/models`` payload (real shape: per-token USD strings).
_OR_PAYLOAD = {
    "data": [
        {
            "id": "openai/gpt-4o-mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "anthropic/claude-sonnet-4.5",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {"id": "junk/no-pricing"},  # malformed row must be skipped, not crash
    ]
}


@pytest.fixture(autouse=True)
def _isolate_openrouter(monkeypatch):
    # Force the live fetch off by default + reset the live cache; keep the suite offline.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pricing.reload_openrouter()
    yield
    pricing.reload_openrouter()


def test_openrouter_payload_parses_into_per_1k_modelprice() -> None:
    table = pricing._prices_from_openrouter(_OR_PAYLOAD)
    # per-token * 1000 = per-1K.
    assert table["openai/gpt-4o-mini"].input_per_1k == pytest.approx(1.5e-4)
    assert table["openai/gpt-4o-mini"].output_per_1k == pytest.approx(6.0e-4)
    assert table["anthropic/claude-sonnet-4.5"].input_per_1k == pytest.approx(3.0e-3)
    # The malformed (no-pricing) row is silently skipped, not present, no crash.
    assert "junk/no-pricing" not in table


def test_openrouter_live_prices_uses_injected_fetch_no_network() -> None:
    calls: list[tuple[str, dict[str, str], float]] = []

    def fake_fetch(url: str, headers: dict[str, str], timeout: float) -> dict:
        calls.append((url, headers, timeout))
        return _OR_PAYLOAD

    table = pricing.openrouter_live_prices(fetch=fake_fetch, now=lambda: 0.0)
    assert table["openai/gpt-4o-mini"].input_per_1k == pytest.approx(1.5e-4)
    # The injected fetch was called with the OpenRouter URL and a bounded timeout.
    assert calls and calls[0][0] == pricing.OPENROUTER_MODELS_URL
    assert calls[0][2] == pricing.OPENROUTER_TIMEOUT_SECONDS


def test_openrouter_live_prices_falls_back_to_empty_when_fetch_raises() -> None:
    def boom(url: str, headers: dict[str, str], timeout: float) -> dict:
        raise RuntimeError("offline / timeout")

    # Any failure → empty table, never raises (offline-first invariant).
    assert pricing.openrouter_live_prices(fetch=boom, now=lambda: 0.0) == {}


def test_openrouter_live_prices_tolerates_junk_payload() -> None:
    def junk(url: str, headers: dict[str, str], timeout: float) -> dict:
        return {"not": "the expected shape"}

    assert pricing.openrouter_live_prices(fetch=junk, now=lambda: 0.0) == {}


def test_openrouter_live_prices_ttl_cache_does_not_refetch() -> None:
    count = {"n": 0}

    def counting_fetch(url: str, headers: dict[str, str], timeout: float) -> dict:
        count["n"] += 1
        return _OR_PAYLOAD

    clock = {"t": 0.0}

    def now() -> float:
        return clock["t"]

    first = pricing.openrouter_live_prices(fetch=counting_fetch, now=now)
    assert count["n"] == 1 and first  # fetched once
    # A second call within the TTL must NOT refetch.
    clock["t"] = pricing.OPENROUTER_TTL_SECONDS - 1.0
    pricing.openrouter_live_prices(fetch=counting_fetch, now=now)
    assert count["n"] == 1
    # Past the TTL it refetches.
    clock["t"] = pricing.OPENROUTER_TTL_SECONDS + 1.0
    pricing.openrouter_live_prices(fetch=counting_fetch, now=now)
    assert count["n"] == 2


def test_openrouter_live_prices_sends_bearer_only_when_key_present(monkeypatch) -> None:
    seen: list[dict[str, str]] = []

    def capture(url: str, headers: dict[str, str], timeout: float) -> dict:
        seen.append(dict(headers))
        return _OR_PAYLOAD

    # No key → no Authorization header.
    pricing.openrouter_live_prices(fetch=capture, now=lambda: 0.0)
    assert "Authorization" not in seen[-1]
    # Key present → bearer header carries it.
    pricing.reload_openrouter()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    pricing.openrouter_live_prices(fetch=capture, now=lambda: 1.0)
    assert seen[-1]["Authorization"] == "Bearer sk-or-secret"


def test_openrouter_live_prices_respects_offline_flag(monkeypatch) -> None:
    monkeypatch.setenv("HIMMY_OFFLINE", "1")

    def should_not_run(url: str, headers: dict[str, str], timeout: float) -> dict:
        raise AssertionError("fetch must be skipped when an injected fetch is absent")

    # With no injected fetch + offline flag set, the real fetch is skipped → empty.
    assert pricing.openrouter_live_prices(now=lambda: 0.0) == {}


def test_openrouter_price_for_prefers_live_then_static_fallback() -> None:
    def fetch(url: str, headers: dict[str, str], timeout: float) -> dict:
        return _OR_PAYLOAD

    live = pricing.openrouter_price_for(
        "anthropic/claude-sonnet-4.5", fetch=fetch, now=lambda: 0.0
    )
    assert live.input_per_1k == pytest.approx(3.0e-3)  # from the live payload
    # A model the live source doesn't list falls back to the static table (gpt-4o-mini
    # is in the bundled snapshot), never crashing.
    pricing.reload_openrouter()
    fallback = pricing.openrouter_price_for(
        "totally-unknown/model-x", fetch=fetch, now=lambda: 0.0
    )
    assert fallback.input_per_1k == 0.0  # unknown everywhere → zero, not a guess
