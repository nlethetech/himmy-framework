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
