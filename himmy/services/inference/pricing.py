"""Model pricing: turn token usage into real USD, with a table that stays current.

``InferenceResponse`` carries token counts from every provider, but dollar cost needs a
per-model price. Rather than hard-code numbers that go stale the moment a provider changes
its pricing, this resolves prices from a layered table:

    explicit override  >  synced file  >  bundled snapshot  >  unpriced ($0)

* **synced file** — ``himmy prices sync`` downloads the community-maintained, ecosystem-
  standard `LiteLLM model-price JSON <https://github.com/BerriAI/litellm>`_ to
  ``~/.himmy/model_prices.json``. That file is updated continuously as providers change
  prices, so a user keeps current without upgrading himmy. ``HIMMY_MODEL_PRICES`` points
  at a custom file.
* **bundled snapshot** — a small, dated offline fallback (``prices.json``) so common
  models price out of the box with no network.

The loader accepts both the LiteLLM flat shape (``{model: {input_cost_per_token, …}}``)
and himmy's ``{"_meta": …, "prices": {…}}`` shape, so a raw LiteLLM file drops straight in.
Model-name lookup is forgiving: a ``provider:`` prefix and a trailing ``-YYYY-MM-DD`` /
``-latest`` date suffix are stripped when an exact match misses.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from himmy.services.inference.models import ModelPrice

#: The canonical, continuously-updated community price source (LiteLLM).
LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

#: Where ``himmy prices sync`` writes the downloaded table.
SYNCED_PRICES_PATH = Path.home() / ".himmy" / "model_prices.json"

_BUNDLED_PRICES_PATH = Path(__file__).with_name("prices.json")
_DATE_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{6,8}|latest)$")

# Process cache of the resolved table (cleared by reload()).
_CACHE: dict[str, ModelPrice] | None = None


def _prices_from_raw(data: dict[str, Any]) -> dict[str, ModelPrice]:
    """Build a ``{model: ModelPrice}`` map from either supported JSON shape."""
    prices = data.get("prices")
    entries = prices if isinstance(prices, dict) else data
    table: dict[str, ModelPrice] = {}
    if not isinstance(entries, dict):
        return table
    for name, spec in entries.items():
        if not isinstance(spec, dict):
            continue  # skip metadata / sample_spec entries
        in_tok = spec.get("input_cost_per_token")
        out_tok = spec.get("output_cost_per_token")
        if in_tok is None and out_tok is None:
            continue
        # Optional per-model prompt-cache rate tier (multipliers of the input rate).
        # Absent -> None -> provider-family default in prompt_cache.resolve_cache_rates.
        read_mult = spec.get("cache_read_multiplier")
        write_mult = spec.get("cache_write_multiplier")
        # LiteLLM is USD-per-token; ModelPrice is per-1K.
        table[name] = ModelPrice(
            input_per_1k=float(in_tok or 0.0) * 1000.0,
            output_per_1k=float(out_tok or 0.0) * 1000.0,
            cache_read_multiplier=(
                float(read_mult) if read_mult is not None else None
            ),
            cache_write_multiplier=(
                float(write_mult) if write_mult is not None else None
            ),
        )
    return table


def _load_file(path: Path) -> dict[str, ModelPrice]:
    try:
        return _prices_from_raw(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}


def load_price_table(*, refresh: bool = False) -> dict[str, ModelPrice]:
    """The merged price table (bundled ← synced ← env override), cached per process."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    table: dict[str, ModelPrice] = {}
    if _BUNDLED_PRICES_PATH.exists():
        table.update(_load_file(_BUNDLED_PRICES_PATH))
    if SYNCED_PRICES_PATH.exists():
        table.update(_load_file(SYNCED_PRICES_PATH))
    env_path = os.environ.get("HIMMY_MODEL_PRICES")
    if env_path and Path(env_path).expanduser().exists():
        table.update(_load_file(Path(env_path).expanduser()))
    _CACHE = table
    return table


def reload() -> None:
    """Drop the cached table (call after ``sync``/config changes)."""
    global _CACHE
    _CACHE = None


def _candidates(model: str) -> list[str]:
    """Lookup keys to try for ``model``: raw, prefix-stripped, then date-stripped."""
    m = (model or "").strip()
    cands = [m]
    if ":" in m:  # "openai:gpt-4o-mini" / "anthropic:claude-..."
        cands.append(m.split(":", 1)[1])
        cands.append(m.rsplit(":", 1)[1])
    if "/" in m:  # "anthropic/claude-3-5-sonnet"
        cands.append(m.rsplit("/", 1)[1])
    base = cands[-1]
    stripped = _DATE_SUFFIX.sub("", base)
    if stripped != base:
        cands.append(stripped)
    return list(dict.fromkeys(c for c in cands if c))


def price_for(model: str) -> ModelPrice:
    """Resolve the :class:`ModelPrice` for a model string (``ModelPrice()`` if unknown)."""
    table = load_price_table()
    for cand in _candidates(model):
        if cand in table:
            return table[cand]
    return ModelPrice()


def is_priced(model: str) -> bool:
    """True when ``model`` has a non-zero price in the table."""
    p = price_for(model)
    return p.input_per_1k > 0 or p.output_per_1k > 0


def sync_prices(
    *,
    url: str = LITELLM_PRICES_URL,
    dest: Path = SYNCED_PRICES_PATH,
    timeout: float = 30.0,
) -> int:
    """Download the canonical price table to ``dest`` and return how many models it has.

    Network-only and opt-in (offline-first stays intact); raises on a failed download so
    the CLI can report it. Refreshes the in-process cache on success.
    """
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    table = _prices_from_raw(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {"source": url, "models": len(table)},
        "prices": {
            name: {
                "input_cost_per_token": p.input_per_1k / 1000.0,
                "output_cost_per_token": p.output_per_1k / 1000.0,
            }
            for name, p in table.items()
        },
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reload()
    return len(table)


__all__ = [
    "LITELLM_PRICES_URL",
    "SYNCED_PRICES_PATH",
    "load_price_table",
    "reload",
    "price_for",
    "is_priced",
    "sync_prices",
]
