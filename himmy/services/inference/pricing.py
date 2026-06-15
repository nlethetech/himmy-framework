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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from himmy.services.inference.models import ModelPrice

#: The canonical, continuously-updated community price source (LiteLLM).
LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

#: OpenRouter's live model catalog (includes per-token USD pricing for every model).
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

#: A fetch seam: ``(url, headers, timeout) -> parsed JSON dict``. The real default hits
#: the network with httpx; tests inject a fake that returns a synthetic payload so the
#: offline-first suite never touches the network.
Fetch = Callable[[str, dict[str, str], float], dict[str, Any]]

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
    reload_openrouter()


# --------------------------------------------------------------- live OpenRouter

#: How long a live OpenRouter price fetch is trusted before a refetch (seconds).
OPENROUTER_TTL_SECONDS = 3600.0

#: Timeout for the live fetch — small, so a hung endpoint never blocks the REPL.
OPENROUTER_TIMEOUT_SECONDS = 3.0

#: Process cache of the live OpenRouter table: (fetched_at_monotonic, table).
_OR_CACHE: tuple[float, dict[str, ModelPrice]] | None = None


def _network_disabled() -> bool:
    """True when an env flag asks himmy to stay fully offline (skip the live fetch)."""
    for var in ("HIMMY_OFFLINE", "HIMMY_NO_NETWORK"):
        val = os.environ.get(var, "").strip().lower()
        if val and val not in ("0", "false", "no", "off"):
            return True
    return False


def _default_fetch(url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """Real network fetch (httpx); only used in live/interactive runs, never in tests."""
    import httpx

    resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _prices_from_openrouter(data: dict[str, Any]) -> dict[str, ModelPrice]:
    """Parse the OpenRouter ``/models`` payload into a ``{model_id: ModelPrice}`` map.

    Shape: ``{"data": [{"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00000015",
    "completion": "0.0000006"}}, ...]}`` — ``prompt``/``completion`` are USD PER TOKEN
    (strings). ModelPrice is per-1K, so multiply by 1000. Malformed rows are skipped.
    """
    table: dict[str, ModelPrice] = {}
    rows = data.get("data")
    if not isinstance(rows, list):
        return table
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        pricing = row.get("pricing")
        if not isinstance(model_id, str) or not isinstance(pricing, dict):
            continue
        try:
            in_tok = float(pricing.get("prompt", 0.0) or 0.0)
            out_tok = float(pricing.get("completion", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        table[model_id] = ModelPrice(
            input_per_1k=in_tok * 1000.0,
            output_per_1k=out_tok * 1000.0,
        )
    return table


def reload_openrouter() -> None:
    """Drop the cached live OpenRouter table (forces a refetch on next access)."""
    global _OR_CACHE
    _OR_CACHE = None


def openrouter_live_prices(
    *,
    fetch: Fetch | None = None,
    now: Callable[[], float] | None = None,
    refresh: bool = False,
) -> dict[str, ModelPrice]:
    """Live OpenRouter ``{model_id: ModelPrice}``, cached in-process for ~1 hour.

    Offline-first is a hard invariant: the fetch is lazy, timeout-bounded, and on ANY
    failure (offline, timeout, non-200, parse error) returns an EMPTY table silently —
    it never raises and never blocks. ``fetch`` is injectable so tests pass a fake JSON
    payload and never touch the network; the default real fetch only runs in live use.
    The ``OPENROUTER_API_KEY`` env var (if present) is sent as a bearer token but never
    logged. Respects ``HIMMY_OFFLINE`` / ``HIMMY_NO_NETWORK`` by skipping the fetch.
    """
    global _OR_CACHE
    clock = now or time.monotonic
    if not refresh and _OR_CACHE is not None:
        fetched_at, cached = _OR_CACHE
        if clock() - fetched_at < OPENROUTER_TTL_SECONDS:
            return cached

    # An injected fetch (tests) overrides the offline flag; the real default respects it.
    if fetch is None and _network_disabled():
        _OR_CACHE = (clock(), {})
        return {}

    do_fetch = fetch or _default_fetch
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        data = do_fetch(OPENROUTER_MODELS_URL, headers, OPENROUTER_TIMEOUT_SECONDS)
        table = _prices_from_openrouter(data if isinstance(data, dict) else {})
    except Exception:  # noqa: BLE001 - offline-first: any failure → empty, never raise
        table = {}
    _OR_CACHE = (clock(), table)
    return table


def openrouter_price_for(
    model: str,
    *,
    fetch: Fetch | None = None,
    now: Callable[[], float] | None = None,
) -> ModelPrice:
    """Resolve an OpenRouter model's price: LIVE table first, static table fallback.

    The OpenRouter ids are slash-qualified (``openai/gpt-4o-mini``); the live table is
    keyed by exactly that id, with an optional ``openrouter:`` prefix stripped. Anything
    the live source doesn't know falls back to the forgiving static :func:`price_for`.
    """
    live = openrouter_live_prices(fetch=fetch, now=now)
    raw = (model or "").strip()
    key = raw.split(":", 1)[1] if raw.startswith("openrouter:") else raw
    hit = live.get(key)
    if hit is not None and (hit.input_per_1k > 0 or hit.output_per_1k > 0):
        return hit
    return price_for(model)


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
    "OPENROUTER_MODELS_URL",
    "SYNCED_PRICES_PATH",
    "Fetch",
    "load_price_table",
    "reload",
    "reload_openrouter",
    "price_for",
    "openrouter_live_prices",
    "openrouter_price_for",
    "is_priced",
    "sync_prices",
]
