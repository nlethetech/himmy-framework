"""The ``/model`` (no-args) picker: render the active model + every available one.

The in-chat REPL command ``/model`` with no arguments is a *picker* — it shows the
currently-active provider/model and ALL currently-available providers + models, each
with its USD cost (input/output per 1M tokens) so a user can see, at a glance, what
they can switch to and what it would cost. ``/model <name> [provider]`` still switches.

The rendering is factored into :func:`render_model_picker`, a pure function that takes
the already-built local catalog (:func:`himmy.services.inference.compare.build_model_catalog`),
the cloud key availability (an ``env`` mapping), a colour palette, and a ``price_for``
lookup — so it can be unit-tested with synthetic inputs and NO live model call or
network. The REPL handler builds those inputs and prints the returned string.

Cloud providers are NOT in the local catalog (it only enumerates ollama + claude-cli),
so a SHORT curated, price-known set is listed per available cloud provider, plus that
provider's configured default model. Prices are ALWAYS looked up via ``price_for`` —
never hard-coded here — so they stay current with the synced LiteLLM table.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from himmy.services.inference.models import ModelPrice

#: Cloud providers gated by an env key, in the order ``himmy doctor`` / the wizard
#: check them. The env var both signals availability AND (for openrouter) names a
#: default model. Easy to edit: add a provider here + a curated list below.
CLOUD_KEY_ENV: tuple[tuple[str, str], ...] = (
    ("openrouter", "OPENROUTER_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
)

#: A SHORT curated, price-known set of models per cloud provider — the ones a user is
#: most likely to want. Prices are looked up at render time (never stored here); a model
#: with no known price renders ``price n/a`` rather than a fake ``$0``. Edit freely.
CURATED_CLOUD_MODELS: dict[str, tuple[str, ...]] = {
    "openrouter": (
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-3.5-haiku",
        "meta-llama/llama-3.3-70b-instruct",
    ),
    "openai": (
        "gpt-4o-mini",
        "gpt-4o",
    ),
    "anthropic": (
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
    ),
}

#: Per-provider env var naming a user-configured default model (listed first, marked).
_DEFAULT_MODEL_ENV: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_MODEL", "HIMMY_EXAMPLES_MODEL"),
    "openai": ("HIMMY_EXAMPLES_MODEL",),
    "anthropic": ("HIMMY_EXAMPLES_MODEL",),
}

#: Providers whose calls never cost API money (local CLI / local server / offline stub).
_FREE_PROVIDERS: frozenset[str] = frozenset({"claude-cli", "ollama", "stub"})


def provider_aware_price_for(
    *,
    fetch: Any | None = None,
    now: Callable[[], float] | None = None,
) -> Callable[[str], ModelPrice]:
    """A ``price_for`` that resolves openrouter targets LIVE, others from the static table.

    The render helper calls ``price_for(f"{provider}:{model}")``. This wrapper inspects
    that ``provider:`` prefix: ``openrouter:`` models go through the live OpenRouter
    resolver (live price first, static fallback, fully offline-safe), and every other
    target uses the existing static :func:`himmy.services.inference.pricing.price_for`.
    ``fetch`` / ``now`` are injectable so the live source can be unit-tested without
    touching the network. The returned callable is pure-by-reference (no hidden state
    beyond the in-process live-price cache).
    """
    from himmy.services.inference import pricing

    def _price(model: str) -> ModelPrice:
        raw = (model or "").strip()
        if raw.startswith("openrouter:"):
            target = raw.split(":", 1)[1]
            return pricing.openrouter_price_for(target, fetch=fetch, now=now)
        return pricing.price_for(model)

    return _price


def cloud_key_env(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """``{provider: key-present}`` for each cloud provider, from ``environ`` (or os.environ).

    Mirrors how ``himmy doctor`` / the init wizard detect cloud availability: a provider
    is available iff its API-key env var is set and non-empty. Never returns or logs the
    key value itself — only the boolean.
    """
    env = environ if environ is not None else os.environ
    return {provider: bool(env.get(key)) for provider, key in CLOUD_KEY_ENV}


def _provider_default_model(
    provider: str, environ: Mapping[str, str]
) -> str | None:
    """A user-configured default model for ``provider`` from env, if any (else ``None``)."""
    for env_key in _DEFAULT_MODEL_ENV.get(provider, ()):  # noqa: SIM110
        val = environ.get(env_key)
        if val:
            return val.strip()
    return None


def _cost_label(provider: str, model: str, price_for: Callable[[str], ModelPrice]) -> str:
    """Cost tag for one model line.

    Local/offline providers cost no API money → ``free · local``. Otherwise look the
    price up (by ``provider:model`` so the table's prefix-stripping applies) and render
    ``$<in>/$<out> per 1M``; an unknown price renders ``price n/a`` (never a fake $0).
    """
    if provider in _FREE_PROVIDERS:
        return "free · local"
    price = price_for(f"{provider}:{model}")
    if price.input_per_1k <= 0 and price.output_per_1k <= 0:
        return "price n/a"
    return f"${price.input_per_1k * 1000:.2f}/${price.output_per_1k * 1000:.2f} per 1M"


def _active_cost_label(
    provider: str | None, model: str | None, price_for: Callable[[str], ModelPrice]
) -> str:
    """Cost tag for the ACTIVE line (same rules, with a None-provider fallback)."""
    if not provider or provider in _FREE_PROVIDERS:
        return "local · no API cost"
    return _cost_label(provider, model or "", price_for)


def _is_active(
    provider: str, model: str, active_provider: str | None, active_model: str | None
) -> bool:
    """True when (provider, model) is the currently-active pair (model match required)."""
    if active_model is None:
        # No explicit active model (default) → mark the provider's first line only by
        # provider match is too loose; require a model too, so nothing is mismarked.
        return False
    return provider == active_provider and model == active_model


def render_model_picker(
    catalog: Sequence[Mapping[str, Any]],
    *,
    active_provider: str | None,
    active_model: str | None,
    color: Mapping[str, str],
    price_for: Callable[[str], ModelPrice],
    cloud_available: Mapping[str, bool],
    environ: Mapping[str, str] | None = None,
) -> str:
    """Render the ``/model`` picker as a single string (pure — no I/O, no network).

    ``catalog`` is the local-provider catalog from
    :func:`himmy.services.inference.compare.build_model_catalog` (ollama + claude-cli).
    ``cloud_available`` is ``{provider: key-present}`` (see :func:`cloud_key_env`).
    ``color`` is the REPL's ``styles()`` palette (``dim``/``snow``/``gold``/``faint``/
    ``green``/``reset`` …); pass an empty-string palette for plain output. ``price_for``
    resolves a :class:`ModelPrice`; ``environ`` (default ``os.environ``) supplies the
    per-provider default-model env vars. Returns the text the handler prints.
    """
    env = environ if environ is not None else os.environ
    c = {k: color.get(k, "") for k in ("dim", "snow", "gold", "faint", "green", "reset")}
    lines: list[str] = []

    # 1) ACTIVE line.
    active_label = " · ".join(p for p in (active_model, active_provider) if p) or "auto"
    active_cost = _active_cost_label(active_provider, active_model, price_for)
    lines.append(
        f"{c['snow']}active:{c['reset']} {c['gold']}{active_label}{c['reset']} "
        f"{c['faint']}({active_cost}){c['reset']}"
    )

    # 2) AVAILABLE providers & models.
    lines.append(f"{c['snow']}available:{c['reset']}")

    def _model_line(provider: str, model: str) -> str:
        cost = _cost_label(provider, model, price_for)
        active = _is_active(provider, model, active_provider, active_model)
        marker = f"{c['green']} (active){c['reset']}" if active else ""
        hint = f"/model {model} {provider}"
        return (
            f"  {c['dim']}{hint}{c['reset']}  "
            f"{c['faint']}{cost}{c['reset']}{marker}"
        )

    rendered_any = False

    # 2a) Local providers from the catalog (ollama, claude-cli).
    for entry in catalog:
        provider = str(entry.get("provider", ""))
        available = bool(entry.get("available"))
        models = entry.get("models") or []
        if not available:
            continue
        if not models:
            continue
        lines.append(f"  {c['snow']}{provider}{c['reset']}")
        for m in models:
            name = str(m.get("name") or m.get("model") or "")
            if not name:
                continue
            lines.append(_model_line(provider, name))
            rendered_any = True

    # 2b) Available cloud providers (key present): default model + curated set.
    for provider, _key in CLOUD_KEY_ENV:
        if not cloud_available.get(provider):
            continue
        default_model = _provider_default_model(provider, env)
        curated = list(CURATED_CLOUD_MODELS.get(provider, ()))
        ordered = ([default_model] if default_model else []) + curated
        deduped = list(dict.fromkeys(ordered))  # order-preserving de-dup
        if not deduped:
            continue
        lines.append(f"  {c['snow']}{provider}{c['reset']}")
        for name in deduped:
            lines.append(_model_line(provider, name))
            rendered_any = True

    if not rendered_any:
        lines.append(
            f"  {c['faint']}(no models detected — start Ollama, sign in to the "
            f"`claude` CLI, or set a cloud API key){c['reset']}"
        )

    # 3) Footer: how to switch + a one-line hint for unavailable cloud providers.
    missing = [p for p, _k in CLOUD_KEY_ENV if not cloud_available.get(p)]
    footer = f"{c['faint']}switch with /model <name> <provider>"
    if missing:
        hints = ", ".join(
            f"set {key} to enable {prov}"
            for prov, key in CLOUD_KEY_ENV
            if prov in missing
        )
        footer += f" · {hints}"
    footer += c["reset"]
    lines.append(footer)

    return "\n".join(lines)


__all__ = [
    "CLOUD_KEY_ENV",
    "CURATED_CLOUD_MODELS",
    "cloud_key_env",
    "provider_aware_price_for",
    "render_model_picker",
]
