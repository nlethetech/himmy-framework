"""CLI: ``himmy models`` (model catalog) + ``himmy compare`` (quick side-by-side).

Both commands run over the SAME shared seam Studio and ``/v1`` use
(:mod:`himmy.services.inference.compare`), so the catalog and the per-target compare
behave identically across the CLI, the web GUI, and the REST API (T3d).

``himmy bench`` stays the rigorous, multi-trial, graded-suite path; ``himmy compare`` is
the quick eyeball — one prompt, N models, output + tokens + cost + latency side-by-side.

The ``provider:model`` STRING lexing (e.g. ``ollama:qwen2.5:3b``, where the model itself
contains a colon) lives HERE in the CLI adapter — the shared helper takes already-split,
structured targets and never parses a user string.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.inference.compare import CompareTargetSpec


def _eprint(*args: object) -> None:
    """Print to stderr (diagnostics, never mixed into machine-readable stdout)."""
    print(*args, file=sys.stderr)


def _parse_target(raw: str) -> CompareTargetSpec | None:
    """Parse one ``provider:model`` token; ``None`` when it is malformed.

    The model may itself contain ``:`` (``ollama:qwen2.5:3b-instruct``), so only the FIRST
    colon separates provider from model — the rest is the model verbatim.
    """
    from himmy.services.inference.compare import CompareTargetSpec

    raw = raw.strip()
    if not raw:
        return None
    provider, sep, model = raw.partition(":")
    provider, model = provider.strip(), model.strip()
    if not sep or not provider or not model:
        return None
    return CompareTargetSpec(provider=provider, model=model)


# ---------------------------------------------------------------- himmy models


def cmd_models(args: argparse.Namespace) -> int:
    """List the available local providers + their models (with cached bench stats).

    Renders the shared :func:`himmy.services.inference.compare.build_model_catalog` — the
    SAME catalog Studio's ``/api/studio/models`` and ``GET /v1/models`` serve. With
    ``--prices`` each model's input/output $/1M is appended from the price table.
    """
    import json

    from himmy.services.inference.compare import build_model_catalog

    catalog = asyncio.run(build_model_catalog())

    if getattr(args, "json", False):
        print(json.dumps(catalog, indent=2))
        return 0

    show_prices = getattr(args, "prices", False)
    price_for = None
    if show_prices:
        from himmy.services.inference import pricing

        price_for = pricing.price_for

    any_models = False
    for entry in catalog:
        provider = entry["provider"]
        available = entry["available"]
        models = cast("list[dict[str, Any]]", entry["models"])
        flag = "ok " if available else "-- "
        print(f"[{flag}] {provider}")
        if not models:
            detail = "no models" if available else "not available"
            print(f"        ({detail})")
            continue
        for m in models:
            any_models = True
            name = m["name"]
            extras: list[str] = []
            acc = m.get("accuracy")
            if acc is not None:
                extras.append(f"acc {float(acc):.0%}")
            lat = m.get("latency")
            if lat is not None:
                extras.append(f"{float(lat):.0f}ms")
            if price_for is not None:
                p = price_for(f"{provider}:{name}")
                extras.append(
                    f"${p.input_per_1k * 1000:.2f}/${p.output_per_1k * 1000:.2f} per 1M"
                )
            suffix = f"  ({', '.join(extras)})" if extras else ""
            print(f"        {name}{suffix}")

    if not any_models:
        print(
            "\nNo local models detected. Start Ollama (`ollama serve` then "
            "`ollama pull llama3.2`) or sign in to the `claude` CLI."
        )
    elif not show_prices:
        print("\nReliability numbers come from `himmy bench`. Add --prices for $/1M.")
    return 0


# --------------------------------------------------------------- himmy compare


def cmd_compare(args: argparse.Namespace) -> int:
    """Run one prompt across several models and print the results side-by-side.

    Targets are ``provider:model`` tokens (``--models`` comma-list, or repeated ``-m``).
    Each is an isolated, tool-less model call through the shared
    :func:`himmy.services.inference.compare.run_compare` seam, so the cells are directly
    comparable; a bad provider/model is an error cell, never a crash.
    """
    import json

    from himmy.services.inference.compare import run_compare

    raw_specs: list[str] = []
    for chunk in getattr(args, "models", None) or []:
        raw_specs.extend(chunk.split(","))

    targets: list[CompareTargetSpec] = []
    for raw in raw_specs:
        if not raw.strip():
            continue
        target = _parse_target(raw)
        if target is None:
            _eprint(f"error: model spec {raw.strip()!r} must be provider:model")
            return 2
        targets.append(target)

    if not targets:
        _eprint(
            "error: at least one model is required, e.g. "
            "himmy compare -m ollama:llama3.2 -m claude-cli:haiku 'say hi'"
        )
        return 2

    prompt = " ".join(getattr(args, "prompt", None) or []).strip()
    if not prompt:
        _eprint("error: a prompt is required, e.g. himmy compare -m stub:a 'say hi'")
        return 2

    cells = asyncio.run(
        run_compare(
            targets,
            prompt=prompt,
            system=getattr(args, "system", None) or None,
            timeout_seconds=float(getattr(args, "timeout", 120.0)),
        )
    )

    if getattr(args, "json", False):
        print(
            json.dumps(
                [
                    {
                        "provider": c.provider,
                        "model": c.model,
                        "ok": c.ok,
                        "output": c.output,
                        "error": c.error,
                        "input_tokens": c.input_tokens,
                        "output_tokens": c.output_tokens,
                        "cost": c.cost,
                        "latency_ms": c.latency_ms,
                    }
                    for c in cells
                ],
                indent=2,
            )
        )
        return 0

    for cell in cells:
        label = f"{cell.provider}:{cell.model}"
        print(f"\n=== {label} ===")
        if not cell.ok:
            print(f"  ERROR: {cell.error}")
            continue
        meta: list[str] = []
        if cell.input_tokens is not None or cell.output_tokens is not None:
            meta.append(f"{cell.input_tokens or 0}→{cell.output_tokens or 0} tok")
        if cell.cost is not None:
            meta.append(f"${cell.cost:.4f}")
        if cell.latency_ms is not None:
            meta.append(f"{cell.latency_ms:.0f}ms")
        if meta:
            print(f"  [{' · '.join(meta)}]")
        print(f"  {cell.output.strip() or '(empty)'}")
    return 0


__all__ = ["cmd_models", "cmd_compare"]
