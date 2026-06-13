"""Shared model surfaces: a quick side-by-side compare + a provider/model catalog.

This is the ONE seam behind three interfaces (T3d): Himmy Studio's ``/api/studio/models``
+ ``/api/studio/compare``, the ``/v1/models`` + ``/v1/models/compare`` REST surfaces, and
the ``himmy models`` + ``himmy compare`` CLI commands. Keeping the per-target model call
and the catalog assembly here means all three stay byte-for-byte consistent and there is a
single place to harden.

* :func:`run_compare` runs one prompt across several ``(provider, model)`` targets
  concurrently, each an isolated, tool-less model call so the cells are directly
  comparable on output, tokens, cost, and latency. Every target is reported per-cell
  (a bad provider/model is an ``ok=False`` cell, never a raised exception), so a caller
  can render a 500-free side-by-side grid.
* :func:`build_model_catalog` assembles the available local providers + their models,
  decorated with any cached ``himmy bench`` reliability stats.

The ``provider:model`` *string* parsing (e.g. ``"ollama:qwen2.5:3b"``) is deliberately
NOT here — that lexing belongs to the CLI adapter that accepts the string; this module
takes already-split, structured targets.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class CompareTargetSpec:
    """One ``(provider, model)`` cell to run the shared prompt against."""

    provider: str
    model: str


@dataclass
class CompareCell:
    """The outcome of running the prompt against one target (interface-agnostic).

    Numeric usage fields are ``None`` on a failed cell (the call never produced a
    usage split) and concrete on success — the same convention every surface renders.
    """

    provider: str
    model: str
    ok: bool
    output: str = ""
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    latency_ms: float | None = None


async def _run_one(
    target: CompareTargetSpec,
    *,
    prompt: str,
    system: str | None,
    timeout_seconds: float,
) -> CompareCell:
    """Run the prompt against a single target; report any failure as an ``ok=False`` cell."""
    from himmy.cli.provider import build_manager_for
    from himmy.services.inference.models import (
        InferenceMessage,
        InferenceRequest,
        InferenceStatus,
    )

    try:
        manager = build_manager_for(target.provider, target.model)
    except Exception as exc:  # noqa: BLE001 - report per-cell, never raise to a 500
        return CompareCell(
            provider=target.provider, model=target.model, ok=False, error=str(exc)
        )

    messages: list[InferenceMessage] = []
    if system:
        messages.append(InferenceMessage(role="system", content=system))
    messages.append(InferenceMessage(role="user", content=prompt))
    request = InferenceRequest(
        model_key=target.model,
        messages=messages,
        timeout_seconds=timeout_seconds,
    )

    try:
        resp = await manager.generate(request)
    except Exception as exc:  # noqa: BLE001 - report per-cell
        return CompareCell(
            provider=target.provider, model=target.model, ok=False, error=str(exc)
        )

    ok = resp.status == InferenceStatus.SUCCESS
    err_msg = resp.error.message if resp.error else "model call failed"
    return CompareCell(
        provider=target.provider,
        model=target.model,
        ok=ok,
        output=resp.output_text or "",
        error=None if ok else err_msg,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost=resp.cost,
        latency_ms=resp.latency_ms,
    )


async def run_compare(
    targets: list[CompareTargetSpec],
    *,
    prompt: str,
    system: str | None = None,
    timeout_seconds: float = 120.0,
) -> list[CompareCell]:
    """Run ``prompt`` across every target concurrently and return one cell per target.

    Each target is an isolated model call (no tools, no run history) so the cells are
    directly comparable on output, tokens, cost, and latency. Results are returned in
    the SAME order as ``targets``. A target that cannot be built or that fails is an
    ``ok=False`` cell — this function never raises for a per-target failure.
    """
    return list(
        await asyncio.gather(
            *(
                _run_one(
                    t, prompt=prompt, system=system, timeout_seconds=timeout_seconds
                )
                for t in targets
            )
        )
    )


# ---- catalog --------------------------------------------------------------


async def _ollama_catalog_models(cap: int = 2) -> list[str]:
    """Names a local Ollama server reports (empty if it isn't running, bounded by ``cap``).

    Shares :mod:`himmy.api.studio_bench`'s probe so the catalog matches what Studio's
    Doctor detects; falls back to an empty list when the ``api`` extra (httpx) is absent
    so the catalog stays usable on a minimal install.
    """
    try:
        from himmy.api import studio_bench
    except Exception:  # noqa: BLE001 - api extra not installed → no local probe
        return []
    return await studio_bench._ollama_models(cap)


def _bench_stats() -> dict[str, dict[str, object]]:
    """``model name → {accuracy, latency}`` from prior ``himmy bench`` runs (best-effort).

    Mirrors the original Studio ``/models`` loop exactly (iterate the newest-first cached
    summaries and assign unconditionally, so the catalog is byte-identical to what Studio
    served before this extraction); a missing/corrupt cache degrades to no stats.
    """
    stats: dict[str, dict[str, object]] = {}
    try:
        from himmy.api import studio_bench
    except Exception:  # noqa: BLE001 - api extra / benchmark cache unavailable
        return stats
    try:
        entries = studio_bench.list_cached()
    except Exception:  # noqa: BLE001 - corrupt/missing cache is non-fatal
        return stats
    for e in entries:
        model = e.get("model")
        if model:
            stats[model] = {"accuracy": e.get("accuracy"), "latency": e.get("latency")}
    return stats


#: The Claude CLI exposes these tier aliases as runnable models.
_CLAUDE_CLI_MODELS: tuple[str, ...] = ("haiku", "sonnet", "opus")


async def build_model_catalog() -> list[dict[str, object]]:
    """Available local providers + their models, decorated with cached bench stats.

    Returns one entry per provider: ``{provider, available, models}`` where each model is
    ``{name, accuracy?, latency?}``. This is the single catalog the Studio ``/models``
    endpoint, ``GET /v1/models`` and ``himmy models`` all render. It carries NO secret
    material — only provider availability + model names + reliability numbers.
    """
    stats = _bench_stats()
    out: list[dict[str, object]] = []

    ollama = await _ollama_catalog_models()
    out.append(
        {
            "provider": "ollama",
            "available": bool(ollama),
            "models": [{"name": m, **stats.get(m, {})} for m in ollama],
        }
    )

    claude = shutil.which("claude") is not None
    out.append(
        {
            "provider": "claude-cli",
            "available": claude,
            "models": (
                [{"name": n, **stats.get(n, {})} for n in _CLAUDE_CLI_MODELS]
                if claude
                else []
            ),
        }
    )
    return out


__all__ = [
    "CompareTargetSpec",
    "CompareCell",
    "run_compare",
    "build_model_catalog",
]
