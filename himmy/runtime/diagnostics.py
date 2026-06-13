"""Environment diagnostics shared by ``himmy doctor`` (CLI) and Himmy Studio (API).

:func:`collect_doctor_report` computes a single structured snapshot of what the
machine can do — installed optional extras, local model providers on ``PATH``,
provider keys in the environment, available guardrails, the active project config,
and the single most useful next action. The CLI renders it as text; the Studio API
returns it as JSON. One source of truth so both stay in lockstep.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Optional extras probed by module import — label is what the user sees.
_EXTRAS: tuple[tuple[str, str], ...] = (
    ("providers (pydantic-ai)", "pydantic_ai"),
    ("api (fastapi)", "fastapi"),
    ("postgres (asyncpg)", "asyncpg"),
    ("connectors (feedparser)", "feedparser"),
    ("connectors (openpyxl)", "openpyxl"),
    ("observability (logfire)", "logfire"),
    ("nepal (nepali-datetime)", "nepali_datetime"),
    ("validation (jsonschema)", "jsonschema"),
)

# Provider API keys we surface (presence only — never the value).
_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "PYDANTIC_AI_GATEWAY_API_KEY",
)


@dataclass
class ExtraStatus:
    label: str
    module: str
    ok: bool


@dataclass
class ProviderStatus:
    name: str
    ok: bool
    path: str | None = None


@dataclass
class KeyStatus:
    name: str
    present: bool


@dataclass
class EmbedderStatus:
    name: str  # backend name: fastembed | ollama | deterministic
    available: bool
    reason: str | None = None


@dataclass
class NextStep:
    kind: str  # "install_model" | "scaffold" | "run"
    message: str


@dataclass
class DoctorReport:
    python: str
    version: str
    extras: list[ExtraStatus] = field(default_factory=list)
    providers: list[ProviderStatus] = field(default_factory=list)
    keys: list[KeyStatus] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    project_config: str | None = None
    has_real_model: bool = False
    has_agent: bool = False
    next_step: NextStep | None = None
    embedder_selected: str = "deterministic"
    embedder_statuses: list[EmbedderStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (nested dataclasses flattened)."""
        return asdict(self)


def _can_import(module: str) -> bool:
    """True when ``module`` imports cleanly (its optional extra is installed)."""
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


#: Default base URL for the local Ollama embedder reachability probe (display only).
_OLLAMA_PROBE_URL = "http://localhost:11434"


def _collect_embedder_statuses() -> tuple[str, list[EmbedderStatus]]:
    """Probe the auto-embedder cascade (fastembed → ollama → deterministic).

    Returns the backend ``"auto"`` would resolve to right now plus a status per
    candidate, in cascade order. The probes are import-/connection-only — they never
    construct a model or trigger a download (``fastembed`` is checked with
    :func:`importlib.util.find_spec`, ``ollama`` with a fast fail-closed localhost
    probe), so ``doctor`` stays cheap and offline.
    """
    from himmy.services.knowledge.local_embedders import (
        default_ollama_embed_model,
        fastembed_available,
        ollama_embed_model_available,
        ollama_reachable,
        resolve_auto_backend,
    )

    selected = resolve_auto_backend()
    fast_ok = fastembed_available()
    ollama_up = ollama_reachable(_OLLAMA_PROBE_URL)
    embed_model = default_ollama_embed_model()
    # "auto" picks Ollama only when the embed model is actually pulled, so the doctor
    # row reflects embed-readiness (and tells the user how to fix a half-ready server).
    ollama_ok = ollama_up and ollama_embed_model_available(base_url=_OLLAMA_PROBE_URL)
    if ollama_ok:
        ollama_reason: str | None = None
    elif not ollama_up:
        ollama_reason = f"{_OLLAMA_PROBE_URL} not reachable"
    else:
        ollama_reason = (
            f"embed model {embed_model!r} not pulled (ollama pull {embed_model})"
        )
    statuses = [
        EmbedderStatus(
            name="fastembed",
            available=fast_ok,
            reason=(
                None if fast_ok else "not installed (pip install 'himmy[embeddings]')"
            ),
        ),
        EmbedderStatus(
            name="ollama",
            available=ollama_ok,
            reason=ollama_reason,
        ),
        EmbedderStatus(name="deterministic", available=True, reason="fallback"),
    ]
    return selected, statuses


def collect_doctor_report() -> DoctorReport:
    """Probe the environment and return a structured diagnostics snapshot."""
    from himmy import __version__
    from himmy.config.project import find_project_config
    from himmy.services.guardrails import BUILTIN_GUARDRAILS

    extras = [
        ExtraStatus(label=label, module=module, ok=_can_import(module))
        for label, module in _EXTRAS
    ]

    providers: list[ProviderStatus] = []
    have_local_provider = False
    for binary in ("claude", "ollama"):
        path = shutil.which(binary)
        have_local_provider = have_local_provider or bool(path)
        providers.append(ProviderStatus(name=binary, ok=bool(path), path=path))

    keys: list[KeyStatus] = []
    have_key = False
    for key in _KEYS:
        present = bool(os.environ.get(key))
        have_key = have_key or present
        keys.append(KeyStatus(name=key, present=present))

    embedder_selected, embedder_statuses = _collect_embedder_statuses()

    cfg = find_project_config()
    has_real_model = have_local_provider or have_key
    has_agent = any(Path(p).exists() for p in ("agent.yaml", "team.yaml"))

    if not has_real_model:
        next_step = NextStep(
            kind="install_model",
            message=(
                "No real model yet. Install one (free, local): `ollama pull llama3.2`, "
                "or set OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY for a "
                "cloud model. New here? The 5-minute quickstart walks you through it: "
                "docs/QUICKSTART.md."
            ),
        )
    elif not has_agent:
        next_step = NextStep(
            kind="scaffold",
            message="Scaffold your first agent: `himmy init my-agent`.",
        )
    else:
        next_step = NextStep(
            kind="run",
            message='Run it: `himmy run -f agent.yaml -p "Say hello."`.',
        )

    return DoctorReport(
        python=sys.version.split()[0],
        version=__version__,
        extras=extras,
        providers=providers,
        keys=keys,
        guardrails=list(BUILTIN_GUARDRAILS),
        project_config=str(cfg) if cfg else None,
        has_real_model=has_real_model,
        has_agent=has_agent,
        next_step=next_step,
        embedder_selected=embedder_selected,
        embedder_statuses=embedder_statuses,
    )


__all__ = [
    "DoctorReport",
    "ExtraStatus",
    "ProviderStatus",
    "KeyStatus",
    "EmbedderStatus",
    "NextStep",
    "collect_doctor_report",
]
