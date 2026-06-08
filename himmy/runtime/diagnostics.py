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

    cfg = find_project_config()
    has_real_model = have_local_provider or have_key
    has_agent = any(Path(p).exists() for p in ("agent.yaml", "team.yaml"))

    if not has_real_model:
        next_step = NextStep(
            kind="install_model",
            message=(
                "No real model yet. Install one (free, local): `ollama pull llama3.2`, "
                "or set OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY for a "
                "cloud model."
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
    )


__all__ = [
    "DoctorReport",
    "ExtraStatus",
    "ProviderStatus",
    "KeyStatus",
    "NextStep",
    "collect_doctor_report",
]
