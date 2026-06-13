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
class StorageHealth:
    """The durable storage backend in use (secrets redacted)."""

    backend: str  # "postgresql" | "sqlite"
    dsn: str | None = None  # redacted (password masked) when Postgres, else None
    code_max_migration: int | None = None  # max migration version in the shipped code


@dataclass
class DbFileHealth:
    """Presence + size of one canonical ``.himmy/*.db`` store (no contents read)."""

    name: str  # logical name: spine | conversations | run_store | v1_approvals
    path: str
    present: bool
    size_bytes: int | None = None


@dataclass
class DiagnosticsReport:
    """Global infra/health snapshot for ``GET /v1/diagnostics`` (NOT per-tenant).

    A read-only, secrets-redacted view of what the *server* can do and where its durable
    state lives — providers/keys/extras/embedders (shared with ``himmy doctor``), the
    storage backend, the canonical ``.himmy`` SQLite stores, and whether the routine
    scheduler loop is active. Per-tenant doctor is not meaningful, so this is global.
    """

    doctor: DoctorReport
    storage: StorageHealth
    db_files: list[DbFileHealth] = field(default_factory=list)
    scheduler_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (nested dataclasses flattened)."""
        return asdict(self)


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


def _redact_dsn(dsn: str) -> str:
    """Return ``dsn`` with any password component replaced by ``***`` (never echo a secret).

    Mirrors ``himmy doctor``'s redaction: rebuilds the netloc with the password masked;
    any parse failure collapses to a fully-masked placeholder rather than risk a leak.
    """
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(dsn)
        if parts.password is None:
            return dsn
        userinfo = parts.username or ""
        if userinfo:
            userinfo = f"{userinfo}:***"
        host = parts.hostname or ""
        if ":" in host:  # re-bracket an IPv6 host that urlsplit stripped
            host = f"[{host}]"
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        netloc = f"{userinfo}@{host}" if userinfo else host
        return urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        )
    except Exception:  # noqa: BLE001 - never risk echoing the raw DSN
        return "<redacted>"


def _collect_storage_health() -> StorageHealth:
    """Resolve the durable storage backend in use (Postgres DSN redacted)."""
    from himmy.config.secrets import get_secret
    from himmy.services.storage.factory import _is_postgres_dsn  # noqa: PLC2701

    dsn = get_secret("HIMMY_DATABASE_URL")
    if _is_postgres_dsn(dsn):
        from himmy.services.storage.postgres import STORAGE_MIGRATIONS

        code_max = max(version for version, _, _ in STORAGE_MIGRATIONS)
        return StorageHealth(
            backend="postgresql",
            dsn=_redact_dsn(dsn or ""),
            code_max_migration=code_max,
        )
    from himmy.services.storage.sqlite import SQLITE_SCHEMA_VERSION

    return StorageHealth(
        backend="sqlite", dsn=None, code_max_migration=SQLITE_SCHEMA_VERSION
    )


def _db_file_health(name: str, path: str) -> DbFileHealth:
    """Presence + size of one canonical SQLite store (never opens or reads contents)."""
    p = Path(path)
    try:
        present = p.is_file()
        size = p.stat().st_size if present else None
    except OSError:
        present, size = False, None
    return DbFileHealth(name=name, path=path, present=present, size_bytes=size)


def _collect_db_files() -> list[DbFileHealth]:
    """The canonical ``.himmy`` SQLite stores (spine, conversations, run store, approvals).

    Path resolution reuses the single decision points in :mod:`himmy.config.project` and
    :mod:`himmy.services.storage.factory`, so the report points at the same files the CLI,
    Studio, and /v1 actually use. Presence/size only — no contents are read.
    """
    import os

    from himmy.config.project import (
        conversations_db_path,
        spine_db_path,
        v1_approvals_db_path,
    )
    from himmy.services.storage.factory import HIMMY_STORE_PATH

    files = [
        _db_file_health("spine", spine_db_path()),
        _db_file_health("conversations", conversations_db_path()),
        _db_file_health("v1_approvals", v1_approvals_db_path()),
    ]
    # The canonical run store (SQLite) — only meaningful when not on Postgres; report the
    # resolved file path either way so an operator sees where runs would land.
    store_path = os.environ.get("HIMMY_STORE_PATH") or HIMMY_STORE_PATH
    files.append(_db_file_health("run_store", store_path))
    return files


def collect_diagnostics_report(*, scheduler_active: bool = False) -> DiagnosticsReport:
    """Probe the global server health: providers/keys, storage backend, and durable stores.

    ``scheduler_active`` is passed in by the caller (the API reads the live
    :class:`~himmy.api.routines.RoutineScheduler` off ``app.state``) so this module stays
    free of an API dependency. Secrets are never included — only presence + redacted DSNs.
    """
    return DiagnosticsReport(
        doctor=collect_doctor_report(),
        storage=_collect_storage_health(),
        db_files=_collect_db_files(),
        scheduler_active=scheduler_active,
    )


__all__ = [
    "DoctorReport",
    "ExtraStatus",
    "ProviderStatus",
    "KeyStatus",
    "EmbedderStatus",
    "NextStep",
    "StorageHealth",
    "DbFileHealth",
    "DiagnosticsReport",
    "collect_doctor_report",
    "collect_diagnostics_report",
]
