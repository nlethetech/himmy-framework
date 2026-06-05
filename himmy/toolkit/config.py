"""Toolkit configuration: where built-in tools read their (non-secret) settings.

A single :class:`ToolkitConfig` carries everything the packs need — the filesystem
jail root, the search backend choice, a database DSN, the sandbox envelope, and the
network safety knobs. :meth:`ToolkitConfig.from_env` reads it from ``HIMMY_*``
environment variables so the CLI can configure the toolkit with no code. Secrets
(search API keys, DB passwords) are only ever read from the environment at call time;
they are not persisted on the model nor projected onto entities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from himmy.services.sandbox.models import SandboxLimits


class ToolkitConfig(BaseModel):
    """Resolved settings for the built-in tool packs."""

    # filesystem pack -------------------------------------------------------
    fs_root: Path = Field(default_factory=Path.cwd)
    fs_allow_write: bool = False

    # web pack --------------------------------------------------------------
    search_backend: str = "duckduckgo"
    search_api_key: str | None = None
    http_timeout: float = 20.0
    http_max_bytes: int = 5_000_000
    allow_private_hosts: bool = False

    # data pack -------------------------------------------------------------
    sqlite_path: str | None = None
    sql_dsn: str | None = None
    sql_read_only: bool = True

    # knowledge / memory embeddings ----------------------------------------
    kb_dsn: str | None = None  # Postgres+pgvector DSN → durable KB; None → in-process
    embedder: str = "deterministic"  # deterministic | ollama | fastembed | openai
    embedder_model: str | None = None
    embedder_dim: int | None = None
    ollama_base_url: str | None = None

    # comms pack ------------------------------------------------------------
    comms_allow_send: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_use_tls: bool = True

    # telegram pack ---------------------------------------------------------
    telegram_bot_token: str | None = None
    telegram_default_chat_id: str | None = None

    # code pack -------------------------------------------------------------
    sandbox_limits: SandboxLimits = Field(default_factory=SandboxLimits)

    # memory pack -----------------------------------------------------------
    memory_path: str | None = None  # sqlite file → durable; None → in-process
    memory_subject: str = "default"

    def build_embedder_and_dim(self) -> tuple[Any, int]:
        """Build the configured embedder and its effective embedding dimension."""
        from himmy.services.knowledge.local_embedders import (
            build_embedder,
            default_dim_for,
        )

        dim = self.embedder_dim or default_dim_for(self.embedder)
        embedder = build_embedder(
            self.embedder,
            model=self.embedder_model,
            dim=dim,
            base_url=self.ollama_base_url,
        )
        return embedder, dim

    @classmethod
    def from_env(cls) -> ToolkitConfig:
        """Build a config from ``HIMMY_*`` environment variables (with defaults).

        Recognized vars: ``HIMMY_FS_ROOT``, ``HIMMY_FS_ALLOW_WRITE``,
        ``HIMMY_SEARCH_BACKEND``, ``HIMMY_SEARCH_API_KEY`` (or ``TAVILY_API_KEY`` /
        ``BRAVE_API_KEY`` by backend), ``HIMMY_HTTP_TIMEOUT``,
        ``HIMMY_HTTP_MAX_BYTES``, ``HIMMY_ALLOW_PRIVATE_HOSTS``,
        ``HIMMY_SQLITE_PATH``, ``HIMMY_SQL_DSN``, ``HIMMY_SQL_READONLY``.
        """
        env = os.environ
        backend = env.get("HIMMY_SEARCH_BACKEND", "duckduckgo").lower()
        api_key = env.get("HIMMY_SEARCH_API_KEY") or env.get(
            f"{backend.upper()}_API_KEY"
        )
        return cls(
            fs_root=Path(env.get("HIMMY_FS_ROOT", str(Path.cwd()))).expanduser(),
            fs_allow_write=_env_bool(env.get("HIMMY_FS_ALLOW_WRITE"), default=False),
            search_backend=backend,
            search_api_key=api_key,
            http_timeout=float(env.get("HIMMY_HTTP_TIMEOUT", "20")),
            http_max_bytes=int(env.get("HIMMY_HTTP_MAX_BYTES", "5000000")),
            allow_private_hosts=_env_bool(
                env.get("HIMMY_ALLOW_PRIVATE_HOSTS"), default=False
            ),
            sqlite_path=env.get("HIMMY_SQLITE_PATH"),
            sql_dsn=env.get("HIMMY_SQL_DSN"),
            sql_read_only=_env_bool(env.get("HIMMY_SQL_READONLY"), default=True),
            kb_dsn=env.get("HIMMY_KB_DSN"),
            embedder=env.get("HIMMY_EMBEDDER", "deterministic"),
            embedder_model=env.get("HIMMY_EMBEDDER_MODEL"),
            embedder_dim=(
                int(env["HIMMY_EMBEDDER_DIM"])
                if env.get("HIMMY_EMBEDDER_DIM")
                else None
            ),
            ollama_base_url=env.get("HIMMY_OLLAMA_URL"),
            comms_allow_send=_env_bool(
                env.get("HIMMY_COMMS_ALLOW_SEND"), default=False
            ),
            smtp_host=env.get("HIMMY_SMTP_HOST"),
            smtp_port=int(env.get("HIMMY_SMTP_PORT", "587")),
            smtp_user=env.get("HIMMY_SMTP_USER"),
            smtp_password=env.get("HIMMY_SMTP_PASSWORD"),
            smtp_from=env.get("HIMMY_SMTP_FROM"),
            smtp_use_tls=_env_bool(env.get("HIMMY_SMTP_USE_TLS"), default=True),
            telegram_bot_token=env.get("HIMMY_TELEGRAM_BOT_TOKEN"),
            telegram_default_chat_id=env.get("HIMMY_TELEGRAM_CHAT_ID"),
            memory_path=env.get("HIMMY_MEMORY_PATH"),
            memory_subject=env.get("HIMMY_MEMORY_SUBJECT", "default"),
        )

    @classmethod
    def from_sources(cls, toml_toolkit: dict[str, Any] | None = None) -> ToolkitConfig:
        """Build from ``himmy.toml`` ``[toolkit]`` defaults overlaid by the environment.

        Precedence is **env > himmy.toml > built-in default**: the toml values seed the
        config, then any field whose ``HIMMY_*`` env var is set wins.
        """
        base = {k: v for k, v in (toml_toolkit or {}).items() if k in cls.model_fields}
        config = cls(**base)
        env_config = cls.from_env()
        overrides = {
            field: getattr(env_config, field)
            for field, env_key in _TOOLKIT_ENV_KEYS.items()
            if env_key in os.environ
        }
        return config.model_copy(update=overrides)


#: Toolkit fields a ``himmy.toml`` may set, paired with the env var that overrides each.
_TOOLKIT_ENV_KEYS: dict[str, str] = {
    "fs_root": "HIMMY_FS_ROOT",
    "fs_allow_write": "HIMMY_FS_ALLOW_WRITE",
    "search_backend": "HIMMY_SEARCH_BACKEND",
    "sqlite_path": "HIMMY_SQLITE_PATH",
    "kb_dsn": "HIMMY_KB_DSN",
    "embedder": "HIMMY_EMBEDDER",
    "embedder_model": "HIMMY_EMBEDDER_MODEL",
    "embedder_dim": "HIMMY_EMBEDDER_DIM",
    "ollama_base_url": "HIMMY_OLLAMA_URL",
    "memory_path": "HIMMY_MEMORY_PATH",
    "memory_subject": "HIMMY_MEMORY_SUBJECT",
}


def _env_bool(value: str | None, *, default: bool) -> bool:
    """Parse a truthy/falsey env string (``1/true/yes/on``), else ``default``."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["ToolkitConfig"]
