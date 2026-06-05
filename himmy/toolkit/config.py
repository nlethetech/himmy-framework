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

    # comms pack ------------------------------------------------------------
    comms_allow_send: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_use_tls: bool = True

    # code pack -------------------------------------------------------------
    sandbox_limits: SandboxLimits = Field(default_factory=SandboxLimits)

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
        api_key = (
            env.get("HIMMY_SEARCH_API_KEY")
            or env.get(f"{backend.upper()}_API_KEY")
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
            comms_allow_send=_env_bool(
                env.get("HIMMY_COMMS_ALLOW_SEND"), default=False
            ),
            smtp_host=env.get("HIMMY_SMTP_HOST"),
            smtp_port=int(env.get("HIMMY_SMTP_PORT", "587")),
            smtp_user=env.get("HIMMY_SMTP_USER"),
            smtp_password=env.get("HIMMY_SMTP_PASSWORD"),
            smtp_from=env.get("HIMMY_SMTP_FROM"),
            smtp_use_tls=_env_bool(env.get("HIMMY_SMTP_USE_TLS"), default=True),
        )


def _env_bool(value: str | None, *, default: bool) -> bool:
    """Parse a truthy/falsey env string (``1/true/yes/on``), else ``default``."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["ToolkitConfig"]
