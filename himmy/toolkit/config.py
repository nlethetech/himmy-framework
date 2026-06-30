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

from himmy.services.sandbox.models import SandboxLimits, SandboxPolicy


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
    egress_allow_hosts: list[str] = Field(default_factory=list)  # [] = no allow-list

    # data pack -------------------------------------------------------------
    sqlite_path: str | None = None
    sql_dsn: str | None = None
    sql_read_only: bool = True

    # knowledge / memory embeddings ----------------------------------------
    kb_dsn: str | None = None  # Postgres+pgvector DSN → durable KB; None → in-process
    # auto | deterministic | ollama | fastembed | openai. "auto" (the default) prefers
    # a real local embedder (fastembed, else a reachable Ollama) and falls back to the
    # offline deterministic embedder, so the default path stays keyless and offline.
    embedder: str = "auto"
    embedder_model: str | None = None
    embedder_dim: int | None = None
    ollama_base_url: str | None = None

    # comms pack ------------------------------------------------------------
    comms_allow_send: bool = False
    #: Opt out of the Gmail/Calendar approval gate (``gmail_send`` / ``gcal_create``).
    #: Defaults to False so mutating Google tools stay human-in-the-loop, mirroring
    #: ``comms_allow_send``; set ``HIMMY_GOOGLE_ALLOW_SEND`` to release the gate.
    google_allow_send: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_use_tls: bool = True

    # telegram pack ---------------------------------------------------------
    telegram_bot_token: str | None = None
    telegram_default_chat_id: str | None = None
    #: Sender/chat allowlist for the inbound bot loop (``HIMMY_TELEGRAM_ALLOWED_CHATS``,
    #: comma-separated chat ids). Empty ⇒ the bot answers EVERY chat that messages it,
    #: so set this for any bot reachable beyond a single trusted chat. Compared as
    #: strings, so it matches both user chat ids and group ids.
    telegram_allowed_chat_ids: list[str] = Field(default_factory=list)

    # code pack -------------------------------------------------------------
    sandbox_limits: SandboxLimits = Field(default_factory=SandboxLimits)
    # off | subprocess | container | gvisor | firecracker (HIMMY_CODE_EXEC). The last
    # two are Linux-only (gVisor) / Linux+KVM-only (Firecracker); see docs/enterprise/sandbox.md.
    code_exec: str = "subprocess"
    sandbox_image: str = "python:3.12-slim"
    sandbox_engine: str = "docker"
    #: Hardened-backend isolation contract (network/ephemeral/limits/host/audit). The
    #: resource envelope is taken from ``sandbox_limits``; the egress fields below feed
    #: the optional mediated allow-list. Defaults are the locked-down posture.
    sandbox_network: str = "none"  # none | egress_proxy (HIMMY_SANDBOX_NETWORK)
    sandbox_egress_network: str | None = None  # HIMMY_SANDBOX_EGRESS_NETWORK
    sandbox_egress_proxy_url: str | None = None  # HIMMY_SANDBOX_EGRESS_PROXY

    def build_sandbox_policy(self) -> SandboxPolicy:
        """Assemble the :class:`SandboxPolicy` the hardened backend enforces.

        The resource envelope comes from ``sandbox_limits``; the network posture from
        the egress fields. An unknown ``sandbox_network`` value falls back to the safe
        default (deny). The remaining OS-level guarantees keep their locked-down
        defaults (read-only rootfs, cap-drop ALL, no-new-privileges, non-root, audit).
        """
        from himmy.services.sandbox.models import NetworkMode

        try:
            network = NetworkMode(self.sandbox_network.strip().lower())
        except ValueError:
            network = NetworkMode.NONE
        return SandboxPolicy(
            limits=self.sandbox_limits,
            network=network,
            egress_network=self.sandbox_egress_network,
            egress_proxy_url=self.sandbox_egress_proxy_url,
        )

    #: The run's owning ``workspace_id`` (P1 tenancy), threaded in by the server run
    #: factory (``build_runtime_for_spec(subject=...)`` → ``from_spec``). On a SHARED
    #: process store — the durable memory ``.himmy/memory.db`` keyed only by subject, and
    #: the durable pgvector KB pinned to a fixed ``(local, local, default)`` scope — every
    #: tenant's runs would otherwise collide on one static subject/KB, a cross-tenant
    #: confused-deputy read (tenant t1's ``remember``/``kb_ingest`` recalled by t2's
    #: ``recall``/``kb_search``). When set, the memory subject and the KB scope keys are
    #: NAMESPACED by this value so a tenant can only ever read its OWN rows. ``None`` — the
    #: one-shot CLI / offline path (one process per tenant, no authenticator) and every
    #: pre-existing caller — leaves both packs on their historical static scope, so the
    #: zero-config path is byte-for-byte unchanged.
    tenant_scope: str | None = None

    #: The run's owning DATA SUBJECT (P1 tenancy, subject axis), threaded in alongside
    #: ``tenant_scope`` by the server run factory for a ``subject_scoped`` principal. On a
    #: deployment that runs PER-USER principals inside ONE shared tenant, ``tenant_scope`` is
    #: identical for every user, so memory/KB/tasks/notes would isolate by tenant but NOT by
    #: user — user-B recalls user-A's facts (a cross-USER confused-deputy within the tenant).
    #: When set, the memory subject, KB scope, and the tasks/notes pack workspace are
    #: additionally NAMESPACED by this subject so two users of one tenant never collide.
    #: ``None`` (offline / one-shot CLI / a non-``subject_scoped`` principal — the common
    #: case) leaves the scope tenant-only, so the historical behaviour is byte-unchanged.
    subject_scope: str | None = None

    # memory pack -----------------------------------------------------------
    memory_path: str | None = None  # sqlite file → durable; None → in-process
    memory_subject: str = "default"
    #: Explicit server-context marker for pack registration: the Studio BFF sets
    #: this when building runtimes so packs default to durable stores without
    #: relying on the lifespan ContextVar crossing task boundaries.
    server_context: bool = False
    # Optional recall floor (HIMMY_MEMORY_MIN_SIM): below it a memory is not
    # recalled. None preserves the always-return-the-top-hit default.
    memory_min_similarity: float | None = None
    # Opt-in audited consolidation on remember (HIMMY_MEMORY_CONSOLIDATE): a new
    # fact is reconciled (ADD/UPDATE/DELETE/NOOP) against existing ones instead of
    # blindly appended. Off by default so a default agent incurs no extra cost.
    memory_consolidate: bool = False

    def _scope_token(self) -> str | None:
        """The combined tenant(+subject) namespace token, or ``None`` for "unscoped".

        ``tenant_scope`` is the tenant axis; ``subject_scope`` (a per-user data subject under
        a ``subject_scoped`` principal) is the within-tenant USER axis. When BOTH are set the
        token is ``<tenant>:s:<subject>`` so two users of one tenant get distinct namespaces;
        tenant-only stays ``<tenant>`` (byte-unchanged for the non-subject-scoped common
        case). ``subject_scope`` alone (no tenant) is folded in too. ``None`` everywhere is
        "no scoping" — the offline / one-shot CLI path is byte-for-byte unchanged.
        """
        if self.tenant_scope and self.subject_scope:
            return f"{self.tenant_scope}:s:{self.subject_scope}"
        if self.tenant_scope:
            return self.tenant_scope
        if self.subject_scope:
            return f"s:{self.subject_scope}"
        return None

    def scoped_memory_subject(self) -> str:
        """The memory subject to actually use, NAMESPACED by the run's tenant+subject when set.

        The by-construction fix for the cross-tenant (and cross-USER) memory leak: on a
        shared process store keyed only by subject, every tenant/user would otherwise
        read/write the one static ``memory_subject`` (``"default"``). When the scope token is
        set (the server multi-tenant / per-user path) the effective subject is prefixed with
        the tenant(+subject) so t1/t2 — and two users of one tenant — never collide; an unset
        token (offline / one-shot CLI / every pre-existing caller) returns ``memory_subject``
        verbatim — byte-for-byte unchanged.
        """
        token = self._scope_token()
        if token:
            return f"t:{token}:{self.memory_subject}"
        return self.memory_subject

    def scoped_kb_keys(self) -> tuple[str, str]:
        """The durable-KB ``(workspace_id, client_id)`` scope, tenant+subject-namespaced when set.

        The knowledge pack pins its KB to a fixed ``(local, local, default)`` scope, so a
        shared durable (``HIMMY_KB_DSN`` / pgvector) backend hands EVERY tenant the same KB
        — t1's ``kb_ingest`` chunks come back from t2's ``kb_search`` (and, under a
        ``subject_scoped`` principal, user-A's come back from user-B's). When the scope token
        is set the scope keys become the tenant's (and user's) own; an unset token keeps the
        historical ``("local", "local")`` so the offline/in-process default is byte-unchanged.
        """
        token = self._scope_token()
        if token:
            return (f"t:{token}", f"t:{token}")
        return ("local", "local")

    def scoped_pack_workspace(self) -> str | None:
        """The ``workspace_id`` the tasks/notes packs scope by — tenant(+subject) namespaced.

        The tasks/notes packs key the singleton SQLite store by a single ``workspace_id``.
        Tenant-only that is the tenant id; under a ``subject_scoped`` principal it becomes the
        combined ``t:<tenant>:s:<subject>`` token so two users of one tenant get distinct,
        non-overlapping task/note partitions (the store's scope clauses treat it as an opaque
        owner string). ``None`` (offline / non-subject-scoped tenant-only … see below) keeps
        the historical behaviour.

        NOTE: when ONLY ``tenant_scope`` is set (no subject) this returns the bare tenant id —
        byte-identical to the previous ``config.tenant_scope`` the packs passed — so the
        tenant-only path is unchanged; the subject namespace engages only for a per-user run.
        """
        token = self._scope_token()
        if self.subject_scope and token:
            # per-user run: opaque combined owner so users of one tenant never collide.
            return f"t:{token}"
        # tenant-only (or unscoped): preserve the historical bare tenant id / None.
        return self.tenant_scope

    def build_embedder_and_dim(self) -> tuple[Any, int]:
        """Build the configured embedder and its effective embedding dimension.

        ``embedder="auto"`` is resolved to a concrete backend exactly once here, so
        the returned ``dim`` matches the embedder that is actually built even when the
        reachability of a local Ollama probe could otherwise differ between calls.
        """
        from himmy.services.knowledge.local_embedders import (
            build_embedder,
            default_dim_for,
            resolve_auto_backend,
        )

        backend = self.embedder
        if backend == "auto":
            backend = resolve_auto_backend(ollama_base_url=self.ollama_base_url)
        dim = self.embedder_dim or default_dim_for(backend)
        embedder = build_embedder(
            backend,
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
        from himmy.config.secrets import get_secret

        env = os.environ
        backend = env.get("HIMMY_SEARCH_BACKEND", "duckduckgo").lower()
        api_key = get_secret("HIMMY_SEARCH_API_KEY") or get_secret(
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
            egress_allow_hosts=[
                h.strip()
                for h in (env.get("HIMMY_EGRESS_ALLOW") or "").split(",")
                if h.strip()
            ],
            sqlite_path=env.get("HIMMY_SQLITE_PATH"),
            sql_dsn=get_secret("HIMMY_SQL_DSN"),
            sql_read_only=_env_bool(env.get("HIMMY_SQL_READONLY"), default=True),
            kb_dsn=get_secret("HIMMY_KB_DSN"),
            embedder=env.get("HIMMY_EMBEDDER", "auto"),
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
            google_allow_send=_env_bool(
                env.get("HIMMY_GOOGLE_ALLOW_SEND"), default=False
            ),
            smtp_host=env.get("HIMMY_SMTP_HOST"),
            smtp_port=int(env.get("HIMMY_SMTP_PORT", "587")),
            smtp_user=env.get("HIMMY_SMTP_USER"),
            smtp_password=get_secret("HIMMY_SMTP_PASSWORD"),
            smtp_from=env.get("HIMMY_SMTP_FROM"),
            smtp_use_tls=_env_bool(env.get("HIMMY_SMTP_USE_TLS"), default=True),
            telegram_bot_token=get_secret("HIMMY_TELEGRAM_BOT_TOKEN"),
            telegram_default_chat_id=env.get("HIMMY_TELEGRAM_CHAT_ID"),
            telegram_allowed_chat_ids=[
                c.strip()
                for c in (env.get("HIMMY_TELEGRAM_ALLOWED_CHATS") or "").split(",")
                if c.strip()
            ],
            code_exec=env.get("HIMMY_CODE_EXEC", "subprocess"),
            sandbox_image=env.get("HIMMY_SANDBOX_IMAGE", "python:3.12-slim"),
            sandbox_engine=env.get("HIMMY_SANDBOX_ENGINE", "docker"),
            sandbox_network=env.get("HIMMY_SANDBOX_NETWORK", "none"),
            sandbox_egress_network=env.get("HIMMY_SANDBOX_EGRESS_NETWORK"),
            sandbox_egress_proxy_url=env.get("HIMMY_SANDBOX_EGRESS_PROXY"),
            memory_path=env.get("HIMMY_MEMORY_PATH"),
            memory_subject=env.get("HIMMY_MEMORY_SUBJECT", "default"),
            memory_min_similarity=(
                float(env["HIMMY_MEMORY_MIN_SIM"])
                if env.get("HIMMY_MEMORY_MIN_SIM")
                else None
            ),
            memory_consolidate=_env_bool(
                env.get("HIMMY_MEMORY_CONSOLIDATE"), default=False
            ),
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
    "memory_min_similarity": "HIMMY_MEMORY_MIN_SIM",
    "memory_consolidate": "HIMMY_MEMORY_CONSOLIDATE",
}


def _env_bool(value: str | None, *, default: bool) -> bool:
    """Parse a truthy/falsey env string (``1/true/yes/on``), else ``default``."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["ToolkitConfig"]
