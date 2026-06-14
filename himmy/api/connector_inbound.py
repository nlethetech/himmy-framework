"""Mount enabled INBOUND connectors on the BFF, each wired to a configured agent.

The generic webhook and the Slack/Discord intake connectors all share the SDK's
:meth:`~himmy.connectors.sdk.InboundChannelConnector.handle_webhook` contract, and
:func:`~himmy.connectors.router.build_connector_router` turns any of them into a FastAPI
router. This module is the BFF glue: at app startup it asks the connector management
service which inbound connectors are ENABLED and capability-OK (their signing secret is
present), builds an agent handler from a configured ``agent.yaml`` (mirroring the CLI's
``himmy telegram`` wiring), and mounts ONE router per such connector.

Default-deny is preserved end to end:

* a connector is mounted ONLY when it is enabled for the ``inbound`` surface AND its
  capability probe passes (the signing secret is configured) — an unsigned public trigger
  is never mounted;
* the connector's own HMAC / signature + allowlist gate still authorizes every delivery;
* when no inbound agent is configured, NOTHING is mounted (the BFF surface is unchanged),
  so the zero-config offline default is untouched.

The agent handler runs the FULL agent loop (act → observe → answer) when the spec binds
tools, so an inbound delivery is genuinely agentic — not a tool-less single shot. One
:class:`ChatThread` per sender id gives each caller continuous context across deliveries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from himmy.config.secrets import get_secret

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

logger = logging.getLogger("himmy.api.connector_inbound")

#: Non-secret config key naming the ``agent.yaml`` an inbound delivery runs. Read through
#: the secrets layer (so a file/keychain backend or the env can set it). When absent, no
#: inbound connector is mounted — the agent that answers an external trigger must be chosen
#: deliberately, never defaulted.
INBOUND_AGENT_PATH_ENV = "HIMMY_INBOUND_AGENT_PATH"

#: Default mount paths per inbound connector (under the guarded ``/v1`` prefix so the
#: Studio rebinding/cross-site guard already protects them).
_INBOUND_PATHS: dict[str, str] = {
    "webhook": "/v1/connectors/webhook",
    "slack": "/v1/connectors/slack",
    "discord": "/v1/connectors/discord",
}


def inbound_agent_path() -> str | None:
    """The configured inbound ``agent.yaml`` path, or None when none is set."""
    value = (get_secret(INBOUND_AGENT_PATH_ENV) or "").strip()
    return value or None


def _build_inbound_handler(agent_path: str) -> Any:
    """Build an async ``(sender_id, text) -> reply`` handler from an ``agent.yaml``.

    Mirrors ``himmy telegram`` (cli/commands.py:cmd_telegram): load the spec, wire the
    runtime + tool registry via the shared spec-builder, and run the agent loop (with
    tools) or a single task (without). One :class:`ChatThread` per sender keeps each
    caller's context across deliveries. ``durable_defaults=True`` so an agent's
    runs/memory land in the durable server stores, matching the rest of the BFF.
    """
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.runtime.from_spec import build_runtime_for_spec, load_spec_file

    spec = load_spec_file(agent_path)
    runtime, registry = build_runtime_for_spec(spec, durable_defaults=True)
    persona = spec.to_persona()
    llm_config = spec.to_llm_config()
    has_tools = registry is not None
    threads: dict[str, Any] = {}

    async def handle(sender_id: str, text: str) -> str:
        thread = threads.get(sender_id)
        if thread is None:
            thread = ChatThread(agent_id=persona.agent_id)
            threads[sender_id] = thread
        task = spec.make_task(text)
        if has_tools:
            loop = await runtime.run_agent_loop(
                persona,
                task,
                thread,
                llm_config=llm_config,
                max_turns=8,
                route_tools=spec.tool_router,
            )
            result = loop.final
        else:
            result = await runtime.run_task_detailed(
                persona, task, thread, llm_config=llm_config
            )
        if getattr(result, "thread", None) is not None:
            threads[sender_id] = result.thread
        return getattr(result, "output_text", "") or ""

    return handle


def _durable_idempotency_store(app: FastAPI) -> Any:
    """Build a durable inbound-dedup store when the app has durable storage, else None (Q4).

    Returns a :class:`~himmy.services.storage.trigger_dedup.DurableIdempotencyStore` backed by
    the active container's storage when that storage exposes the ``trigger_dedup`` TTL-CAS
    surface (a durable SQLite/Postgres deployment), so a webhook delivery id seen before a
    restart is still deduped after it. Returns ``None`` for the zero-config offline default
    (the in-memory ``StorageService`` DOES implement the surface, but its dedup is process-
    local and lost on exit just like the connector's built-in in-RAM store — so there is no
    durability to gain and we keep the lighter built-in path). The check is therefore "is the
    storage a DURABLE backend?", not merely "does it have the methods?".
    """
    container = getattr(app.state, "container", None)
    storage = getattr(container, "storage", None)
    if storage is None:
        return None
    # Only a durable backend (file SQLite / Postgres) gains anything; the in-memory facade's
    # dedup dies with the process exactly like the connector's built-in store.
    from himmy.services.storage.postgres import PostgresStorageService
    from himmy.services.storage.sqlite import SqliteStorageService

    if not isinstance(storage, (SqliteStorageService, PostgresStorageService)):
        return None
    if not hasattr(storage, "dedup_try_claim"):  # defensive: surface must be present
        return None
    from himmy.services.storage.trigger_dedup import DurableIdempotencyStore

    return DurableIdempotencyStore(storage, scope="webhook")


def mount_inbound_connectors(app: FastAPI) -> list[str]:
    """Mount a router for each ENABLED + capability-OK inbound connector. Returns the names.

    No-op (returns ``[]``) when no inbound ``agent.yaml`` is configured or the secrets
    backend can't resolve one — the BFF surface is then byte-identical to today. A
    connector is mounted only when it is enabled for ``inbound`` AND its signing secret is
    present (capability OK); a disabled or unconfigured connector is skipped, preserving
    default-deny. Mounting is best-effort per connector: one connector that fails to build
    is logged and skipped, never aborting the app startup.

    When the app runs on DURABLE storage (a SQLite/Postgres deployment), inbound
    de-duplication is backed by the durable ``trigger_dedup`` ledger (Q4) so a re-delivered
    id stays deduped across a restart; the zero-config offline default keeps the connector's
    process-local in-RAM dedup (nothing durable to gain). The store is injected through the
    verified ``build_inbound(**kwargs) -> idempotency=`` seam.
    """
    from himmy.connectors.manage import ConnectorService
    from himmy.connectors.router import build_connector_router
    from himmy.connectors.sdk import ConnectorKind

    agent_path = inbound_agent_path()
    if not agent_path:
        return []

    service = ConnectorService()
    idempotency = _durable_idempotency_store(app)
    mounted: list[str] = []
    handler: Any = None
    for name in service.names():
        descriptor = service.descriptor(name)
        if descriptor.kind != ConnectorKind.INBOUND:
            continue
        if not service.is_enabled(name, "inbound"):
            continue
        if not service.status(name).available:
            # missing signing secret — never mount an unsigned public trigger
            continue
        try:
            if handler is None:
                # Build the agent handler lazily, once, only when something will mount.
                handler = _build_inbound_handler(agent_path)
            # Pass the durable dedup store via the **kwargs seam ONLY to connectors that
            # accept it (the webhook connector does). ``None`` leaves the built-in in-RAM
            # store, preserving the offline default byte-for-byte.
            build_kwargs: dict[str, Any] = {}
            if idempotency is not None and name == "webhook":
                build_kwargs["idempotency"] = idempotency
            connector = service.build_inbound(name, handler, **build_kwargs)
            path = _INBOUND_PATHS.get(name, f"/v1/connectors/{name}")
            app.include_router(build_connector_router(connector, path=path))
            mounted.append(name)
            logger.info(
                "mounted inbound connector %r at %s%s",
                name,
                path,
                " (durable dedup)" if build_kwargs.get("idempotency") else "",
            )
        except Exception:  # noqa: BLE001 - one bad connector must not abort startup
            logger.warning(
                "failed to mount inbound connector %r; skipping", name, exc_info=True
            )
    return mounted


__all__ = [
    "INBOUND_AGENT_PATH_ENV",
    "inbound_agent_path",
    "mount_inbound_connectors",
]
