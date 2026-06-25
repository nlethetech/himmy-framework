"""Generic inbound-connector FastAPI router: one signed HTTP trigger for any channel.

:func:`build_webhook_router` (webhook.py) mounts the *generic* webhook connector. This
module generalizes it to **any** :class:`~himmy.connectors.sdk.InboundChannelConnector` —
the generic webhook AND the Slack Events / Discord Interactions connectors — so the BFF
mounts every enabled inbound channel through ONE code path with one security posture:

* the RAW request body is read and handed to the connector's
  :meth:`~himmy.connectors.sdk.InboundChannelConnector.handle_webhook`, so the connector's
  HMAC / signature check runs over exactly the bytes the sender signed (no re-serialization);
* the connector's own default-deny gate (signature → allowlist → de-dupe → run the agent)
  is what authorizes the request — the router never relaxes it;
* a rejected delivery maps onto an HTTP status (``401`` bad signature, ``403`` denied
  sender, ``413`` oversize, ``400`` otherwise) with a generic, secret-free reason — the
  request body and any internal detail are NEVER echoed back.

The only channel-specific wrinkle is the **provider response body** an accepted delivery
must return: a plain webhook returns the agent reply; Slack returns ``{"challenge": …}``
for the URL-verification handshake; Discord returns the interaction-response object
(``{"type": 1}`` PONG, ``{"type": 4, …}`` command reply). A small per-connector
``provider_response`` mapper handles that without forking the router.

Nothing here imports FastAPI at module load — it stays a lazy dependency so importing
:mod:`himmy.connectors.router` is free in an air-gapped / CLI-only install.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps fastapi lazy
    from fastapi import APIRouter

    from himmy.connectors.sdk import InboundChannelConnector

# HTTP status mapping for a connector's (secret-free) rejection reasons. Mirrors
# build_webhook_router so every inbound channel rejects with the same codes.
_REASON_STATUS: dict[str, int] = {
    "invalid signature": 401,
    "source not allowed": 403,
    "sender not allowed": 403,
    "payload too large": 413,
}

#: A mapper from a connector's ``handle_webhook`` result dict to the JSON body returned to
#: the PROVIDER on an accepted delivery (None ⇒ return the whole result dict, the webhook /
#: Slack default — Slack reads ``challenge`` from it, a generic sender reads ``reply``).
ProviderResponse = Callable[[Mapping[str, Any]], Any]


def _discord_provider_response(result: Mapping[str, Any]) -> Any:
    """Return the Discord interaction-response object (PONG / command reply) as the body.

    Discord reads the HTTP body as the interaction response itself, so we hand back the
    connector's ``response`` field (``{"type": 1}`` for a PING, ``{"type": 4, …}`` for a
    command reply). When the connector ran but produced no interaction response (an
    acknowledged-but-ignored interaction) we fall back to a deferred-ack so Discord does
    not show an error.
    """
    response = result.get("response")
    if isinstance(response, Mapping):
        return dict(response)
    # type 5 = DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE: a valid "received, nothing to say"
    return {"type": 5}


#: Per-connector provider-response mappers. A name absent here uses the default (return the
#: whole result dict), which is correct for the generic webhook and for Slack.
_PROVIDER_RESPONSE: dict[str, ProviderResponse] = {
    "discord": _discord_provider_response,
    "discord-interactions": _discord_provider_response,
}


def provider_response_for(
    connector: InboundChannelConnector,
) -> ProviderResponse | None:
    """The provider-response mapper for ``connector`` (None ⇒ return the whole result)."""
    return _PROVIDER_RESPONSE.get(getattr(connector, "name", ""))


def build_connector_router(
    connector: InboundChannelConnector,
    *,
    path: str,
    ack: int | None = None,
    provider_response: ProviderResponse | None = None,
    tags: Collection[str] = ("connectors",),
) -> APIRouter:
    """Mount any inbound channel ``connector`` as a FastAPI router on ``path``.

    Generalizes :func:`himmy.connectors.webhook.build_webhook_router` over the whole
    :class:`~himmy.connectors.sdk.InboundChannelConnector` family. The route reads the RAW
    body (so the signature is verified over the exact bytes sent) and delegates to
    :meth:`InboundChannelConnector.handle_webhook`; the connector's own default-deny gate
    authorizes the request. A rejection is mapped onto an HTTP status with a generic,
    secret-free reason — the body is never echoed.

    ``ack`` is the status code for an accepted delivery the connector marked ``accepted``
    (async-callback mode); it defaults to ``202`` when the connector advertises a result
    callback, else ``200``. ``provider_response`` maps the connector's result dict to the
    body the provider expects; when omitted it is resolved from the connector name (Discord
    returns its interaction response; webhook/Slack return the whole result dict).

    Mount on the BFF::

        app.include_router(build_connector_router(conn, path="/v1/connectors/slack"))

    The route inherits the app's global dependencies (the Studio rebinding guard, auth,
    rate-limit) when mounted on the BFF; the connector's OWN signature + allowlist gate is
    independent and always applies, so it is equally safe on a bare standalone app.
    """
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse

    from himmy.connectors.webhook import (
        DEFAULT_MAX_BODY_BYTES,
        BodyTooLarge,
        read_body_capped,
    )

    router = APIRouter(tags=list(tags))
    mapper = provider_response or provider_response_for(connector)
    # 202 only makes sense when a reply is delivered out-of-band (an async callback).
    has_callback = getattr(connector, "_result_callback", None) is not None
    ack_status = ack if ack is not None else (202 if has_callback else 200)
    # Per-connector cap when it sets one (the webhook connector does); else the shared
    # default. Bounds the body BEFORE it is buffered, so an oversized delivery on any
    # inbound channel can't pin memory ahead of the connector's signature check.
    max_body_bytes = getattr(connector, "_max_body_bytes", DEFAULT_MAX_BODY_BYTES)

    async def receive(request: Request) -> JSONResponse:
        """Receive a signed delivery, run the agent (gated), return the provider body."""
        try:
            body = await read_body_capped(request, max_body_bytes)
        except BodyTooLarge:
            # Reject an over-cap (or over-declared) body BEFORE buffering the whole thing.
            return JSONResponse(
                status_code=413, content={"ok": False, "reason": "payload too large"}
            )
        result = await connector.handle_webhook(body, dict(request.headers))
        if result.get("ok"):
            status = ack_status if result.get("accepted") else 200
            content = mapper(result) if mapper is not None else dict(result)
            return JSONResponse(status_code=status, content=content)
        reason = str(result.get("reason") or "rejected")
        status_code = _REASON_STATUS.get(reason, 400)
        return JSONResponse(
            status_code=status_code, content={"ok": False, "reason": reason}
        )

    # ``from __future__ import annotations`` stringifies every annotation, so FastAPI would
    # evaluate ``"Request"`` against this module's globals — where fastapi is NOT imported
    # (lazy dependency) — and mistake the parameter for a query field (422). Bind the
    # already-resolved class objects so FastAPI sees the real types without a module import.
    receive.__annotations__["request"] = Request
    receive.__annotations__["return"] = JSONResponse
    router.add_api_route(path, receive, methods=["POST"], include_in_schema=True)
    return router


__all__ = [
    "ProviderResponse",
    "build_connector_router",
    "provider_response_for",
]
