"""Connectors: the secure pattern that bridges agents to external systems.

The :mod:`himmy.connectors.sdk` defines the abstraction every connector follows —
:class:`InboundChannelConnector` (receive a channel message → run the agent → reply,
with a default-deny sender allowlist + HMAC webhook verification) and
:class:`OutboundToolConnector` (expose typed tools whose credentials come from the
secrets layer, whose HTTP is SSRF-guarded, and whose side effects are idempotent), plus
a :class:`ConnectorRegistry`, capability detection, per-connector rate limiting +
bounded retries, secret-safe error handling, and a structured audit event per action.
:mod:`himmy.connectors.spec` adds a declarative YAML connector (:class:`ConnectorSpec`).

Concrete built-ins: Nepali **news** (curated outlet RSS feeds; also a standalone MCP
server via ``python -m himmy.connectors.news_mcp_server``) and **NRB** (foreign-exchange
API + monthly macroeconomic reports + Excel workbook parsing). Every connector fetches
directly from the public source through the injectable :class:`Fetcher` seam, so the
whole layer is testable offline. Live fetching needs the ``connectors`` extra
(``pip install 'himmy[connectors]'`` — feedparser + openpyxl).

See ``docs/connectors/authoring.md`` to write your own.
"""

from __future__ import annotations

from himmy.connectors.discord import (
    DISCORD_API_HOST,
    DISCORD_API_ROOT,
    DISCORD_BOT_TOKEN_SECRET,
    DISCORD_PUBLIC_KEY_SECRET,
    DiscordClient,
    DiscordGatewayListener,
    DiscordInteractionsConnector,
    DiscordOutboundConnector,
    register_discord_connectors,
    verify_ed25519_signature,
)
from himmy.connectors.fetcher import DEFAULT_USER_AGENT, Fetcher, HttpxFetcher
from himmy.connectors.models import (
    FeedFailure,
    ForexRate,
    MacroReport,
    NewsFetchResult,
    NewsItem,
    NewsSource,
    Workbook,
)
from himmy.connectors.news import NEPAL_NEWS_SOURCES, NewsFetcher
from himmy.connectors.news_tools import NEWS_TOOL_NAMES, register_news_tools
from himmy.connectors.nrb import NRB_FOREX_API, NRB_MACRO_FEED, NRBClient
from himmy.connectors.nrb_tools import register_nrb_tools
from himmy.connectors.sdk import (
    Capability,
    Connector,
    ConnectorContext,
    ConnectorError,
    ConnectorKind,
    ConnectorRegistry,
    IdempotencyStore,
    InboundChannelConnector,
    InboundMessage,
    OutboundToolConnector,
    RateLimiter,
    RetryPolicy,
    capability,
    default_registry,
    redact_args,
    register_connector,
    safe_error,
    verify_hmac_signature,
)
from himmy.connectors.slack import (
    SLACK_API_HOST,
    SLACK_API_ROOT,
    SLACK_APP_TOKEN_SECRET,
    SLACK_BOT_TOKEN_SECRET,
    SLACK_SIGNING_SECRET,
    SlackClient,
    SlackEventsConnector,
    SlackOutboundConnector,
    SlackSocketModeListener,
    register_slack_connectors,
    verify_slack_signature,
)
from himmy.connectors.spec import (
    ConnectorSpec,
    SpecToolConnector,
    register_connector_specs,
)

__all__ = [
    "Fetcher",
    "HttpxFetcher",
    "DEFAULT_USER_AGENT",
    # SDK
    "Connector",
    "ConnectorKind",
    "ConnectorContext",
    "ConnectorError",
    "ConnectorRegistry",
    "InboundChannelConnector",
    "InboundMessage",
    "OutboundToolConnector",
    "RateLimiter",
    "RetryPolicy",
    "IdempotencyStore",
    "Capability",
    "capability",
    "verify_hmac_signature",
    "safe_error",
    "redact_args",
    "default_registry",
    "register_connector",
    # declarative spec
    "ConnectorSpec",
    "SpecToolConnector",
    "register_connector_specs",
    "NewsSource",
    "NewsItem",
    "FeedFailure",
    "NewsFetchResult",
    "ForexRate",
    "MacroReport",
    "Workbook",
    "NEPAL_NEWS_SOURCES",
    "NewsFetcher",
    "NRBClient",
    "NRB_FOREX_API",
    "NRB_MACRO_FEED",
    "register_nrb_tools",
    "register_news_tools",
    "NEWS_TOOL_NAMES",
    # Discord connector
    "DiscordClient",
    "DiscordOutboundConnector",
    "DiscordInteractionsConnector",
    "DiscordGatewayListener",
    "verify_ed25519_signature",
    "register_discord_connectors",
    "DISCORD_API_ROOT",
    "DISCORD_API_HOST",
    "DISCORD_BOT_TOKEN_SECRET",
    "DISCORD_PUBLIC_KEY_SECRET",
    # Slack connector
    "SlackClient",
    "SlackOutboundConnector",
    "SlackEventsConnector",
    "SlackSocketModeListener",
    "verify_slack_signature",
    "register_slack_connectors",
    "SLACK_API_ROOT",
    "SLACK_API_HOST",
    "SLACK_BOT_TOKEN_SECRET",
    "SLACK_SIGNING_SECRET",
    "SLACK_APP_TOKEN_SECRET",
]
