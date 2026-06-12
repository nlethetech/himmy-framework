# How to write a Himmy connector

A **connector** is the bridge between a Himmy agent and an external system — a chat
channel it lives in, or a third-party API it can call. Himmy ships a small SDK
(`himmy/connectors/sdk.py`) so every connector inherits the same security posture
instead of re-deriving it (and re-opening the same holes) each time. This guide is the
plain-English version: read it, copy the closest example, and you have a connector that
is allow-listed, signature-verified, secret-safe, SSRF-guarded, rate-limited, idempotent,
and audited — for free.

There are exactly **two kinds** of connector:

1. **Inbound channel** — a message arrives *from* an external channel (Telegram, Slack,
   a webhook), you run the agent, and you reply. Base class:
   `InboundChannelConnector`.
2. **Outbound tool** — you expose typed tools the agent can *call out* with (GitHub,
   Slack, a search API). Base class: `OutboundToolConnector`.

If your connector is "call a REST API with a token", you may not need to write Python at
all — skip to [The no-code path](#the-no-code-path-declarative-yaml).

---

## The five rules (these are not optional)

Every connector MUST follow these. The base classes enforce them so long as you use the
provided seams; reach around them and you are on your own.

1. **Credentials come from the secrets layer.** Read every token/key with
   `self.secret("MY_TOKEN")` (outbound) or `get_secret("MY_TOKEN")` — never
   `os.environ[...]` directly, never a constructor argument the model could fill. A
   secret is never logged, never echoed in an error, never put on an event.
2. **All outbound HTTP goes through the guarded fetcher.** Use `self.fetcher` and
   `self.guard(url)`. They block requests to loopback / private / link-local / cloud-
   metadata addresses (SSRF) and re-check every redirect hop. Never build your own bare
   `httpx.Client`.
3. **Authenticate the sender; verify the signature.** Inbound connectors are
   **default-deny**: an empty allowlist answers *no one*. For webhooks, set a
   `signing_secret` so the HMAC is verified on the raw body *before* the agent runs.
4. **Side effects are idempotent.** A POST that times out must not be blindly retried
   into a double-send. Wrap mutating calls with `self.call_idempotent(key, fn)` and use
   a `RetryPolicy` that only retries idempotent calls.
5. **Never leak a secret in an error or an event.** Use `safe_error(exc)` to turn an
   exception into a loggable string, and `redact_args(args)` before putting arguments
   into an audit detail.

---

## The no-code path (declarative YAML)

For the common "call a REST API with a bearer token" connector, declare it — no Python.
A `ConnectorSpec` (`himmy/connectors/spec.py`) becomes a fully-secure connector:

```yaml
connectors:
  - name: github
    description: GitHub REST API.
    base_url: https://api.github.com
    auth: { type: bearer, secret: GITHUB_TOKEN }   # secret NAME, never a literal
    egress_allow_hosts: [api.github.com]           # pin the connector to its host
    rate_limit: { rate: 30, per_seconds: 60 }      # self-throttle
    tools:
      - name: github_get_issue                     # a read → read_only, retryable
        method: GET
        path: /repos/{owner}/{repo}/issues/{number}
      - name: github_create_issue                  # a write → approval-gated
        method: POST
        path: /repos/{owner}/{repo}/issues
        body: [title, body]
        requires_approval: true
        idempotency_arg: title                     # dedupe a re-issued create
```

What you get automatically:

- The credential is resolved from the secrets layer; if `GITHUB_TOKEN` is absent the
  connector reports **unavailable** and is skipped cleanly — it does not crash the rest.
- Path placeholders are percent-encoded (a model-supplied `owner` can't escape the path
  or pivot to another host), and the final URL is SSRF-guarded against your
  `egress_allow_hosts` before any request.
- `GET`/`HEAD` are marked `read_only` and may be retried; other methods are not retried
  unless you give them an `idempotency_arg`, in which case a repeat with the same key
  returns the first result *without re-firing the effect*.
- Each tool flows through the normal tool pipeline (arg validation, approval gating,
  events, lineage).

Register them onto a tool registry:

```python
from himmy.connectors.spec import ConnectorSpec, register_connector_specs

specs = [ConnectorSpec(**raw) for raw in yaml_doc["connectors"]]
register_connector_specs(registry, specs)   # unavailable ones are skipped, not fatal
```

Write Python only when you need something the spec can't express (a non-HTTP transport,
a polling inbound channel, multi-step pagination, a bespoke auth dance).

---

## Writing an outbound tool connector (Python)

Subclass `OutboundToolConnector`. Declare the secrets/modules you need so capability
detection can skip you cleanly when they're absent, then register your tools.

```python
from himmy.connectors.sdk import OutboundToolConnector
from himmy.connectors.fetcher import get_json
from himmy.services.tools.registry import ToolRegistry, register_local_tool


class WeatherConnector(OutboundToolConnector):
    name = "weather"
    required_secrets = ("WEATHER_API_KEY",)     # → capability() reports unavailable if absent
    required_modules = ()                       # e.g. ("some_optional_dep",)

    def register(self, registry: ToolRegistry) -> list[str]:
        async def get_weather(args: dict) -> dict:
            self._throttle()                                  # rate-limit token
            key = self.secret("WEATHER_API_KEY")              # from the secrets layer
            city = str(args["city"])
            url = self.guard(f"https://api.example.com/w?city={city}&key={key}")
            data = await asyncio.to_thread(get_json, self.fetcher, url)  # guarded fetch
            self._audit("get_weather", "allow", detail=f"city={city}")  # audit event
            return {"temp_c": data.get("temp_c")}

        register_local_tool(
            registry,
            name="get_weather",
            handler=get_weather,
            description="Current weather for a city.",
            args_json_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            read_only=True,
            metadata={"connector": self.name},
        )
        return ["get_weather"]
```

Call `connector.register_tools(registry)` (not `register` directly): it runs the
capability gate first, so a connector missing its key registers **zero** tools and
audits the skip instead of raising.

### Side-effecting calls (the idempotency rule)

A tool that *changes* something must not double-fire on a retry. Wrap it:

```python
def send(args: dict) -> dict:
    # `idem` is a client-supplied key; the same key fires the effect once.
    return self.call_idempotent(
        f"send:{args['idem']}",
        lambda: self._post("/messages", {"text": args["text"]}),
    )
```

`call_idempotent` runs the effect at most once per key and caches the result; a repeated
key returns the cached result. Combine with a `RetryPolicy` that has
`idempotent=True` only for reads or idempotent-keyed writes — never blind-retry a raw
mutation.

---

## Writing an inbound channel connector (Python)

Subclass `InboundChannelConnector` and implement the channel seams. The SDK gives you
the security gate, the poll/offset loop, and the webhook path; you implement how to pull
updates, parse them, and send a reply.

```python
from typing import Any
from himmy.connectors.sdk import InboundChannelConnector, InboundMessage


class MyChannelConnector(InboundChannelConnector):
    name = "mychannel"

    def __init__(self, handler, client, **kw):
        super().__init__(handler, **kw)      # handler(sender_id, text) -> reply
        self._client = client

    # --- polling delivery (Telegram-style offset loop) ---
    async def fetch_updates(self, offset):
        return await self._client.get_updates(offset=offset)

    def parse_update(self, update) -> InboundMessage | None:
        msg = update.get("message") or {}
        text, sender = msg.get("text"), (msg.get("from") or {}).get("id")
        if not text or sender is None:
            return None
        return InboundMessage(sender_id=str(sender), text=text, raw=update)

    def next_offset(self, update, current):
        return int(update["update_id"]) + 1

    async def send_reply(self, msg: InboundMessage, reply: str) -> None:
        await self._client.send(msg.sender_id, reply)

    # --- webhook delivery (optional; signature is verified by the SDK first) ---
    def parse_webhook(self, body: bytes, headers) -> InboundMessage | None:
        import json
        payload = json.loads(body)
        return InboundMessage(sender_id=str(payload["from"]), text=payload["text"])
```

Wire it with the security policy:

```python
bot = MyChannelConnector(
    handler,
    client,
    allowed_senders=["123", "456"],          # default-deny allowlist
    signing_secret=get_secret("MYCHANNEL_WEBHOOK_SECRET"),  # enables HMAC verify
    signature_header="X-Signature",
    signature_prefix="sha256=",              # strip a scheme marker if the channel sends one
)

# polling:
await bot.run(stop=should_stop)

# OR webhook (e.g. from a FastAPI route): signature → allowlist → handler
result = await bot.handle_webhook(raw_body, request.headers)
```

The order on the webhook path is load-bearing and the SDK enforces it: **verify the
HMAC over the raw body → check the sender allowlist → run the agent**. A bad signature
or an unlisted sender is denied and audited; the agent never sees it.

### Allowlist semantics (default-deny)

- **Non-empty allowlist** → only listed senders are answered.
- **Empty allowlist** → *no one* is answered (the safe default for anything publicly
  reachable). A private poll bot that you knowingly want to answer everyone can pass
  `allow_empty_allowlist=True`.

---

## Discovery: the connector registry

Register a connector *factory* (a zero-arg callable) so capability is evaluated lazily:

```python
from himmy.connectors.sdk import register_connector, default_registry

register_connector("weather", lambda: WeatherConnector())

# later, wire only what can actually run in this environment:
for connector in default_registry().discover(available_only=True):
    ...
```

`discover(available_only=True)` skips any connector whose dependency or credential is
missing, so you never wire a connector that would fail on first use.

---

## Capability detection (skip cleanly)

Declare what you need and the SDK probes it without importing heavy deps or reading a
secret value:

```python
class SlackConnector(OutboundToolConnector):
    name = "slack"
    required_secrets = ("SLACK_BOT_TOKEN",)
    required_modules = ()        # add an optional import here if you have one

# elsewhere:
cap = connector.capability()
if not cap.ok:
    log.info("slack connector unavailable: %s", cap.reason)  # names the MISSING thing
```

`cap.reason` names the missing module or the missing secret *variable* — never a secret
value. This is what lets `register_tools` / `discover` skip a connector cleanly.

---

## Rate limiting, retries, and audit (the shared context)

Pass a `ConnectorContext` to give a connector a rate limit, a retry policy, and an audit
sink:

```python
from himmy.connectors.sdk import ConnectorContext, RateLimiter, RetryPolicy, AuditSink
from himmy.services.audit.log import SecurityAuditLog

ctx = ConnectorContext(
    audit=AuditSink(SecurityAuditLog(entity_registry).record),  # structured audit events
    rate_limiter=RateLimiter(rate=30, per_seconds=60),          # token bucket
    retry=RetryPolicy(attempts=3, retry_on=(httpx.TransportError,)),  # bounded, idempotent-only
)
connector = WeatherConnector(context=ctx)
```

- **Audit** — every connector action records a `SecurityEvent` (kind
  `connector_action`) on the existing tamper-evident audit spine: `allow`/`deny`, the
  connector name, the action, and a secret-safe detail. A denied sender, a bad
  signature, a blocked URL, and a successful call all leave a trail.
- **Rate limit** — `self._throttle()` consumes a token; over budget raises
  `ConnectorError`. This is the connector's own self-throttle so a runaway agent or a
  webhook flood can't hammer the upstream and get the credential banned.
- **Retry** — `RetryPolicy.run(call, idempotent=...)` retries *only* idempotent calls,
  *only* on the exception classes you list, with bounded exponential backoff. A
  non-idempotent call is attempted exactly once.

All three are optional: a connector built with the bare default audits nothing,
throttles at a sane rate, and does not retry — the safe baseline.

---

## Error handling that never leaks

Surface failures with `safe_error`:

```python
from himmy.connectors.sdk import safe_error

try:
    ...
except Exception as exc:                       # noqa: BLE001
    detail = safe_error(exc, secrets=[token])   # type-only for unknown exceptions; scrubs literals
    self._audit("send", "deny", detail=detail)
    raise ConnectorError("send failed")         # ConnectorError messages are authored safe
```

`safe_error` reduces an *unknown* exception to its class name (so a credential a library
echoed into `str(exc)` can't escape), preserves the message of the SDK's own
secret-safe `ConnectorError` / `ToolSecurityError`, and scrubs any literal secret you
pass in `secrets=`.

---

## Testing your connector

Connectors are built around injectable seams, so they test fully offline.

- **Inject the transport.** Outbound: pass a fixture `Fetcher` (see
  `tests/connectors/_fixtures.py`) or an `httpx.MockTransport`. Inbound: pass a fake
  client that serves scripted update batches.
- **Assert the security spine.** Allowlist default-deny, a wrong HMAC signature is
  rejected *before* the handler runs, the egress guard blocks a private/off-allowlist
  host, the credential is read from the secrets layer (install a test provider with
  `configure_secrets(...)`), and a duplicate idempotency key fires the effect once.
  See `tests/connectors/test_sdk.py` and `tests/connectors/test_connector_spec.py` for
  the full pattern.
- **Contract-test external services.** A connector that hits Slack/GitHub/Discord can't
  be truly end-to-end tested without a live token. Mock the transport and assert the
  exact request it *would* make — URL, method, auth header, body — plus the
  SSRF/allowlist/idempotency enforcement. That proves the contract; verifying against
  the real service then needs only a live token.

---

## Checklist before you ship

- [ ] Credentials read via `self.secret(...)` / `get_secret(...)` — never `os.environ`,
      never a constructor arg, never logged.
- [ ] All HTTP through `self.fetcher` / `self.guard(url)` — no bare `httpx.Client`.
- [ ] Inbound: a sender allowlist (default-deny) and, for webhooks, a `signing_secret`.
- [ ] Side-effecting calls wrapped in `call_idempotent`; retries are idempotent-only.
- [ ] `required_secrets` / `required_modules` declared so capability detection can skip
      you cleanly.
- [ ] Errors surfaced via `safe_error`; audit details use `redact_args`.
- [ ] A `ConnectorContext` with a rate limit (and, in production, an audit sink).
- [ ] Offline tests covering the security spine; external-service connectors
      contract-tested and clearly marked "needs a live token to verify".

## Related docs

- [connectors](../services/connectors.md) — the built-in Nepal connectors and the
  `Fetcher` seam.
- [tools](../services/tools.md) — the pipeline a connector's tools run through.
- [mcp](../services/mcp.md) — plugging the wider MCP ecosystem in as tools.
- [Live on Telegram](../advanced.md#live-on-telegram) — the canonical inbound channel.
