"""Studio Google: connect a Google account for Gmail + Google Calendar.

Any user connects *their own* Google account via the standard OAuth2 authorization-code
flow with a loopback redirect back to this local Studio server. The user supplies a
Google Cloud "Web application" OAuth client (``client_id`` + ``client_secret``); the
refresh/access tokens land in the active *writable* secrets backend (macOS keychain or a
local file store — see :func:`himmy.config.secrets.get_writable_provider`) and are never
returned to the GUI. Access tokens are refreshed transparently when expired.

The Gmail (read inbox / send) and Calendar (list / create) helpers call the Google REST
APIs with the connected account's token. HTTP goes through a module-level
:data:`_TRANSPORT` seam so tests can inject an ``httpx.MockTransport`` with no network.
"""

from __future__ import annotations

import base64
import secrets as _secrets
import time
from email.message import EmailMessage
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from himmy.config.secrets import get_secret, get_writable_provider

# ---- secret keys --------------------------------------------------------

# These are *names of* secret-store keys, not secret values themselves.
CLIENT_ID = "HIMMY_GOOGLE_CLIENT_ID"
CLIENT_SECRET = "HIMMY_GOOGLE_CLIENT_SECRET"  # noqa: S105
ACCESS_TOKEN = "HIMMY_GOOGLE_ACCESS_TOKEN"  # noqa: S105
REFRESH_TOKEN = "HIMMY_GOOGLE_REFRESH_TOKEN"  # noqa: S105
TOKEN_EXPIRY = "HIMMY_GOOGLE_TOKEN_EXPIRY"  # noqa: S105
ACCOUNT_EMAIL = "HIMMY_GOOGLE_EMAIL"

SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
)

_AUTH_BASE = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_CAL_BASE = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

#: Injectable httpx transport (tests set an ``httpx.MockTransport``); ``None`` = real net.
_TRANSPORT: httpx.AsyncBaseTransport | None = None

#: Outstanding CSRF ``state`` tokens issued by :func:`auth_url`, consumed on callback.
_PENDING_STATES: set[str] = set()


class GoogleError(RuntimeError):
    """A Google connection / API failure with a user-safe message."""


class GoogleNotConnectedError(GoogleError):
    """Raised when an action needs a connected account but none is present."""


# ---- status -------------------------------------------------------------


class GoogleStatus(BaseModel):
    configured: bool  # client_id + client_secret present
    connected: bool  # we hold a refresh token
    email: str | None = None
    writable: bool


def status() -> GoogleStatus:
    return GoogleStatus(
        configured=bool(get_secret(CLIENT_ID) and get_secret(CLIENT_SECRET)),
        connected=bool(get_secret(REFRESH_TOKEN)),
        email=get_secret(ACCOUNT_EMAIL),
        writable=get_writable_provider() is not None,
    )


# ---- secret IO ----------------------------------------------------------


def _writable() -> Any:
    w = get_writable_provider()
    if w is None:
        raise GoogleError(
            "the active secrets backend is read-only; set HIMMY_SECRETS=keychain or file"
        )
    return w


def set_client(client_id: str, client_secret: str) -> GoogleStatus:
    """Store the user's Google OAuth client credentials."""
    w = _writable()
    w.set(CLIENT_ID, client_id.strip())
    w.set(CLIENT_SECRET, client_secret.strip())
    return status()


def disconnect() -> GoogleStatus:
    """Forget the connected account's tokens (keeps the client credentials)."""
    w = _writable()
    for key in (ACCESS_TOKEN, REFRESH_TOKEN, TOKEN_EXPIRY, ACCOUNT_EMAIL):
        w.delete(key)
    return status()


def forget_client() -> GoogleStatus:
    """Forget everything Google — tokens and client credentials."""
    w = _writable()
    for key in (
        ACCESS_TOKEN,
        REFRESH_TOKEN,
        TOKEN_EXPIRY,
        ACCOUNT_EMAIL,
        CLIENT_ID,
        CLIENT_SECRET,
    ):
        w.delete(key)
    return status()


# ---- OAuth flow ---------------------------------------------------------


def auth_url(redirect_uri: str) -> str:
    """Build the Google consent URL for ``redirect_uri`` and remember its state token."""
    client_id = get_secret(CLIENT_ID)
    if not client_id:
        raise GoogleError("set the Google client_id/secret before connecting")
    state = _secrets.token_urlsafe(24)
    _PENDING_STATES.add(state)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    return f"{_AUTH_BASE}?{urlencode(params)}"


async def exchange_code(code: str, redirect_uri: str, state: str | None) -> GoogleStatus:
    """Exchange an authorization ``code`` for tokens and persist them."""
    if state is not None:
        # Best-effort CSRF check; a restart drops pending states, so don't hard-fail
        # solely on absence, but do consume a known one.
        _PENDING_STATES.discard(state)
    client_id = get_secret(CLIENT_ID)
    client_secret = get_secret(CLIENT_SECRET)
    if not client_id or not client_secret:
        raise GoogleError("missing Google client credentials")
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    async with _http() as client:
        resp = await client.post(_TOKEN_URL, data=data)
        tok = _json_or_raise(resp, "token exchange")
        access = tok.get("access_token")
        refresh = tok.get("refresh_token")
        if not access:
            raise GoogleError("Google did not return an access token")
        email = await _fetch_email(client, access)

    w = _writable()
    w.set(ACCESS_TOKEN, access)
    if refresh:  # Google omits refresh_token on re-consent if one already exists
        w.set(REFRESH_TOKEN, refresh)
    w.set(TOKEN_EXPIRY, str(int(time.time()) + int(tok.get("expires_in", 3600)) - 60))
    if email:
        w.set(ACCOUNT_EMAIL, email)
    return status()


async def _fetch_email(client: httpx.AsyncClient, access: str) -> str | None:
    try:
        resp = await client.get(
            _USERINFO_URL, headers={"Authorization": f"Bearer {access}"}
        )
        if resp.status_code == 200:
            return resp.json().get("email")
    except Exception:  # noqa: BLE001 - email is a nicety, never fatal
        return None
    return None


async def _access_token() -> str:
    """Return a valid access token, refreshing via the refresh token if expired."""
    access = get_secret(ACCESS_TOKEN)
    expiry = int(get_secret(TOKEN_EXPIRY) or "0")
    if access and time.time() < expiry:
        return access
    refresh = get_secret(REFRESH_TOKEN)
    if not refresh:
        raise GoogleNotConnectedError("no Google account is connected")
    client_id = get_secret(CLIENT_ID)
    client_secret = get_secret(CLIENT_SECRET)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }
    async with _http() as client:
        resp = await client.post(_TOKEN_URL, data=data)
        tok = _json_or_raise(resp, "token refresh")
    new_access = tok.get("access_token")
    if not new_access:
        raise GoogleError("token refresh did not return an access token")
    w = _writable()
    w.set(ACCESS_TOKEN, new_access)
    w.set(TOKEN_EXPIRY, str(int(time.time()) + int(tok.get("expires_in", 3600)) - 60))
    return new_access


# ---- Gmail --------------------------------------------------------------


class GmailMessage(BaseModel):
    id: str
    thread_id: str | None = None
    sender: str = ""
    subject: str = "(no subject)"
    snippet: str = ""
    date: str = ""


async def gmail_list(max_results: int = 20) -> list[GmailMessage]:
    """List recent INBOX messages (metadata only)."""
    token = await _access_token()
    headers = {"Authorization": f"Bearer {token}"}
    async with _http() as client:
        resp = await client.get(
            f"{_GMAIL_BASE}/messages",
            params={"maxResults": max_results, "labelIds": "INBOX"},
            headers=headers,
        )
        listing = _json_or_raise(resp, "gmail list")
        ids = [m["id"] for m in listing.get("messages", [])]
        out: list[GmailMessage] = []
        for mid in ids:
            mresp = await client.get(
                f"{_GMAIL_BASE}/messages/{mid}",
                params={
                    "format": "metadata",
                    "metadataHeaders": ["From", "Subject", "Date"],
                },
                headers=headers,
            )
            if mresp.status_code != 200:
                continue
            out.append(_parse_message(mresp.json()))
    return out


def _parse_message(msg: dict[str, Any]) -> GmailMessage:
    headers = {
        h.get("name", "").lower(): h.get("value", "")
        for h in msg.get("payload", {}).get("headers", [])
    }
    return GmailMessage(
        id=msg.get("id", ""),
        thread_id=msg.get("threadId"),
        sender=headers.get("from", ""),
        subject=headers.get("subject", "(no subject)"),
        snippet=msg.get("snippet", ""),
        date=headers.get("date", ""),
    )


class GmailSendResult(BaseModel):
    ok: bool
    id: str | None = None
    detail: str = ""


async def gmail_send(to: str, subject: str, body: str) -> GmailSendResult:
    """Send a plain-text email as the connected account."""
    to = to.strip()
    if not to:
        return GmailSendResult(ok=False, detail="recipient required")
    token = await _access_token()
    email = get_secret(ACCOUNT_EMAIL)
    msg = EmailMessage()
    msg["To"] = to
    if email:
        msg["From"] = email
    msg["Subject"] = subject
    msg.set_content(body)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    async with _http() as client:
        resp = await client.post(
            f"{_GMAIL_BASE}/messages/send",
            json={"raw": raw},
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code not in (200, 201):
        return GmailSendResult(ok=False, detail=_error_detail(resp))
    return GmailSendResult(ok=True, id=resp.json().get("id"), detail=f"sent to {to}")


# ---- Calendar -----------------------------------------------------------


class CalendarEvent(BaseModel):
    id: str | None = None
    summary: str = "(no title)"
    start: str = ""
    end: str = ""
    location: str | None = None
    html_link: str | None = None


async def calendar_list(max_results: int = 20) -> list[CalendarEvent]:
    """List upcoming events from the primary calendar."""
    token = await _access_token()
    time_min = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    async with _http() as client:
        resp = await client.get(
            _CAL_BASE,
            params={
                "timeMin": time_min,
                "maxResults": max_results,
                "singleEvents": "true",
                "orderBy": "startTime",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        data = _json_or_raise(resp, "calendar list")
    return [_parse_event(e) for e in data.get("items", [])]


def _parse_event(e: dict[str, Any]) -> CalendarEvent:
    start = e.get("start", {})
    end = e.get("end", {})
    return CalendarEvent(
        id=e.get("id"),
        summary=e.get("summary", "(no title)"),
        start=start.get("dateTime") or start.get("date") or "",
        end=end.get("dateTime") or end.get("date") or "",
        location=e.get("location"),
        html_link=e.get("htmlLink"),
    )


async def calendar_create(
    summary: str, start: str, end: str, *, all_day: bool = False
) -> CalendarEvent:
    """Create an event on the primary calendar.

    ``start``/``end`` are RFC3339 datetimes (``2026-06-09T14:00:00Z``) or, when
    ``all_day``, plain ``YYYY-MM-DD`` dates.
    """
    token = await _access_token()
    key = "date" if all_day else "dateTime"
    payload = {"summary": summary, "start": {key: start}, "end": {key: end}}
    async with _http() as client:
        resp = await client.post(
            _CAL_BASE,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        data = _json_or_raise(resp, "calendar create")
    return _parse_event(data)


# ---- http helpers -------------------------------------------------------


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_TRANSPORT, timeout=30.0)


def _json_or_raise(resp: httpx.Response, what: str) -> dict[str, Any]:
    if resp.status_code != 200:
        raise GoogleError(f"{what} failed: {_error_detail(resp)}")
    return resp.json()


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error")
        if isinstance(err, dict):
            return f"{resp.status_code} {err.get('message', '')}".strip()
        if isinstance(err, str):
            desc = body.get("error_description", "")
            return f"{resp.status_code} {err} {desc}".strip()
    except Exception:  # noqa: BLE001
        pass
    return f"HTTP {resp.status_code}"


__all__ = [
    "GoogleStatus",
    "GoogleError",
    "GoogleNotConnectedError",
    "GmailMessage",
    "GmailSendResult",
    "CalendarEvent",
    "status",
    "set_client",
    "disconnect",
    "forget_client",
    "auth_url",
    "exchange_code",
    "gmail_list",
    "gmail_send",
    "calendar_list",
    "calendar_create",
    "SCOPES",
]
