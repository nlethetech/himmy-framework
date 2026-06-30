"""API kernel: Studio notifications router + the in-process notification sink.

Mounted under ``/api/studio/notify`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`).

THE NOTIFICATION CONTRACT
-------------------------
:func:`record_notification` is the one shared write path every other Studio
lane (missions, routines, MCP, projects, …) imports from day one::

    from himmy.api.routers.studio_notify import record_notification
    record_notification("mission", "Mission finished", body="…", link="/missions")

Guarantees the callers rely on — keep these when hardening:

* the signature ``record_notification(kind, title, body="", link="") -> None``
  never changes;
* it NEVER raises (a broken notification must never break the caller's work);
* storage is bounded (deque of 500) and thread-safe, so any thread —
  request handlers, background workers — may call it.

HARDENING (the notifications lane)
----------------------------------
Behind the unchanged signature the sink now also:

* mirrors every notification into a small SQLite store (``.himmy/notify.db``,
  same cwd-keyed pattern as the other Studio stores) so the bell survives a
  restart — the DB is OPTIONAL: any storage failure degrades to memory-only;
* tracks per-item read state (``POST /notify/{id}/read``, ``/notify/read-all``)
  and serves an incremental poll (``GET /notify?after=<id>``) with the unread
  count;
* optionally forwards each notification to Telegram (setting
  ``{"forward_telegram": true}``, persisted with the store) through the same
  bot credentials the Connections screen manages. Forwarding is fire-and-forget
  on a daemon thread: a Telegram outage can never break recording.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel

from himmy.api.auth import scoped_read, singleton_read_filter
from himmy.api.routers.studio_common import build_studio_router, studio_permission

router = build_studio_router("notify", tag="studio-notify")

#: Marking notifications read and updating notification settings are mutations gated by
#: ``studio.notify:write`` (admin-only by default), additively on top of the router's
#: ``studio.notify:read`` baseline so a read-only role can view but not mutate state.
_notify_write = Depends(studio_permission("studio.notify", "write"))


# ---- in-process notification sink ------------------------------------------

#: Ring size for both the in-memory deque and the durable mirror.
RING_SIZE = 500
#: Telegram forwards are alerts, not transcripts — keep them terse.
_FORWARD_TEXT_CAP = 400

#: Bounded in-memory store: oldest entries fall off past 500. Authoritative for
#: reads within this process; the SQLite mirror only seeds it once at startup.
_NOTIFICATIONS: deque[dict[str, Any]] = deque(maxlen=RING_SIZE)
_LOCK = threading.Lock()
_NEXT_ID = 0  # monotonic per-process id, guarded by _LOCK

# Durable mirror state (all guarded by _LOCK). The DB is best-effort: when it
# cannot be opened or written, the in-memory contract above still holds.
_CONN: sqlite3.Connection | None = None
_CONN_PATH: str | None = None
_DB_FAILED_PATH: str | None = None  # don't retry a broken path on every write
_HYDRATED = False  # the deque was seeded from disk (or there was nothing to seed)
_FORWARD_TELEGRAM = False  # the one notify setting, mirrored to the DB

# K5: under a Postgres DSN the durable mirror lives in Postgres (no .himmy/notify.db
# sidecar). The in-memory deque stays the live read truth exactly as on the SQLite path;
# this holder is the K5 PostgresNotifyStore the durability helpers route to instead of
# SQLite. ``None`` until first resolved; ``_PG_FAILED`` mirrors ``_DB_FAILED_PATH`` so a
# transient pool failure degrades to memory-only and is not retried every write.
_PG_STORE: object | None = None
_PG_FAILED = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL DEFAULT '',
    link         TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    read         INTEGER NOT NULL DEFAULT 0,
    workspace_id TEXT
);
CREATE TABLE IF NOT EXISTS notify_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def notify_db_path() -> str:
    """The durable store's path (``HIMMY_NOTIFY_PATH`` overrides for tests).

    Resolved against the current working directory so a project-root change
    (tests chdir per case) re-keys the connection instead of silently writing
    into the old root's store.
    """
    env = os.environ.get("HIMMY_NOTIFY_PATH")
    if env:
        return env
    return str(Path.cwd() / ".himmy" / "notify.db")


def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    return {
        "id": int(row["id"]),
        "kind": row["kind"],
        "title": row["title"],
        "body": row["body"],
        "link": row["link"],
        "created_at": row["created_at"],
        "read": bool(row["read"]),
        # workspace_id is the additive tenant-isolation column; a pre-migration row (no
        # such column) reads back as None = "single local tenant / predates tenant binding".
        "workspace_id": row["workspace_id"] if "workspace_id" in keys else None,
    }


def _pg_enabled() -> bool:
    """True when the notify mirror should route to Postgres (``HIMMY_DATABASE_URL`` is PG)."""
    try:
        from himmy.services.storage.aux_store_factory import aux_postgres_enabled

        return aux_postgres_enabled()
    except Exception:  # noqa: BLE001 - never let backend selection break recording
        return False


def _pg_store_locked() -> object | None:
    """Resolve (once) the Postgres notify mirror; ``None`` on any failure (lock held).

    The first touch hydrates the deque + the forward setting + the id counter from
    Postgres so the bell survives a restart, mirroring ``_ensure_db_locked``'s SQLite
    hydration. Best-effort: a pool failure marks the mirror broken and degrades to
    memory-only, never raising into the caller.
    """
    global _PG_STORE, _PG_FAILED, _HYDRATED, _NEXT_ID, _FORWARD_TELEGRAM
    if _PG_FAILED:
        return None
    if _PG_STORE is not None:
        return _PG_STORE
    try:
        from himmy.services.storage.postgres_aux import PostgresNotifyStore

        store = PostgresNotifyStore(tenant="local")
        _NEXT_ID = max(_NEXT_ID, store.max_id())
        if not _HYDRATED:
            items, forward = store.hydrate(RING_SIZE)
            _NOTIFICATIONS.clear()
            for item in items:
                _NOTIFICATIONS.append(item)
            _FORWARD_TELEGRAM = forward
        _HYDRATED = True
        _PG_STORE = store
        return store
    except Exception:  # noqa: BLE001 - storage is optional; memory-only is fine
        _PG_FAILED = True
        _HYDRATED = True  # the open was attempted; memory is the only truth from here
        return None


def _ensure_db_locked(*, create: bool) -> sqlite3.Connection | None:
    """Open (and once per process, hydrate from) the durable mirror.

    Caller holds ``_LOCK``. Never raises — any failure marks the path broken
    and falls back to memory-only. With ``create=False`` a missing DB file is
    left uncreated (reads/marks never conjure a database into existence).

    Under a Postgres DSN (K5) this delegates to :func:`_pg_store_locked` and returns
    ``None`` for the SQLite handle — so NO ``.himmy/notify.db`` sidecar is created — while
    still seeding the deque from Postgres.
    """
    if _pg_enabled():
        _pg_store_locked()
        return None
    global _CONN, _CONN_PATH, _DB_FAILED_PATH, _HYDRATED, _NEXT_ID
    global _FORWARD_TELEGRAM
    path = notify_db_path()
    if _DB_FAILED_PATH == path:
        return None
    if _CONN is not None and _CONN_PATH == path:
        return _CONN
    if _CONN is not None:  # the project root changed — reopen against it
        try:
            _CONN.close()
        except Exception:  # noqa: BLE001 - closing a stale handle is best-effort
            pass
        _CONN, _CONN_PATH = None, None
    if not create and path != ":memory:" and not Path(path).exists():
        return None  # reads never conjure a DB; hydration stays pending
    try:
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        from himmy.core.sqlite_util import connect_hardened

        conn = connect_hardened(path)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        # Additive + nullable tenant-isolation column so an old single-box notify.db (whose
        # CREATE predates the column) opens + reads unchanged; legacy rows carry NULL.
        from himmy.api.studio_tenant_scope import ensure_workspace_column

        ensure_workspace_column(conn, "notifications")
        # Ids must keep climbing past anything already persisted at this path.
        cur = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM notifications")
        _NEXT_ID = max(_NEXT_ID, int(cur.fetchone()["m"]))
        if not _HYDRATED:
            # First touch in this process: restore the ring + settings so the
            # bell survives a restart. Later path changes never re-seed (the
            # in-memory deque is the live truth once the process is running).
            rows = conn.execute(
                "SELECT * FROM notifications ORDER BY id DESC LIMIT ?",
                (RING_SIZE,),
            ).fetchall()
            _NOTIFICATIONS.clear()
            for r in reversed(rows):
                _NOTIFICATIONS.append(_row_to_item(r))
            srow = conn.execute(
                "SELECT value FROM notify_settings WHERE key = 'forward_telegram'"
            ).fetchone()
            _FORWARD_TELEGRAM = bool(srow) and srow["value"] == "1"
        _HYDRATED = True
        _CONN, _CONN_PATH = conn, path
        return conn
    except Exception:  # noqa: BLE001 - storage is optional; memory-only is fine
        _DB_FAILED_PATH = path
        # The open was attempted: from here on memory is the only truth, and a
        # later successful open must never clear what memory accumulated.
        _HYDRATED = True
        return None


def _persist_item_locked(item: dict[str, Any]) -> None:
    """Mirror one freshly recorded item to the durable store (best-effort, lock held)."""
    if _pg_enabled():
        store = _pg_store_locked()
        if store is None:
            return
        try:
            store.persist_item(item, RING_SIZE)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - never let the mirror break recording
            pass
        return
    conn = _ensure_db_locked(create=True)
    if conn is None:
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO notifications "
            "(id, kind, title, body, link, created_at, read, workspace_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                item["id"],
                item["kind"],
                item["title"],
                item["body"],
                item["link"],
                item["created_at"],
                int(item["read"]),
                item.get("workspace_id"),
            ),
        )
        # Keep the mirror ring-bounded too (ids are monotonic).
        conn.execute(
            "DELETE FROM notifications WHERE id <= ?", (item["id"] - RING_SIZE,)
        )
        conn.commit()
    except Exception:  # noqa: BLE001 - never let the mirror break recording
        pass


def _persist_read_locked(ids: list[int] | None) -> None:
    """Mirror read-state to the durable store; ``None`` marks everything read (lock held)."""
    if _pg_enabled():
        store = _pg_store_locked()
        if store is None:
            return
        try:
            store.mark_read(ids)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return
    conn = _ensure_db_locked(create=False)
    if conn is None:
        return
    try:
        if ids is None:
            conn.execute("UPDATE notifications SET read = 1")
        else:
            conn.executemany(
                "UPDATE notifications SET read = 1 WHERE id = ?",
                [(i,) for i in ids],
            )
        conn.commit()
    except Exception:  # noqa: BLE001
        pass


def reset_notify_state() -> None:
    """Drop every in-process handle/cache (tests change cwd between cases)."""
    global _CONN, _CONN_PATH, _DB_FAILED_PATH, _HYDRATED, _NEXT_ID
    global _FORWARD_TELEGRAM, _PG_STORE, _PG_FAILED
    with _LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:  # noqa: BLE001
                pass
        if _PG_STORE is not None:
            try:
                _PG_STORE.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        _CONN, _CONN_PATH, _DB_FAILED_PATH = None, None, None
        _PG_STORE, _PG_FAILED = None, False
        _HYDRATED = False
        _NEXT_ID = 0
        _FORWARD_TELEGRAM = False
        _NOTIFICATIONS.clear()


# ---- telegram forwarding -----------------------------------------------------


def _telegram_target() -> tuple[str, str] | None:
    """The configured ``(bot_token, chat_id)`` pair, or ``None`` if incomplete."""
    from himmy.toolkit.config import ToolkitConfig

    cfg = ToolkitConfig.from_env()
    if cfg.telegram_bot_token and cfg.telegram_default_chat_id:
        return str(cfg.telegram_bot_token), str(cfg.telegram_default_chat_id)
    return None


def _dispatch_telegram(token: str, chat_id: str, text: str) -> None:
    """Fire-and-forget ``sendMessage`` on a daemon thread (never raises)."""

    def _run() -> None:
        try:
            import asyncio

            from himmy.toolkit.telegram import TelegramClient

            asyncio.run(TelegramClient(token, timeout=10.0).send_message(chat_id, text))
        except Exception:  # noqa: BLE001 - a Telegram outage must stay silent
            pass

    try:
        threading.Thread(target=_run, name="himmy-notify-telegram", daemon=True).start()
    except Exception:  # noqa: BLE001
        pass


def _forward_to_telegram(item: dict[str, Any]) -> None:
    """Forward one recorded item if forwarding is on and a bot is configured."""
    try:
        target = _telegram_target()
        if target is None:
            return
        text = str(item["title"])
        if item["body"]:
            text += "\n" + str(item["body"])
        text = text[:_FORWARD_TEXT_CAP]
        _dispatch_telegram(target[0], target[1], f"[himmy] {text}")
    except Exception:  # noqa: BLE001 - forwarding never breaks recording
        return


# ---- the write contract -------------------------------------------------------


def record_notification(
    kind: str,
    title: str,
    body: str = "",
    link: str = "",
    *,
    workspace_id: str | None = None,
) -> None:
    """Record one notification. Thread-safe, bounded, and never raises.

    ``kind`` is a short machine tag (``"mission"``, ``"routine"``, …), ``title``
    the human one-liner, ``body`` optional detail, ``link`` an optional in-app
    route (``"/missions"``). Failures are swallowed by design: notifying must
    never take down the work being notified about.

    ``workspace_id`` (keyword-only, defaulting to ``None``) is the tenant the notification
    belongs to. The default keeps EVERY existing importer call-site
    (missions/routines/MCP/projects/…) compiling and stamping ``NULL`` =
    "single local tenant" — so the offline / single-box bell is **byte-unchanged**. A
    tenant-aware caller threads its resolved workspace so the list endpoint can hide it from
    other tenants once an authenticator is configured.
    """
    global _NEXT_ID
    try:
        item = {
            "id": 0,  # assigned under the lock below
            "kind": str(kind),
            "title": str(title),
            "body": str(body),
            "link": str(link),
            "created_at": datetime.now(UTC).isoformat(),
            "read": False,
            "workspace_id": workspace_id,
        }
        with _LOCK:
            # Hydrate-before-mutate: the first touch seeds the ring + the id
            # counter from disk, so this item lands AFTER what's restored.
            _ensure_db_locked(create=True)
            _NEXT_ID += 1
            item["id"] = _NEXT_ID
            _NOTIFICATIONS.append(item)
            _persist_item_locked(item)
            forward = _FORWARD_TELEGRAM
        if forward:
            _forward_to_telegram(dict(item))
    except Exception:  # noqa: BLE001 - contract: never raise into the caller
        return


# ---- API surface ----------------------------------------------------------------


class NotifySettings(BaseModel):
    """The notify lane's one persisted preference."""

    forward_telegram: bool = False


def _item_in_scope(item: dict[str, Any], scope: frozenset[str] | None) -> bool:
    """Whether a notification is visible to a ``scope``-restricted reader.

    ``scope is None`` (unrestricted / offline / ``all_tenants``) → always True
    (byte-unchanged); a tenant-bound reader sees its own workspaces' notifications AND any
    legacy ``NULL``-tenant one (predates tenant binding, belongs to no tenant).
    """
    if scope is None:
        return True
    ws = item.get("workspace_id")
    return ws is None or ws in scope


@router.get("", dependencies=[Depends(scoped_read)])
async def list_notifications(
    request: Request,
    after: int = Query(0, ge=0, le=2**53),
) -> dict[str, Any]:
    """Recorded notifications, newest first (bounded at 500), tenant-scoped.

    ``after=<id>`` polls incrementally — only items newer than that id are
    returned, while ``unread`` / ``latest_id`` always describe the whole ring.
    A tenant-bound principal sees only its own workspaces' notifications (plus legacy
    NULL-tenant ones); the offline / ``all_tenants`` path is byte-unchanged.
    """
    scope = singleton_read_filter(request)
    with _LOCK:
        _ensure_db_locked(create=False)  # restore the ring after a restart
        visible = [i for i in _NOTIFICATIONS if _item_in_scope(i, scope)]
        # workspace_id is an internal isolation column, NOT part of the public notification
        # contract — strip it so the API response stays byte-identical to pre-migration.
        items = [
            {k: v for k, v in i.items() if k != "workspace_id"}
            for i in visible
            if int(i["id"]) > after
        ]
        unread = sum(1 for i in visible if not i["read"])
        latest = max((int(i["id"]) for i in visible), default=0)
        forward = _FORWARD_TELEGRAM
    items.reverse()
    return {
        "items": items,
        "unread": unread,
        "latest_id": latest,
        "settings": {"forward_telegram": forward},
    }


@router.post("/read-all", dependencies=[_notify_write])
async def mark_all_read(request: Request) -> dict[str, Any]:
    """Mark every (in-scope) notification read (idempotent).

    A tenant-bound caller marks only its OWN workspaces' notifications read — it cannot
    flip another tenant's read state. NO-OP-scoped offline / ``all_tenants``.
    """
    scope = singleton_read_filter(request)
    with _LOCK:
        _ensure_db_locked(create=False)
        changed_ids: list[int] = []
        for item in _NOTIFICATIONS:
            if not item["read"] and _item_in_scope(item, scope):
                item["read"] = True
                changed_ids.append(int(item["id"]))
        # Scoped read-mark: when unrestricted, mark all (None); else only the in-scope ids.
        _persist_read_locked(None if scope is None else changed_ids)
    return {"ok": True, "marked": len(changed_ids)}


@router.post("/{notification_id}/read", dependencies=[_notify_write])
async def mark_read(request: Request, notification_id: int) -> dict[str, Any]:
    """Mark one notification read; 404 when the id is unknown (or out of the caller's tenant).

    A tenant-bound caller cannot mark (or even probe the existence of) a foreign tenant's
    notification — an out-of-scope id is a uniform 404, mirroring the by-id run reader.
    """
    scope = singleton_read_filter(request)
    with _LOCK:
        _ensure_db_locked(create=False)
        for item in _NOTIFICATIONS:
            if int(item["id"]) == notification_id and _item_in_scope(item, scope):
                item["read"] = True
                _persist_read_locked([notification_id])
                return {"ok": True}
    raise HTTPException(status_code=404, detail="unknown notification")


@router.post("/settings", dependencies=[_notify_write])
async def set_settings(settings: NotifySettings, request: Request) -> dict[str, Any]:
    """Persist the notify settings (currently just Telegram forwarding).

    The forward-Telegram flag is a single process-global (one shared ``notify_settings``
    row, no ``workspace_id`` key), so a tenant-bound principal flipping it would toggle
    notification forwarding for EVERY co-tenant. Restrict the write to an operator /
    ``all_tenants`` principal (the same posture gate as :func:`reveal_host_posture`): the
    offline / single-box default is ``all_tenants`` so it stays byte-unchanged, while a
    tenant-bound principal can no longer mutate this deployment-global toggle.
    """
    from himmy.api.auth import reveal_host_posture

    if not reveal_host_posture(request):
        raise HTTPException(
            status_code=403,
            detail="notify forwarding is a deployment-global setting (operator-only)",
        )
    global _FORWARD_TELEGRAM
    with _LOCK:
        conn = _ensure_db_locked(create=True)  # hydrate first, then overwrite
        _FORWARD_TELEGRAM = settings.forward_telegram
        value = "1" if settings.forward_telegram else "0"
        if _pg_enabled():
            store = _pg_store_locked()
            if store is not None:
                try:
                    store.set_setting("forward_telegram", value)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 - setting still applies in-memory
                    pass
        elif conn is not None:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO notify_settings (key, value) "
                    "VALUES ('forward_telegram', ?)",
                    (value,),
                )
                conn.commit()
            except Exception:  # noqa: BLE001 - setting still applies in-memory
                pass
    return {"settings": {"forward_telegram": settings.forward_telegram}}


def _bootstrap() -> None:
    """Seed the ring from a pre-existing DB at import (never creates one)."""
    try:
        with _LOCK:
            _ensure_db_locked(create=False)
    except Exception:  # noqa: BLE001 - startup hydration is best-effort
        pass


_bootstrap()

__all__ = [
    "RING_SIZE",
    "NotifySettings",
    "notify_db_path",
    "record_notification",
    "reset_notify_state",
    "router",
]
