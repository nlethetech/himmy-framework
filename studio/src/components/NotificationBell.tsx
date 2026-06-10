// The notification bell — a quiet glyph in the sidebar footer that opens a
// hairline ledger popover. Polls every 20s (and on focus); while the tab is
// hidden, NEW items also fire native browser notifications (permission is
// requested on the first popover open, never on load).

import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BellIcon } from "./icons";
import { relativeTime } from "../lib/format";
import {
  fetchNotifications,
  markAllNotificationsRead,
  markNotificationRead,
  setNotifySettings,
  type NotificationItem,
} from "../lib/notifyApi";
import "../styles/notify.css";

const POLL_MS = 20_000;

export default function NotificationBell() {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<NotificationItem[]>([]);
  const [unread, setUnread] = useState(0);
  const [forwardTg, setForwardTg] = useState(false);
  const [loaded, setLoaded] = useState(false); // first successful poll done
  const [failed, setFailed] = useState(false); // last poll errored
  const navigate = useNavigate();

  const rootRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const [anchor, setAnchor] = useState<{ left: number; bottom: number }>({
    left: 8,
    bottom: 48,
  });
  // Native notifications: only items newer than this id may ping the OS.
  // Seeded by the first poll so restored history never re-alerts.
  const alertedUpTo = useRef<number | null>(null);
  const askedPermission = useRef(false);

  const poll = useCallback(() => {
    fetchNotifications()
      .then((feed) => {
        setItems(feed.items);
        setUnread(feed.unread);
        setForwardTg(feed.settings.forward_telegram);
        setFailed(false);
        setLoaded(true);
        const seen = alertedUpTo.current;
        if (seen === null) {
          alertedUpTo.current = feed.latest_id;
          return;
        }
        if (feed.latest_id > seen) {
          alertedUpTo.current = feed.latest_id;
          if (
            document.hidden &&
            typeof Notification !== "undefined" &&
            Notification.permission === "granted"
          ) {
            for (const n of feed.items.filter((i) => i.id > seen).reverse()) {
              try {
                new Notification(n.title, {
                  body: n.body || undefined,
                  tag: `himmy-notify-${n.id}`,
                });
              } catch {
                /* notification construction can throw on some platforms */
              }
            }
          }
        }
      })
      .catch(() => setFailed(true));
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_MS);
    window.addEventListener("focus", poll);
    return () => {
      clearInterval(id);
      window.removeEventListener("focus", poll);
    };
  }, [poll]);

  // Close on outside click / Escape (the PickMenu pattern).
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const toggle = () => {
    if (!open) {
      const rect = btnRef.current?.getBoundingClientRect();
      if (rect) {
        setAnchor({
          left: Math.max(8, rect.left - 4),
          bottom: window.innerHeight - rect.top + 8,
        });
      }
      poll(); // fresh ledger the moment it opens
      if (
        !askedPermission.current &&
        typeof Notification !== "undefined" &&
        Notification.permission === "default"
      ) {
        askedPermission.current = true;
        Notification.requestPermission().catch(() => {});
      }
    }
    setOpen((o) => !o);
  };

  const onItemClick = (n: NotificationItem) => {
    if (!n.read) {
      setItems((xs) =>
        xs.map((x) => (x.id === n.id ? { ...x, read: true } : x)),
      );
      setUnread((u) => Math.max(0, u - 1));
      markNotificationRead(n.id).catch(() => {});
    }
    if (n.link) {
      setOpen(false);
      navigate(n.link);
    }
  };

  const onMarkAll = () => {
    setItems((xs) => xs.map((x) => ({ ...x, read: true })));
    setUnread(0);
    markAllNotificationsRead().catch(() => {});
  };

  const onToggleForward = () => {
    const next = !forwardTg;
    setForwardTg(next); // optimistic; the next poll reconciles
    setNotifySettings({ forward_telegram: next }).catch(() =>
      setForwardTg(!next),
    );
  };

  return (
    <div className="nb" ref={rootRef}>
      <button
        ref={btnRef}
        type="button"
        className={"rail-btn nb-btn" + (open ? " active" : "")}
        style={{ width: 24, height: 24 }}
        onClick={toggle}
        title="Notifications"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={
          unread > 0 ? `Notifications (${unread} unread)` : "Notifications"
        }
      >
        <BellIcon className="ico" />
        {unread > 0 && (
          <span className="sb-badge nb-count" aria-hidden>
            {unread > 99 ? "99+" : unread}
          </span>
        )}
      </button>

      {open && (
        <div
          className="nb-pop"
          role="dialog"
          aria-label="Notifications"
          style={{ left: anchor.left, bottom: anchor.bottom }}
        >
          <div className="nb-head">
            <span>Notifications</span>
            {unread > 0 && <span className="nb-head-n">{unread} unread</span>}
          </div>

          <div className="nb-list">
            {!loaded && !failed && <div className="nb-quiet">loading…</div>}
            {failed && (
              <div className="nb-quiet nb-err">couldn't reach the studio</div>
            )}
            {loaded && !failed && items.length === 0 && (
              <div className="nb-quiet">nothing yet — events land here</div>
            )}
            {items.map((n) => (
              <button
                type="button"
                key={n.id}
                className={"nb-row" + (n.read ? "" : " unread")}
                onClick={() => onItemClick(n)}
                title={n.body || n.title}
              >
                <span className="nb-row-title">{n.title}</span>
                <span className="nb-row-meta">
                  {relativeTime(n.created_at)}
                </span>
              </button>
            ))}
          </div>

          <div className="nb-foot">
            <button
              type="button"
              className="nb-check"
              role="checkbox"
              aria-checked={forwardTg}
              onClick={onToggleForward}
              title="Also send each notification to the configured Telegram bot"
            >
              <span className={"nb-box" + (forwardTg ? " on" : "")}>
                {forwardTg ? "✓" : ""}
              </span>
              forward to telegram
            </button>
            <button
              type="button"
              className="nb-act"
              onClick={onMarkAll}
              disabled={unread === 0}
            >
              mark all read
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
