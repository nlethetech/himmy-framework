import { useEffect, useState } from "react";
import {
  googleStatus,
  googleSetClient,
  googleAuthUrl,
  googleDisconnect,
  googleForgetClient,
  type GoogleStatus,
} from "../lib/api";
import { MailIcon, CalendarIcon, PlugIcon } from "./icons";
import { useToast } from "./ui/Toast";

/**
 * Connect a Google account (Gmail + Calendar) via the OAuth2 loopback flow.
 *
 * The user supplies their own Google Cloud "Web application" OAuth client
 * (client_id + secret, redirect URI = this server's /api/studio/google/callback),
 * clicks Connect → a popup walks Google's consent → the callback page posts back
 * `himmy-google-connected`, and we re-read status. Tokens live in the keychain.
 */
export function GoogleCard({ onChange }: { onChange?: () => void }) {
  const [st, setStatus] = useState<GoogleStatus | null>(null);
  const [editing, setEditing] = useState(false);
  const [cid, setCid] = useState("");
  const [csecret, setCsecret] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const load = () =>
    googleStatus()
      .then((s) => {
        setStatus(s);
        if (!s.configured) setEditing(true);
      })
      .catch(() => setStatus(null));
  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.data === "himmy-google-connected") {
        load();
        onChange?.();
        toast.show("Google account connected", "ok");
      }
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onChange]);

  const saveClient = async () => {
    setBusy(true);
    try {
      const s = await googleSetClient(cid.trim(), csecret.trim());
      setStatus(s);
      setEditing(false);
      setCsecret("");
      toast.show("Google client saved", "ok");
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setBusy(false);
    }
  };

  const connect = async () => {
    setBusy(true);
    try {
      const { url } = await googleAuthUrl();
      window.open(url, "himmy-google", "width=520,height=680");
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    setBusy(true);
    try {
      setStatus(await googleDisconnect());
      toast.show("Google disconnected", "info");
      onChange?.();
    } finally {
      setBusy(false);
    }
  };

  const forget = async () => {
    setBusy(true);
    try {
      setStatus(await googleForgetClient());
      setEditing(true);
      onChange?.();
    } finally {
      setBusy(false);
    }
  };

  const redirect = `${window.location.origin}/api/studio/google/callback`;

  return (
    <div className="conn-card">
      <div className="conn-head">
        <span className="conn-icon" style={{ background: "#2a1f1d", color: "#ea4335" }}>
          G
        </span>
        <div className="conn-titles">
          <div className="conn-title">Google</div>
          <div className="conn-desc">Gmail &amp; Calendar for any account</div>
        </div>
      </div>

      <div className="conn-status">
        <span
          className="dot"
          style={{ color: st?.connected ? "var(--ok)" : "var(--text-dim)" }}
        />
        {st?.connected ? (
          <span className="conn-id mono">{st.email ?? "Connected"}</span>
        ) : (
          <span className="conn-off">Not connected</span>
        )}
      </div>

      {st?.connected ? (
        <>
          <div className="google-caps">
            <span className="cap-pill">
              <MailIcon /> Gmail
            </span>
            <span className="cap-pill">
              <CalendarIcon /> Calendar
            </span>
          </div>
          <div className="conn-actions">
            <button className="btn" onClick={disconnect} disabled={busy}>
              Disconnect
            </button>
            <button className="btn ghost" onClick={forget} disabled={busy}>
              Remove client
            </button>
          </div>
        </>
      ) : editing || !st?.configured ? (
        <div className="google-setup">
          {st && !st.writable && (
            <div className="google-hint err">
              Secrets backend is read-only — set <code>HIMMY_SECRETS=keychain</code>.
            </div>
          )}
          <div className="google-hint">
            Create a Google Cloud OAuth client (Web application) with this redirect URI:
            <code className="google-redirect">{redirect}</code>
          </div>
          <label className="field">
            <span className="field-label">Client ID</span>
            <input
              className="input"
              value={cid}
              onChange={(e) => setCid(e.target.value)}
              placeholder="…apps.googleusercontent.com"
            />
          </label>
          <label className="field">
            <span className="field-label">Client secret</span>
            <input
              className="input"
              type="password"
              value={csecret}
              onChange={(e) => setCsecret(e.target.value)}
              placeholder="GOCSPX-…"
            />
          </label>
          <div className="conn-actions">
            <button
              className="btn btn-primary"
              onClick={saveClient}
              disabled={busy || !cid.trim() || !csecret.trim() || !st?.writable}
            >
              Save client
            </button>
            {st?.configured && (
              <button className="btn" onClick={() => setEditing(false)}>
                Cancel
              </button>
            )}
          </div>
        </div>
      ) : (
        <>
          <div className="google-hint">
            Client configured. Connect your account to enable Gmail and Calendar.
          </div>
          <div className="conn-actions">
            <button className="btn btn-primary" onClick={connect} disabled={busy}>
              <PlugIcon /> Connect Google
            </button>
            <button className="btn ghost" onClick={() => setEditing(true)}>
              Edit client
            </button>
          </div>
        </>
      )}
    </div>
  );
}
