import { useEffect, useState } from "react";
import {
  googleStatus,
  gmailList,
  gmailSend,
  type GoogleStatus,
  type GmailMessage,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { useToast } from "../components/ui/Toast";
import { MailIcon, RefreshIcon, SendIcon, PlugIcon } from "../components/icons";
import { Modal } from "../components/ui/Modal";
import { Link } from "react-router-dom";

export default function Email() {
  const [st, setStatus] = useState<GoogleStatus | null>(null);
  const [msgs, setMsgs] = useState<GmailMessage[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [composing, setComposing] = useState(false);
  const toast = useToast();

  const load = () => {
    setLoading(true);
    googleStatus()
      .then((s) => {
        setStatus(s);
        if (s.connected) {
          return gmailList(25)
            .then(setMsgs)
            .catch((e) => {
              toast.show((e as Error).message, "err");
              setMsgs([]);
            });
        }
        setMsgs(null);
      })
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  return (
    <>
      <Topbar
        title="Email"
        sub={st?.email ? `Inbox · ${st.email}` : "Your Gmail inbox"}
        actions={
          st?.connected && (
            <>
              <button className="btn" onClick={load} disabled={loading}>
                <RefreshIcon /> Refresh
              </button>
              <button
                className="btn btn-primary"
                onClick={() => setComposing(true)}
              >
                <SendIcon /> Compose
              </button>
            </>
          )
        }
      />
      <Page>
        {!st?.connected ? (
          <EmptyState
            icon={<PlugIcon />}
            title="Connect Google to read your inbox"
            action={
              <Link className="btn btn-primary" to="/connections">
                Go to Connections
              </Link>
            }
          >
            Email is powered by your own Google account. Connect it once on the
            Connections page.
          </EmptyState>
        ) : msgs && msgs.length === 0 ? (
          <EmptyState icon={<MailIcon />} title="Inbox is empty" />
        ) : (
          <div className="mail-list">
            {(msgs ?? []).map((m) => (
              <div className="mail-row" key={m.id}>
                <div className="mail-from">{cleanFrom(m.sender)}</div>
                <div className="mail-main">
                  <span className="mail-subj">{m.subject}</span>
                  <span className="mail-snip">{m.snippet}</span>
                </div>
                <div className="mail-date mono">{shortDate(m.date)}</div>
              </div>
            ))}
          </div>
        )}
      </Page>
      {composing && (
        <ComposeEmail
          onClose={() => setComposing(false)}
          onSent={() => {
            setComposing(false);
            load();
          }}
        />
      )}
    </>
  );
}

function cleanFrom(from: string): string {
  const m = from.match(/^\s*"?([^"<]+?)"?\s*<.+>$/);
  return (m ? m[1] : from).trim() || from;
}
function shortDate(d: string): string {
  if (!d) return "";
  const parts = d.replace(/^[A-Za-z]{3},\s*/, "").split(" ");
  return parts.slice(0, 3).join(" ");
}

function ComposeEmail({
  onClose,
  onSent,
}: {
  onClose: () => void;
  onSent: () => void;
}) {
  const [to, setTo] = useState("");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [sending, setSending] = useState(false);
  const toast = useToast();

  const send = async () => {
    setSending(true);
    try {
      const r = await gmailSend(to.trim(), subject, body);
      if (r.ok) {
        toast.show(r.detail || "sent", "ok");
        onSent();
      } else {
        toast.show(r.detail || "send failed", "err");
      }
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setSending(false);
    }
  };

  return (
    <Modal
      title="New message"
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={send}
            disabled={sending || !to.trim()}
          >
            {sending ? <span className="spinner" /> : <SendIcon />} Send
          </button>
        </>
      }
    >
      <div className="compose-form">
        <input
          className="input"
          placeholder="To"
          value={to}
          onChange={(e) => setTo(e.target.value)}
        />
        <input
          className="input"
          placeholder="Subject"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
        />
        <textarea
          className="textarea"
          rows={10}
          placeholder="Write your message…"
          value={body}
          onChange={(e) => setBody(e.target.value)}
        />
      </div>
    </Modal>
  );
}
