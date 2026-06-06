import { useState } from "react";
import { Link } from "react-router-dom";
import { sendViaConnection, type ConnectionStatus } from "../lib/api";
import { Modal } from "./ui/Modal";
import { useToast } from "./ui/Toast";

export type ComposeMode = "email" | "telegram";

export function ComposeModal({
  mode,
  connection,
  onClose,
}: {
  mode: ComposeMode;
  connection: ConnectionStatus | undefined;
  onClose: () => void;
}) {
  const [fields, setFields] = useState<Record<string, string>>({});
  const [sending, setSending] = useState(false);
  const toast = useToast();
  const connected = connection?.configured;
  const title = mode === "email" ? "Send email" : "Message Telegram";

  const set = (k: string, v: string) => setFields((f) => ({ ...f, [k]: v }));

  const send = async () => {
    setSending(true);
    try {
      const payload =
        mode === "email"
          ? { to: fields.to, subject: fields.subject, body: fields.body }
          : { chat_id: fields.chat_id, text: fields.text };
      const r = await sendViaConnection(mode, payload);
      if (r.ok) {
        toast.show(`${title}: ${r.detail}`, "ok");
        onClose();
      } else {
        toast.show(`Couldn't send: ${r.detail}`, "err");
      }
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setSending(false);
    }
  };

  return (
    <Modal
      title={title}
      onClose={onClose}
      width={520}
      footer={
        connected ? (
          <>
            <button className="btn" onClick={onClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={send} disabled={sending}>
              {sending ? <span className="spinner" /> : "Send"}
            </button>
          </>
        ) : (
          <Link className="btn btn-primary" to="/connections" onClick={onClose}>
            Connect {mode === "email" ? "Email" : "Telegram"}
          </Link>
        )
      }
    >
      {!connected ? (
        <div className="compose-need">
          Connect your {mode === "email" ? "email account" : "Telegram bot"} first,
          then come back here to send.
        </div>
      ) : mode === "email" ? (
        <div className="form-grid">
          <label className="field">
            <span className="field-label">To</span>
            <input
              className="input"
              placeholder="someone@example.com"
              value={fields.to ?? ""}
              onChange={(e) => set("to", e.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Subject</span>
            <input
              className="input"
              value={fields.subject ?? ""}
              onChange={(e) => set("subject", e.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Body</span>
            <textarea
              className="input"
              rows={6}
              value={fields.body ?? ""}
              onChange={(e) => set("body", e.target.value)}
            />
          </label>
        </div>
      ) : (
        <div className="form-grid">
          <label className="field">
            <span className="field-label">
              Chat id
              <span className="field-opt">defaults to your saved chat</span>
            </span>
            <input
              className="input"
              placeholder="987654321"
              value={fields.chat_id ?? ""}
              onChange={(e) => set("chat_id", e.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Message</span>
            <textarea
              className="input"
              rows={5}
              value={fields.text ?? ""}
              onChange={(e) => set("text", e.target.value)}
            />
          </label>
        </div>
      )}
    </Modal>
  );
}
