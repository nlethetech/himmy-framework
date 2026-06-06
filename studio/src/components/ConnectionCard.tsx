import { useState } from "react";
import {
  setConnection,
  testConnection,
  deleteConnection,
  type ConnectionStatus,
} from "../lib/api";
import { Modal } from "./ui/Modal";
import { MailIcon, TelegramIcon, GlobeIcon, PlugIcon } from "./icons";
import { useToast } from "./ui/Toast";

const ICONS: Record<string, (p: { className?: string }) => JSX.Element> = {
  email: MailIcon,
  telegram: TelegramIcon,
  web_search: GlobeIcon,
};

export function ConnectionCard({
  conn,
  onChange,
}: {
  conn: ConnectionStatus;
  onChange: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testOk, setTestOk] = useState<boolean | null>(null);
  const toast = useToast();
  const Icon = ICONS[conn.type] ?? PlugIcon;

  const runTest = async () => {
    setTesting(true);
    setTestOk(null);
    try {
      const r = await testConnection(conn.type);
      setTestOk(r.ok);
      toast.show(
        `${conn.title}: ${r.detail}`,
        r.ok ? "ok" : "err",
      );
    } catch (e) {
      setTestOk(false);
      toast.show(`${conn.title} test failed: ${(e as Error).message}`, "err");
    } finally {
      setTesting(false);
    }
  };

  const disconnect = async () => {
    try {
      await deleteConnection(conn.type);
      toast.show(`${conn.title} disconnected`, "info");
      onChange();
    } catch (e) {
      toast.show((e as Error).message, "err");
    }
  };

  return (
    <div className="conn-card">
      <div className="conn-head">
        <span
          className="conn-icon"
          style={{
            background: `color-mix(in srgb, var(--hue-${conn.hue}) 16%, transparent)`,
            color: `var(--hue-${conn.hue})`,
          }}
        >
          <Icon />
        </span>
        <div className="conn-titles">
          <div className="conn-title">{conn.title}</div>
          <div className="conn-desc">{conn.description}</div>
        </div>
      </div>

      <div className="conn-status">
        <span
          className="dot"
          style={{
            color:
              testOk === false
                ? "var(--err)"
                : conn.configured
                  ? "var(--ok)"
                  : "var(--text-dim)",
          }}
        />
        {conn.configured ? (
          <span className="conn-id mono">
            {conn.fields.find((f) => f.masked_hint)?.masked_hint ?? "Connected"}
          </span>
        ) : (
          <span className="conn-off">Not connected</span>
        )}
      </div>

      <div className="conn-actions">
        {conn.configured ? (
          <>
            <button className="btn" onClick={runTest} disabled={testing}>
              {testing ? <span className="spinner" /> : "Test"}
            </button>
            <button className="btn" onClick={() => setEditing(true)}>
              Edit
            </button>
            <button className="btn ghost" onClick={disconnect}>
              Disconnect
            </button>
          </>
        ) : (
          <button
            className="btn btn-primary"
            onClick={() => setEditing(true)}
            disabled={!conn.writable}
            title={conn.writable ? "" : "Active secrets backend is read-only"}
          >
            Connect
          </button>
        )}
      </div>

      {editing && (
        <ConnectModal
          conn={conn}
          onClose={() => setEditing(false)}
          onSaved={() => {
            setEditing(false);
            onChange();
          }}
        />
      )}
    </div>
  );
}

function ConnectModal({
  conn,
  onClose,
  onSaved,
}: {
  conn: ConnectionStatus;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const toast = useToast();

  const save = async () => {
    setSaving(true);
    try {
      await setConnection(conn.type, values);
      toast.show(`${conn.title} connected`, "ok");
      onSaved();
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      title={`Connect ${conn.title}`}
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saving ? <span className="spinner" /> : "Save"}
          </button>
        </>
      }
    >
      <div className="form-grid">
        {conn.fields.map((f) => (
          <label className="field" key={f.name}>
            <span className="field-label">
              {f.label}
              {!f.required && <span className="field-opt">optional</span>}
            </span>
            {f.kind === "enum" ? (
              <select
                className="input"
                value={values[f.name] ?? ""}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [f.name]: e.target.value }))
                }
              >
                <option value="">default ({f.options[0]})</option>
                {f.options.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            ) : f.kind === "bool" ? (
              <select
                className="input"
                value={values[f.name] ?? ""}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [f.name]: e.target.value }))
                }
              >
                <option value="">default</option>
                <option value="true">on</option>
                <option value="false">off</option>
              </select>
            ) : (
              <input
                className="input"
                type={f.kind === "password" ? "password" : "text"}
                placeholder={
                  f.present ? "•••• (leave blank to keep)" : f.placeholder
                }
                value={values[f.name] ?? ""}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [f.name]: e.target.value }))
                }
              />
            )}
          </label>
        ))}
      </div>
    </Modal>
  );
}
