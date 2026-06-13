import { useEffect, useMemo, useState } from "react";
import {
  api,
  getTelegramListener,
  startTelegramListener,
  stopTelegramListener,
  type AgentSummary,
  type ConnectionStatus,
  type TelegramListenerStatus,
} from "../lib/api";
import { PickMenu } from "./ui/PickMenu";
import { TelegramIcon } from "./icons";
import { useToast } from "./ui/Toast";

// The inbound Telegram long-poll listener control (T3g). Studio already configures
// the OUTBOUND Telegram connection; this card starts/stops the INBOUND bot — the loop
// that turns each message a user sends the bot into an agent reply (recorded as a run,
// guardrails applied). The bot token + chat allowlist live on the Telegram Connection
// card; this only picks which agent to run and toggles the loop on/off.
export function TelegramListenerCard({ telegram }: { telegram?: ConnectionStatus }) {
  const toast = useToast();
  const [status, setStatus] = useState<TelegramListenerStatus | null>(null);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [agentPath, setAgentPath] = useState("");
  const [allowEveryone, setAllowEveryone] = useState(false);
  const [busy, setBusy] = useState(false);

  const refresh = () => {
    getTelegramListener()
      .then(setStatus)
      .catch(() => setStatus(null));
  };

  useEffect(() => {
    refresh();
    api
      .get<AgentSummary[]>("/agents")
      .then((a) => {
        const usable = a.filter((x) => !x.error);
        setAgents(usable);
        setAgentPath((p) => p || usable[0]?.path || "");
      })
      .catch(() => setAgents([]));
  }, []);

  const agentGroups = useMemo(
    () => [
      {
        label: "agents",
        options: agents.map((a) => ({
          value: a.path,
          label: a.name,
          meta: a.path,
        })),
      },
    ],
    [agents],
  );

  const tokenConfigured = telegram?.configured ?? false;
  const active = status?.active ?? false;
  const effectiveAgent = active ? (status?.agent_path ?? agentPath) : agentPath;

  const start = async () => {
    if (!agentPath) {
      toast.show("pick an agent to run the bot as", "err");
      return;
    }
    setBusy(true);
    try {
      const next = await startTelegramListener({
        agent_path: agentPath,
        allow_everyone: allowEveryone,
      });
      setStatus(next);
      toast.show("Telegram bot is live", "ok");
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      setStatus(await stopTelegramListener());
      toast.show("Telegram bot stopped", "info");
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="conn-card">
      <div className="conn-head">
        <span
          className="conn-icon"
          style={{
            background:
              "color-mix(in srgb, var(--hue-telegram) 16%, transparent)",
            color: "var(--hue-telegram)",
          }}
        >
          <TelegramIcon />
        </span>
        <div className="conn-titles">
          <div className="conn-title">Telegram bot (inbound)</div>
          <div className="conn-desc">
            Run an agent as a live bot — it replies to messages and records each turn
            as a run.
          </div>
        </div>
      </div>

      <div className="conn-status">
        <span
          className="dot"
          style={{ color: active ? "var(--ok)" : "var(--text-dim)" }}
        />
        {active ? (
          <span className="conn-id">
            Listening as <strong>{effectiveAgent || "an agent"}</strong>
          </span>
        ) : tokenConfigured ? (
          <span className="conn-off">Stopped</span>
        ) : (
          <span className="conn-off">Connect Telegram above to enable</span>
        )}
      </div>

      {active ? (
        <>
          {status && (status.allowed_chat_ids.length > 0 || status.allow_everyone) && (
            <div className="conn-allowlist">
              <span className="conn-allowlist-label">Chats</span>
              <span className="conn-allowlist-value mono">
                {status.allowed_chat_ids.length > 0
                  ? status.allowed_chat_ids.join(", ")
                  : "open to everyone"}
              </span>
            </div>
          )}
          {status?.error && (
            <div className="conn-status">
              <span className="conn-off">{status.error}</span>
            </div>
          )}
          <div className="conn-toggle-row">
            <button className="btn" onClick={stop} disabled={busy}>
              {busy ? "Stopping…" : "Stop bot"}
            </button>
          </div>
        </>
      ) : tokenConfigured ? (
        <>
          <div className="conn-allowlist">
            <span className="conn-allowlist-label">Agent</span>
            <PickMenu
              value={agentPath}
              groups={agentGroups}
              searchable
              placeholder={agents.length ? "Pick an agent…" : "no agents found"}
              onChange={setAgentPath}
            />
          </div>
          <div className="conn-toggle-row">
            <label
              className="conn-toggle"
              title="Run an open bot with no chat allowlist (only for a private bot you control)"
            >
              <input
                type="checkbox"
                checked={allowEveryone}
                onChange={(e) => setAllowEveryone(e.target.checked)}
              />
              <span>Allow everyone (no chat allowlist)</span>
            </label>
          </div>
          <div className="conn-status">
            <span className="conn-off">
              By default only allow-listed chat ids (set on the Telegram connection)
              can drive the bot; starting with none is refused unless you tick the box.
            </span>
          </div>
          <div className="conn-toggle-row">
            <button className="btn" onClick={start} disabled={busy || !agentPath}>
              {busy ? "Starting…" : "Start bot"}
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}
