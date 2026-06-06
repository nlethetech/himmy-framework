import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  streamRun,
  type AgentSummary,
  type RunEvent,
} from "../lib/api";
import { Topbar } from "../components/Page";
import { SendIcon, RefreshIcon } from "../components/icons";

const PROVIDERS = [
  { value: "", label: "Auto (spec default)" },
  { value: "stub", label: "Stub (offline)" },
  { value: "ollama", label: "Ollama (local)" },
  { value: "claude-cli", label: "Claude CLI" },
  { value: "pydantic-ai", label: "Cloud (key)" },
];

interface Msg {
  role: "user" | "agent";
  text: string;
  tools?: string[];
  streaming?: boolean;
  runId?: string;
}

export default function Chat() {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [agentPath, setAgentPath] = useState<string>("");
  const [provider, setProvider] = useState<string>("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadAgents = () => {
    api
      .get<AgentSummary[]>("/agents")
      .then((a) => {
        setAgents(a);
        setAgentPath((cur) => cur || a[0]?.path || "");
        setLoadErr(null);
      })
      .catch((e) => setLoadErr(String(e.message ?? e)));
  };
  useEffect(loadAgents, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const current = agents.find((a) => a.path === agentPath);

  const send = async () => {
    const prompt = input.trim();
    if (!prompt || !agentPath || busy) return;
    setInput("");

    // History = the committed turns so far (exclude the in-flight assistant msg).
    const history = messages.map((m) => ({
      role: m.role === "agent" ? ("assistant" as const) : ("user" as const),
      content: m.text,
    }));

    setMessages((m) => [
      ...m,
      { role: "user", text: prompt },
      { role: "agent", text: "", streaming: true, tools: [] },
    ]);
    setBusy(true);

    const ac = new AbortController();
    abortRef.current = ac;
    const patchLast = (fn: (m: Msg) => Msg) =>
      setMessages((ms) => {
        const copy = ms.slice();
        copy[copy.length - 1] = fn(copy[copy.length - 1]);
        return copy;
      });

    try {
      await streamRun(
        { agent_path: agentPath, prompt, provider: provider || null, history },
        (e: RunEvent) => {
          if (e.type === "token") {
            patchLast((m) => ({ ...m, text: m.text + e.delta }));
          } else if (e.type === "tool") {
            patchLast((m) => ({ ...m, tools: [...(m.tools ?? []), e.name] }));
          } else if (e.type === "message") {
            patchLast((m) => ({ ...m, text: e.text }));
          } else if (e.type === "done") {
            patchLast((m) => ({
              ...m,
              text: e.output_text || m.text,
              streaming: false,
              runId: e.run_id,
            }));
          } else if (e.type === "error") {
            patchLast((m) => ({
              ...m,
              text: (m.text ? m.text + "\n\n" : "") + "⚠ " + e.message,
              streaming: false,
            }));
          }
        },
        ac.signal,
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      patchLast((m) => ({
        ...m,
        text: (m.text ? m.text + "\n\n" : "") + "⚠ " + msg,
        streaming: false,
      }));
    } finally {
      patchLast((m) => ({ ...m, streaming: false }));
      setBusy(false);
      abortRef.current = null;
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="chat-wrap">
      <Topbar
        title="Chat"
        sub={current ? current.description || current.name : "talk to an agent"}
      />

      <div className="chat-bar">
        <label className="row gap6">
          <span className="dim" style={{ fontSize: 12 }}>
            Agent
          </span>
          <select
            className="select"
            style={{ width: "auto", minWidth: 180 }}
            value={agentPath}
            onChange={(e) => setAgentPath(e.target.value)}
          >
            {agents.length === 0 && <option value="">No agents found</option>}
            {agents.map((a) => (
              <option key={a.path} value={a.path}>
                {a.name} {a.has_tools ? "· tools" : ""} ({a.path})
              </option>
            ))}
          </select>
        </label>

        <label className="row gap6">
          <span className="dim" style={{ fontSize: 12 }}>
            Provider
          </span>
          <select
            className="select"
            style={{ width: "auto" }}
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
          >
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        <button className="btn" onClick={loadAgents} title="Reload agents">
          <RefreshIcon />
        </button>
        {messages.length > 0 && (
          <button
            className="btn"
            onClick={() => setMessages([])}
            disabled={busy}
            style={{ marginLeft: "auto" }}
          >
            New chat
          </button>
        )}
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {loadErr ? (
          <div className="chat-thread">
            <div className="banner" style={{ borderColor: "var(--err)" }}>
              <div
                className="b-ico"
                style={{ background: "var(--err-soft)", color: "var(--err)" }}
              >
                !
              </div>
              <div>
                <div className="b-title">Couldn't load agents</div>
                <div className="b-msg mono">{loadErr}</div>
              </div>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="chat-thread">
            <div className="empty">
              {agents.length === 0 ? (
                <>
                  No agents found in this project.
                  <br />
                  Create one in the <b>Agents</b> tab, or run{" "}
                  <code>himmy init my-agent</code>.
                </>
              ) : (
                <>
                  Chatting with <b>{current?.name}</b>. Say something below.
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map((m, i) => (
              <div className={"msg " + m.role} key={i}>
                <div className="avatar">{m.role === "user" ? "you" : "H"}</div>
                <div className="body">
                  <div className="who">{m.role === "user" ? "You" : current?.name ?? "Agent"}</div>
                  {m.tools && m.tools.length > 0 && (
                    <div className="row wrap gap6" style={{ marginBottom: 8 }}>
                      {m.tools.map((t, j) => (
                        <span className="tool-event" key={j}>
                          ⚙ {t}
                        </span>
                      ))}
                    </div>
                  )}
                  {m.text}
                  {m.streaming && <span className="caret" />}
                  {m.role === "agent" && m.runId && !m.streaming && (
                    <div className="mt8">
                      <Link
                        to={`/runs/${m.runId}`}
                        className="dim"
                        style={{ fontSize: 12 }}
                      >
                        ↗ view trace
                      </Link>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="chat-compose">
        <div className="compose-inner">
          <textarea
            className="textarea"
            placeholder={
              agents.length === 0
                ? "Create an agent first…"
                : "Message your agent…  (Enter to send, Shift+Enter for newline)"
            }
            value={input}
            disabled={busy || agents.length === 0}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
          />
          <button
            className="btn btn-primary"
            onClick={send}
            disabled={busy || !input.trim() || agents.length === 0}
          >
            {busy ? <span className="spinner" /> : <SendIcon />}
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
