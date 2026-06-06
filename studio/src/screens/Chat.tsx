import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  streamRun,
  streamTeamRun,
  type AgentSummary,
  type TeamSummary,
  type RunEvent,
} from "../lib/api";
import { Topbar } from "../components/Page";
import { SendIcon, RefreshIcon } from "../components/icons";

// Domain-agnostic starter prompts for the empty state (fill the input, don't send).
const EXAMPLES = [
  "Give me a quick status summary",
  "What should I prioritize today?",
  "What can you help me with?",
];

const PROVIDERS = [
  { value: "", label: "Auto (spec default)" },
  { value: "stub", label: "Stub (offline)" },
  { value: "ollama", label: "Ollama (local)" },
  { value: "claude-cli", label: "Claude CLI" },
  { value: "pydantic-ai", label: "Cloud (key)" },
];

interface Step {
  label: string;
  kind: "tool" | "delegate" | "handoff";
}
interface Msg {
  role: "user" | "agent";
  text: string;
  steps?: Step[];
  streaming?: boolean;
  runId?: string;
}

type Pick =
  | { kind: "agent"; item: AgentSummary }
  | { kind: "team"; item: TeamSummary };

export default function Chat() {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [teams, setTeams] = useState<TeamSummary[]>([]);
  const [path, setPath] = useState<string>("");
  const [provider, setProvider] = useState<string>("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = () => {
    Promise.all([
      api.get<AgentSummary[]>("/agents"),
      api.get<TeamSummary[]>("/teams"),
    ])
      .then(([a, t]) => {
        setAgents(a);
        setTeams(t);
        setPath((cur) => cur || t[0]?.path || a[0]?.path || "");
        setLoadErr(null);
      })
      .catch((e) => setLoadErr(String(e.message ?? e)));
  };
  useEffect(load, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const picked: Pick | null = (() => {
    const team = teams.find((t) => t.path === path);
    if (team) return { kind: "team", item: team };
    const agent = agents.find((a) => a.path === path);
    if (agent) return { kind: "agent", item: agent };
    return null;
  })();
  const isTeam = picked?.kind === "team";
  const hasAny = agents.length + teams.length > 0;

  const send = async () => {
    const prompt = input.trim();
    if (!prompt || !path || busy) return;
    setInput("");

    const history = messages.map((m) => ({
      role: m.role === "agent" ? ("assistant" as const) : ("user" as const),
      content: m.text,
    }));

    setMessages((m) => [
      ...m,
      { role: "user", text: prompt },
      { role: "agent", text: "", streaming: true, steps: [] },
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
    const addStep = (s: Step) =>
      patchLast((m) => ({ ...m, steps: [...(m.steps ?? []), s] }));

    const onEvent = (e: RunEvent) => {
      switch (e.type) {
        case "token":
          patchLast((m) => ({ ...m, text: m.text + e.delta }));
          break;
        case "tool":
          addStep({ label: e.name, kind: "tool" });
          break;
        case "delegate":
          addStep({ label: `${e.worker}${e.task ? ` — ${e.task}` : ""}`, kind: "delegate" });
          break;
        case "handoff":
          addStep({ label: e.to, kind: "handoff" });
          break;
        case "message":
          patchLast((m) => ({ ...m, text: e.text }));
          break;
        case "done":
          patchLast((m) => ({
            ...m,
            text: e.output_text || m.text,
            streaming: false,
            runId: e.run_id,
          }));
          break;
        case "error":
          patchLast((m) => ({
            ...m,
            text: (m.text ? m.text + "\n\n" : "") + "⚠ " + e.message,
            streaming: false,
            runId: e.run_id,
          }));
          break;
      }
    };

    try {
      if (isTeam) {
        await streamTeamRun({ team_path: path, prompt }, onEvent, ac.signal);
      } else {
        await streamRun(
          { agent_path: path, prompt, provider: provider || null, history },
          onEvent,
          ac.signal,
        );
      }
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

  const sub = picked
    ? isTeam
      ? `team · ${(picked.item as TeamSummary).members.length} agents`
      : (picked.item as AgentSummary).description || picked.item.name
    : "talk to an agent or team";

  return (
    <div className="chat-wrap">
      <Topbar title="Chat" sub={sub} />

      <div className="chat-bar">
        <label className="row gap6">
          <span className="dim" style={{ fontSize: 12 }}>
            Run
          </span>
          <select
            className="select"
            style={{ width: "auto", minWidth: 220 }}
            value={path}
            onChange={(e) => setPath(e.target.value)}
          >
            {!hasAny && <option value="">No agents or teams found</option>}
            {teams.length > 0 && (
              <optgroup label="Teams (manager + workers)">
                {teams.map((t) => (
                  <option key={t.path} value={t.path}>
                    {t.name} · team ({t.members.length})
                  </option>
                ))}
              </optgroup>
            )}
            {agents.length > 0 && (
              <optgroup label="Agents">
                {agents.map((a) => (
                  <option key={a.path} value={a.path}>
                    {a.name} {a.has_tools ? "· tools" : ""}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
        </label>

        {!isTeam && (
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
        )}
        {isTeam && (
          <span className="pill dim" title="members run on their own providers">
            {(picked!.item as TeamSummary).members
              .map((m) => `${m.name}:${m.provider ?? "auto"}`)
              .join("  ·  ")}
          </span>
        )}

        <button className="btn" onClick={load} title="Reload">
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
          <div className="chat-thread grow">
            <div className="banner" style={{ borderColor: "var(--err)" }}>
              <div
                className="b-ico"
                style={{ background: "var(--err-soft)", color: "var(--err)" }}
              >
                !
              </div>
              <div>
                <div className="b-title">Couldn't load</div>
                <div className="b-msg mono">{loadErr}</div>
              </div>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="chat-thread grow">
            {!hasAny ? (
              <div className="chat-hero">
                <div className="hero-avatar">∅</div>
                <div className="hero-title">Nothing to run yet</div>
                <div className="hero-sub">
                  Create an agent in the <b>Agents</b> tab, or drop a{" "}
                  <code>team.yaml</code> into this project.
                </div>
              </div>
            ) : (
              <div className="chat-hero">
                <div className="hero-avatar">
                  {(picked?.item.name ?? "H").charAt(0).toUpperCase()}
                </div>
                <div className="hero-title">{picked?.item.name}</div>
                <div className="hero-sub">
                  {isTeam
                    ? `A manager that delegates to ${
                        (picked!.item as TeamSummary).members.length - 1
                      } specialists. Ask it to look something up or take an action.`
                    : (picked?.item as AgentSummary)?.description ||
                      "Send a message to get started."}
                </div>
                <div className="hero-prompts">
                  {EXAMPLES.map((ex) => (
                    <button
                      key={ex}
                      className="hero-chip"
                      onClick={() => setInput(ex)}
                    >
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map((m, i) => (
              <div className={"msg " + m.role} key={i}>
                <div className="avatar">{m.role === "user" ? "you" : "H"}</div>
                <div className="body">
                  <div className="who">
                    {m.role === "user" ? "You" : picked?.item.name ?? "Agent"}
                  </div>
                  {m.steps && m.steps.length > 0 && (
                    <div className="row wrap gap6" style={{ marginBottom: 8 }}>
                      {m.steps.map((s, j) => (
                        <span className="tool-event" key={j}>
                          {s.kind === "delegate"
                            ? "→ "
                            : s.kind === "handoff"
                              ? "⇒ "
                              : "⚙ "}
                          {s.label}
                        </span>
                      ))}
                    </div>
                  )}
                  {m.text}
                  {m.streaming &&
                    (m.text ? (
                      <span className="caret" />
                    ) : (
                      <span className="thinking" aria-label="working">
                        <span />
                        <span />
                        <span />
                      </span>
                    ))}
                  {m.role === "agent" && m.runId && !m.streaming && (
                    <div className="mt8">
                      <Link to={`/runs/${m.runId}`} className="dim" style={{ fontSize: 12 }}>
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
              !hasAny
                ? "Create an agent or add a team first…"
                : isTeam
                  ? "Ask the team to manage something…  (Enter to send)"
                  : "Message your agent…  (Enter to send, Shift+Enter for newline)"
            }
            value={input}
            disabled={busy || !hasAny}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
          />
          <button
            className="btn btn-primary"
            onClick={send}
            disabled={busy || !input.trim() || !hasAny}
          >
            {busy ? <span className="spinner" /> : <SendIcon />}
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
