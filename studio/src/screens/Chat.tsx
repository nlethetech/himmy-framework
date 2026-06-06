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
import { Markdown } from "../components/Markdown";
import {
  CognitionTrace,
  WorldLedger,
  GroundingPanel,
  SafetyPanel,
  OrchestrationGraph,
  type CogStep,
  type GraphMember,
} from "../components/Cognition";
import {
  UsageHud,
  emptyUsage,
  foldUsage,
  type LiveUsage,
} from "../components/Usage";
import { SendIcon, RefreshIcon, PlusIcon } from "../components/icons";

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

interface Msg {
  role: "user" | "agent";
  text: string;
  steps?: CogStep[];
  active?: string | null; // currently-active agent (for the team graph)
  team?: boolean;
  usage?: LiveUsage;
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
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow the composer textarea with its content (and reset when cleared).
  useEffect(() => {
    const ta = inputRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [input]);

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
  const teamMembers: GraphMember[] = isTeam
    ? (picked!.item as TeamSummary).members.map((m) => ({
        name: m.name,
        role: m.role,
      }))
    : [];
  const teamEntry = isTeam ? (picked!.item as TeamSummary).entry : "";

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
      {
        role: "agent",
        text: "",
        streaming: true,
        steps: [],
        team: isTeam,
        usage: emptyUsage(),
      },
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
    const addStep = (s: CogStep) =>
      patchLast((m) => ({ ...m, steps: [...(m.steps ?? []), s] }));

    const onEvent = (e: RunEvent) => {
      switch (e.type) {
        case "token":
          patchLast((m) => ({ ...m, text: m.text + e.delta }));
          break;
        case "agent":
          patchLast((m) => ({ ...m, active: e.name }));
          addStep({ kind: "agent", agent: e.name, model: e.model });
          break;
        case "reason":
          addStep({ kind: "reason", agent: e.agent, text: e.text });
          break;
        case "tool":
          addStep({
            kind: "tool",
            name: e.name,
            agent: e.agent,
            args: e.args ?? null,
            result: e.result ?? null,
            outcome: e.outcome ?? null,
            read_only: e.read_only ?? null,
            latency_ms: e.latency_ms ?? null,
          });
          break;
        case "delegate":
          patchLast((m) => ({ ...m, active: e.worker }));
          addStep({ kind: "delegate", worker: e.worker, task: e.task, agent: e.from });
          break;
        case "handoff":
          patchLast((m) => ({ ...m, active: e.to }));
          addStep({ kind: "handoff", to: e.to, agent: e.from });
          break;
        case "grounding":
          addStep({
            kind: "grounding",
            agent: e.agent,
            source: e.source,
            query: e.query,
            citations: e.citations,
          });
          break;
        case "safety":
          addStep({
            kind: "safety",
            agent: e.agent,
            stage: e.stage,
            blocked: e.blocked,
            redacted: e.redacted,
            flags: e.flags,
            reasons: e.reasons,
          });
          break;
        case "usage":
          patchLast((m) => ({
            ...m,
            usage: foldUsage(m.usage ?? emptyUsage(), e),
          }));
          break;
        case "approval_required":
          patchLast((m) => ({
            ...m,
            text:
              (m.text ? m.text + "\n\n" : "") +
              `⏸ Paused — needs your approval to run **${e.tools.join(", ")}**. ` +
              `Review it in the **Approvals** tab.`,
          }));
          break;
        case "paused":
          patchLast((m) => ({ ...m, streaming: false, active: null }));
          break;
        case "message":
          patchLast((m) => ({ ...m, text: e.text }));
          break;
        case "done":
          patchLast((m) => ({
            ...m,
            text: e.output_text || m.text,
            streaming: false,
            active: null,
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
                  {m.role === "agent" &&
                    m.team &&
                    teamMembers.length > 1 &&
                    (m.steps?.length ?? 0) > 0 && (
                      <OrchestrationGraph
                        members={teamMembers}
                        entry={teamEntry}
                        steps={m.steps ?? []}
                        active={m.streaming ? m.active : null}
                      />
                    )}
                  {m.steps && m.steps.length > 0 && (
                    <CognitionTrace steps={m.steps} />
                  )}
                  {m.role === "agent" && m.steps && (
                    <SafetyPanel steps={m.steps} />
                  )}
                  {m.role === "agent" && m.steps && (
                    <GroundingPanel steps={m.steps} />
                  )}
                  {m.role === "agent" && m.usage && (
                    <UsageHud usage={m.usage} live={m.streaming} />
                  )}
                  {m.text &&
                    (m.role === "agent" ? (
                      <Markdown>{m.text}</Markdown>
                    ) : (
                      m.text
                    ))}
                  {m.role === "agent" && !m.streaming && m.steps && (
                    <WorldLedger steps={m.steps} />
                  )}
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
                      <Link to={`/activity/${m.runId}`} className="dim" style={{ fontSize: 12 }}>
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
        <div className="composer">
          <textarea
            ref={inputRef}
            className="composer-input"
            rows={1}
            placeholder={
              !hasAny
                ? "Create an agent or add a team first…"
                : isTeam
                  ? "Ask the team to manage something…"
                  : "Message your agent…"
            }
            value={input}
            disabled={busy || !hasAny}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
          />
          <div className="composer-bar">
            <div className="composer-tools">
              <select
                className="composer-pick"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                title="Choose an agent or team"
              >
                {!hasAny && <option value="">No agents or teams</option>}
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
                        {a.name}
                        {a.has_tools ? " · tools" : ""}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>

              {!isTeam && (
                <select
                  className="composer-pick"
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  title="Inference provider"
                >
                  {PROVIDERS.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              )}

              <button
                className="composer-icon"
                onClick={load}
                title="Reload agents & teams"
                type="button"
              >
                <RefreshIcon />
              </button>
              {messages.length > 0 && (
                <button
                  className="composer-icon"
                  onClick={() => setMessages([])}
                  disabled={busy}
                  title="New chat"
                  type="button"
                >
                  <PlusIcon />
                </button>
              )}
            </div>

            <button
              className="composer-send"
              onClick={send}
              disabled={busy || !input.trim() || !hasAny}
              title="Send (Enter)"
              type="button"
            >
              {busy ? <span className="spinner" /> : <SendIcon />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
