import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  api,
  streamRun,
  streamTeamRun,
  getChat,
  saveChat,
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
  type CogStep,
} from "../components/Cognition";
import {
  UsageHud,
  emptyUsage,
  foldUsage,
  type LiveUsage,
} from "../components/Usage";
import { SendIcon, RefreshIcon, PlusIcon } from "../components/icons";
import { highlightAll } from "../lib/highlight";
import { PickMenu } from "../components/ui/PickMenu";

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
  const [sessionId, setSessionId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [params, setParams] = useSearchParams();
  // Search deep link (?hl=): highlight every hit once the session has rendered.
  const [pendingHl, setPendingHl] = useState<string | null>(null);
  const pendingHlRef = useRef<string | null>(null);
  pendingHlRef.current = pendingHl;

  // A Cookbook recipe (or any deep link) can prefill the agent + prompt via
  // ?agent=<path>&q=<prompt>. The Chats screen resumes a saved conversation via
  // ?session=<id>. Consume these once on mount.
  useEffect(() => {
    const a = params.get("agent");
    const q = params.get("q");
    const s = params.get("session");
    const hl = params.get("hl");
    if (hl) setPendingHl(hl);
    if (s) {
      getChat(s)
        .then((sess) => {
          setSessionId(sess.id);
          if (sess.agent_path) setPath(sess.agent_path);
          if (sess.provider) setProvider(sess.provider);
          setMessages(
            sess.messages.map((m) => ({ role: m.role, text: m.text })),
          );
        })
        .catch(() => undefined);
    }
    if (a) setPath(a);
    if (q) setInput(q);
    if (a || q || s || hl) setParams({}, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  const messagesRef = useRef<Msg[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
    if (pendingHlRef.current) return; // a search jump owns the scroll position
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  // Honor the ?hl= deep link: after the resumed session renders, wrap every
  // occurrence in <mark class="hl">, pulse the first one, and center it.
  useEffect(() => {
    if (!pendingHl || messages.length === 0) return;
    const root = scrollRef.current;
    if (!root) return;
    const raf = requestAnimationFrame(() => {
      const marks = highlightAll(root, pendingHl);
      if (marks.length > 0) {
        marks[0].classList.add("hl-first");
        marks[0].scrollIntoView({ block: "center" });
      }
      setPendingHl(null);
    });
    return () => cancelAnimationFrame(raf);
  }, [messages, pendingHl]);

  // Persist the current transcript so it shows up under Chats and can be resumed.
  const persist = async () => {
    const rows = messagesRef.current
      .filter((m) => m.text.trim())
      .map((m) => ({ role: m.role, text: m.text }));
    if (rows.length === 0) return;
    try {
      const saved = await saveChat({
        id: sessionId,
        agent_path: path || null,
        provider: provider || null,
        messages: rows,
      });
      if (!sessionId) setSessionId(saved.id);
    } catch {
      /* saving is best-effort; never block the chat */
    }
  };

  // Persist when a run finishes (busy true→false). Doing it here — rather than in
  // send()'s finally — guarantees the final reply has committed to messagesRef, so
  // the saved transcript always includes the assistant turn (not just the user's).
  const wasBusy = useRef(false);
  useEffect(() => {
    if (wasBusy.current && !busy) void persist();
    wasBusy.current = busy;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy]);

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
      // NOTE: do NOT persist() here — patchLast/setBusy are async, so messagesRef
      // hasn't yet flushed the final reply and we'd save a partial transcript
      // (e.g. the user turn without the assistant answer). The busy→false effect
      // below persists once state has committed and messagesRef is current.
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
                <div className="hero-title">Nothing to run yet</div>
                <div className="hero-sub">
                  Create an agent in the <b>Agents</b> tab, or drop a{" "}
                  <code>team.yaml</code> into this project.
                </div>
              </div>
            ) : (
              <div className="chat-hero">
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
                  {/* While running, the trace IS the feedback. Once the answer
                      lands it folds to one quiet line — the answer takes the room. */}
                  {m.streaming && m.steps && m.steps.length > 0 && (
                    <CognitionTrace steps={m.steps} />
                  )}
                  {m.role === "agent" &&
                    !m.streaming &&
                    m.steps &&
                    m.steps.length > 0 && (
                      <details className="trace-fold">
                        <summary>
                          <span className="tf-chev">›</span>
                          activity
                          <span className="tf-meta">
                            {m.steps.length} step
                            {m.steps.length === 1 ? "" : "s"}
                          </span>
                        </summary>
                        <CognitionTrace steps={m.steps} />
                        <SafetyPanel steps={m.steps} />
                        <GroundingPanel steps={m.steps} />
                        <WorldLedger steps={m.steps} />
                      </details>
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
              <PickMenu
                value={path}
                onChange={setPath}
                title="Choose an agent or team"
                placeholder={hasAny ? "Choose an agent or team" : "No agents or teams"}
                groups={[
                  ...(teams.length > 0
                    ? [
                        {
                          label: "Teams",
                          options: teams.map((t) => ({
                            value: t.path,
                            label: t.name,
                            meta: `team · ${t.members.length}`,
                          })),
                        },
                      ]
                    : []),
                  ...(agents.length > 0
                    ? [
                        {
                          label: "Agents",
                          options: agents.map((a) => ({
                            value: a.path,
                            label: a.name,
                            meta: a.has_tools ? "tools" : undefined,
                          })),
                        },
                      ]
                    : []),
                ]}
              />

              {!isTeam && (
                <PickMenu
                  value={provider}
                  onChange={setProvider}
                  title="Inference provider"
                  groups={[
                    {
                      label: "Provider",
                      options: PROVIDERS.map((p) => ({
                        value: p.value,
                        label: p.label,
                      })),
                    },
                  ]}
                />
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
                  onClick={() => {
                    setMessages([]);
                    setSessionId(null);
                  }}
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
