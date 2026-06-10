import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  api,
  streamTeamRun,
  getChat,
  saveChat,
  type AgentSummary,
  type TeamSummary,
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
import { PickMenu, type PickGroup } from "../components/ui/PickMenu";
import {
  fetchOpenRouterCatalog,
  fmtModelPrice,
  fmtCtx,
  type OpenRouterModel,
} from "../lib/modelsApi";
import { ApprovalCard, type ApprovalState } from "../components/ApprovalCard";
import { ArtifactCard } from "../components/ArtifactCard";
import {
  artifactsFromSteps,
  resolveApprovalStream,
} from "../lib/artifactsApi";
import { PlanCard } from "../components/PlanCard";
import {
  interruptMission,
  startMission,
  steerMission,
  streamMission,
  type MissionEvent,
  type PlanStep,
} from "../lib/missionsApi";
import "../styles/chatsurfaces.css";
import "../styles/missions.css";

// Quiet inline glyph for the "run in background" composer action (no icon in
// the shared set carries this meaning; mono-weight strokes match the others).
const BgIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v10" />
    <path d="m8 9 4 4 4-4" />
    <path d="M4 17h16v4H4z" />
  </svg>
);

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
  { value: "openrouter", label: "OpenRouter" },
  { value: "anthropic", label: "Anthropic API" },
  { value: "openai", label: "OpenAI API" },
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
  approvals?: ApprovalState[]; // inline HITL pauses raised during this turn
  missionId?: string; // single-agent runs flow through the missions registry
  plan?: PlanStep[]; // plan-first checklist (updates in place per plan frame)
  steers?: string[]; // guidance queued mid-run (rendered as quiet rows)
  backgrounded?: boolean; // a mission chip entry, not an inline stream
  agentLabel?: string; // chip label (the agent that runs in the background)
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
  // Plan-first: ask the agent to publish + maintain a step checklist.
  const [planMode, setPlanMode] = useState<boolean>(
    () => localStorage.getItem("himmy.chat.planmode") === "1",
  );
  const togglePlanMode = () => {
    setPlanMode((v) => {
      localStorage.setItem("himmy.chat.planmode", v ? "0" : "1");
      return !v;
    });
  };
  // The mission backing the CURRENT foreground stream (single-agent runs all
  // flow through the missions registry — one code path for fg/bg/steering).
  const [activeMission, setActiveMission] = useState<string | null>(null);
  const activeMissionRef = useRef<string | null>(null);
  // A ?project=<id> deep link (from Projects → new chat) assigns saved
  // transcripts to that project. Held in a ref so the persist closure always
  // sees the latest value without re-subscribing the busy→idle effect.
  const projectIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [params, setParams] = useSearchParams();
  // OpenRouter model picker: catalog is fetched lazily the first time the
  // provider is selected; the chosen model persists across sessions.
  const [orModel, setOrModel] = useState<string>(
    () => localStorage.getItem("himmy.chat.openrouter.model") ?? "",
  );
  const [orCatalog, setOrCatalog] = useState<OpenRouterModel[] | null>(null);
  const [orError, setOrError] = useState<string | null>(null);
  useEffect(() => {
    if (provider !== "openrouter" || orCatalog !== null) return;
    fetchOpenRouterCatalog()
      .then((r) => {
        setOrCatalog(r.models);
        setOrError(null);
      })
      .catch((e) => setOrError(String(e.message ?? e)));
  }, [provider, orCatalog]);
  const pickOrModel = (v: string) => {
    setOrModel(v);
    localStorage.setItem("himmy.chat.openrouter.model", v);
  };
  const orGroups: PickGroup[] = (() => {
    const def = {
      value: "",
      label: "Default (Mistral Small)",
      meta: "openrouter default",
    };
    if (orError) {
      return [
        { options: [def] },
        { label: "catalog unavailable", options: [] },
      ];
    }
    if (orCatalog === null) {
      return [{ options: [def, { value: "__loading", label: "loading catalog…" }] }];
    }
    const toOpt = (m: OpenRouterModel) => ({
      value: m.id,
      label: m.name,
      meta: [fmtModelPrice(m), fmtCtx(m)].filter(Boolean).join(" · "),
    });
    return [
      { options: [def] },
      { label: "Free", options: orCatalog.filter((m) => m.free).map(toOpt) },
      { label: "Paid", options: orCatalog.filter((m) => !m.free).map(toOpt) },
    ];
  })();

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
    const project = params.get("project");
    if (project) projectIdRef.current = project;
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
    if (a || q || s || hl || project) setParams({}, { replace: true });
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
        project_id: projectIdRef.current,
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

  // Patch one message by index — resumes can land on a message that is no
  // longer the last one (the user may have sent more turns while paused).
  const patchAt = (idx: number, fn: (m: Msg) => Msg) =>
    setMessages((ms) => {
      if (!ms[idx]) return ms;
      const copy = ms.slice();
      copy[idx] = fn(copy[idx]);
      return copy;
    });

  const setApprovalAt = (
    idx: number,
    checkpointId: string,
    patch: Partial<ApprovalState>,
  ) =>
    patchAt(idx, (m) => ({
      ...m,
      approvals: (m.approvals ?? []).map((a) =>
        a.checkpointId === checkpointId ? { ...a, ...patch } : a,
      ),
    }));

  // One SSE-frame applier for both the initial run and an approval resume.
  // A resume's message/done frames carry only the CONTINUATION text, so they
  // are appended to the text captured when the pause happened, never replace it.
  const makeEventHandler = (idx: number, resume?: { base: string }) => {
    let failed = false;
    let firstToken = true;
    const sep = resume && resume.base ? "\n\n" : "";
    const addStep = (s: CogStep) =>
      patchAt(idx, (m) => ({ ...m, steps: [...(m.steps ?? []), s] }));

    const handler = (e: MissionEvent) => {
      switch (e.type) {
        case "plan":
          // Plan-first checklist — replaced in place as new frames arrive.
          patchAt(idx, (m) => ({ ...m, plan: e.steps }));
          break;
        case "steer":
          // Guidance queued mid-run, surfaced as a quiet row in the entry.
          patchAt(idx, (m) => ({ ...m, steers: [...(m.steers ?? []), e.text] }));
          break;
        case "token": {
          const lead = resume && firstToken ? sep : "";
          firstToken = false;
          patchAt(idx, (m) => ({ ...m, text: m.text + lead + e.delta }));
          break;
        }
        case "agent":
          patchAt(idx, (m) => ({ ...m, active: e.name }));
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
          patchAt(idx, (m) => ({ ...m, active: e.worker }));
          addStep({ kind: "delegate", worker: e.worker, task: e.task, agent: e.from });
          break;
        case "handoff":
          patchAt(idx, (m) => ({ ...m, active: e.to }));
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
          patchAt(idx, (m) => ({
            ...m,
            usage: foldUsage(m.usage ?? emptyUsage(), e),
          }));
          break;
        case "approval_required":
          // Inline card in this ledger entry (a resume can pause again).
          patchAt(idx, (m) =>
            (m.approvals ?? []).some((a) => a.checkpointId === e.checkpoint_id)
              ? m
              : {
                  ...m,
                  approvals: [
                    ...(m.approvals ?? []),
                    {
                      checkpointId: e.checkpoint_id,
                      tools: e.tools,
                      status: "pending",
                    },
                  ],
                },
          );
          break;
        case "paused":
          patchAt(idx, (m) => ({ ...m, streaming: false, active: null }));
          break;
        case "message":
          patchAt(idx, (m) => ({
            ...m,
            text: resume ? resume.base + sep + e.text : e.text,
          }));
          break;
        case "done":
          patchAt(idx, (m) => ({
            ...m,
            text: resume
              ? e.output_text
                ? resume.base + sep + e.output_text
                : m.text
              : e.output_text || m.text,
            streaming: false,
            active: null,
            runId: e.run_id,
          }));
          break;
        case "error":
          failed = true;
          patchAt(idx, (m) => ({
            ...m,
            text: (m.text ? m.text + "\n\n" : "") + "⚠ " + e.message,
            streaming: false,
            runId: e.run_id ?? m.runId,
          }));
          break;
      }
    };
    return { handler, failed: () => failed };
  };

  // Approve/Deny from an inline card: resolve the checkpoint and render the
  // resumed run's stream back into the same ledger entry.
  const resolveInline = async (
    idx: number,
    checkpointId: string,
    approve: boolean,
  ) => {
    if (busy) return;
    setApprovalAt(idx, checkpointId, { status: "working" });
    setBusy(true);
    const base = messagesRef.current[idx]?.text ?? "";
    patchAt(idx, (m) => ({ ...m, streaming: true }));
    const { handler, failed } = makeEventHandler(idx, { base });
    try {
      await resolveApprovalStream(checkpointId, approve, handler);
      setApprovalAt(
        idx,
        checkpointId,
        failed()
          ? { status: "error", note: "the resumed run reported an error" }
          : { status: approve ? "approved" : "denied" },
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setApprovalAt(idx, checkpointId, { status: "error", note: msg });
    } finally {
      patchAt(idx, (m) => ({ ...m, streaming: false }));
      setBusy(false);
    }
  };

  const chatModel = () =>
    provider === "openrouter" && orModel && orModel !== "__loading"
      ? orModel
      : null;

  const historyOf = () =>
    messages.map((m) => ({
      role: m.role === "agent" ? ("assistant" as const) : ("user" as const),
      content: m.text,
    }));

  // While a foreground mission streams, the composer is a STEERING input:
  // submitting queues the text, which the loop injects before its next turn.
  const queueSteer = async (text: string) => {
    const mid = activeMissionRef.current;
    if (!mid) return;
    setInput("");
    try {
      await steerMission(mid, text.slice(0, 4000));
      // the `steer` frame arrives over the live stream and renders the row
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      const idx = messagesRef.current.length - 1;
      patchAt(idx, (m) => ({
        ...m,
        steers: [...(m.steers ?? []), `(not delivered — ${msg})`],
      }));
    }
  };

  const send = async () => {
    const prompt = input.trim();
    if (!prompt || !path) return;
    if (busy) {
      // Streaming → the same box steers (single-agent missions only).
      if (activeMissionRef.current) void queueSteer(prompt);
      return;
    }
    setInput("");
    const history = historyOf();

    // The agent reply lands right after the user turn we are about to append.
    const agentIdx = messages.length + 1;
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
    const { handler: onEvent } = makeEventHandler(agentIdx);
    const patchLast = (fn: (m: Msg) => Msg) => patchAt(agentIdx, fn);

    try {
      if (isTeam) {
        await streamTeamRun({ team_path: path, prompt }, onEvent, ac.signal);
      } else {
        // Single-agent runs flow through the missions registry — ONE code path
        // for foreground/background, steering, interrupt, and reconnects.
        const { mission_id } = await startMission({
          agent_path: path,
          prompt,
          provider: provider || null,
          model: chatModel(),
          history,
          plan_mode: planMode,
        });
        activeMissionRef.current = mission_id;
        setActiveMission(mission_id);
        patchLast((m) => ({ ...m, missionId: mission_id }));
        await streamMission(mission_id, onEvent, ac.signal);
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
      activeMissionRef.current = null;
      setActiveMission(null);
      // NOTE: do NOT persist() here — patchLast/setBusy are async, so messagesRef
      // hasn't yet flushed the final reply and we'd save a partial transcript
      // (e.g. the user turn without the assistant answer). The busy→false effect
      // below persists once state has committed and messagesRef is current.
    }
  };

  // Quiet background action: run the CURRENT prompt as a mission and drop a
  // chip entry in the thread instead of streaming inline.
  const sendBackground = async () => {
    const prompt = input.trim();
    if (!prompt || !path || busy || isTeam) return;
    setInput("");
    const history = historyOf();
    const label = picked?.item.name ?? "agent";
    const chipIdx = messages.length + 1;
    setMessages((m) => [
      ...m,
      { role: "user", text: prompt },
      { role: "agent", text: "", backgrounded: true, agentLabel: label },
    ]);
    try {
      const { mission_id } = await startMission({
        agent_path: path,
        prompt,
        provider: provider || null,
        model: chatModel(),
        history,
        plan_mode: planMode,
      });
      patchAt(chipIdx, (m) => ({ ...m, missionId: mission_id }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      patchAt(chipIdx, (m) => ({
        ...m,
        backgrounded: false,
        text: "⚠ " + msg,
      }));
    }
  };

  // Stop replaces send while streaming. Missions are interrupted server-side
  // (checkpoint-pause when one exists, else an honest cooperative cancel);
  // team runs just stop the stream.
  const stop = async () => {
    const mid = activeMissionRef.current;
    if (!mid) {
      abortRef.current?.abort();
      return;
    }
    try {
      const r = await interruptMission(mid);
      if (r.mode === "checkpoint") {
        const idx = messagesRef.current.length - 1;
        patchAt(idx, (m) => ({
          ...m,
          text:
            (m.text ? m.text + "\n\n" : "") +
            "paused — resume from Approvals/Missions",
        }));
      }
      // mode "cancelled": the stream delivers the interrupted error frame.
    } catch {
      /* already finished — the stream is about to end on its own */
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
                  {/* A backgrounded run renders a quiet mission chip, not a stream. */}
                  {m.backgrounded && (
                    <div className="mission-chip">
                      <span className="mc-agent">{m.agentLabel}</span>
                      <span className="mc-meta">
                        running in background ·
                      </span>
                      <Link to="/missions">view in Missions</Link>
                    </div>
                  )}
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
                  {/* Plan-first checklist — updates in place per plan frame. */}
                  {m.role === "agent" && m.plan && <PlanCard steps={m.plan} />}
                  {/* Steering queued mid-run, in the order it was given. */}
                  {m.role === "agent" &&
                    (m.steers ?? []).map((s, k) => (
                      <div className="steer-row" key={k}>
                        <span className="steer-tag">steer</span>
                        <span>{s}</span>
                      </div>
                    ))}
                  {m.text &&
                    (m.role === "agent" ? (
                      <Markdown>{m.text}</Markdown>
                    ) : (
                      m.text
                    ))}
                  {m.role === "agent" &&
                    (m.approvals ?? []).map((a) => (
                      <ApprovalCard
                        key={a.checkpointId}
                        approval={a}
                        onResolve={(approve) =>
                          resolveInline(i, a.checkpointId, approve)
                        }
                      />
                    ))}
                  {m.role === "agent" && !m.streaming && (
                    <ArtifactCard artifacts={artifactsFromSteps(m.steps)} />
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
                : busy && activeMission
                  ? "Steer the run — guidance lands before its next turn…"
                  : isTeam
                    ? "Ask the team to manage something…"
                    : "Message your agent…"
            }
            value={input}
            disabled={!hasAny || (busy && !activeMission)}
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
              {!isTeam && provider === "openrouter" && (
                <PickMenu
                  value={orModel === "__loading" ? "" : orModel}
                  onChange={pickOrModel}
                  title={orError ?? "OpenRouter model — price per 1M tokens"}
                  placeholder="Default (Mistral Small)"
                  searchable
                  groups={orGroups}
                />
              )}

              {!isTeam && (
                <button
                  className={"plan-toggle" + (planMode ? " on" : "")}
                  onClick={togglePlanMode}
                  title="Plan-first: the agent publishes a step checklist and keeps it current"
                  type="button"
                >
                  plan
                </button>
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

            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              {!isTeam && !busy && (
                <button
                  className="composer-icon"
                  onClick={() => void sendBackground()}
                  disabled={!input.trim() || !hasAny}
                  title="Run in background — steer and follow it under Missions"
                  type="button"
                >
                  <BgIcon />
                </button>
              )}
              {busy ? (
                <button
                  className="composer-send"
                  onClick={() => void stop()}
                  title={
                    activeMission
                      ? "Stop — pauses at a checkpoint when one exists, else cancels"
                      : "Stop"
                  }
                  type="button"
                >
                  <span className="stop-glyph" />
                </button>
              ) : (
                <button
                  className="composer-send"
                  onClick={send}
                  disabled={!input.trim() || !hasAny}
                  title="Send (Enter)"
                  type="button"
                >
                  <SendIcon />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
