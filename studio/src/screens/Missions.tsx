import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { Markdown } from "../components/Markdown";
import {
  CognitionTrace,
  GroundingPanel,
  SafetyPanel,
  WorldLedger,
  type CogStep,
} from "../components/Cognition";
import { PlanCard } from "../components/PlanCard";
import { relativeTime } from "../lib/format";
import {
  interruptMission,
  listMissions,
  getMission,
  steerMission,
  streamMission,
  type MissionEvent,
  type MissionView,
  type PlanStep,
} from "../lib/missionsApi";
import "../styles/missions.css";

// One steer/text row reconstructed from the frame stream.
interface SteerRow {
  text: string;
}

interface DetailState {
  text: string;
  steps: CogStep[];
  plan: PlanStep[] | null;
  steers: SteerRow[];
  awaitingApproval: boolean;
  streamErr: string | null;
}

const EMPTY_DETAIL: DetailState = {
  text: "",
  steps: [],
  plan: null,
  steers: [],
  awaitingApproval: false,
  streamErr: null,
};

function statusMeta(m: MissionView): JSX.Element {
  const when = relativeTime(m.finished_at ?? m.created_at);
  return (
    <span className="ml-meta">
      {m.agent} · {m.plan_mode ? "plan · " : ""}
      {m.status === "error" ? (
        <span className="st-error">error</span>
      ) : (
        m.status
      )}{" "}
      · {when}
    </span>
  );
}

export default function Missions() {
  const [items, setItems] = useState<MissionView[] | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<MissionView | null>(null);
  const [detail, setDetail] = useState<DetailState>(EMPTY_DETAIL);
  const [steer, setSteer] = useState("");
  const [actErr, setActErr] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const selectedRef = useRef<string | null>(null);

  const refresh = useCallback(() => {
    listMissions()
      .then((r) => {
        setItems(r.items);
        setLoadErr(null);
        // Keep the open detail's status row honest while it runs.
        const cur = selectedRef.current;
        if (cur) {
          const fresh = r.items.find((m) => m.id === cur);
          if (fresh) setSelected(fresh);
        }
      })
      .catch((e) => setLoadErr(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  // Stream the selected mission: replay the buffer, then follow live. The
  // same cognition components chat uses render the frames.
  const open = (m: MissionView) => {
    abortRef.current?.abort();
    selectedRef.current = m.id;
    setSelected(m);
    setDetail(EMPTY_DETAIL);
    setActErr(null);
    const ac = new AbortController();
    abortRef.current = ac;

    const patch = (fn: (d: DetailState) => DetailState) =>
      setDetail((d) => fn(d));
    const addStep = (s: CogStep) =>
      patch((d) => ({ ...d, steps: [...d.steps, s] }));

    const onEvent = (e: MissionEvent) => {
      switch (e.type) {
        case "token":
          patch((d) => ({ ...d, text: d.text + e.delta }));
          break;
        case "agent":
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
        case "plan":
          patch((d) => ({ ...d, plan: e.steps }));
          break;
        case "steer":
          patch((d) => ({ ...d, steers: [...d.steers, { text: e.text }] }));
          break;
        case "approval_required":
          patch((d) => ({ ...d, awaitingApproval: true }));
          break;
        case "message":
          patch((d) => ({ ...d, text: e.text }));
          break;
        case "done":
          patch((d) => ({ ...d, text: e.output_text || d.text }));
          break;
        case "error":
          patch((d) => ({ ...d, streamErr: e.message }));
          break;
        default:
          break;
      }
    };

    streamMission(m.id, onEvent, ac.signal)
      .then(() => getMission(m.id))
      .then((fresh) => {
        if (selectedRef.current === m.id) setSelected(fresh);
        refresh();
      })
      .catch((e: unknown) => {
        if (ac.signal.aborted) return;
        const msg = e instanceof Error ? e.message : String(e);
        patch((d) => ({ ...d, streamErr: msg }));
      });
  };

  const close = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    selectedRef.current = null;
    setSelected(null);
    setDetail(EMPTY_DETAIL);
  };
  useEffect(() => () => abortRef.current?.abort(), []);

  const doSteer = async () => {
    if (!selected || !steer.trim()) return;
    const text = steer.trim().slice(0, 4000);
    setSteer("");
    setActErr(null);
    try {
      await steerMission(selected.id, text);
      // the steer frame arrives via the live stream and renders itself
    } catch (e: unknown) {
      setActErr(e instanceof Error ? e.message : String(e));
    }
  };

  const doInterrupt = async () => {
    if (!selected) return;
    setActErr(null);
    try {
      const r = await interruptMission(selected.id);
      setActErr(
        r.mode === "checkpoint"
          ? "paused at a checkpoint — resume from Approvals"
          : null,
      );
      const fresh = await getMission(selected.id);
      setSelected(fresh);
      refresh();
    } catch (e: unknown) {
      setActErr(e instanceof Error ? e.message : String(e));
    }
  };

  // ---- detail view ---------------------------------------------------------
  if (selected) {
    const running = selected.status === "running";
    return (
      <>
        <Topbar title="Missions" sub={selected.agent} />
        <div className="missions-col">
          <button className="mission-back" onClick={close} type="button">
            ‹ all missions
          </button>
          <div className="mission-detail-head">
            <div className="md-prompt">{selected.prompt}</div>
            <span className="md-meta">
              {selected.agent} ·{" "}
              {selected.status === "error" ? (
                <span className="st-error">error</span>
              ) : (
                selected.status
              )}{" "}
              · {relativeTime(selected.finished_at ?? selected.created_at)}
              {running && <span className="caret" />}
            </span>
          </div>

          {detail.plan && <PlanCard steps={detail.plan} />}
          {detail.steps.length > 0 && <CognitionTrace steps={detail.steps} />}
          {detail.steers.map((s, i) => (
            <div className="steer-row" key={i}>
              <span className="steer-tag">steer</span>
              <span>{s.text}</span>
            </div>
          ))}
          {detail.awaitingApproval && (
            <div className="missions-note">
              awaiting approval —{" "}
              <Link to="/approvals">resolve it under Approvals</Link>
            </div>
          )}
          {detail.text && (
            <div className="mission-output">
              <Markdown>{detail.text}</Markdown>
            </div>
          )}
          {running && !detail.text && detail.steps.length === 0 && (
            <div className="missions-note">
              working — frames stream in as the agent acts
              <span className="caret" />
            </div>
          )}
          {detail.streamErr && (
            <div className="mission-error">{detail.streamErr}</div>
          )}
          {!running && detail.steps.length > 0 && (
            <>
              <SafetyPanel steps={detail.steps} />
              <GroundingPanel steps={detail.steps} />
              <WorldLedger steps={detail.steps} />
            </>
          )}
          {!running && selected.run_id && (
            <div className="missions-note">
              <Link to={`/activity/${selected.run_id}`}>↗ view full trace</Link>
            </div>
          )}

          {running && (
            <div className="mission-actions">
              <input
                className="input steer-input"
                placeholder="steer the agent — lands before its next turn…"
                value={steer}
                maxLength={4000}
                onChange={(e) => setSteer(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void doSteer();
                }}
              />
              <button className="btn" onClick={() => void doSteer()} type="button">
                Steer
              </button>
              <button
                className="btn"
                onClick={() => void doInterrupt()}
                type="button"
              >
                Interrupt
              </button>
            </div>
          )}
          {actErr && <div className="mission-error">{actErr}</div>}
        </div>
      </>
    );
  }

  // ---- list view -----------------------------------------------------------
  return (
    <>
      <Topbar title="Missions" sub="background runs you can steer" />
      {loadErr ? (
        <div className="missions-col">
          <div className="mission-error">{loadErr}</div>
        </div>
      ) : items === null ? (
        <div className="missions-col">
          <div className="missions-note">loading…</div>
        </div>
      ) : items.length === 0 ? (
        <EmptyState icon="✦" title="No missions yet">
          Start one from Chat — the background action next to send runs your
          prompt as a mission you can steer, interrupt, and reconnect to here.
        </EmptyState>
      ) : (
        <div className="missions-col">
          <div className="home-sec">
            <div className="home-sec-head">
              <span>missions</span>
              <span>
                {items.filter((m) => m.status === "running").length} running
              </span>
            </div>
            {items.map((m) => (
              <button
                key={m.id}
                className="mission-line"
                onClick={() => open(m)}
                type="button"
              >
                <span className="ml-prompt">{m.prompt}</span>
                {statusMeta(m)}
                {m.status === "running" && <span className="caret" />}
              </button>
            ))}
            <div className="missions-note">
              running missions live in this server process — finished ones are
              kept in Activity
            </div>
          </div>
        </div>
      )}
    </>
  );
}
