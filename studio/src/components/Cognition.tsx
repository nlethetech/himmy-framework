import { useState } from "react";

// One step of an agent's thinking, shared by the live chat stream and the run
// detail. Mirrors the backend CognitionStep (himmy/api/studio_runs.py) but every
// field is optional so it can be assembled incrementally from live SSE frames.
export interface CogStep {
  kind: "agent" | "reason" | "tool" | "delegate" | "handoff";
  agent?: string | null;
  model?: string | null;
  text?: string | null;
  name?: string | null;
  args?: Record<string, unknown> | null;
  result?: string | null;
  outcome?: string | null;
  read_only?: boolean | null;
  latency_ms?: number | null;
  worker?: string | null;
  task?: string | null;
  to?: string | null;
}

function fmtMs(ms?: number | null): string | null {
  if (ms == null || !isFinite(ms)) return null;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)}s`;
}

function fmtArgs(args?: Record<string, unknown> | null): string {
  if (!args || Object.keys(args).length === 0) return "";
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("  ");
}

// ---- Tool step (the unit of "what it did to the world") -----------------

function ToolStep({ s }: { s: CogStep }) {
  const [open, setOpen] = useState(false);
  const write = s.read_only === false;
  const failed = s.outcome && s.outcome !== "success";
  const latency = fmtMs(s.latency_ms);
  const argline = fmtArgs(s.args);
  const expandable = Boolean(argline || (s.result && s.result.length > 0));
  return (
    <div className={"cog-tool" + (write ? " write" : "") + (failed ? " failed" : "")}>
      <button
        type="button"
        className="cog-tool-head"
        onClick={() => expandable && setOpen((o) => !o)}
        style={{ cursor: expandable ? "pointer" : "default" }}
      >
        <span className={"cog-intent " + (write ? "w" : "r")}>
          {write ? "WRITE" : "READ"}
        </span>
        <span className="cog-tool-name mono">{s.name}</span>
        {argline && !open && <span className="cog-tool-args mono">{argline}</span>}
        <span className="cog-tool-meta">
          {failed && <span className="cog-fail">{s.outcome}</span>}
          {latency && <span className="cog-lat mono">{latency}</span>}
          {expandable && <span className="cog-caret">{open ? "▾" : "▸"}</span>}
        </span>
      </button>
      {open && (
        <div className="cog-tool-body">
          {argline && (
            <div className="cog-kv">
              <span className="cog-k">args</span>
              <code className="cog-v mono">{argline}</code>
            </div>
          )}
          {s.result && (
            <div className="cog-kv">
              <span className="cog-k">result</span>
              <code className="cog-v mono">{s.result}</code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ReasonStep({ s }: { s: CogStep }) {
  const text = (s.text || "").trim();
  const [open, setOpen] = useState(false);
  const long = text.length > 220;
  const shown = open || !long ? text : text.slice(0, 219) + "…";
  return (
    <div className="cog-reason">
      <span className="cog-reason-mark">“</span>
      <span className="cog-reason-text">{shown}</span>
      {long && (
        <button type="button" className="cog-more" onClick={() => setOpen((o) => !o)}>
          {open ? "less" : "more"}
        </button>
      )}
    </div>
  );
}

// The full think → act → observe trace for one turn.
export function CognitionTrace({ steps }: { steps: CogStep[] }) {
  if (!steps || steps.length === 0) return null;
  let lastAgent: string | null = null;
  return (
    <div className="cog-trace">
      {steps.map((s, i) => {
        // Show an agent divider only when the active agent actually changes
        // (so a single-agent run never shows a redundant header).
        if (s.kind === "agent") {
          if (s.agent === lastAgent) return null;
          lastAgent = s.agent ?? null;
          return (
            <div className="cog-agent" key={i}>
              <span className="cog-dot" />
              <span className="cog-agent-name">{s.agent}</span>
              {s.model && <span className="cog-agent-model mono">{s.model}</span>}
            </div>
          );
        }
        if (s.kind === "reason") return <ReasonStep s={s} key={i} />;
        if (s.kind === "tool") return <ToolStep s={s} key={i} />;
        if (s.kind === "delegate")
          return (
            <div className="cog-route delegate" key={i}>
              <span className="cog-arrow">→</span>
              <span className="cog-route-to">{s.worker}</span>
              {s.task && <span className="cog-route-task">{s.task}</span>}
            </div>
          );
        if (s.kind === "handoff")
          return (
            <div className="cog-route handoff" key={i}>
              <span className="cog-arrow">⇒</span>
              <span className="cog-route-to">{s.to}</span>
            </div>
          );
        return null;
      })}
    </div>
  );
}

// ---- "What it did to the world" ledger ----------------------------------

// Just the write actions (read_only === false) — the side effects this run had.
export function WorldLedger({ steps }: { steps: CogStep[] }) {
  const writes = (steps || []).filter(
    (s) => s.kind === "tool" && s.read_only === false,
  );
  if (writes.length === 0) return null;
  return (
    <div className="ledger">
      <div className="ledger-head">
        <span className="ledger-dot" />
        Changed {writes.length} thing{writes.length === 1 ? "" : "s"}
      </div>
      {writes.map((s, i) => {
        const failed = s.outcome && s.outcome !== "success";
        return (
          <div className={"ledger-row" + (failed ? " failed" : "")} key={i}>
            <span className="ledger-verb mono">{s.name}</span>
            {fmtArgs(s.args) && (
              <span className="ledger-args mono">{fmtArgs(s.args)}</span>
            )}
            <span className="ledger-status mono">
              {failed ? s.outcome : "✓"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---- Live orchestration graph (manager → workers) -----------------------

export interface GraphMember {
  name: string;
  role?: string | null;
}

// A compact node graph of a team: the entry/manager links to each worker; an
// edge lights up while that worker is being delegated to, and a node glows when
// it's the active agent. Pure SVG, no deps.
export function OrchestrationGraph({
  members,
  entry,
  steps,
  active,
}: {
  members: GraphMember[];
  entry: string;
  steps: CogStep[];
  active?: string | null;
}) {
  if (!members || members.length < 2) return null;
  const workers = members.filter((m) => m.name !== entry);

  // Which workers have been delegated to (edge fired) + delegation counts.
  const fired = new Map<string, number>();
  for (const s of steps) {
    if (s.kind === "delegate" && s.worker) {
      fired.set(s.worker, (fired.get(s.worker) || 0) + 1);
    }
  }

  const W = 520;
  const colW = W / Math.max(workers.length, 1);
  const managerX = W / 2;
  const managerY = 34;
  const workerY = 150;
  const pos = (i: number) => colW * i + colW / 2;

  return (
    <div className="orch">
      <svg viewBox={`0 0 ${W} 190`} className="orch-svg" role="img">
        {workers.map((w, i) => {
          const x = pos(i);
          const hit = fired.get(w.name) || 0;
          return (
            <line
              key={"e" + i}
              x1={managerX}
              y1={managerY + 18}
              x2={x}
              y2={workerY - 18}
              className={"orch-edge" + (hit ? " on" : "")}
            />
          );
        })}
        {/* manager */}
        <g className={"orch-node mgr" + (active === entry ? " active" : "")}>
          <rect
            x={managerX - 56}
            y={managerY - 16}
            width={112}
            height={34}
            rx={8}
          />
          <text x={managerX} y={managerY + 5} className="orch-label">
            {entry}
          </text>
        </g>
        {/* workers */}
        {workers.map((w, i) => {
          const x = pos(i);
          const hit = fired.get(w.name) || 0;
          const isActive = active === w.name;
          return (
            <g
              key={"n" + i}
              className={
                "orch-node" +
                (hit ? " hit" : "") +
                (isActive ? " active" : "")
              }
            >
              <rect x={x - 48} y={workerY - 16} width={96} height={34} rx={8} />
              <text x={x} y={workerY + 1} className="orch-label">
                {w.name}
              </text>
              {hit > 1 && (
                <text x={x + 38} y={workerY - 8} className="orch-badge">
                  {hit}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
