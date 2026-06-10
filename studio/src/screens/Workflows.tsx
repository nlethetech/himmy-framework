import { useEffect, useRef, useState } from "react";
import {
  api,
  listWorkflows,
  type AgentSummary,
  type WorkflowInfo,
} from "../lib/api";
import {
  streamWorkflow,
  type WorkflowNodeState,
  type WorkflowStreamEvent,
  type WorkflowStepRow,
} from "../lib/teamsApi";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { RunsIcon } from "../components/icons";
import { PickMenu, type PickGroup } from "../components/ui/PickMenu";
import { useToast } from "../components/ui/Toast";
import "../styles/teams.css";

/* Workflow runner as a ledger: workflows are rows, the selected one shows its
   steps read-only, and a run streams node-status rows (running = the thinking
   dots, failures red) followed by the final output block. */

type NodeRow = {
  name: string;
  state: WorkflowNodeState | "pending";
  error?: string | null;
};

type RunPhase = "idle" | "running" | "done" | "error";

interface RunState {
  phase: RunPhase;
  nodes: NodeRow[];
  status: string | null; // final workflow status (completed|failed|partial|cancelled)
  steps: WorkflowStepRow[];
  message: string | null; // terminal error message
}

const IDLE_RUN: RunState = {
  phase: "idle",
  nodes: [],
  status: null,
  steps: [],
  message: null,
};

export default function Workflows() {
  const [workflows, setWorkflows] = useState<WorkflowInfo[] | null>(null);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [sel, setSel] = useState<string>("");
  const [agent, setAgent] = useState<string>("");
  const [run, setRun] = useState<RunState>(IDLE_RUN);
  const abortRef = useRef<AbortController | null>(null);
  const toast = useToast();

  useEffect(() => {
    listWorkflows()
      .then((w) => {
        setWorkflows(w);
        setSel((c) => c || w[0]?.path || "");
      })
      .catch(() => setWorkflows([]));
    api
      .get<AgentSummary[]>("/agents")
      .then((a) => {
        setAgents(a);
        setAgent((c) => c || a[0]?.path || "");
      })
      .catch(() => setAgents([]));
    return () => abortRef.current?.abort();
  }, []);

  const current = workflows?.find((w) => w.path === sel);
  const running = run.phase === "running";

  const onEvent = (e: WorkflowStreamEvent) => {
    if (e.type === "workflow_start") {
      setRun((r) => ({
        ...r,
        nodes: e.steps.map((name) => ({ name, state: "pending" as const })),
      }));
    } else if (e.type === "node") {
      setRun((r) => ({
        ...r,
        nodes: r.nodes.some((n) => n.name === e.name)
          ? r.nodes.map((n) =>
              n.name === e.name ? { ...n, state: e.state, error: e.error } : n,
            )
          : [...r.nodes, { name: e.name, state: e.state, error: e.error }],
      }));
    } else if (e.type === "done") {
      setRun((r) => ({
        ...r,
        phase: "done",
        status: e.status,
        steps: e.step_results,
      }));
    } else if (e.type === "error") {
      setRun((r) => ({ ...r, phase: "error", message: e.message }));
    }
  };

  const go = async () => {
    if (!sel || !agent || running) return;
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setRun({ ...IDLE_RUN, phase: "running" });
    try {
      await streamWorkflow(
        { workflow_path: sel, agent_path: agent },
        onEvent,
        ctl.signal,
      );
      // Stream closed without a terminal frame — surface it instead of hanging.
      setRun((r) =>
        r.phase === "running"
          ? { ...r, phase: "error", message: "stream ended unexpectedly" }
          : r,
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setRun((r) => ({ ...r, phase: "error", message: "stopped" }));
        return;
      }
      const msg = (e as Error).message;
      setRun((r) => ({ ...r, phase: "error", message: msg }));
      toast.show(msg, "err");
    }
  };

  const stop = () => abortRef.current?.abort();

  const selectWorkflow = (path: string) => {
    if (running) return;
    setSel(path);
    setRun(IDLE_RUN);
  };

  const agentGroups: PickGroup[] = [
    {
      options: agents.map((a) => ({
        value: a.path,
        label: a.name,
        meta: a.provider ?? undefined,
      })),
    },
  ];

  // The final output: the last step that produced text (state threads forward).
  const finalText =
    [...run.steps].reverse().find((s) => s.output_text)?.output_text ?? null;
  const failedStep = run.steps.find((s) => s.status !== "completed");

  return (
    <>
      <Topbar title="Workflows" sub="Declarative multi-step runs with shared state" />
      <div className="home-col teams-col">
        {workflows === null ? (
          <div className="home-empty">Loading workflows…</div>
        ) : workflows.length === 0 ? (
          <EmptyState icon={<RunsIcon />} title="No workflows found">
            Add a <code>*.workflow.yaml</code> (a <code>name</code> + ordered{" "}
            <code>steps</code>, each a <code>subtask</code> that can reference prior{" "}
            <code>output_key</code>s with <code>{"{placeholders}"}</code>) to your
            project, then run it here.
          </EmptyState>
        ) : (
          <>
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Workflows</span>
              </div>
              {workflows.map((w) => (
                <button
                  type="button"
                  className={"run-line wfl-line" + (w.path === sel ? " on" : "")}
                  key={w.path}
                  onClick={() => selectWorkflow(w.path)}
                  title={w.description || w.path}
                >
                  <span className="run-line-prompt">{w.name}</span>
                  <span className="run-line-meta mono">
                    {w.steps.length} step{w.steps.length === 1 ? "" : "s"} · {w.path}
                  </span>
                </button>
              ))}
            </section>

            {current && (
              <section className="home-sec">
                <div className="home-sec-head">
                  <span>Steps — {current.name}</span>
                </div>
                {current.description && (
                  <div className="home-empty">{current.description}</div>
                )}
                {current.steps.map((s, i) => (
                  <div key={s.name + i}>
                    <div className="run-line no-rule">
                      <span className="wfl-step-idx">{i + 1}</span>
                      <span className="run-line-prompt">{s.name}</span>
                      <span className="run-line-meta mono">
                        {s.tools.length > 0 ? s.tools.join(", ") : ""}
                        {s.output_key
                          ? `${s.tools.length > 0 ? " · " : ""}→ {${s.output_key}}`
                          : ""}
                      </span>
                    </div>
                    <span className="wfl-step-task" title={s.subtask}>
                      {s.subtask}
                    </span>
                  </div>
                ))}
              </section>
            )}

            <section className="home-sec">
              <div className="home-sec-head">
                <span>Run</span>
              </div>
              <div className="tb-row">
                <span className="tb-row-label">agent</span>
                <div className="tb-row-main">
                  {agents.length > 0 ? (
                    <PickMenu
                      value={agent}
                      groups={agentGroups}
                      onChange={setAgent}
                      placeholder="pick an agent…"
                      title="The agent persona the steps run with"
                    />
                  ) : (
                    <span className="tb-note">
                      no agents found — add an agent.yaml first
                    </span>
                  )}
                  <span style={{ flex: 1 }} />
                  {running ? (
                    <button type="button" className="btn" onClick={stop}>
                      Stop
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={go}
                      disabled={!sel || !agent}
                    >
                      Run
                    </button>
                  )}
                </div>
              </div>

              {run.nodes.length > 0 &&
                run.nodes.map((n) => (
                  <div key={n.name}>
                    <div className={"run-line" + (n.error ? " no-rule" : "")}>
                      <span className="run-line-prompt mono">{n.name}</span>
                      {n.state === "running" ? (
                        <span className="thinking" aria-label="running">
                          <span />
                          <span />
                          <span />
                        </span>
                      ) : (
                        <span
                          className={
                            "wfl-node-state" +
                            (n.state === "failed" ? " failed" : "")
                          }
                        >
                          {n.state === "pending" ? "—" : n.state}
                        </span>
                      )}
                    </div>
                    {n.error && <span className="wfl-node-err">{n.error}</span>}
                  </div>
                ))}
              {run.phase === "error" && (
                <ul className="tb-errors">
                  <li>{run.message ?? "run failed"}</li>
                </ul>
              )}
            </section>

            {run.phase === "done" && (
              <section className="home-sec">
                <div className="home-sec-head">
                  <span>Output</span>
                  {run.status !== "completed" && (
                    <span className="pill err">{run.status}</span>
                  )}
                </div>
                {finalText ? (
                  <pre className="wfl-out">{finalText}</pre>
                ) : failedStep ? (
                  <pre className="wfl-out failed">
                    {failedStep.step_name}: {failedStep.error ?? "step failed"}
                  </pre>
                ) : (
                  <div className="home-empty">The run produced no output text.</div>
                )}
                <div className="wfl-meta">
                  {run.status} · {run.steps.filter((s) => s.status === "completed").length}
                  /{run.steps.length} steps completed
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </>
  );
}
