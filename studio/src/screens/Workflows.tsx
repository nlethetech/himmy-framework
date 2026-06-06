import { useEffect, useState } from "react";
import {
  api,
  listWorkflows,
  runWorkflow,
  type WorkflowInfo,
  type AgentSummary,
  type WorkflowResult,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { RunsIcon } from "../components/icons";
import { useToast } from "../components/ui/Toast";

export default function Workflows() {
  const [workflows, setWorkflows] = useState<WorkflowInfo[] | null>(null);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [sel, setSel] = useState<string>("");
  const [agent, setAgent] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<WorkflowResult | null>(null);
  const toast = useToast();

  useEffect(() => {
    listWorkflows().then((w) => {
      setWorkflows(w);
      setSel((c) => c || w[0]?.path || "");
    });
    api.get<AgentSummary[]>("/agents").then((a) => {
      setAgents(a);
      setAgent((c) => c || a[0]?.path || "");
    });
  }, []);

  const current = workflows?.find((w) => w.path === sel);

  const go = async () => {
    if (!sel || !agent) return;
    setRunning(true);
    setResult(null);
    try {
      setResult(await runWorkflow(sel, agent));
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setRunning(false);
    }
  };

  return (
    <>
      <Topbar title="Workflows" sub="Declarative multi-step runs with shared state" />
      <Page>
        {workflows === null ? null : workflows.length === 0 ? (
          <EmptyState icon={<RunsIcon />} title="No workflows found">
            Add a <code>*.workflow.yaml</code> (a <code>name</code> + ordered{" "}
            <code>steps</code>, each a <code>subtask</code> that can reference prior{" "}
            <code>output_key</code>s with <code>{"{placeholders}"}</code>) to your
            project, then run it here.
          </EmptyState>
        ) : (
          <>
            <div className="card card-pad eval-bar">
              <label className="field" style={{ margin: 0, flex: 1 }}>
                <span className="field-label">Workflow</span>
                <select className="input" value={sel} onChange={(e) => setSel(e.target.value)}>
                  {workflows.map((w) => (
                    <option key={w.path} value={w.path}>
                      {w.name} · {w.steps.length} step{w.steps.length === 1 ? "" : "s"}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field" style={{ margin: 0, flex: 1 }}>
                <span className="field-label">Agent</span>
                <select className="input" value={agent} onChange={(e) => setAgent(e.target.value)}>
                  {agents.map((a) => (
                    <option key={a.path} value={a.path}>
                      {a.name}
                    </option>
                  ))}
                </select>
              </label>
              <button className="btn btn-primary" onClick={go} disabled={running}>
                {running ? <span className="spinner" /> : "Run"}
              </button>
            </div>

            {current && (
              <div className="card card-pad" style={{ marginBottom: 16 }}>
                <span className="section-title">{current.name}</span>
                {current.description && (
                  <div className="dim" style={{ marginBottom: 8 }}>
                    {current.description}
                  </div>
                )}
                <div className="wf-steps">
                  {current.steps.map((s, i) => {
                    const res = result?.step_results.find((r) => r.step_name === s.name);
                    return (
                      <div className="wf-step" key={s.name}>
                        <div className="wf-step-num">{i + 1}</div>
                        <div className="wf-step-body">
                          <div className="row spread">
                            <span className="wf-step-name">{s.name}</span>
                            {res && (
                              <span
                                className={
                                  "pill " + (res.status === "completed" ? "ok" : "err")
                                }
                              >
                                {res.status}
                              </span>
                            )}
                            {running && !res && <span className="spinner" />}
                          </div>
                          <div className="wf-step-task">{s.subtask}</div>
                          {s.tools.length > 0 && (
                            <div className="wf-step-tools mono">
                              tools: {s.tools.join(", ")}
                            </div>
                          )}
                          {res?.output_text && (
                            <div className="eval-output">{res.output_text}</div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                {result && (
                  <div className="row gap10 mt8">
                    <span
                      className={
                        "pill " + (result.status === "completed" ? "ok" : "warn")
                      }
                    >
                      workflow {result.status}
                    </span>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </Page>
    </>
  );
}
