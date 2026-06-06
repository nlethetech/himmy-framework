import { useEffect, useState } from "react";
import {
  api,
  listEvalSuites,
  runEval,
  type EvalSuiteInfo,
  type AgentSummary,
  type EvalRun,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { GlobeIcon } from "../components/icons";
import { useToast } from "../components/ui/Toast";

export default function Evaluation() {
  const [suites, setSuites] = useState<EvalSuiteInfo[] | null>(null);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [suite, setSuite] = useState("");
  const [agent, setAgent] = useState("");
  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<EvalRun | null>(null);
  const toast = useToast();

  useEffect(() => {
    listEvalSuites().then((s) => {
      setSuites(s);
      setSuite((c) => c || s[0]?.path || "");
    });
    api.get<AgentSummary[]>("/agents").then((a) => {
      setAgents(a);
      setAgent((c) => c || a[0]?.path || "");
    });
  }, []);

  const go = async () => {
    if (!suite || !agent) return;
    setRunning(true);
    setRun(null);
    try {
      setRun(await runEval(suite, agent));
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setRunning(false);
    }
  };

  const pct = (n: number) => Math.round(n * 100) + "%";

  return (
    <>
      <Topbar title="Evaluation" sub="Score an agent against a test suite" />
      <Page>
        {suites === null ? null : suites.length === 0 ? (
          <EmptyState icon={<GlobeIcon />} title="No eval suites found">
            Add a <code>*.eval.yaml</code> with a <code>name</code> and a list of{" "}
            <code>cases</code> (each an <code>input</code> + <code>expected_output</code>{" "}
            + <code>metric_weights</code>) to your project, then run it here.
          </EmptyState>
        ) : (
          <>
            <div className="card card-pad eval-bar">
              <label className="field" style={{ margin: 0, flex: 1 }}>
                <span className="field-label">Suite</span>
                <select className="input" value={suite} onChange={(e) => setSuite(e.target.value)}>
                  {suites.map((s) => (
                    <option key={s.path} value={s.path}>
                      {s.name} · {s.cases} case{s.cases === 1 ? "" : "s"}
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
                {running ? <span className="spinner" /> : "Run eval"}
              </button>
            </div>

            {running && (
              <div className="dim" style={{ padding: 16 }}>
                Running the agent over each case…
              </div>
            )}

            {run && (
              <div className="card">
                <div className="card-head">
                  <h2>{run.suite_name}</h2>
                  <span
                    className={
                      "pill " +
                      (run.aggregate_score >= 0.8
                        ? "ok"
                        : run.aggregate_score >= 0.5
                          ? "warn"
                          : "err")
                    }
                  >
                    {pct(run.aggregate_score)} overall
                  </span>
                </div>
                <div className="card-pad">
                  <div className="eval-cases">
                    {run.case_results.map((c) => (
                      <div className="eval-case" key={c.case_id}>
                        <div className="row spread">
                          <span className="mono dim" style={{ fontSize: 12 }}>
                            case {c.case_id.slice(0, 8)}
                          </span>
                          <span className={"pill " + (c.passed ? "ok" : "err")}>
                            {c.passed ? "pass" : "fail"} · {pct(c.aggregate)}
                          </span>
                        </div>
                        <div className="eval-metrics">
                          {c.metric_scores.map((m) => (
                            <div className="eval-metric" key={m.metric}>
                              <span className="eval-metric-name">{m.metric}</span>
                              <span className="eval-bar-track">
                                <span
                                  className="eval-bar-fill"
                                  style={{
                                    width: pct(m.score),
                                    background: m.passed
                                      ? "var(--ok)"
                                      : "var(--warn)",
                                  }}
                                />
                              </span>
                              <span className="mono eval-metric-score">
                                {pct(m.score)}
                              </span>
                            </div>
                          ))}
                        </div>
                        {c.actual_output && (
                          <div className="eval-output">{c.actual_output}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </Page>
    </>
  );
}
