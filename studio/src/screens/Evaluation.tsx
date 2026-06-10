import { useEffect, useRef, useState } from "react";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { PickMenu } from "../components/ui/PickMenu";
import { useToast } from "../components/ui/Toast";
import { GlobeIcon } from "../components/icons";
import {
  getBaseline,
  listRunnableSuites,
  streamEvalRun,
  type BaselineRow,
  type BaselineView,
  type EvalCaseEvent,
  type EvalSummaryEvent,
  type RunnableSuite,
} from "../lib/evalApi";
import "../styles/eval.css";

/* Evaluation as a single centered ledger column: SUITES (what's runnable),
   RUN (per-case rows streaming in), BASELINE (the gate table, with a delta
   column against the just-finished bench run). Offline stub by default —
   a live provider only when explicitly picked. */

const PROVIDERS = [
  {
    options: [
      { value: "stub", label: "offline · stub", meta: "deterministic" },
      { value: "claude-cli", label: "claude-cli", meta: "live" },
      { value: "ollama", label: "ollama", meta: "live" },
    ],
  },
];

const pct = (n: number) => Math.round(n * 100) + "%";

export default function Evaluation() {
  const [suites, setSuites] = useState<RunnableSuite[] | null>(null);
  const [suitesError, setSuitesError] = useState("");
  const [baseline, setBaseline] = useState<BaselineView | null>(null);
  const [provider, setProvider] = useState("stub");

  const [running, setRunning] = useState(false);
  const [runLabel, setRunLabel] = useState("");
  const [cases, setCases] = useState<EvalCaseEvent[]>([]);
  const [summary, setSummary] = useState<EvalSummaryEvent | null>(null);
  const [runError, setRunError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const toast = useToast();

  useEffect(() => {
    listRunnableSuites()
      .then((r) => setSuites(r.suites))
      .catch((e) => {
        setSuites([]);
        setSuitesError((e as Error).message);
      });
    getBaseline().catch(() => null).then((b) => setBaseline(b ?? null));
    return () => abortRef.current?.abort();
  }, []);

  const startRun = async (s: RunnableSuite) => {
    if (running) return;
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setRunning(true);
    setRunLabel(s.name);
    setCases([]);
    setSummary(null);
    setRunError("");
    try {
      await streamEvalRun(
        { suite: s.id, provider: provider === "stub" ? null : provider },
        (e) => {
          if (e.type === "case") setCases((c) => [...c, e]);
          else if (e.type === "summary") setSummary(e);
          else if (e.type === "error") setRunError(e.message);
        },
        ctl.signal,
      );
    } catch (e) {
      if (!ctl.signal.aborted) {
        setRunError((e as Error).message);
        toast.show((e as Error).message, "err");
      }
    } finally {
      if (abortRef.current === ctl) setRunning(false);
    }
  };

  // Delta vs the baseline only makes sense for a finished bench-gate run.
  const runMetric = (metric: string): number | null => {
    if (summary?.kind !== "bench" || !summary.metrics) return null;
    if (metric === "accuracy") return summary.metrics.accuracy;
    if (metric === "tool_call_accuracy") return summary.metrics.tool_call_accuracy;
    if (metric === "error_rate") return summary.metrics.error_rate;
    return null;
  };

  const caseMeta = (c: EvalCaseEvent) => {
    const failing = c.metrics.filter((m) => !m.passed).map((m) => m.metric);
    return pct(c.score) + (failing.length ? " · " + failing.join(", ") : "");
  };
  const caseTitle = (c: EvalCaseEvent) =>
    c.metrics
      .map((m) => `${m.metric} ${pct(m.score)} ${m.passed ? "pass" : "fail"}`)
      .join("\n");

  const nothingToShow =
    suites !== null && suites.length === 0 && baseline !== null && !baseline.exists;

  return (
    <>
      <Topbar title="Evaluation" sub="Run suites, compare to baseline" />
      {nothingToShow ? (
        <EmptyState icon={<GlobeIcon />} title="Nothing to evaluate">
          Add a <code>*.eval.yaml</code> suite (a <code>name</code> plus{" "}
          <code>cases</code> of <code>input</code> / <code>expected_output</code> /{" "}
          <code>metric_weights</code>) to the project, or check in a{" "}
          <code>benchmarks/baseline.json</code> to run the bench gate.
          {suitesError ? <> — {suitesError}</> : null}
        </EmptyState>
      ) : (
        <div className="home-col">
          {/* ---- SUITES ---- */}
          <section className="home-sec">
            <div className="home-sec-head">
              <span>Suites</span>
              <span className="eval-pick">
                <PickMenu
                  value={provider}
                  groups={PROVIDERS}
                  onChange={setProvider}
                  title="Provider for runs — offline stub completes in seconds"
                />
              </span>
            </div>
            {suites === null ? (
              <div className="home-empty">Loading suites…</div>
            ) : suites.length === 0 ? (
              <div className="home-empty">
                {suitesError ||
                  "No suites found — add a *.eval.yaml to the project root."}
              </div>
            ) : (
              suites.map((s) => (
                <div className="run-line" key={s.id}>
                  <span className="run-line-prompt" title={s.source}>
                    {s.name}
                  </span>
                  <span className="run-line-meta mono">
                    {s.cases} case{s.cases === 1 ? "" : "s"} ·{" "}
                    {s.kind === "bench" ? "gate" : "eval"}
                  </span>
                  <button
                    className="eval-run-act"
                    onClick={() => startRun(s)}
                    disabled={running}
                  >
                    run →
                  </button>
                </div>
              ))
            )}
          </section>

          {/* ---- RUN ---- */}
          <section className="home-sec">
            <div className="home-sec-head">
              <span>Run</span>
              {runLabel && (
                <span className="mono">
                  {runLabel}
                  {running ? " · running…" : ""}
                </span>
              )}
            </div>
            {!runLabel ? (
              <div className="home-empty">
                No run yet — pick a suite above. The default provider is the
                offline stub, so a run finishes in seconds.
              </div>
            ) : (
              <>
                {cases.map((c) => (
                  <div
                    className={"run-line" + (c.passed ? "" : " eval-row-fail")}
                    key={c.id + c.index}
                    title={caseTitle(c)}
                  >
                    <span className="run-line-prompt">{c.case}</span>
                    <span className="run-line-meta mono">{caseMeta(c)}</span>
                    <span className={"mono eval-verdict" + (c.passed ? "" : " fail")}>
                      {c.passed ? "pass" : "fail"}
                    </span>
                  </div>
                ))}
                {running && (
                  <div className="home-empty mono">
                    running case {cases.length + 1}…
                  </div>
                )}
                {runError && <div className="eval-error">{runError}</div>}
                {summary && (
                  <div className="eval-summary">
                    {summary.passed}/{summary.cases} passed · aggregate{" "}
                    {pct(summary.aggregate_score)} ·{" "}
                    {summary.duration_s.toFixed(1)}s
                    {summary.model ? ` · ${summary.model}` : ""}
                    {summary.note ? ` · ${summary.note}` : ""}
                  </div>
                )}
              </>
            )}
          </section>

          {/* ---- BASELINE ---- */}
          <section className="home-sec">
            <div className="home-sec-head">
              <span>Baseline</span>
              {baseline?.exists && (
                <span className="mono">
                  {baseline.sha.slice(0, 12)} · {baseline.date} · {baseline.trials}{" "}
                  trials
                </span>
              )}
            </div>
            {baseline === null ? (
              <div className="home-empty">Loading baseline…</div>
            ) : !baseline.exists ? (
              <div className="home-empty">{baseline.reason}</div>
            ) : (
              baseline.models.map((m) => (
                <div key={m.name}>
                  <div className="eval-bl-model">{m.name}</div>
                  <div className="eval-bl-hrow">
                    <span>metric</span>
                    <span>floor</span>
                    <span>measured</span>
                    <span>delta</span>
                  </div>
                  {m.rows.map((r) => (
                    <BaselineRowLine key={r.metric} row={r} run={runMetric(r.metric)} />
                  ))}
                </div>
              ))
            )}
          </section>
        </div>
      )}
    </>
  );
}

function BaselineRowLine({ row, run }: { row: BaselineRow; run: number | null }) {
  const delta = run != null && row.measured != null ? run - row.measured : null;
  // a fresh run breaching the gate bound is a hit — red's third job
  const breach =
    run != null &&
    row.limit != null &&
    (row.bound === "floor" ? run < row.limit : run > row.limit);
  return (
    <div className="eval-bl-row">
      <span className="eval-bl-name" title={row.bound}>
        {row.metric}
      </span>
      <span className="eval-bl-num">
        {row.limit == null ? "—" : pct(row.limit) + (row.bound === "ceiling" ? " max" : "")}
      </span>
      <span className="eval-bl-num">
        {row.measured == null ? "—" : pct(row.measured)}
      </span>
      <span className={"eval-bl-num" + (breach ? " fail" : "")}>
        {delta == null
          ? "—"
          : (delta >= 0 ? "+" : "−") + Math.abs(Math.round(delta * 100)) + "pt"}
      </span>
    </div>
  );
}
