import { useEffect, useRef, useState } from "react";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { PickMenu } from "../components/ui/PickMenu";
import { useToast } from "../components/ui/Toast";
import { GlobeIcon } from "../components/icons";
import {
  getBaseline,
  getHistory,
  listRunnableSuites,
  streamEvalRun,
  type BaselineRow,
  type BaselineView,
  type EvalCaseEvent,
  type EvalSummaryEvent,
  type HistoryPoint,
  type HistorySeries,
  type HistoryView,
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
  const [history, setHistory] = useState<HistoryView | null>(null);
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
    getHistory().catch(() => null).then((h) => setHistory(h ?? null));
    return () => abortRef.current?.abort();
  }, []);

  // After a bench-gate run lands a new point, refresh the trend series.
  const refreshHistory = () =>
    getHistory().catch(() => null).then((h) => setHistory(h ?? null));

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
      if (s.kind === "bench") void refreshHistory();
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

          {/* ---- HISTORY / TRENDS ---- */}
          <section className="home-sec">
            <div className="home-sec-head">
              <span>History</span>
              {history?.exists && history.series.length > 0 && (
                <span className="mono">
                  {history.total_records} run
                  {history.total_records === 1 ? "" : "s"}
                </span>
              )}
            </div>
            {history === null ? (
              <div className="home-empty">Loading history…</div>
            ) : !history.exists || history.series.length === 0 ? (
              <div className="home-empty">
                {history?.reason ||
                  "Run `himmy bench` from the CLI to start collecting history (Studio runs are not recorded)."}
              </div>
            ) : (
              history.series.map((s) => (
                <HistorySeriesLine key={s.model + "·" + s.suite} series={s} />
              ))
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

/* One model+suite trend: an accuracy sparkline over recent runs, the latest
   value, and the latest-vs-previous delta (red when it regressed). */
function HistorySeriesLine({ series }: { series: HistorySeries }) {
  const acc = series.points.map((p) => p.accuracy);
  const accTrend = series.trends.find((t) => t.metric === "accuracy");
  const latest = accTrend?.latest ?? null;
  const delta = accTrend?.delta ?? null;
  const regressed = accTrend?.regressed ?? false;
  return (
    <div className="eval-hist-row">
      <span className="eval-hist-model" title={series.model + " · " + series.suite}>
        {series.model}
        {series.suite ? <span className="eval-hist-suite">{series.suite}</span> : null}
      </span>
      <span className="eval-hist-spark" title={`${series.runs} runs`}>
        <Sparkline points={series.points} regressed={regressed} />
      </span>
      <span className="eval-hist-val mono">{latest == null ? "—" : pct(latest)}</span>
      <span
        className={"eval-hist-delta mono" + (regressed ? " fail" : "")}
        title={
          accTrend?.previous != null
            ? `prev ${pct(accTrend.previous)}`
            : "no previous run"
        }
      >
        {delta == null || acc.length < 2
          ? "—"
          : (delta >= 0 ? "+" : "−") +
            Math.abs(Math.round(delta * 100)) +
            "pt" +
            (regressed ? " ↓" : "")}
      </span>
    </div>
  );
}

/* A minimal inline-SVG accuracy sparkline (0–1 on the y-axis). No charting
   dependency — just a polyline + the latest point dotted. */
function Sparkline({
  points,
  regressed,
}: {
  points: HistoryPoint[];
  regressed: boolean;
}) {
  const W = 96;
  const H = 22;
  const PAD = 2;
  const vals = points.map((p) => (typeof p.accuracy === "number" ? p.accuracy : null));
  const known = vals.filter((v): v is number => v != null);
  if (known.length === 0) {
    return <svg className="eval-spark" width={W} height={H} aria-hidden="true" />;
  }
  if (known.length === 1) {
    const cy = PAD + (1 - known[0]) * (H - 2 * PAD);
    return (
      <svg className="eval-spark" width={W} height={H} aria-hidden="true">
        <circle cx={W - PAD} cy={cy} r={2} className="eval-spark-dot" />
      </svg>
    );
  }
  const n = vals.length;
  const x = (i: number) => PAD + (i / (n - 1)) * (W - 2 * PAD);
  const y = (v: number) => PAD + (1 - v) * (H - 2 * PAD);
  const pts = vals
    .map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`))
    .filter((p): p is string => p != null)
    .join(" ");
  let lastIdx = -1;
  let lastVal: number | null = null;
  vals.forEach((v, i) => {
    if (v != null) {
      lastIdx = i;
      lastVal = v;
    }
  });
  return (
    <svg className="eval-spark" width={W} height={H} aria-hidden="true">
      <polyline
        className={"eval-spark-line" + (regressed ? " fail" : "")}
        points={pts}
        fill="none"
      />
      {lastVal != null && (
        <circle
          cx={x(lastIdx)}
          cy={y(lastVal)}
          r={2}
          className={"eval-spark-dot" + (regressed ? " fail" : "")}
        />
      )}
    </svg>
  );
}
