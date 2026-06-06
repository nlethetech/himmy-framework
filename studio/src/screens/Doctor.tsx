import { useEffect, useState } from "react";
import {
  api,
  type DoctorReport,
  type BenchmarkEntry,
  type ProbeResult,
} from "../lib/api";
import { Topbar, Page, Loading, ErrorState } from "../components/Page";
import { CheckIcon, XIcon, RefreshIcon } from "../components/icons";

function pct(x: number | null): string {
  return x == null ? "—" : `${Math.round(x * 100)}%`;
}
function accClass(x: number | null): "ok" | "warn" | "err" | "dim" {
  if (x == null) return "dim";
  if (x >= 0.8) return "ok";
  if (x >= 0.55) return "warn";
  return "err";
}

function Benchmarks() {
  const [entries, setEntries] = useState<BenchmarkEntry[]>([]);
  const [running, setRunning] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const load = () =>
    api
      .get<{ entries: BenchmarkEntry[] }>("/benchmarks")
      .then((r) => setEntries(r.entries))
      .catch(() => {});
  useEffect(() => {
    load();
  }, []);

  const probe = async () => {
    setRunning(true);
    setNote(null);
    try {
      const r = await api.post<ProbeResult>("/benchmarks/probe", {});
      if (!r.ran) setNote(r.reason ?? "No local models detected.");
      else await load();
    } catch (e) {
      setNote(String(e instanceof Error ? e.message : e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="card">
      <div className="card-head">
        <h2>Model reliability</h2>
        <button className="btn" onClick={probe} disabled={running}>
          {running ? <span className="spinner" /> : <RefreshIcon />}
          {running ? "Running…" : "Run quick check"}
        </button>
      </div>
      {running && (
        <div className="statrow dim" style={{ fontSize: 12.5 }}>
          Running a small tool-focused suite against your local models — this calls the
          models for real, so it can take a minute.
        </div>
      )}
      {note && (
        <div className="statrow" style={{ color: "var(--warn)", fontSize: 13 }}>
          {note}
        </div>
      )}
      {entries.length === 0 && !running ? (
        <div className="statrow dim" style={{ fontSize: 13 }}>
          No benchmarks yet. Run a quick check, or{" "}
          <code>himmy bench --models ollama:qwen2.5:3b-instruct</code>.
        </div>
      ) : (
        <>
          <div className="statrow" style={{ color: "var(--text-dim)", fontSize: 11.5, textTransform: "uppercase", letterSpacing: 0.5 }}>
            <span style={{ flex: 1 }}>model</span>
            <span style={{ width: 90, textAlign: "right" }}>tool-call</span>
            <span style={{ width: 70, textAlign: "right" }}>accuracy</span>
            <span style={{ width: 70, textAlign: "right" }}>p50</span>
          </div>
          {entries.map((e) => (
            <div className="statrow" key={`${e.model}:${e.suite}`}>
              <span className="name mono" style={{ flex: 1, fontSize: 12.5 }}>
                {e.model}
              </span>
              <span style={{ width: 90, textAlign: "right" }}>
                <span className={"pill " + accClass(e.tool_call_accuracy)}>
                  {pct(e.tool_call_accuracy)}
                </span>
              </span>
              <span className="mono" style={{ width: 70, textAlign: "right", fontSize: 12.5 }}>
                {pct(e.accuracy)}
              </span>
              <span className="dim mono" style={{ width: 70, textAlign: "right", fontSize: 12 }}>
                {e.p50_latency_s.toFixed(1)}s
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function Status({ ok }: { ok: boolean }) {
  return ok ? (
    <span className="pill ok">
      <CheckIcon /> ready
    </span>
  ) : (
    <span className="pill dim">
      <XIcon /> absent
    </span>
  );
}

export default function Doctor() {
  const [report, setReport] = useState<DoctorReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    setError(null);
    api
      .get<DoctorReport>("/doctor")
      .then(setReport)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const nextStepTone =
    report?.next_step?.kind === "run" ? "ok" : "accent";

  return (
    <>
      <Topbar
        title="Doctor"
        sub="environment & setup"
        actions={
          <button className="btn" onClick={load}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      <Page>
        {loading && !report ? (
          <Loading label="Probing environment…" />
        ) : error ? (
          <ErrorState message={error} />
        ) : report ? (
          <div className="stack gap16">
            {report.next_step && (
              <div className={"banner " + (nextStepTone === "ok" ? "" : "accent")}>
                <div className="b-ico">→</div>
                <div>
                  <div className="b-title">Next step</div>
                  <div
                    className="b-msg"
                    dangerouslySetInnerHTML={{
                      __html: renderInlineCode(report.next_step.message),
                    }}
                  />
                </div>
              </div>
            )}

            <div className="grid grid-2">
              <div className="card">
                <div className="card-head">
                  <h2>Local model providers</h2>
                  <span className={"pill " + (report.has_real_model ? "ok" : "warn")}>
                    {report.has_real_model ? "available" : "stub only"}
                  </span>
                </div>
                {report.providers.map((p) => (
                  <div className="statrow" key={p.name}>
                    <span>
                      <span className="name">{p.name}</span>
                      {p.path && <span className="path">{p.path}</span>}
                    </span>
                    <Status ok={p.ok} />
                  </div>
                ))}
                {report.keys.map((k) => (
                  <div className="statrow" key={k.name}>
                    <span className="name mono" style={{ fontSize: 12.5 }}>
                      {k.name}
                    </span>
                    <Status ok={k.present} />
                  </div>
                ))}
              </div>

              <div className="card">
                <div className="card-head">
                  <h2>Optional extras</h2>
                </div>
                {report.extras.map((e) => (
                  <div className="statrow" key={e.module}>
                    <span className="name">{e.label}</span>
                    <Status ok={e.ok} />
                  </div>
                ))}
              </div>
            </div>

            <Benchmarks />

            <div className="card card-pad">
              <div className="row spread">
                <div className="stack gap6">
                  <span className="section-title" style={{ margin: 0 }}>
                    Runtime
                  </span>
                  <span className="muted">
                    Python <code>{report.python}</code> · himmy{" "}
                    <code>{report.version}</code>
                  </span>
                </div>
                <div className="stack gap6" style={{ textAlign: "right" }}>
                  <span className="section-title" style={{ margin: 0 }}>
                    Project config
                  </span>
                  <span className="muted mono" style={{ fontSize: 12 }}>
                    {report.project_config ?? "(env + defaults)"}
                  </span>
                </div>
              </div>
              <div className="mt16">
                <span className="section-title">Guardrails available</span>
                <div className="row wrap gap6">
                  {report.guardrails.map((g) => (
                    <span className="chip" key={g}>
                      {g}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </Page>
    </>
  );
}

// Render `inline code` spans from the next-step message (the only markup we allow).
function renderInlineCode(text: string): string {
  const esc = (s: string) =>
    s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  return esc(text).replace(/`([^`]+)`/g, "<code>$1</code>");
}
