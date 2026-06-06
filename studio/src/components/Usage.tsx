import type { RunAnalytics, ModelUsage } from "../lib/api";

// ---- formatters ---------------------------------------------------------

export function fmtTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return (n / 1000).toFixed(n < 10_000 ? 1 : 0) + "k";
  return (n / 1_000_000).toFixed(1) + "M";
}

export function fmtUsd(n: number): string {
  if (!n) return "$0";
  if (n < 0.01) return "$" + n.toFixed(4);
  if (n < 1) return "$" + n.toFixed(3);
  return "$" + n.toFixed(2);
}

export function fmtMs(ms?: number | null): string {
  if (ms == null || !isFinite(ms)) return "—";
  if (ms < 1000) return Math.round(ms) + "ms";
  return (ms / 1000).toFixed(ms < 10_000 ? 2 : 1) + "s";
}

// ---- live usage accumulator (built from "usage" SSE frames) -------------

export interface LiveUsage {
  input_tokens: number;
  output_tokens: number;
  cost: number;
  inferences: number;
  latency_ms: number; // cumulative inference latency
  by_model: Record<
    string,
    { input: number; output: number; cost: number; inferences: number }
  >;
}

export function emptyUsage(): LiveUsage {
  return {
    input_tokens: 0,
    output_tokens: 0,
    cost: 0,
    inferences: 0,
    latency_ms: 0,
    by_model: {},
  };
}

// Fold one "usage" frame into the running totals. The frame's `total_*` fields
// are authoritative for the run-level rollup; the per-frame delta drives the
// per-model split.
export function foldUsage(
  u: LiveUsage,
  f: {
    model: string;
    input_tokens: number;
    output_tokens: number;
    cost: number;
    latency_ms?: number | null;
    total_input_tokens: number;
    total_output_tokens: number;
    total_cost: number;
    inferences: number;
  },
): LiveUsage {
  const by_model = { ...u.by_model };
  const m = by_model[f.model] || { input: 0, output: 0, cost: 0, inferences: 0 };
  by_model[f.model] = {
    input: m.input + f.input_tokens,
    output: m.output + f.output_tokens,
    cost: m.cost + f.cost,
    inferences: m.inferences + 1,
  };
  return {
    input_tokens: f.total_input_tokens,
    output_tokens: f.total_output_tokens,
    cost: f.total_cost,
    inferences: f.inferences,
    latency_ms: u.latency_ms + (f.latency_ms || 0),
    by_model,
  };
}

// ---- live HUD (compact strip on an agent message) -----------------------

export function UsageHud({
  usage,
  live,
}: {
  usage: LiveUsage;
  live?: boolean;
}) {
  if (usage.inferences === 0) return null;
  const models = Object.entries(usage.by_model);
  const free = usage.cost === 0;
  return (
    <div className={"hud" + (live ? " live" : "")}>
      <span className="hud-item" title="tokens in → out">
        <span className="hud-ico">🪙</span>
        <span className="mono">
          {fmtTokens(usage.input_tokens)}
          <span className="hud-arrow">→</span>
          {fmtTokens(usage.output_tokens)}
        </span>
      </span>
      <span
        className={"hud-item" + (free ? "" : " cost")}
        title={free ? "local model — no API cost" : "estimated API cost"}
      >
        <span className="hud-ico">{free ? "🟢" : "💲"}</span>
        <span className="mono">{free ? "local" : fmtUsd(usage.cost)}</span>
      </span>
      <span className="hud-item" title="total inference latency">
        <span className="hud-ico">⚡</span>
        <span className="mono">{fmtMs(usage.latency_ms)}</span>
      </span>
      <span className="hud-item" title="model calls">
        <span className="mono hud-dim">
          {usage.inferences} call{usage.inferences === 1 ? "" : "s"}
        </span>
      </span>
      {models.length > 1 &&
        models.map(([name, m]) => (
          <span className="hud-model mono" key={name} title={`${name}`}>
            {name.split(":").pop()} · {fmtTokens(m.input + m.output)}
          </span>
        ))}
    </div>
  );
}

// ---- persisted per-run usage (RunDetail) --------------------------------

export function RunUsage({
  inputTokens,
  outputTokens,
  cost,
  byModel,
}: {
  inputTokens: number;
  outputTokens: number;
  cost: number;
  byModel: ModelUsage[];
}) {
  if (inputTokens === 0 && outputTokens === 0 && cost === 0) return null;
  return (
    <div className="card card-pad">
      <span className="section-title">Cost &amp; tokens</span>
      <div className="usage-grid">
        <div className="usage-cell">
          <div className="usage-num mono">{fmtTokens(inputTokens)}</div>
          <div className="usage-lab">input</div>
        </div>
        <div className="usage-cell">
          <div className="usage-num mono">{fmtTokens(outputTokens)}</div>
          <div className="usage-lab">output</div>
        </div>
        <div className="usage-cell">
          <div className="usage-num mono">
            {cost === 0 ? "local" : fmtUsd(cost)}
          </div>
          <div className="usage-lab">cost</div>
        </div>
      </div>
      {byModel.length > 0 && (
        <div className="usage-models">
          {byModel.map((m) => (
            <div className="usage-model-row" key={m.model}>
              <span className="mono usage-model-name">{m.model}</span>
              <span className="mono usage-model-tok">
                {fmtTokens(m.input_tokens)} → {fmtTokens(m.output_tokens)}
              </span>
              <span className="mono usage-model-cost">
                {m.cost === 0 ? "—" : fmtUsd(m.cost)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- analytics dashboard (Runs page header) -----------------------------

export function AnalyticsPanel({ a }: { a: RunAnalytics }) {
  if (a.total_runs === 0) return null;
  const maxDayCost = Math.max(...a.by_day.map((d) => d.cost), 0);
  const maxDayRuns = Math.max(...a.by_day.map((d) => d.runs), 1);
  return (
    <div className="analytics">
      <div className="kpis">
        <Kpi label="runs" value={String(a.total_runs)} />
        <Kpi
          label="success"
          value={Math.round(a.success_rate * 100) + "%"}
          tone={a.success_rate >= 0.9 ? "ok" : a.success_rate >= 0.6 ? "warn" : "err"}
        />
        <Kpi
          label="cost"
          value={a.total_cost === 0 ? "local" : fmtUsd(a.total_cost)}
        />
        <Kpi
          label="tokens"
          value={fmtTokens(a.total_input_tokens + a.total_output_tokens)}
        />
        <Kpi label="p50" value={fmtMs(a.p50_latency_ms)} />
        <Kpi label="p95" value={fmtMs(a.p95_latency_ms)} />
      </div>

      <div className="analytics-cols">
        {a.by_day.length > 0 && (
          <div className="analytics-card">
            <div className="analytics-head">Activity</div>
            <div className="spark">
              {a.by_day.map((d) => {
                const h =
                  maxDayCost > 0
                    ? Math.max(6, (d.cost / maxDayCost) * 100)
                    : Math.max(6, (d.runs / maxDayRuns) * 100);
                return (
                  <div
                    className="spark-bar"
                    key={d.day}
                    style={{ height: h + "%" }}
                    title={`${d.day}: ${d.runs} runs · ${
                      d.cost === 0 ? "local" : fmtUsd(d.cost)
                    } · ${fmtTokens(d.tokens)} tok`}
                  />
                );
              })}
            </div>
            <div className="spark-axis mono">
              <span>{a.by_day[0]?.day.slice(5)}</span>
              <span>{a.by_day[a.by_day.length - 1]?.day.slice(5)}</span>
            </div>
          </div>
        )}

        {a.by_model.length > 0 && (
          <div className="analytics-card">
            <div className="analytics-head">By model</div>
            <div className="model-table">
              {a.by_model.map((m) => (
                <div className="model-row" key={m.model}>
                  <span className="mono model-name">{m.model}</span>
                  <span className="mono model-runs">{m.runs}×</span>
                  <span className="mono model-tok">
                    {fmtTokens(m.input_tokens + m.output_tokens)}
                  </span>
                  <span className="mono model-cost">
                    {m.cost === 0 ? "—" : fmtUsd(m.cost)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Kpi({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "warn" | "err";
}) {
  return (
    <div className={"kpi" + (tone ? " " + tone : "")}>
      <div className="kpi-val mono">{value}</div>
      <div className="kpi-lab">{label}</div>
    </div>
  );
}
