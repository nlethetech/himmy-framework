import { useEffect, useState } from "react";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import {
  getLearning,
  setOverride,
  setTrustFeedback,
  type LearningReport,
  type LearningTool,
  type OverrideAction,
} from "../lib/learningApi";
import "../styles/learning.css";

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

/** Map a tool to a status tier shared by the pill colour and the score-bar fill. */
function tierOf(t: LearningTool): { label: string; cls: "ok" | "warn" | "err" | "dim" } {
  if (!t.has_min_samples)
    return { label: `learning ${t.completed + t.failed}/3`, cls: "dim" };
  if (t.flaky) return { label: "flaky · demoted", cls: "err" };
  if (t.failed > 0) return { label: `watch · ${t.failed} failed`, cls: "warn" };
  return { label: "reliable", cls: "ok" };
}

export default function Learning() {
  const [report, setReport] = useState<LearningReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = () => {
    setError(null);
    getLearning()
      .then(setReport)
      .catch((e) => setError(String(e?.message || e)));
  };
  useEffect(load, []);

  // Apply an override (pin/mute/reset/clear) then adopt the refreshed report the
  // server returns — so the table reflects exactly what the runtime will honour.
  const applyOverride = (tool: string, action: OverrideAction) => {
    setBusy(true);
    setError(null);
    setOverride(tool, action)
      .then(setReport)
      .catch((e) => setError(String(e?.message || e)))
      .finally(() => setBusy(false));
  };

  const toggleTrust = (enabled: boolean) => {
    setBusy(true);
    setError(null);
    setTrustFeedback(enabled)
      .then(setReport)
      .catch((e) => setError(String(e?.message || e)))
      .finally(() => setBusy(false));
  };

  const tools = report?.tools ?? [];
  // Muted (ignored) tools leave the flaky bucket — they are de-emphasised and out
  // of auto-learning, so they never count toward "needs attention".
  const needsAttention = tools.filter(
    (t) =>
      t.override !== "ignored" &&
      (t.flaky || (t.has_min_samples && t.failed > 0)),
  );
  const shown = showAll ? tools : needsAttention;
  const hidden = tools.length - shown.length;

  return (
    <>
      <Topbar
        title="Learning"
        sub="what Himmy learned about its tools — and the reorders it applied"
      />
      <div className="home-col">
        {error && (
          <div className="rt-error mono">
            {error}{" "}
            <button className="rt-line-act" onClick={load}>
              retry
            </button>
          </div>
        )}

        {!report && !error && <div className="dimtext">Loading…</div>}

        {report && !report.has_data && (
          <EmptyState icon="✦" title="Nothing learned yet">
            Himmy learns as an agent with <code>self_learning: true</code> runs and
            calls tools. Come back after a few runs to see which tools it trusts —
            and which it has quietly demoted.
          </EmptyState>
        )}

        {report && report.has_data && (
          <>
            <div className="learn-summary mono">
              {tools.length} tool(s) tracked ·{" "}
              <span className={report.flaky_count ? "learn-flaky" : "learn-ok"}>
                {report.flaky_count} flaky
              </span>{" "}
              <span className="dimtext">(below {pct(report.floor)} caution floor)</span>
              {report.outcome_weight > 0 && (
                <span className="dimtext">
                  {" · "}answer-quality blended @ {report.outcome_weight.toFixed(2)}
                </span>
              )}
            </div>

            <div className="learn-trust">
              <label className="learn-toggle">
                <input
                  type="checkbox"
                  checked={report.trust_feedback}
                  disabled={busy}
                  onChange={(e) => toggleTrust(e.target.checked)}
                />
                <span>Blend feedback into reputation</span>
              </label>
              <span className="learn-trust-help dimtext">
                When on, your 👍/👎 on runs nudges each tool's score. Off keeps
                reputation purely behavioural (Sprint-1 default).
              </span>
            </div>

            {shown.map((t) => {
              const s = tierOf(t);
              const pinned = t.override === "pinned";
              const ignored = t.override === "ignored";
              const reset = t.override === "reset";
              return (
                <div
                  key={t.tool_name}
                  className={`learn-row${ignored ? " learn-row-ignored" : ""}`}
                >
                  <div className="learn-name mono" title={t.tool_name}>
                    {t.tool_name}
                  </div>
                  <div className="learn-bar">
                    <div
                      className={`learn-bar-fill ${s.cls}`}
                      style={{ width: pct(t.score) }}
                    />
                  </div>
                  <div className="learn-score mono">{pct(t.score)}</div>
                  <div className="learn-counts mono dimtext">
                    {t.completed}✓ {t.failed}✗
                  </div>
                  {pinned ? (
                    <span className="chip chip-pinned">pinned</span>
                  ) : ignored ? (
                    <span className="chip chip-ignored">ignored</span>
                  ) : reset ? (
                    <span className="chip chip-reset">rebuilding</span>
                  ) : (
                    <span className={`pill ${s.cls}`}>{s.label}</span>
                  )}
                  <div className="learn-actions">
                    {!pinned && (
                      <button
                        className="learn-act"
                        disabled={busy}
                        title="Force-trust this tool — never demoted or flagged flaky"
                        onClick={() => applyOverride(t.tool_name, "pin")}
                      >
                        Pin
                      </button>
                    )}
                    {!ignored && (
                      <button
                        className="learn-act"
                        disabled={busy}
                        title="Exclude this tool from auto-learning"
                        onClick={() => applyOverride(t.tool_name, "mute")}
                      >
                        Mute
                      </button>
                    )}
                    <button
                      className="learn-act"
                      disabled={busy}
                      title="Forget accumulated reputation — rebuild from now"
                      onClick={() => applyOverride(t.tool_name, "reset")}
                    >
                      Reset
                    </button>
                    {t.override && (
                      <button
                        className="learn-act"
                        disabled={busy}
                        title="Remove the override — back to pure auto-learning"
                        onClick={() => applyOverride(t.tool_name, "clear")}
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>
              );
            })}

            {!showAll && needsAttention.length === 0 && (
              <div className="dimtext">
                All {tools.length} tracked tool(s) are reliable.{" "}
                <button className="rt-line-act" onClick={() => setShowAll(true)}>
                  show all
                </button>
              </div>
            )}
            {!showAll && needsAttention.length > 0 && hidden > 0 && (
              <button className="rt-line-act" onClick={() => setShowAll(true)}>
                show {hidden} reliable tool(s)
              </button>
            )}
            {showAll && (
              <button className="rt-line-act" onClick={() => setShowAll(false)}>
                show problems only
              </button>
            )}

            {report.recent_reorders.length > 0 && (
              <section className="home-sec" style={{ marginTop: 18 }}>
                <div className="home-sec-head">
                  <span>Recent reorders</span>
                </div>
                {report.recent_reorders.slice(0, 8).map((r, i) => (
                  <div key={i} className="learn-reorder mono">
                    <span className="dimtext">
                      {(r.when || "").slice(0, 19).replace("T", " ")}
                    </span>{" "}
                    demoted{" "}
                    <span className="learn-flaky">
                      {r.deprioritised_tools.join(", ") || "(none)"}
                    </span>
                  </div>
                ))}
              </section>
            )}
          </>
        )}
      </div>
    </>
  );
}
