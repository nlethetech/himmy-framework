import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  api,
  type RunListResponse,
  type RunSummary,
  type RunAnalytics,
} from "../lib/api";
import { Topbar, Page, Loading, ErrorState } from "../components/Page";
import { RefreshIcon } from "../components/icons";
import { AnalyticsPanel, fmtTokens, fmtUsd } from "../components/Usage";
import { relativeTime, duration, statusClass, statusLabel } from "../lib/format";

export default function Activity() {
  const [data, setData] = useState<RunListResponse | null>(null);
  const [analytics, setAnalytics] = useState<RunAnalytics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const nav = useNavigate();

  const load = () => {
    setLoading(true);
    setError(null);
    api
      .get<RunListResponse>("/runs?limit=100")
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
    api
      .get<RunAnalytics>("/runs/analytics")
      .then(setAnalytics)
      .catch(() => setAnalytics(null));
  };
  useEffect(load, []);

  return (
    <>
      <Topbar
        title="Activity"
        sub={
          data ? `${data.total} run${data.total === 1 ? "" : "s"}` : "runs & analytics"
        }
        actions={
          <button className="btn" onClick={load}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      <Page>
        {loading && !data ? (
          <Loading label="Loading activity…" />
        ) : error ? (
          <ErrorState message={error} />
        ) : data && data.items.length === 0 ? (
          <div className="card">
            <div className="empty">
              No activity yet. Head to <b>Chat</b> or use a quick action on{" "}
              <b>Home</b> — every run shows up here with its full trace.
            </div>
          </div>
        ) : data ? (
          <>
            {analytics && <AnalyticsPanel a={analytics} />}
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Runs</span>
                <span>{data.total} total</span>
              </div>
              {data.items.map((r: RunSummary) => {
                const tok = (r.input_tokens || 0) + (r.output_tokens || 0);
                const meta = [
                  r.agent_name ?? "agent",
                  duration(r.duration_ms),
                  tok > 0 ? fmtTokens(tok) : null,
                  (r.cost || 0) > 0 ? fmtUsd(r.cost) : null,
                  r.tool_count > 0 ? `${r.tool_count} tools` : null,
                  relativeTime(r.created_at),
                ]
                  .filter(Boolean)
                  .join(" · ");
                return (
                  <div
                    className="run-line"
                    role="button"
                    tabIndex={0}
                    key={r.id}
                    onClick={() => nav(`/activity/${r.id}`)}
                    style={{ cursor: "pointer" }}
                  >
                    <span className="run-line-prompt">
                      {r.prompt || "(no prompt)"}
                    </span>
                    {statusClass(r.status) !== "ok" && (
                      <span className={"pill " + statusClass(r.status)}>
                        {statusLabel(r.status)}
                      </span>
                    )}
                    <span className="run-line-meta mono">{meta}</span>
                  </div>
                );
              })}
            </section>
          </>
        ) : null}
      </Page>
    </>
  );
}
