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
import { relativeTime, duration, statusClass } from "../lib/format";

export default function Runs() {
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
        title="Runs"
        sub={data ? `${data.total} run${data.total === 1 ? "" : "s"}` : "past runs & traces"}
        actions={
          <button className="btn" onClick={load}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      <Page>
        {loading && !data ? (
          <Loading label="Loading runs…" />
        ) : error ? (
          <ErrorState message={error} />
        ) : data && data.items.length === 0 ? (
          <div className="card">
            <div className="empty">
              No runs yet. Head to <b>Chat</b> and talk to an agent — every run
              shows up here with its full trace.
            </div>
          </div>
        ) : data ? (
          <>
            {analytics && <AnalyticsPanel a={analytics} />}
            <div className="card">
              {data.items.map((r: RunSummary) => (
                <div
                  className="list-row"
                  key={r.id}
                  onClick={() => nav(`/runs/${r.id}`)}
                >
                  <div className="lead">
                    <div className="row gap10">
                      <span className="title">{r.agent_name ?? "agent"}</span>
                      <span className={"pill " + statusClass(r.status)}>
                        <span className="dot" />
                        {r.status}
                      </span>
                      {r.tool_count > 0 && (
                        <span className="pill dim">⚙ {r.tool_count}</span>
                      )}
                    </div>
                    <span className="meta">{r.prompt || "(no prompt)"}</span>
                  </div>
                  <div className="stack gap6" style={{ textAlign: "right" }}>
                    <span className="meta">{relativeTime(r.created_at)}</span>
                    <span className="meta">
                      {r.provider ?? "auto"} · {duration(r.duration_ms)}
                      {r.input_tokens + r.output_tokens > 0 && (
                        <>
                          {" · "}
                          <span className="mono">
                            {fmtTokens(r.input_tokens + r.output_tokens)} tok
                          </span>
                          {r.cost > 0 && (
                            <>
                              {" · "}
                              <span className="mono">{fmtUsd(r.cost)}</span>
                            </>
                          )}
                        </>
                      )}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </Page>
    </>
  );
}
