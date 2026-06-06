import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, type RunDetailT, type IoCapture } from "../lib/api";
import { Topbar, Loading, ErrorState } from "../components/Page";
import { Markdown } from "../components/Markdown";
import {
  CognitionTrace,
  WorldLedger,
  GroundingPanel,
} from "../components/Cognition";
import { RunUsage, fmtTokens, fmtUsd } from "../components/Usage";
import { BackIcon } from "../components/icons";
import { relativeTime, duration, statusClass } from "../lib/format";

function IoInspector({ io }: { io: IoCapture }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="io-box">
      <button className="io-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} raw I/O
        {io.model && <span className="io-model">{io.model}</span>}
      </button>
      {open && (
        <div className="io-body">
          {io.tools.length > 0 && (
            <div className="io-row">
              <span className="io-key">bound tools</span>
              <span className="io-val mono">{io.tools.join(", ")}</span>
            </div>
          )}
          <div className="io-key">prompt sent</div>
          {io.messages.map((m, i) => (
            <div className="io-msg" key={i}>
              <span className={"io-role " + m.role}>{m.role}</span>
              <pre className="io-pre">{m.content}</pre>
            </div>
          ))}
          <div className="io-key">model output</div>
          <pre className="io-pre">{io.response_text || "(empty — only tool calls)"}</pre>
          {io.tool_calls.length > 0 && (
            <>
              <div className="io-key">parsed tool calls</div>
              <pre className="io-pre">{JSON.stringify(io.tool_calls, null, 2)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function RunDetail() {
  const { runId } = useParams();
  const [run, setRun] = useState<RunDetailT | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const nav = useNavigate();

  useEffect(() => {
    setLoading(true);
    api
      .get<RunDetailT>(`/runs/${runId}`)
      .then(setRun)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [runId]);

  return (
    <>
      <Topbar
        title={run ? run.agent_name ?? "Run" : "Run"}
        sub={run ? relativeTime(run.created_at) : "trace timeline"}
        actions={
          <button className="btn" onClick={() => nav("/runs")}>
            <BackIcon /> Back
          </button>
        }
      />
      {loading ? (
        <div className="run-status">
          <Loading label="Loading run…" />
        </div>
      ) : error ? (
        <div className="run-status">
          <ErrorState message={error} />
        </div>
      ) : run ? (
        <div className="run-detail">
          {/* Left: transcript + meta (scrolls independently) */}
          <div className="run-col">
              <div className="card card-pad">
                <div className="row wrap gap6">
                  <span className={"pill " + statusClass(run.status)}>
                    <span className="dot" />
                    {run.status}
                  </span>
                  <span className="pill dim">{run.provider ?? "auto"}</span>
                  {run.model && <span className="pill dim">{run.model}</span>}
                  <span className="pill dim">{duration(run.duration_ms)}</span>
                  {run.tool_count > 0 && (
                    <span className="pill dim">⚙ {run.tool_count} tools</span>
                  )}
                  {run.input_tokens + run.output_tokens > 0 && (
                    <span className="pill dim">
                      {fmtTokens(run.input_tokens + run.output_tokens)} tok
                    </span>
                  )}
                  {run.cost > 0 && (
                    <span className="pill dim">{fmtUsd(run.cost)}</span>
                  )}
                </div>
                {run.agent_path && (
                  <div className="dim mono mt8" style={{ fontSize: 12 }}>
                    {run.agent_path}
                  </div>
                )}
              </div>

              <div className="card">
                <div className="card-head">
                  <h2>Transcript</h2>
                </div>
                <div className="card-pad stack gap16">
                  {run.messages.map((m, i) => (
                    <div className={"msg " + (m.role === "user" ? "user" : "agent")} key={i}>
                      <div className="avatar">
                        {m.role === "user" ? "you" : "H"}
                      </div>
                      <div className="body">
                        <div className="who">{m.role}</div>
                        {m.role === "user" ? (
                          m.content
                        ) : (
                          <Markdown>{m.content}</Markdown>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {run.steps && run.steps.length > 0 && (
                <div className="card">
                  <div className="card-head">
                    <h2>Cognition</h2>
                    <span className="dim" style={{ fontSize: 12 }}>
                      think → act → observe
                    </span>
                  </div>
                  <div className="card-pad">
                    <CognitionTrace steps={run.steps} />
                    <WorldLedger steps={run.steps} />
                  </div>
                </div>
              )}

              {run.steps && run.steps.some((s) => s.kind === "grounding") && (
                <div className="card card-pad">
                  <span className="section-title">Grounding</span>
                  <GroundingPanel steps={run.steps} />
                </div>
              )}

              <RunUsage
                inputTokens={run.input_tokens}
                outputTokens={run.output_tokens}
                cost={run.cost}
                byModel={run.usage_by_model}
              />

              {run.tools.length > 0 && (
                <div className="card card-pad">
                  <span className="section-title">Tools used</span>
                  <div className="row wrap gap6">
                    {run.tools.map((t) => (
                      <span className="chip on" key={t}>
                        ⚙ {t}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

          {/* Right: timeline (scrolls independently) */}
          <div className="run-col">
            <div className="card">
              <div className="card-head">
                <h2>Timeline</h2>
                <span className="dim" style={{ fontSize: 12 }}>
                  {run.timeline.length} steps
                </span>
              </div>
              <div className="card-pad">
                <div className="timeline">
                  {run.timeline.map((s) => (
                    <div className="tl-item" key={s.seq}>
                      <div className="tl-dot" />
                      <div className="tl-body">
                        <div className="row spread">
                          <span className="tl-label">{s.label}</span>
                          {s.ts && (
                            <span className="tl-time">{relativeTime(s.ts)}</span>
                          )}
                        </div>
                        {s.detail && <div className="tl-detail">{s.detail}</div>}
                        {s.io && <IoInspector io={s.io} />}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
