import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { api, type RunDetailT, type IoCapture } from "../lib/api";
import { Topbar, Loading, ErrorState } from "../components/Page";
import { Markdown } from "../components/Markdown";
import {
  CognitionTrace,
  WorldLedger,
  GroundingPanel,
  SafetyPanel,
} from "../components/Cognition";
import { RunUsage, fmtTokens, fmtUsd } from "../components/Usage";
import { BackIcon } from "../components/icons";
import { relativeTime, duration, statusClass } from "../lib/format";

/* Run detail as a single centered ledger column: a meta line between rules,
   then TRANSCRIPT / ACTIVITY / TIMELINE sections. No boxes, no dots. */

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
          {(io.tools || []).length > 0 && (
            <div className="io-row">
              <span className="io-key">bound tools</span>
              <span className="io-val mono">{(io.tools || []).join(", ")}</span>
            </div>
          )}
          <div className="io-key">prompt sent</div>
          {(io.messages || []).map((m, i) => (
            <div className="io-msg" key={i}>
              <span className={"io-role " + m.role}>{m.role}</span>
              <pre className="io-pre">{m.content}</pre>
            </div>
          ))}
          <div className="io-key">model output</div>
          <pre className="io-pre">{io.response_text || "(empty — only tool calls)"}</pre>
          {(io.tool_calls || []).length > 0 && (
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

  const meta = run
    ? [
        run.provider ?? "auto",
        run.model,
        duration(run.duration_ms),
        run.tool_count > 0 ? `${run.tool_count} tools` : null,
        (run.input_tokens || 0) + (run.output_tokens || 0) > 0
          ? `${fmtTokens((run.input_tokens || 0) + (run.output_tokens || 0))} tok`
          : null,
        (run.cost || 0) > 0 ? fmtUsd(run.cost) : null,
        relativeTime(run.created_at),
      ]
        .filter(Boolean)
        .join(" · ")
    : "";

  return (
    <>
      <Topbar
        title={run ? (run.agent_name ?? "Run") : "Run"}
        sub={run ? relativeTime(run.created_at) : "trace timeline"}
        actions={
          <button className="btn" onClick={() => nav("/activity")}>
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
        <div className="run-page">
          {/* meta line between rules — the run's key facts */}
          <div className="run-meta">
            <div className="run-meta-line">
              {statusClass(run.status) !== "ok" && (
                <span className={"pill " + statusClass(run.status)}>
                  {run.status}
                </span>
              )}
              <span className="mono run-meta-facts">{meta}</span>
              <Link
                className="mono run-meta-link"
                title="How the agent reached this conclusion"
                to={
                  run.thread_id
                    ? `/advanced/lineage?entity=${encodeURIComponent(run.thread_id)}&run=${encodeURIComponent(run.id)}`
                    : `/advanced/lineage?run=${encodeURIComponent(run.id)}`
                }
              >
                lineage →
              </Link>
            </div>
            {run.agent_path && (
              <div className="run-meta-path mono">{run.agent_path}</div>
            )}
          </div>

          <section className="home-sec">
            <div className="home-sec-head">
              <span>Transcript</span>
            </div>
            {(run.messages || []).map((m, i) => (
              <div
                className={"msg " + (m.role === "user" ? "user" : "agent")}
                key={i}
              >
                <div className="body">
                  <div className="who">{m.role === "user" ? "You" : m.role}</div>
                  {m.role === "user" ? m.content : <Markdown>{m.content}</Markdown>}
                </div>
              </div>
            ))}
          </section>

          {run.steps && run.steps.length > 0 && (
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Activity</span>
                <span>think → act → observe</span>
              </div>
              <div className="run-sec-body">
                <CognitionTrace steps={run.steps} />
                <WorldLedger steps={run.steps} />
              </div>
            </section>
          )}

          {run.steps && run.steps.some((s) => s.kind === "safety") && (
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Guardrails</span>
              </div>
              <div className="run-sec-body">
                <SafetyPanel steps={run.steps} />
              </div>
            </section>
          )}

          {run.steps && run.steps.some((s) => s.kind === "grounding") && (
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Grounding</span>
              </div>
              <div className="run-sec-body">
                <GroundingPanel steps={run.steps} />
              </div>
            </section>
          )}

          <section className="home-sec">
            <div className="home-sec-head">
              <span>Timeline</span>
              <span>{(run.timeline || []).length} steps</span>
            </div>
            {(run.timeline || []).map((s, i) => (
              <div className="tl-row" key={s.seq}>
                <span className="tl-idx mono">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className="tl-main">
                  <div className="tl-head">
                    <span className="tl-label">{s.label}</span>
                    {s.ts && <span className="tl-time mono">{relativeTime(s.ts)}</span>}
                  </div>
                  {s.detail && <div className="tl-detail">{s.detail}</div>}
                  {s.io && <IoInspector io={s.io} />}
                </div>
              </div>
            ))}
          </section>

          {((run.usage_by_model && run.usage_by_model.length > 0) ||
            (run.input_tokens || 0) + (run.output_tokens || 0) > 0) && (
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Usage</span>
              </div>
              <div className="run-sec-body">
                <RunUsage
                  inputTokens={run.input_tokens}
                  outputTokens={run.output_tokens}
                  cost={run.cost}
                  byModel={run.usage_by_model}
                />
              </div>
            </section>
          )}

          {(run.tools || []).length > 0 && (
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Tools used</span>
              </div>
              <div className="run-tools">
                {(run.tools || []).map((t) => (
                  <span className="chip mono" key={t}>
                    {t}
                  </span>
                ))}
              </div>
            </section>
          )}
        </div>
      ) : null}
    </>
  );
}
