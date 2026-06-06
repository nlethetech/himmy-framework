import { useEffect, useState } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { api, getLineage, type RunSummary, type LineageView } from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { GlobeIcon } from "../components/icons";
import { fmtTokens, fmtUsd } from "../components/Usage";

export default function Lineage() {
  const [params, setParams] = useSearchParams();
  const runId = params.get("run") ?? "";
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [view, setView] = useState<LineageView | null>(null);

  useEffect(() => {
    api.get<{ items: RunSummary[] }>("/runs?limit=50").then((r) => setRuns(r.items));
  }, []);
  useEffect(() => {
    if (!runId) {
      setView(null);
      return;
    }
    getLineage(runId).then(setView).catch(() => setView(null));
  }, [runId]);

  return (
    <>
      <Topbar title="Lineage" sub="Trace an answer back to what produced it" />
      <Page>
        <div className="card card-pad" style={{ marginBottom: 16, maxWidth: 480 }}>
          <span className="field-label">Run</span>
          <select
            className="input"
            value={runId}
            onChange={(e) => setParams(e.target.value ? { run: e.target.value } : {})}
          >
            <option value="">Choose a run…</option>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.agent_name ?? "agent"} — {r.prompt.slice(0, 50) || "(no prompt)"}
              </option>
            ))}
          </select>
        </div>

        {!view ? (
          <EmptyState icon={<GlobeIcon />} title="Pick a run">
            Choose a run to see its provenance — the answer traced back through the
            agent, prompt, tools it called, evidence it grounded on, and any
            specialists it delegated to.
          </EmptyState>
        ) : (
          <div className="lin">
            <div className="lin-node answer">
              <div className="lin-kind">Answer</div>
              <div className="lin-label">{view.answer || "(no answer)"}</div>
              <div className="lin-meta">
                {fmtTokens(view.tokens)} tok
                {view.cost > 0 ? ` · ${fmtUsd(view.cost)}` : ""}
              </div>
            </div>

            <div className="lin-branch">
              <div className="lin-rel">produced by</div>
              <div className="lin-node agent">
                <div className="lin-kind">Agent</div>
                <div className="lin-label">
                  {view.agent ?? "agent"}
                  <span className="dim"> · {view.provider ?? "auto"}</span>
                </div>
                {view.models.length > 0 && (
                  <div className="lin-meta mono">{view.models.join(", ")}</div>
                )}
              </div>
            </div>

            <LinGroup rel="answering" items={[view.prompt]} kind="Prompt" />

            {view.delegates.length > 0 && (
              <div className="lin-section">
                <div className="lin-rel">delegated to</div>
                {view.delegates.map((d, i) => (
                  <div className="lin-leaf" key={i}>
                    <span className="lin-leaf-name">{d.worker}</span>
                    {d.task && <span className="lin-leaf-detail">{d.task}</span>}
                  </div>
                ))}
              </div>
            )}

            {view.tools.length > 0 && (
              <div className="lin-section">
                <div className="lin-rel">using tools</div>
                {view.tools.map((t, i) => (
                  <div className="lin-leaf" key={i}>
                    <span className={"cog-intent " + (t.read_only === false ? "w" : "r")}>
                      {t.read_only === false ? "WRITE" : "READ"}
                    </span>
                    <span className="lin-leaf-name mono">{t.name}</span>
                    {t.result && (
                      <span className="lin-leaf-detail mono">{t.result.slice(0, 80)}</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {view.evidence.length > 0 && (
              <div className="lin-section">
                <div className="lin-rel">grounded on</div>
                {view.evidence.map((e, i) => (
                  <div className="lin-leaf" key={i}>
                    <span className="lin-ev-icon">
                      {e.source === "memory" ? "🧠" : "📚"}
                    </span>
                    <span className="lin-leaf-detail">{e.text}</span>
                    {e.source_uri && (
                      <span className="cite-src mono">{e.source_uri.split("/").pop()}</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="mt8">
              <Link to={`/activity/${view.run_id}`} className="dim" style={{ fontSize: 13 }}>
                ↗ open the full run
              </Link>
            </div>
          </div>
        )}
      </Page>
    </>
  );
}

function LinGroup({ rel, items, kind }: { rel: string; items: string[]; kind: string }) {
  const real = items.filter(Boolean);
  if (real.length === 0) return null;
  return (
    <div className="lin-branch">
      <div className="lin-rel">{rel}</div>
      <div className="lin-node prompt">
        <div className="lin-kind">{kind}</div>
        <div className="lin-label">{real[0]}</div>
      </div>
    </div>
  );
}
