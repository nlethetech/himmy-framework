import { useEffect, useState } from "react";
import { api, type DoctorReport } from "../lib/api";
import { Topbar, Page, Loading, ErrorState } from "../components/Page";
import { CheckIcon, XIcon, RefreshIcon } from "../components/icons";

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
