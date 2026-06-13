import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type PackInfo, type SkillInfo } from "../lib/api";
import { Topbar, Page, Loading } from "../components/Page";
import {
  getGuardrails,
  type GuardrailInfo,
  type GuardrailFiring,
} from "../lib/guardrailsApi";
import { relativeTime } from "../lib/format";

const whenAgo = (iso: string) => {
  try {
    return relativeTime(iso);
  } catch {
    return iso.slice(0, 10);
  }
};

export default function Tools() {
  const [packs, setPacks] = useState<PackInfo[] | null>(null);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [guardrails, setGuardrails] = useState<GuardrailInfo[]>([]);
  const [firings, setFirings] = useState<GuardrailFiring[]>([]);

  useEffect(() => {
    api.get<PackInfo[]>("/tools").then(setPacks).catch(() => setPacks([]));
    api.get<SkillInfo[]>("/skills").then(setSkills).catch(() => {});
    getGuardrails()
      .then((g) => {
        setGuardrails(g.builtins);
        setFirings(g.firings);
      })
      .catch(() => {});
  }, []);

  return (
    <>
      <Topbar title="Tools" sub="The built-in capabilities your agents can switch on" />
      <Page>
        {packs === null ? (
          <Loading label="Loading tools…" />
        ) : (
          <>
            <div className="section-title">Tool packs</div>
            <div className="tools-grid">
              {packs.map((p) => (
                <div className="tools-card" key={p.name}>
                  <div className="tools-name mono">{p.name}</div>
                  <div className="tools-desc">{p.description}</div>
                  <div className="tools-chips">
                    {p.tools.map((t) => (
                      <span className="chip mono" key={t}>
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {skills.length > 0 && (
              <>
                <div className="section-title" style={{ marginTop: 24 }}>
                  Skills <span className="dim">— tools + know-how, bundled</span>
                </div>
                <div className="tools-grid">
                  {skills.map((s) => (
                    <div className="tools-card" key={s.name}>
                      <div className="tools-name mono">
                        {s.name}
                        {s.builtin && <span className="tools-builtin">built-in</span>}
                      </div>
                      <div className="tools-desc">{s.description}</div>
                      {s.when_to_use && (
                        <div className="tools-when dim">{s.when_to_use}</div>
                      )}
                      <div className="tools-chips">
                        {s.tool_packs.map((tp) => (
                          <span className="chip mono" key={tp}>
                            {tp}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {guardrails.length > 0 && (
              <>
                <div className="section-title" style={{ marginTop: 24 }}>
                  Guardrails{" "}
                  <span className="dim">— the safety layer your agents can switch on</span>
                </div>
                <div className="tools-grid">
                  {guardrails.map((g) => (
                    <div className="tools-card" key={g.name}>
                      <div className="tools-name mono">
                        <span className="tools-shield">🛡</span> {g.name}
                      </div>
                      <div className="tools-desc">{g.description}</div>
                    </div>
                  ))}
                </div>

                <div className="section-title" style={{ marginTop: 24 }}>
                  Recent firings{" "}
                  <span className="dim">
                    — guardrails that fired in past runs
                  </span>
                </div>
                {firings.length === 0 ? (
                  <div className="tools-when dim">
                    No guardrail has fired yet — a firing appears here (and as a
                    🛡 line in the run trace) the moment one redacts or blocks.
                  </div>
                ) : (
                  <div className="guardrail-firings">
                    {firings.map((f) => (
                      <Link
                        className="guardrail-firing"
                        key={`${f.run_id}-${f.seq}`}
                        to={`/activity/${f.run_id}`}
                        title={`Open run ${f.run_id}`}
                      >
                        <span className="guardrail-firing-shield">🛡</span>
                        <span className="guardrail-firing-name mono">{f.name}</span>
                        <span
                          className={
                            "guardrail-firing-action" +
                            (f.blocked ? " blocked" : "")
                          }
                        >
                          {f.action}
                        </span>
                        {f.detail && (
                          <span className="guardrail-firing-detail">
                            {f.detail}
                          </span>
                        )}
                        {f.stage && (
                          <span className="guardrail-firing-stage mono">
                            {f.stage}
                          </span>
                        )}
                        <span className="guardrail-firing-meta mono">
                          {f.agent_name ? `${f.agent_name} · ` : ""}
                          {whenAgo(f.created_at)}
                        </span>
                      </Link>
                    ))}
                  </div>
                )}
              </>
            )}
          </>
        )}
      </Page>
    </>
  );
}
