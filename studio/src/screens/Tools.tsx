import { useEffect, useState } from "react";
import { api, type PackInfo, type SkillInfo } from "../lib/api";
import { Topbar, Page, Loading } from "../components/Page";

export default function Tools() {
  const [packs, setPacks] = useState<PackInfo[] | null>(null);
  const [skills, setSkills] = useState<SkillInfo[]>([]);

  useEffect(() => {
    api.get<PackInfo[]>("/tools").then(setPacks).catch(() => setPacks([]));
    api.get<SkillInfo[]>("/skills").then(setSkills).catch(() => {});
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
          </>
        )}
      </Page>
    </>
  );
}
