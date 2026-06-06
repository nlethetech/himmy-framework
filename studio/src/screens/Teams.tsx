import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type TeamSummary } from "../lib/api";
import { Topbar, Page, Loading } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { OrchestrationGraph, type CogStep } from "../components/Cognition";
import { BuildIcon, ChatIcon } from "../components/icons";

export default function Teams() {
  const [teams, setTeams] = useState<TeamSummary[] | null>(null);
  const nav = useNavigate();

  useEffect(() => {
    api
      .get<TeamSummary[]>("/teams")
      .then(setTeams)
      .catch(() => setTeams([]));
  }, []);

  return (
    <>
      <Topbar
        title="Teams"
        sub="Multi-agent teams — a manager delegating to specialists"
      />
      <Page>
        {teams === null ? (
          <Loading label="Loading teams…" />
        ) : teams.length === 0 ? (
          <EmptyState icon={<BuildIcon />} title="No teams yet">
            Drop a <code>team.yaml</code> (an <code>entry</code> manager + a list of{" "}
            <code>members</code> with <code>delegates</code>) into your project and it
            shows up here. Run one from Chat.
          </EmptyState>
        ) : (
          <div className="team-grid">
            {teams.map((t) => {
              // Synthesize delegate "steps" so the graph draws every manager→worker
              // edge for the static view.
              const entryMember = t.members.find((m) => m.name === t.entry);
              const steps: CogStep[] = (entryMember?.delegates ?? []).map((w) => ({
                kind: "delegate" as const,
                worker: w,
              }));
              return (
                <div className="card card-pad team-card" key={t.path}>
                  <div className="row spread">
                    <div>
                      <div className="team-name">{t.name}</div>
                      <div className="dim mono" style={{ fontSize: 12 }}>
                        {t.path}
                      </div>
                    </div>
                    <button className="btn btn-primary" onClick={() => nav("/chat")}>
                      <ChatIcon /> Run in Chat
                    </button>
                  </div>

                  <OrchestrationGraph
                    members={t.members.map((m) => ({ name: m.name, role: m.role }))}
                    entry={t.entry}
                    steps={steps}
                    active={null}
                  />

                  <div className="team-members">
                    {t.members.map((m) => (
                      <div className="team-member" key={m.name}>
                        <span className="team-member-name">
                          {m.name}
                          {m.name === t.entry && (
                            <span className="team-tag">manager</span>
                          )}
                        </span>
                        <span className="team-member-role">{m.role ?? "specialist"}</span>
                        <span className="team-member-model mono">
                          {m.provider ?? "auto"}
                          {m.model && m.model !== "default" ? ` · ${m.model}` : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Page>
    </>
  );
}
