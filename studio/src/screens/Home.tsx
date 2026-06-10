import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  api,
  listConnections,
  type RunSummary,
  type ConnectionStatus,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { ComposeModal, type ComposeMode } from "../components/ComposeModal";
import { MailIcon, TelegramIcon, GlobeIcon, ChatIcon } from "../components/icons";
import { relativeTime, statusClass } from "../lib/format";

export default function Home() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [conns, setConns] = useState<ConnectionStatus[]>([]);
  const [compose, setCompose] = useState<ComposeMode | null>(null);
  const nav = useNavigate();

  useEffect(() => {
    api
      .get<{ items: RunSummary[] }>("/runs?limit=6")
      .then((r) => setRuns(r.items))
      .catch(() => setRuns([]));
    listConnections()
      .then(setConns)
      .catch(() => setConns([]));
  }, []);

  const byType = (t: string) => conns.find((c) => c.type === t);
  const needs = (t: string) => conns.length > 0 && !byType(t)?.configured;

  const tiles = [
    {
      key: "email",
      label: "Send email",
      hue: "email",
      Icon: MailIcon,
      onClick: () => setCompose("email"),
      missing: needs("email"),
    },
    {
      key: "telegram",
      label: "Message Telegram",
      hue: "telegram",
      Icon: TelegramIcon,
      onClick: () => setCompose("telegram"),
      missing: needs("telegram"),
    },
    {
      key: "research",
      label: "Research",
      hue: "web",
      Icon: GlobeIcon,
      onClick: () => nav("/chat"),
      missing: false,
    },
    {
      key: "ask",
      label: "Ask / Run",
      hue: "memory",
      Icon: ChatIcon,
      onClick: () => nav("/chat"),
      missing: false,
    },
  ];

  const now = new Date();
  const hour = now.getHours();
  const greeting =
    hour < 5 ? "Working late" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  const dateline = now
    .toLocaleDateString("en-GB", {
      weekday: "long",
      day: "numeric",
      month: "long",
      year: "numeric",
    })
    .toUpperCase();

  return (
    <>
      <Topbar title="Home" sub="Your workspace" />
      <Page>
        <div className="masthead">
          <div className="masthead-date">{dateline}</div>
          <h2 className="masthead-greeting">{greeting}.</h2>
        </div>

        <div className="quick-row">
          {tiles.map((t) => (
            <button className="qtile" key={t.key} onClick={t.onClick}>
              <t.Icon />
              <span className="qtile-label">{t.label}</span>
              {t.missing && (
                <span className="qtile-warn" title="Needs a connection">
                  set up
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="home-cols">
          <div className="home-main">
            <div className="card">
              <div className="card-head">
                <h2>Recent activity</h2>
                <Link to="/activity" className="dim" style={{ fontSize: 13 }}>
                  View all →
                </Link>
              </div>
              {runs.length === 0 ? (
                <div className="card-pad dim">
                  No runs yet. Start one from Chat or a quick action above.
                </div>
              ) : (
                runs.map((r) => (
                  <Link className="list-row" to={`/activity/${r.id}`} key={r.id}>
                    <div className="lead">
                      <div className="row gap10">
                        <span className="title">{r.agent_name ?? "agent"}</span>
                        <span className={"pill " + statusClass(r.status)}>
                          <span className="dot" />
                          {r.status}
                        </span>
                      </div>
                      <span className="meta">{r.prompt || "(no prompt)"}</span>
                    </div>
                    <span className="meta">{relativeTime(r.created_at)}</span>
                  </Link>
                ))
              )}
            </div>
          </div>

          <div className="home-side">
            <div className="card card-pad">
              <span className="section-title">Connections</span>
              <div className="conn-mini">
                {conns.map((c) => (
                  <Link className="conn-mini-row" to="/connections" key={c.type}>
                    <span
                      className="dot"
                      style={{
                        color: c.configured ? "var(--ok)" : "var(--text-dim)",
                      }}
                    />
                    <span className="conn-mini-name">{c.title}</span>
                    <span className="conn-mini-state">
                      {c.configured ? "connected" : "connect"}
                    </span>
                  </Link>
                ))}
              </div>
            </div>

            <div className="card card-pad">
              <span className="section-title">Templates</span>
              <div className="tpl-list">
                {[
                  { name: "Email assistant", to: "/agents" },
                  { name: "Research agent", to: "/agents" },
                  { name: "Telegram bot", to: "/agents" },
                ].map((t) => (
                  <Link className="tpl-row" to={t.to} key={t.name}>
                    {t.name}
                    <span className="dim">→</span>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      </Page>

      {compose && (
        <ComposeModal
          mode={compose}
          connection={byType(compose)}
          onClose={() => setCompose(null)}
        />
      )}
    </>
  );
}
