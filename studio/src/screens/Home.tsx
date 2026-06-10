import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  api,
  listConnections,
  type RunSummary,
  type ConnectionStatus,
} from "../lib/api";
import { Topbar } from "../components/Page";
import { ComposeModal, type ComposeMode } from "../components/ComposeModal";
import { MailIcon, TelegramIcon, GlobeIcon, ChatIcon } from "../components/icons";
import { relativeTime, statusClass } from "../lib/format";

/* Home as a single centered column — everything on one axis. No card boxes:
   sections are a micro-label over a hairline, ledger rows beneath. Status is
   only shown when it is NOT ok (six green tags in a row is noise; one red
   tag is information). */

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
      Icon: MailIcon,
      onClick: () => setCompose("email" as ComposeMode),
      missing: needs("email"),
    },
    {
      key: "telegram",
      label: "Message Telegram",
      Icon: TelegramIcon,
      onClick: () => setCompose("telegram" as ComposeMode),
      missing: needs("telegram"),
    },
    {
      key: "research",
      label: "Research",
      Icon: GlobeIcon,
      onClick: () => nav("/chat"),
      missing: false,
    },
    {
      key: "ask",
      label: "Ask / Run",
      Icon: ChatIcon,
      onClick: () => nav("/chat"),
      missing: false,
    },
  ];

  const now = new Date();
  const hour = now.getHours();
  const greeting =
    hour < 5
      ? "Working late"
      : hour < 12
        ? "Good morning"
        : hour < 18
          ? "Good afternoon"
          : "Good evening";
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
      <div className="home-col">
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

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Recent</span>
            <Link to="/activity">view all →</Link>
          </div>
          {runs.length === 0 ? (
            <div className="home-empty">
              No runs yet — start one from Chat or an action above.
            </div>
          ) : (
            runs.map((r) => (
              <Link className="run-line" to={`/activity/${r.id}`} key={r.id}>
                <span className="run-line-prompt">
                  {r.prompt || "(no prompt)"}
                </span>
                {statusClass(r.status) !== "ok" && (
                  <span className={"pill " + statusClass(r.status)}>
                    {r.status}
                  </span>
                )}
                <span className="run-line-meta mono">
                  {r.agent_name ?? "agent"} · {relativeTime(r.created_at)}
                </span>
              </Link>
            ))
          )}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Connections</span>
            <Link to="/connections">manage →</Link>
          </div>
          <div className="home-inline">
            {conns.map((c) => (
              <Link className="home-inline-item" to="/connections" key={c.type}>
                {c.title}
                <span className={"mono state" + (c.configured ? " on" : "")}>
                  {c.configured ? "connected" : "connect"}
                </span>
              </Link>
            ))}
          </div>
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Templates</span>
            <Link to="/agents">all agents →</Link>
          </div>
          <div className="home-inline">
            {["Email assistant", "Research agent", "Telegram bot"].map((t) => (
              <Link className="home-inline-item" to="/agents" key={t}>
                {t}
                <span className="mono state">→</span>
              </Link>
            ))}
          </div>
        </section>
      </div>

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
