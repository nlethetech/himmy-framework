import { useState } from "react";
import { Link } from "react-router-dom";
import { Topbar } from "../components/Page";
import { CheckIcon } from "../components/icons";
import { APP_DEFS, enabledApps, setAppEnabled } from "../lib/apps";
import { resetFirstRun } from "../onboarding/firstRun";

/* Settings: the one door to everything that isn't daily work — which apps
   show in the sidebar, the builder screens, and the advanced machinery. */

const BUILD = [
  { to: "/agents", label: "Agents", sub: "create and edit agent specs" },
  { to: "/tools", label: "Tools", sub: "the built-in capability packs" },
  { to: "/cookbook", label: "Cookbook", sub: "ready-made recipes" },
  { to: "/models", label: "Models", sub: "providers and model routing" },
  { to: "/compare", label: "Compare", sub: "run two agents side by side" },
  { to: "/connections", label: "Connections", sub: "email, Telegram, Google" },
  { to: "/mcp", label: "MCP servers", sub: "connect external tool servers" },
];

// Automation surfaces: long-running and scheduled work plus project grouping.
// Listed here so they stay reachable even when their sidebar apps are hidden.
const AUTOMATION = [
  { to: "/missions", label: "Missions", sub: "long-running autonomous runs" },
  { to: "/routines", label: "Routines", sub: "scheduled agent runs" },
  { to: "/projects", label: "Projects", sub: "group chats, knowledge, agents" },
];

const ADVANCED = [
  { to: "/advanced/teams", label: "Teams", sub: "multi-agent orchestration" },
  { to: "/advanced/workflows", label: "Workflows", sub: "graph runs" },
  { to: "/advanced/knowledge", label: "Knowledge", sub: "RAG document bases" },
  { to: "/advanced/memory", label: "Memory", sub: "durable agent memory" },
  { to: "/advanced/eval", label: "Evaluation", sub: "test suites and scores" },
  { to: "/advanced/learning", label: "Learning", sub: "what Himmy learned about its tools" },
  { to: "/advanced/lineage", label: "Lineage", sub: "the entity audit trail" },
  {
    to: "/advanced/enterprise",
    label: "Enterprise",
    sub: "governance, cost and audit console",
  },
  { to: "/advanced/doctor", label: "Doctor", sub: "system health checks" },
  { to: "/theme", label: "Appearance", sub: "theme and accent" },
];

export default function Settings() {
  const [enabled, setEnabled] = useState<Set<string>>(enabledApps());

  const toggle = (key: string) =>
    setEnabled(new Set(setAppEnabled(key, !enabled.has(key))));

  return (
    <>
      <Topbar title="Settings" sub="apps, builders, and advanced machinery" />
      <div className="home-col">
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Sidebar apps</span>
            <span>{enabled.size} shown</span>
          </div>
          {APP_DEFS.map((a) => (
            <button
              key={a.key}
              className="run-line set-toggle"
              onClick={() => toggle(a.key)}
              type="button"
            >
              <span
                className={"task-check" + (enabled.has(a.key) ? " on" : "")}
                aria-hidden
              >
                <CheckIcon />
              </span>
              <span className="run-line-prompt">{a.label}</span>
              <span className="run-line-meta mono">
                {enabled.has(a.key) ? "shown" : "hidden"}
              </span>
            </button>
          ))}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Privacy</span>
          </div>
          <Link className="run-line" to="/privacy">
            <span className="run-line-prompt">Privacy &amp; audit</span>
            <span className="run-line-meta mono">
              subjects, consent, signed audit bundles
            </span>
          </Link>
          <button
            className="run-line set-toggle"
            type="button"
            onClick={() => {
              resetFirstRun();
              window.location.assign("/");
            }}
          >
            <span className="run-line-prompt">Replay onboarding tour</span>
            <span className="run-line-meta mono">
              show the first-run wizard and tour again
            </span>
          </button>
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Automation</span>
          </div>
          {AUTOMATION.map((b) => (
            <Link className="run-line" to={b.to} key={b.to}>
              <span className="run-line-prompt">{b.label}</span>
              <span className="run-line-meta mono">{b.sub}</span>
            </Link>
          ))}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Build</span>
          </div>
          {BUILD.map((b) => (
            <Link className="run-line" to={b.to} key={b.to}>
              <span className="run-line-prompt">{b.label}</span>
              <span className="run-line-meta mono">{b.sub}</span>
            </Link>
          ))}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Advanced</span>
          </div>
          {ADVANCED.map((b) => (
            <Link className="run-line" to={b.to} key={b.to}>
              <span className="run-line-prompt">{b.label}</span>
              <span className="run-line-meta mono">{b.sub}</span>
            </Link>
          ))}
        </section>
      </div>
    </>
  );
}
