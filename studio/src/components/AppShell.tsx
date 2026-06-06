import { useEffect, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { listConnections, listApprovals } from "../lib/api";
import {
  HomeIcon,
  ChatIcon,
  BellIcon,
  RunsIcon,
  BuildIcon,
  PlugIcon,
  DoctorIcon,
  ChevronIcon,
  MemoryIcon,
  BookIcon,
  GlobeIcon,
} from "./icons";
import { useTheme } from "./ui/useTheme";

type NavItem = {
  to: string;
  label: string;
  Icon: (p: { className?: string }) => ReactNode;
  badge?: number;
};

// Primary daily-loop sections. New screens (Home/Approvals/Connections) light up
// as their phases land; until then the sidebar only links what exists.
const WORKSPACE: NavItem[] = [
  { to: "/", label: "Home", Icon: HomeIcon },
  { to: "/chat", label: "Chat", Icon: ChatIcon },
  { to: "/approvals", label: "Approvals", Icon: BellIcon },
  { to: "/activity", label: "Activity", Icon: RunsIcon },
];

const BUILD: NavItem[] = [
  { to: "/agents", label: "Agents", Icon: BuildIcon },
  { to: "/connections", label: "Connections", Icon: PlugIcon },
];

const ADVANCED: NavItem[] = [
  { to: "/advanced/teams", label: "Teams", Icon: BuildIcon },
  { to: "/advanced/workflows", label: "Workflows", Icon: RunsIcon },
  { to: "/advanced/knowledge", label: "Knowledge", Icon: BookIcon },
  { to: "/advanced/memory", label: "Memory", Icon: MemoryIcon },
  { to: "/advanced/eval", label: "Evaluation", Icon: GlobeIcon },
  { to: "/advanced/lineage", label: "Lineage", Icon: GlobeIcon },
  { to: "/advanced/doctor", label: "Doctor", Icon: DoctorIcon },
];

function SidebarLink({ item }: { item: NavItem }) {
  return (
    <NavLink
      to={item.to}
      end={item.to === "/"}
      className={({ isActive }) => "sb-link" + (isActive ? " active" : "")}
    >
      <item.Icon className="ico" />
      <span className="sb-link-label">{item.label}</span>
      {item.badge ? <span className="sb-badge">{item.badge}</span> : null}
    </NavLink>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [connectionsOk, setConnectionsOk] = useState<number | null>(null);
  const [approvalsCount, setApprovalsCount] = useState(0);
  const { theme, toggleTheme } = useTheme();

  // Live connection count for the footer (refreshed when the window regains focus,
  // so connecting an account updates the badge).
  useEffect(() => {
    const refresh = () =>
      listConnections()
        .then((c) => setConnectionsOk(c.filter((x) => x.configured).length))
        .catch(() => setConnectionsOk(null));
    refresh();
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, []);

  // Poll pending approvals so the sidebar badge stays current.
  useEffect(() => {
    const poll = () =>
      listApprovals()
        .then((a) => setApprovalsCount(a.length))
        .catch(() => {});
    poll();
    const id = setInterval(poll, 15000);
    window.addEventListener("focus", poll);
    return () => {
      clearInterval(id);
      window.removeEventListener("focus", poll);
    };
  }, []);

  const workspace = WORKSPACE.map((i) =>
    i.to === "/approvals" ? { ...i, badge: approvalsCount } : i,
  );

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sb-brand">
          <div className="sb-logo">H</div>
          <div className="sb-title">Himmy Studio</div>
        </div>

        <div className="sb-group">Workspace</div>
        {workspace.map((i) => (
          <SidebarLink key={i.to} item={i} />
        ))}

        <div className="sb-group">Build</div>
        {BUILD.map((i) => (
          <SidebarLink key={i.to} item={i} />
        ))}

        <button
          type="button"
          className={"sb-disclosure" + (advancedOpen ? " open" : "")}
          onClick={() => setAdvancedOpen((o) => !o)}
        >
          <ChevronIcon className="chev" />
          Advanced
        </button>
        {advancedOpen && ADVANCED.map((i) => <SidebarLink key={i.to} item={i} />)}

        <div className="sb-foot">
          <span
            className="dot"
            style={{
              color: connectionsOk == null ? "var(--text-dim)" : "var(--ok)",
            }}
          />
          <span style={{ flex: 1 }}>
            {connectionsOk == null
              ? "offline-first · local"
              : `${connectionsOk} connection${connectionsOk === 1 ? "" : "s"}`}
          </span>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? "☾" : "☀"}
          </button>
        </div>
      </aside>

      <main className="main">{children}</main>
    </div>
  );
}
