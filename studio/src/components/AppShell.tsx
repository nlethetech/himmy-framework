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
  CalendarIcon,
  SearchIcon,
  PlusIcon,
  CheckIcon,
  MailIcon,
} from "./icons";
import { useTheme } from "./ui/useTheme";
import { useAccent } from "./ui/useAccent";

type NavItem = {
  to: string;
  label: string;
  Icon: (p: { className?: string }) => ReactNode;
  badge?: number;
};

const WORKSPACE: NavItem[] = [
  { to: "/", label: "Home", Icon: HomeIcon },
  { to: "/chat", label: "Chat", Icon: ChatIcon },
  { to: "/chats", label: "Chats", Icon: ChatIcon },
  { to: "/research", label: "Research", Icon: SearchIcon },
  { to: "/approvals", label: "Approvals", Icon: BellIcon },
  { to: "/activity", label: "Activity", Icon: RunsIcon },
];

const APPS: NavItem[] = [
  { to: "/calendar", label: "Calendar", Icon: CalendarIcon },
  { to: "/email", label: "Email", Icon: MailIcon },
  { to: "/tasks", label: "Tasks", Icon: CheckIcon },
  { to: "/notes", label: "Notes", Icon: BookIcon },
  { to: "/brain", label: "Brain", Icon: MemoryIcon },
  { to: "/tools", label: "Tools", Icon: BuildIcon },
  { to: "/library", label: "Library", Icon: BookIcon },
  { to: "/theme", label: "Theme", Icon: GlobeIcon },
];

const BUILD: NavItem[] = [
  { to: "/agents", label: "Agents", Icon: BuildIcon },
  { to: "/cookbook", label: "Cookbook", Icon: BookIcon },
  { to: "/models", label: "Models", Icon: DoctorIcon },
  { to: "/compare", label: "Compare", Icon: DoctorIcon },
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

function SidebarItem({ item }: { item: NavItem }) {
  return (
    <NavLink
      to={item.to}
      end={item.to === "/"}
      className={({ isActive }) => "sb-item" + (isActive ? " active" : "")}
    >
      <item.Icon className="ico" />
      <span className="sb-item-label">{item.label}</span>
      {item.badge ? <span className="sb-badge">{item.badge}</span> : null}
    </NavLink>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [connectionsOk, setConnectionsOk] = useState<number | null>(null);
  const [approvalsCount, setApprovalsCount] = useState(0);
  const { theme, toggleTheme } = useTheme();
  useAccent(); // apply the saved brand accent on load

  useEffect(() => {
    const refresh = () =>
      listConnections()
        .then((c) => setConnectionsOk(c.filter((x) => x.configured).length))
        .catch(() => setConnectionsOk(null));
    refresh();
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, []);

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
      {/* Labeled sidebar */}
      <aside className="sidebar">
        <div className="sb-brand">
          <span className="sb-logo">⛰</span>
          <span className="sb-title">Himmy</span>
        </div>

        <NavLink to="/chat" className="sb-newchat">
          <PlusIcon className="ico" /> New Chat
        </NavLink>

        {workspace.map((i) => (
          <SidebarItem key={i.to} item={i} />
        ))}

        <NavLink
          to="/search"
          className={({ isActive }) => "sb-item" + (isActive ? " active" : "")}
        >
          <SearchIcon className="ico" />
          <span className="sb-item-label">Search</span>
        </NavLink>

        <div className="sb-section" style={{ cursor: "default" }}>
          Apps
        </div>
        {APPS.map((i) => (
          <SidebarItem key={i.to} item={i} />
        ))}

        <div className="sb-section" style={{ cursor: "default" }}>
          Build
        </div>
        {BUILD.map((i) => (
          <SidebarItem key={i.to} item={i} />
        ))}

        <button
          type="button"
          className={"sb-section" + (advancedOpen ? " open" : "")}
          onClick={() => setAdvancedOpen((o) => !o)}
        >
          Advanced
          <ChevronIcon className="chev" />
        </button>
        {advancedOpen && ADVANCED.map((i) => <SidebarItem key={i.to} item={i} />)}

        <div className="sb-foot">
          <span className="sb-foot-avatar" />
          <span>
            {connectionsOk == null
              ? "local"
              : `${connectionsOk} connection${connectionsOk === 1 ? "" : "s"}`}
          </span>
          <button
            className="rail-btn theme-toggle"
            onClick={toggleTheme}
            title="Toggle theme"
            style={{ width: 24, height: 24 }}
          >
            {theme === "dark" ? "☾" : "☀"}
          </button>
        </div>
      </aside>

      <main className="main">{children}</main>
    </div>
  );
}
