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

// Claude-clean IA: the conversation is the product, so only chat-centric and
// oversight items live at the top level. Everything else is grouped and
// collapsible — Apps open by default (daily use), Build & Advanced tucked
// away. Routes all stay alive; only the nav moved. Library/Theme left the
// sidebar (Knowledge lives under Advanced + Brain; the footer toggles theme,
// and /theme stays reachable under Advanced).
const WORKSPACE: NavItem[] = [
  { to: "/", label: "Home", Icon: HomeIcon },
  { to: "/chat", label: "Chat", Icon: ChatIcon },
  // Chats and Search are one screen: the history list that searches inside
  // every transcript as you type — hence the search icon.
  { to: "/chats", label: "Chats", Icon: SearchIcon },
  { to: "/approvals", label: "Approvals", Icon: BellIcon },
  { to: "/activity", label: "Activity", Icon: RunsIcon },
];

const APPS: NavItem[] = [
  { to: "/calendar", label: "Calendar", Icon: CalendarIcon },
  { to: "/email", label: "Email", Icon: MailIcon },
  { to: "/tasks", label: "Tasks", Icon: CheckIcon },
  { to: "/notes", label: "Notes", Icon: BookIcon },
  { to: "/research", label: "Research", Icon: SearchIcon },
  { to: "/brain", label: "Brain", Icon: MemoryIcon },
];

const BUILD: NavItem[] = [
  { to: "/agents", label: "Agents", Icon: BuildIcon },
  { to: "/tools", label: "Tools", Icon: BuildIcon },
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
  { to: "/theme", label: "Theme", Icon: GlobeIcon },
];

function SidebarItem({ item }: { item: NavItem }) {
  return (
    <NavLink
      to={item.to}
      end={item.to === "/"}
      className={({ isActive }) => "sb-item" + (isActive ? " active" : "")}
      data-label={item.label}
    >
      <item.Icon className="ico" />
      <span className="sb-item-label">{item.label}</span>
      {item.badge ? <span className="sb-badge">{item.badge}</span> : null}
    </NavLink>
  );
}

function useSection(key: string, defaultOpen: boolean) {
  const [open, setOpen] = useState<boolean>(() => {
    const saved = localStorage.getItem("himmy.sb." + key);
    return saved === null ? defaultOpen : saved === "1";
  });
  const toggle = () =>
    setOpen((o) => {
      localStorage.setItem("himmy.sb." + key, o ? "0" : "1");
      return !o;
    });
  return { open, toggle };
}

function Section({
  label,
  items,
  state,
}: {
  label: string;
  items: NavItem[];
  state: { open: boolean; toggle: () => void };
}) {
  return (
    <>
      <button
        type="button"
        className={"sb-section" + (state.open ? " open" : "")}
        onClick={state.toggle}
      >
        {label}
        <ChevronIcon className="chev" />
      </button>
      {state.open && items.map((i) => <SidebarItem key={i.to} item={i} />)}
    </>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const apps = useSection("apps", true);
  const build = useSection("build", false);
  const advanced = useSection("advanced", false);
  const [collapsed, setCollapsed] = useState<boolean>(
    () => localStorage.getItem("himmy.sb.collapsed") === "1",
  );
  const toggleCollapsed = () =>
    setCollapsed((c) => {
      localStorage.setItem("himmy.sb.collapsed", c ? "0" : "1");
      return !c;
    });
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
    <div className={"shell" + (collapsed ? " sb-collapsed" : "")}>
      {/* Labeled sidebar — collapses to an icon rail */}
      <aside className="sidebar">
        <div className="sb-brand">
          <span className="sb-logo">⛰</span>
          <span className="sb-title">Himmy</span>
        </div>

        <NavLink to="/chat" className="sb-newchat" data-label="New Chat">
          <PlusIcon className="ico" />
          <span className="sb-item-label">New Chat</span>
        </NavLink>

        {workspace.map((i) => (
          <SidebarItem key={i.to} item={i} />
        ))}

        {!collapsed && (
          <>
            <Section label="Apps" items={APPS} state={apps} />
            <Section label="Build" items={BUILD} state={build} />
            <Section label="Advanced" items={ADVANCED} state={advanced} />
          </>
        )}

        <div className="sb-foot">
          <span className="sb-foot-avatar" />
          <span className="sb-foot-text">
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
          <button
            className="rail-btn sb-collapse"
            onClick={toggleCollapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            style={{ width: 24, height: 24 }}
          >
            <ChevronIcon className="chev" />
          </button>
        </div>
      </aside>

      <main className="main">{children}</main>
    </div>
  );
}
