import { NavLink, Outlet } from "react-router-dom";
import {
  ChatIcon,
  BuildIcon,
  RunsIcon,
  DoctorIcon,
} from "./components/icons";

const NAV = [
  { to: "/chat", label: "Chat", Icon: ChatIcon },
  { to: "/agents", label: "Agents", Icon: BuildIcon },
  { to: "/runs", label: "Runs", Icon: RunsIcon },
  { to: "/doctor", label: "Doctor", Icon: DoctorIcon },
];

export default function App() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">H</div>
          <div>
            <div className="brand-name">Himmy</div>
            <div className="brand-sub">Studio</div>
          </div>
        </div>

        <nav className="stack gap6">
          {NAV.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                "nav-link" + (isActive ? " active" : "")
              }
            >
              <Icon className="ico" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-foot">
          <span className="dot" style={{ color: "var(--ok)" }} />
          <span>offline-first · local</span>
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
