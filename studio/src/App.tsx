import { NavLink, Outlet } from "react-router-dom";
import { ChatIcon, BuildIcon, RunsIcon, DoctorIcon } from "./components/icons";

const NAV = [
  { to: "/chat", label: "Chat", Icon: ChatIcon },
  { to: "/agents", label: "Agents", Icon: BuildIcon },
  { to: "/runs", label: "Runs", Icon: RunsIcon },
  { to: "/doctor", label: "Doctor", Icon: DoctorIcon },
];

export default function App() {
  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail-brand" title="Himmy Studio">
          H
        </div>
        <nav className="rail-nav">
          {NAV.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              data-label={label}
              aria-label={label}
              className={({ isActive }) =>
                "rail-link" + (isActive ? " active" : "")
              }
            >
              <Icon className="ico" />
            </NavLink>
          ))}
        </nav>
        <div className="rail-foot" data-label="offline-first · local">
          <span className="dot" style={{ color: "var(--ok)" }} />
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
