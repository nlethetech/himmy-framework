import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider, Navigate } from "react-router-dom";
import App from "./App";
import Doctor from "./screens/Doctor";
import Chat from "./screens/Chat";
import Runs from "./screens/Runs";
import RunDetail from "./screens/RunDetail";
import Builder from "./screens/Builder";
import ComingSoon from "./screens/ComingSoon";
import "./index.css";

// New task-first IA. Screens not yet built (Home/Connections/Approvals) render a
// ComingSoon placeholder so the nav is fully navigable; each phase swaps in the
// real screen. Old /runs and /doctor paths redirect to their new homes.
const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <ComingSoon title="Home" /> },
      { path: "chat", element: <Chat /> },
      { path: "approvals", element: <ComingSoon title="Approvals" /> },
      { path: "activity", element: <Runs /> },
      { path: "activity/:runId", element: <RunDetail /> },
      { path: "agents", element: <Builder /> },
      { path: "agents/:name", element: <Builder /> },
      { path: "connections", element: <ComingSoon title="Connections" /> },
      { path: "advanced/doctor", element: <Doctor /> },
      // legacy redirects (one release)
      { path: "runs", element: <Navigate to="/activity" replace /> },
      { path: "runs/:runId", element: <Navigate to="/activity" replace /> },
      { path: "doctor", element: <Navigate to="/advanced/doctor" replace /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
