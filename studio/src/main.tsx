import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider, Navigate } from "react-router-dom";
import App from "./App";
import Doctor from "./screens/Doctor";
import Chat from "./screens/Chat";
import Activity from "./screens/Activity";
import RunDetail from "./screens/RunDetail";
import Builder from "./screens/Builder";
import Connections from "./screens/Connections";
import Home from "./screens/Home";
import Approvals from "./screens/Approvals";
import Memory from "./screens/Memory";
import Knowledge from "./screens/Knowledge";
import Teams from "./screens/Teams";
import Evaluation from "./screens/Evaluation";
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
      { index: true, element: <Home /> },
      { path: "chat", element: <Chat /> },
      { path: "approvals", element: <Approvals /> },
      { path: "activity", element: <Activity /> },
      { path: "activity/:runId", element: <RunDetail /> },
      { path: "agents", element: <Builder /> },
      { path: "agents/:name", element: <Builder /> },
      { path: "connections", element: <Connections /> },
      { path: "advanced/doctor", element: <Doctor /> },
      { path: "advanced/teams", element: <Teams /> },
      {
        path: "advanced/workflows",
        element: (
          <ComingSoon
            title="Workflows"
            note="Deterministic multi-step workflows (himmy.orchestrators) run via the CLI today. A DAG editor + run timeline will live here."
          />
        ),
      },
      { path: "advanced/knowledge", element: <Knowledge /> },
      { path: "advanced/memory", element: <Memory /> },
      { path: "advanced/eval", element: <Evaluation /> },
      {
        path: "advanced/lineage",
        element: (
          <ComingSoon
            title="Lineage"
            note="Every answer traces back to its prompt, persona, and evidence (the /v1/runs/{id}/lineage graph). An interactive provenance viewer will live here."
          />
        ),
      },
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
