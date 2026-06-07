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
import Workflows from "./screens/Workflows";
import Lineage from "./screens/Lineage";
import Calendar from "./screens/Calendar";
import Notes from "./screens/Notes";
import Cookbook from "./screens/Cookbook";
import Models from "./screens/Models";
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
      { path: "calendar", element: <Calendar /> },
      { path: "notes", element: <Notes /> },
      { path: "cookbook", element: <Cookbook /> },
      { path: "models", element: <Models /> },
      { path: "approvals", element: <Approvals /> },
      { path: "activity", element: <Activity /> },
      { path: "activity/:runId", element: <RunDetail /> },
      { path: "agents", element: <Builder /> },
      { path: "agents/:name", element: <Builder /> },
      { path: "connections", element: <Connections /> },
      { path: "advanced/doctor", element: <Doctor /> },
      { path: "advanced/teams", element: <Teams /> },
      { path: "advanced/workflows", element: <Workflows /> },
      { path: "advanced/knowledge", element: <Knowledge /> },
      { path: "advanced/memory", element: <Memory /> },
      { path: "advanced/eval", element: <Evaluation /> },
      { path: "advanced/lineage", element: <Lineage /> },
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
