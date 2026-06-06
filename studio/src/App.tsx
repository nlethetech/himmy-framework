import { Outlet } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { ToastProvider } from "./components/ui/Toast";

export default function App() {
  return (
    <ToastProvider>
      <AppShell>
        <Outlet />
      </AppShell>
    </ToastProvider>
  );
}
