import { Outlet } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import OnboardingGate from "./components/OnboardingGate";
import { ToastProvider } from "./components/ui/Toast";

export default function App() {
  return (
    <ToastProvider>
      <OnboardingGate />
      <AppShell>
        <Outlet />
      </AppShell>
    </ToastProvider>
  );
}
