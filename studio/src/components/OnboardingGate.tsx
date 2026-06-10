// First-run onboarding gate. Rendered inside ToastProvider above the AppShell.
//
// Zero cost on normal loads: the synchronous localStorage flag short-circuits
// before any effect runs, and the wizard/tour bundles are lazy chunks that are
// never even fetched once onboarding is done. When the flag is absent we
// confirm with one cheap signal (zero saved chats) so existing users on a new
// browser are settled silently and permanently.
import { lazy, Suspense, useEffect, useState } from "react";
import {
  checkFirstRun,
  isOnboardingDone,
  isTourDone,
} from "../onboarding/firstRun";

const Wizard = lazy(() => import("../onboarding/Wizard"));
const Tour = lazy(() => import("../onboarding/Tour"));

type Phase = "checking" | "wizard" | "tour" | "off";

export default function OnboardingGate() {
  const [phase, setPhase] = useState<Phase>(() =>
    isOnboardingDone() ? "off" : "checking",
  );

  useEffect(() => {
    if (phase !== "checking") return;
    let alive = true;
    checkFirstRun().then((show) => {
      if (alive) setPhase(show ? "wizard" : "off");
    });
    return () => {
      alive = false;
    };
  }, [phase]);

  if (phase === "wizard") {
    return (
      <Suspense fallback={null}>
        <Wizard
          onDone={({ tour }) =>
            setPhase(tour && !isTourDone() ? "tour" : "off")
          }
        />
      </Suspense>
    );
  }
  if (phase === "tour") {
    return (
      <Suspense fallback={null}>
        <Tour onDone={() => setPhase("off")} />
      </Suspense>
    );
  }
  return null;
}
