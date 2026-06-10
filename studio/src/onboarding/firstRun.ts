// First-run detection + onboarding persistence.
//
// The wizard only ever shows when TWO signals agree:
//   1. the localStorage flag is absent (this browser never completed/skipped it)
//   2. the workspace has zero saved chats (an existing user on a fresh browser
//      must never be walked through their own product)
// The check degrades to "not first run" when the API is unreachable, so a
// broken backend can never trap the app behind the wizard.

import { listChats } from "../lib/api";

const DONE_KEY = "himmy.onboarding.done";
const TOUR_KEY = "himmy.onboarding.tour.done";
const PROVIDER_KEY = "himmy.onboarding.provider";

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* private mode / storage disabled — onboarding simply won't persist */
  }
}

/** Synchronous, zero-cost gate: true once the wizard was completed or skipped. */
export function isOnboardingDone(): boolean {
  // If storage is unavailable we report "done" — never loop the wizard.
  try {
    return localStorage.getItem(DONE_KEY) === "1";
  } catch {
    return true;
  }
}

/** Completing OR skipping the wizard both mark it done — it never nags. */
export function markDone(): void {
  write(DONE_KEY, "1");
}

export function isTourDone(): boolean {
  return read(TOUR_KEY) === "1";
}

export function markTourDone(): void {
  write(TOUR_KEY, "1");
}

const FORCE_KEY = "himmy.onboarding.force";

/**
 * Forget that onboarding ever ran (wizard + tour) so it shows again on the
 * next load. Sets a one-shot FORCE flag so an explicit replay bypasses the
 * zero-chats heuristic — an existing user pressing "Replay onboarding tour"
 * means it, even though their workspace is full of chats.
 */
export function resetFirstRun(): void {
  write(DONE_KEY, null);
  write(TOUR_KEY, null);
  write(FORCE_KEY, "1");
}

/** The provider chosen in the wizard — the chat default for new conversations. */
export function setPreferredProvider(provider: string): void {
  write(PROVIDER_KEY, provider);
}

export function getPreferredProvider(): string | null {
  const v = read(PROVIDER_KEY);
  return v && v.length > 0 ? v : null;
}

/**
 * Async first-run check. Cheap: the flag short-circuits without any network;
 * otherwise one GET /chats decides. Existing users (any saved chat) are
 * marked done permanently so the wizard never re-evaluates.
 */
export async function checkFirstRun(): Promise<boolean> {
  if (isOnboardingDone()) return false;
  if (read(FORCE_KEY) === "1") {
    write(FORCE_KEY, null); // one-shot: consumed by this replay
    return true;
  }
  try {
    const chats = await listChats();
    if (chats.length > 0) {
      markDone(); // existing workspace — settle it once, never ask again
      return false;
    }
    return true;
  } catch {
    // Backend down or unauthorized: don't show the wizard, don't mark done —
    // we'll re-check on the next healthy load.
    return false;
  }
}
