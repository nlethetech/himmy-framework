/* The customizable Apps registry: which app screens show in the sidebar.

   The set of enabled apps persists in localStorage and changes broadcast a
   window event so the sidebar updates live while Settings is open. */

export interface AppDef {
  key: string;
  to: string;
  label: string;
}

export const APP_DEFS: AppDef[] = [
  { key: "calendar", to: "/calendar", label: "Calendar" },
  { key: "email", to: "/email", label: "Email" },
  { key: "tasks", to: "/tasks", label: "Tasks" },
  { key: "notes", to: "/notes", label: "Notes" },
  { key: "research", to: "/research", label: "Research" },
  { key: "files", to: "/files", label: "Files" },
  { key: "brain", to: "/brain", label: "Brain" },
];

const KEY = "himmy.apps.enabled";
export const APPS_CHANGED = "himmy:apps-changed";

export function enabledApps(): Set<string> {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw === null) return new Set(APP_DEFS.map((a) => a.key)); // all on by default
    return new Set(JSON.parse(raw) as string[]);
  } catch {
    return new Set(APP_DEFS.map((a) => a.key));
  }
}

export function setAppEnabled(key: string, on: boolean): Set<string> {
  const next = enabledApps();
  if (on) next.add(key);
  else next.delete(key);
  localStorage.setItem(KEY, JSON.stringify([...next]));
  window.dispatchEvent(new CustomEvent(APPS_CHANGED));
  return next;
}
