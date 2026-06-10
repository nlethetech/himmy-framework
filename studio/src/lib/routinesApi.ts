// Typed client for /api/studio/routines — scheduled agent runs.

import { api } from "./api";

export interface RoutineSchedule {
  kind: "daily" | "every";
  at?: string | null; // "HH:MM" when kind === "daily"
  hours?: number | null; // 1..168 when kind === "every"
}

export type DeliverKind = "none" | "telegram" | "email";

export interface Routine {
  id: string;
  name: string;
  agent_path: string;
  prompt: string;
  schedule: RoutineSchedule;
  provider: string | null;
  model: string | null;
  deliver: DeliverKind;
  enabled: boolean;
  created_at: string;
  updated_at: string;
  last_run_at: string | null;
  last_status: string | null; // running | ok | error | timeout | awaiting_approval
  last_preview: string;
  last_error: string | null;
  last_delivery: string | null; // delivery failure note; null = fine / not asked
  running: boolean;
}

export interface RoutineDraft {
  name: string;
  agent_path: string;
  prompt: string;
  schedule: RoutineSchedule;
  deliver: DeliverKind;
  enabled?: boolean;
  provider?: string | null;
  model?: string | null;
}

export type RoutinePatch = Partial<RoutineDraft>;

export const listRoutines = () => api.get<Routine[]>("/routines");
export const createRoutine = (draft: RoutineDraft) =>
  api.post<Routine>("/routines", draft);
export const updateRoutine = (id: string, patch: RoutinePatch) =>
  api.patch<Routine>(`/routines/${id}`, patch);
export const deleteRoutine = (id: string) =>
  api.del<{ ok: boolean }>(`/routines/${id}`);
export const runRoutineNow = (id: string) =>
  api.post<Routine>(`/routines/${id}/run-now`, {});

export function describeSchedule(s: RoutineSchedule): string {
  if (s.kind === "daily") return `daily ${s.at ?? "--:--"}`;
  return `every ${s.hours ?? "?"}h`;
}
