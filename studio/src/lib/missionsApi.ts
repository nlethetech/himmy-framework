// Typed client for Studio Missions (background runs + steering + plan-first),
// served at /api/studio/missions/*. Missions reuse the chat run's SSE frame
// vocabulary (RunEvent) plus two mission-specific frames: `plan` (the live
// plan-first checklist) and `steer` (guidance queued between turns).

import { ApiError, api, type RunEvent, type RunTurn } from "./api";

// ---- types -----------------------------------------------------------------

export type PlanStatus = "pending" | "active" | "done" | "skipped";

export interface PlanStep {
  title: string;
  status: PlanStatus;
}

export type MissionEvent =
  | RunEvent
  | { type: "plan"; agent?: string | null; steps: PlanStep[] }
  | { type: "steer"; text: string; ts?: string };

export type MissionStatus = "running" | "paused" | "done" | "error";

export interface MissionView {
  id: string;
  agent: string;
  agent_path: string;
  prompt: string;
  status: MissionStatus;
  created_at: string;
  finished_at: string | null;
  result_preview: string;
  error: string | null;
  run_id: string | null;
  checkpoint_id: string | null;
  plan_mode: boolean;
  frame_count: number;
}

export interface MissionListResponse {
  items: MissionView[];
  running: number;
  limit: number;
}

export interface MissionStartRequest {
  agent_path: string;
  prompt: string;
  provider?: string | null;
  model?: string | null;
  history?: RunTurn[];
  plan_mode?: boolean;
}

export interface InterruptResult {
  status: MissionStatus;
  mode: "checkpoint" | "cancelled";
  detail: string;
}

// ---- REST ------------------------------------------------------------------

export const startMission = (body: MissionStartRequest) =>
  api.post<{ mission_id: string }>("/missions", body);

export const listMissions = () => api.get<MissionListResponse>("/missions");

export const getMission = (id: string) =>
  api.get<MissionView>(`/missions/${id}`);

export const steerMission = (id: string, text: string) =>
  api.post<MissionView>(`/missions/${id}/steer`, { text });

export const interruptMission = (id: string) =>
  api.post<InterruptResult>(`/missions/${id}/interrupt`, {});

// ---- SSE stream (replay + live follow; reconnect-safe) ----------------------

// GET-based SSE via fetch (mirrors api.ts streamSse, which is POST-shaped and
// not exported). Each reconnect replays the mission's whole buffered history
// first, then follows live until the mission leaves `running`.
export async function streamMission(
  id: string,
  onEvent: (e: MissionEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api/studio/missions/${id}/stream`, { signal });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (!res.body) throw new ApiError(0, "no response body to stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)) as MissionEvent);
    }
  }
}
