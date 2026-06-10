// Typed client for the team builder + streaming workflow runner
// (/api/studio/teams and /api/studio/workflows/run-stream).

import { api, ApiError, type TeamSummary } from "./api";

// ---- Team builder ----------------------------------------------------------

// One member as the builder form edits it (mirrors TeamMemberForm).
export interface TeamMemberDraft {
  name: string;
  description: string;
  instructions: string[];
  role: string | null;
  provider: string | null;
  model: string;
  tools: string[];
  tool_packs: string[];
  handoffs: string[];
  delegates: string[];
}

export interface TeamSaveBody {
  name: string;
  entry: string;
  members: TeamMemberDraft[];
  path?: string | null;
  overwrite?: boolean;
}

export interface TeamValidationResult {
  ok: boolean;
  errors: string[];
}

export interface TeamDetail {
  path: string;
  name: string;
  spec: { entry: string; members: TeamMemberDraft[] };
  has_advanced: boolean;
}

export const validateTeam = (body: {
  name: string;
  entry: string;
  members: TeamMemberDraft[];
}) => api.post<TeamValidationResult>("/teams/validate", body);

export const saveTeam = (body: TeamSaveBody) =>
  api.post<TeamSummary>("/teams", body);

export const getTeamDetail = (path: string) =>
  api.get<TeamDetail>(`/teams/detail?path=${encodeURIComponent(path)}`);

// ---- Workflow runner (streaming) ------------------------------------------

export type WorkflowNodeState = "running" | "completed" | "failed";

export interface WorkflowStepRow {
  step_name: string;
  status: string;
  output_text: string | null;
  error: string | null;
  attempts: number;
}

export type WorkflowStreamEvent =
  | {
      type: "workflow_start";
      workflow: string;
      steps: string[];
      timeout_seconds: number;
    }
  | { type: "node"; name: string; state: WorkflowNodeState; error?: string | null }
  | {
      type: "done";
      workflow_name: string;
      status: string;
      final_state: Record<string, unknown>;
      step_results: WorkflowStepRow[];
    }
  | { type: "error"; message: string };

export interface WorkflowStreamBody {
  workflow_path: string;
  agent_path: string;
  provider?: string | null;
  model?: string | null;
  initial_state?: Record<string, unknown>;
  timeout_seconds?: number;
}

// Stream a workflow run over SSE (fetch + ReadableStream; EventSource can't POST).
export async function streamWorkflow(
  body: WorkflowStreamBody,
  onEvent: (e: WorkflowStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/studio/workflows/run-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const b = await res.json();
      if (typeof b?.detail === "string") detail = b.detail;
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
      if (line) onEvent(JSON.parse(line.slice(6)) as WorkflowStreamEvent);
    }
  }
}
