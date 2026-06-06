// Thin typed client over the Himmy Studio API (served at /api/studio/*).
//
// In dev, Vite proxies /api → the FastAPI BFF on :8000. In production the SPA is
// served by that same BFF, so requests are same-origin either way.

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/studio${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
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
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

// ---- Shared types (mirror the Python response shapes) --------------------

export interface ExtraStatus {
  label: string;
  module: string;
  ok: boolean;
}
export interface ProviderStatus {
  name: string;
  ok: boolean;
  path: string | null;
}
export interface KeyStatus {
  name: string;
  present: boolean;
}
export interface NextStep {
  kind: "install_model" | "scaffold" | "run";
  message: string;
}
export interface DoctorReport {
  python: string;
  version: string;
  extras: ExtraStatus[];
  providers: ProviderStatus[];
  keys: KeyStatus[];
  guardrails: string[];
  project_config: string | null;
  has_real_model: boolean;
  has_agent: boolean;
  next_step: NextStep | null;
}

export interface BenchmarkEntry {
  model: string;
  provider: string;
  suite: string;
  when: string;
  accuracy: number;
  accuracy_ci?: number[];
  tool_call_accuracy: number | null;
  p50_latency_s: number;
  p95_latency_s: number;
  error_rate: number;
  total_trials: number;
}

export interface ProbeResult {
  ran: boolean;
  reason?: string;
  suite?: string;
  results: BenchmarkEntry[];
}

export interface AgentSummary {
  name: string;
  path: string;
  description: string;
  provider: string | null;
  model: string;
  skills: string[];
  tool_packs: string[];
  has_tools: boolean;
  error: string | null;
}

export interface TeamMemberInfo {
  name: string;
  role: string | null;
  provider: string | null;
  model: string;
  delegates: string[];
  handoffs: string[];
}

export interface TeamSummary {
  name: string;
  path: string;
  entry: string;
  members: TeamMemberInfo[];
  is_team: boolean;
}

export interface RunTurn {
  role: "user" | "assistant";
  content: string;
}

export interface RunRequest {
  agent_path: string;
  prompt: string;
  provider?: string | null;
  model?: string | null;
  history?: RunTurn[];
}

// One streamed event from POST /api/studio/run or /run-team.
export type RunEvent =
  | { type: "start"; agent: string; streaming: boolean; team?: boolean }
  | { type: "token"; delta: string }
  | { type: "tool"; name: string }
  | { type: "delegate"; worker: string; task: string }
  | { type: "handoff"; to: string }
  | { type: "message"; text: string }
  | {
      type: "done";
      output_text: string;
      thread_id?: string;
      run_id: string;
      succeeded: boolean;
    }
  | { type: "error"; message: string; run_id?: string };

export interface RunTeamRequest {
  team_path: string;
  prompt: string;
}

// Stream a team run (manager → workers) over SSE.
export async function streamTeamRun(
  body: RunTeamRequest,
  onEvent: (e: RunEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamSse("/api/studio/run-team", body, onEvent, signal);
}

export interface PackInfo {
  name: string;
  description: string;
  tools: string[];
}

export interface SkillInfo {
  name: string;
  description: string;
  when_to_use: string | null;
  tool_packs: string[];
  tools: string[];
  requires_skills: string[];
  builtin: boolean;
}

// The editable spec is an open map of agent.yaml fields.
export type AgentFields = Record<string, unknown>;

export interface AgentDetail {
  path: string;
  spec: AgentFields;
  has_advanced: boolean;
}

export interface ValidationResult {
  ok: boolean;
  errors: string[];
}

export interface RunSummary {
  id: string;
  created_at: string;
  agent_name: string | null;
  agent_path: string | null;
  provider: string | null;
  model: string | null;
  prompt: string;
  output_preview: string;
  status: string;
  duration_ms: number | null;
  tool_count: number;
}

export interface IoCapture {
  model: string | null;
  messages: { role: string; content: string }[];
  tools: string[];
  response_text: string;
  tool_calls: { tool: string; args: unknown }[];
}

export interface TimelineStep {
  seq: number;
  type: string;
  label: string;
  detail: string;
  ts: string | null;
  io: IoCapture | null;
}

export interface TranscriptMessage {
  role: string;
  content: string;
}

export interface RunDetailT extends RunSummary {
  output: string;
  thread_id: string | null;
  tools: string[];
  messages: TranscriptMessage[];
  timeline: TimelineStep[];
}

export interface RunListResponse {
  items: RunSummary[];
  total: number;
  limit: number;
  offset: number;
  next_offset: number | null;
}

// Stream an SSE endpoint (fetch + ReadableStream; EventSource can't POST).
// Calls `onEvent` for each `data:` frame; resolves when the stream ends.
async function streamSse(
  path: string,
  body: unknown,
  onEvent: (e: RunEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const b = await res.json();
      if (b?.detail) detail = b.detail;
    } catch {
      /* non-JSON */
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
      if (line) onEvent(JSON.parse(line.slice(6)) as RunEvent);
    }
  }
}

export function streamRun(
  body: RunRequest,
  onEvent: (e: RunEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamSse("/api/studio/run", body, onEvent, signal);
}
