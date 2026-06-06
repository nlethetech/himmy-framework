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

// ---- Connections ---------------------------------------------------------

export interface ConnectionFieldStatus {
  name: string;
  label: string;
  present: boolean;
  secret: boolean;
  required: boolean;
  kind: string; // text | password | number | bool | enum
  placeholder: string;
  options: string[];
  masked_hint: string | null;
}

export interface ConnectionStatus {
  type: string;
  title: string;
  description: string;
  hue: string;
  configured: boolean;
  writable: boolean;
  fields: ConnectionFieldStatus[];
}

export interface ConnectionTestResult {
  ok: boolean;
  detail: string;
  checked: string;
}

export const listConnections = () =>
  api.get<ConnectionStatus[]>("/connections");
export const getConnection = (type: string) =>
  api.get<ConnectionStatus>(`/connections/${type}`);
export const setConnection = (type: string, fields: Record<string, unknown>) =>
  api.put<ConnectionStatus>(`/connections/${type}`, { fields });
export const deleteConnection = (type: string) =>
  api.del<ConnectionStatus>(`/connections/${type}`);
export const testConnection = (type: string) =>
  api.post<ConnectionTestResult>(`/connections/${type}/test`, {});
export const sendViaConnection = (type: string, payload: Record<string, unknown>) =>
  api.post<{ ok: boolean; detail: string }>(`/connections/${type}/send`, {
    payload,
  });

// ---- Approvals (HITL inbox) ----------------------------------------------

export interface PendingToolView {
  tool_name: string;
  args: Record<string, unknown>;
}
export interface ApprovalSummary {
  checkpoint_id: string;
  status: string;
  created_at: string;
  tools: string[];
  run_id: string | null;
  agent: string | null;
  prompt: string;
}
export interface ApprovalDetail extends ApprovalSummary {
  pending_tool_calls: PendingToolView[];
  thread_preview: { role: string; content: string }[];
}

export const listApprovals = () => api.get<ApprovalSummary[]>("/approvals");
export const getApproval = (id: string) =>
  api.get<ApprovalDetail>(`/approvals/${id}`);

// approve/reject return an SSE stream of the resumed run
export function resolveApproval(
  id: string,
  approve: boolean,
  onEvent: (e: RunEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamSse(
    `/approvals/${id}/${approve ? "approve" : "reject"}`,
    {},
    onEvent,
    signal,
  );
}

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
  | { type: "agent"; name: string; model?: string | null }
  | { type: "reason"; agent?: string | null; text: string }
  | {
      type: "tool";
      name: string;
      agent?: string | null;
      args?: Record<string, unknown> | null;
      result?: string | null;
      outcome?: string | null;
      read_only?: boolean | null;
      latency_ms?: number | null;
    }
  | { type: "delegate"; worker: string; task: string; from?: string | null }
  | { type: "handoff"; to: string; from?: string | null }
  | {
      type: "grounding";
      agent?: string | null;
      source: "memory" | "knowledge";
      query?: string | null;
      citations: Citation[];
    }
  | {
      type: "safety";
      agent?: string | null;
      stage: "input" | "output";
      blocked: boolean;
      redacted: boolean;
      flags: string[];
      reasons: string[];
    }
  | {
      type: "usage";
      agent?: string | null;
      model: string;
      input_tokens: number;
      output_tokens: number;
      cost: number;
      latency_ms?: number | null;
      total_input_tokens: number;
      total_output_tokens: number;
      total_cost: number;
      inferences: number;
    }
  | {
      type: "approval_required";
      agent?: string | null;
      checkpoint_id: string;
      tools: string[];
    }
  | { type: "paused"; checkpoint_id: string | null; run_id: string }
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
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

export interface ModelUsage {
  model: string;
  input_tokens: number;
  output_tokens: number;
  cost: number;
  inferences: number;
}

export interface ModelAnalytics {
  model: string;
  runs: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
}

export interface DayAnalytics {
  day: string;
  runs: number;
  cost: number;
  tokens: number;
}

export interface RunAnalytics {
  total_runs: number;
  ok_runs: number;
  error_runs: number;
  success_rate: number;
  total_cost: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  by_model: ModelAnalytics[];
  by_day: DayAnalytics[];
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

export interface Citation {
  text: string;
  similarity?: number | null;
  source_uri?: string | null;
}

// One step in the live cognition trace (think → act → observe).
export interface CognitionStep {
  seq: number;
  kind:
    | "agent"
    | "reason"
    | "tool"
    | "delegate"
    | "handoff"
    | "grounding"
    | "safety";
  agent?: string | null;
  model?: string | null;
  text?: string | null;
  name?: string | null;
  args?: Record<string, unknown> | null;
  result?: string | null;
  outcome?: string | null;
  read_only?: boolean | null;
  latency_ms?: number | null;
  worker?: string | null;
  task?: string | null;
  to?: string | null;
  source?: string | null;
  query?: string | null;
  citations?: Citation[] | null;
  stage?: string | null;
  blocked?: boolean | null;
  redacted?: boolean | null;
  flags?: string[] | null;
  reasons?: string[] | null;
  ts?: string | null;
}

export interface RunDetailT extends RunSummary {
  output: string;
  thread_id: string | null;
  tools: string[];
  messages: TranscriptMessage[];
  timeline: TimelineStep[];
  steps: CognitionStep[];
  usage_by_model: ModelUsage[];
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
