// Typed client for the Studio Evaluation surface (/api/studio/eval/*):
// runnable suites, the streamed run (SSE), and the checked-in bench baseline.

import { ApiError } from "./api";

const BASE = "/api/studio/eval";

// ---- GET /suites -----------------------------------------------------------

export interface RunnableSuite {
  id: string; // pass back to streamEvalRun as `suite`
  kind: "eval" | "bench";
  name: string;
  cases: number;
  source: string;
}
export interface SuitesResponse {
  suites: RunnableSuite[];
}

// ---- GET /baseline ----------------------------------------------------------

export interface BaselineRow {
  metric: string;
  bound: "floor" | "ceiling";
  limit: number | null;
  measured: number | null;
}
export interface BaselineModel {
  name: string;
  rows: BaselineRow[];
}
export interface BaselineView {
  exists: boolean;
  reason: string;
  suite: string;
  sha: string;
  date: string;
  trials: number;
  min_trials: number;
  gate_tasks: string[];
  models: BaselineModel[];
}

// ---- GET /history -----------------------------------------------------------

export interface HistoryPoint {
  when: string;
  sha: string | null;
  trials: number;
  accuracy: number | null;
  tool_call_accuracy: number | null;
  error_rate: number | null;
  p50_latency_s: number | null;
}
export interface HistoryTrendRow {
  metric: string;
  latest: number | null;
  previous: number | null;
  delta: number | null;
  regressed: boolean;
}
export interface HistorySeries {
  model: string;
  suite: string;
  runs: number;
  latest_when: string;
  previous_when: string | null;
  regressed: boolean;
  points: HistoryPoint[];
  trends: HistoryTrendRow[];
}
export interface HistoryView {
  exists: boolean;
  reason: string;
  total_records: number;
  threshold: number;
  series: HistorySeries[];
}

// ---- POST /run (SSE) ---------------------------------------------------------

export interface EvalMetricScore {
  metric: string;
  score: number;
  passed: boolean;
  detail: string;
}
export interface EvalCaseEvent {
  type: "case";
  index: number;
  id: string;
  case: string;
  score: number;
  passed: boolean;
  meta?: string;
  metrics: EvalMetricScore[];
}
export interface EvalSummaryEvent {
  type: "summary";
  kind: "eval" | "bench";
  suite: string;
  aggregate_score: number;
  passed: number;
  failed: number;
  cases: number;
  duration_s: number;
  model?: string;
  metrics?: {
    accuracy: number;
    tool_call_accuracy: number | null;
    error_rate: number;
  };
  note?: string;
}
export type EvalRunEvent =
  | {
      type: "start";
      kind: "eval" | "bench";
      suite: string;
      cases: number;
      provider: string;
      model?: string;
    }
  | EvalCaseEvent
  | EvalSummaryEvent
  | { type: "error"; message: string };

export interface EvalRunBody {
  suite: string;
  provider?: string | null;
  model?: string | null;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
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
  return (await res.json()) as T;
}

export const listRunnableSuites = () => getJson<SuitesResponse>("/suites");
export const getBaseline = () => getJson<BaselineView>("/baseline");
export const getHistory = () => getJson<HistoryView>("/history");

// Stream a suite run over SSE (fetch + ReadableStream; EventSource can't POST).
// Calls `onEvent` per `data:` frame; resolves when the stream ends.
export async function streamEvalRun(
  body: EvalRunBody,
  onEvent: (e: EvalRunEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/run`, {
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
      if (line) onEvent(JSON.parse(line.slice(6)) as EvalRunEvent);
    }
  }
}
