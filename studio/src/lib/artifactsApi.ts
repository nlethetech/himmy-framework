// Client for the Studio file-sandbox (/api/studio/files) + the chat-side
// helpers behind inline approvals and artifact cards: a correct-path SSE
// resolver for approvals and artifact extraction from finished tool steps.

import { api, ApiError, type RunEvent } from "./api";
import type { CogStep } from "../components/Cognition";

// ---- Sandbox files ---------------------------------------------------------

export interface SandboxFile {
  name: string;
  path: string; // relative to the sandbox root
  size: number;
  mtime: string; // ISO-8601 UTC
}

export interface FileListResponse {
  root: string;
  items: SandboxFile[];
  total: number;
  truncated: boolean;
}

export const listSandboxFiles = (limit = 200) =>
  api.get<FileListResponse>(`/files?limit=${limit}`);

export const fileDownloadUrl = (path: string) =>
  `/api/studio/files/download?path=${encodeURIComponent(path)}`;

// Fetch-then-save so a missing/forbidden file surfaces as a catchable error
// (a bare <a href> would navigate the tab to a JSON error body instead).
export async function downloadSandboxFile(path: string): Promise<void> {
  const res = await fetch(fileDownloadUrl(path));
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
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = path.split("/").pop() || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function formatBytes(n: number | null | undefined): string {
  if (n == null || !isFinite(n) || n < 0) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v >= 10 ? Math.round(v) : v.toFixed(1)} ${units[u]}`;
}

// ---- Approvals: resolve with the resumed run streamed back ----------------

// POST /api/studio/approvals/{id}/{approve|reject} returns an SSE stream of
// the resumed run (start/tool/usage/.../done frames). The whole stream must be
// drained — approve/reject is not a single response.
export async function resolveApprovalStream(
  checkpointId: string,
  approve: boolean,
  onEvent: (e: RunEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(
    `/api/studio/approvals/${encodeURIComponent(checkpointId)}/${
      approve ? "approve" : "reject"
    }`,
    { method: "POST", headers: { "Content-Type": "application/json" }, signal },
  );
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
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

// ---- Artifacts: file-producing tool steps → cards --------------------------

export interface Artifact {
  name: string;
  path: string; // relative to the sandbox root
  bytes: number | null; // from the tool's result when parseable
}

// write_file's result is a stringified JSON {"path": ..., "bytes_written": ...}
// (capped server-side, so parsing may fail on long results — fall back to args).
function parseWriteResult(result: string | null | undefined): {
  path?: string;
  bytes_written?: number;
} {
  if (!result) return {};
  try {
    const parsed = JSON.parse(result) as unknown;
    if (parsed && typeof parsed === "object") {
      const o = parsed as Record<string, unknown>;
      return {
        path: typeof o.path === "string" ? o.path : undefined,
        bytes_written:
          typeof o.bytes_written === "number" ? o.bytes_written : undefined,
      };
    }
  } catch {
    /* truncated or non-JSON result */
  }
  return {};
}

// Pull the artifacts a turn produced out of its cognition steps: successful
// write_file calls, deduped by path (the last write wins for size).
export function artifactsFromSteps(steps: CogStep[] | undefined): Artifact[] {
  const byPath = new Map<string, Artifact>();
  for (const s of steps ?? []) {
    if (s.kind !== "tool" || s.name !== "write_file") continue;
    if (s.outcome && s.outcome !== "success") continue;
    const fromResult = parseWriteResult(s.result);
    const path =
      fromResult.path ??
      (typeof s.args?.path === "string" ? (s.args.path as string) : null);
    if (!path) continue;
    byPath.set(path, {
      name: path.split("/").pop() || path,
      path,
      bytes: fromResult.bytes_written ?? null,
    });
  }
  return [...byPath.values()];
}
