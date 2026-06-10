// Typed client for the Studio models surface (/api/studio/models/*):
// provider detection, API-key save/validate, and local Ollama tags + pulls.

import { api, ApiError } from "./api";

// ---- providers ------------------------------------------------------------

export interface ProviderInfo {
  name: string;
  label: string;
  kind: "local" | "key";
  configured: boolean;
  detected_via: "server" | "binary" | "env" | "secret" | null;
  status: string; // "detected via env" / "key saved" / "not configured" / ...
  key_name: string | null;
  cost_hint: number; // 0.0 = free local; routing orders ascending
  validate_supported: boolean;
  hint: string | null;
}

export interface ProvidersResponse {
  providers: ProviderInfo[];
  secrets_writable: boolean;
  default_provider: string;
  default_detail: string;
}

export const getProviders = () =>
  api.get<ProvidersResponse>("/models/providers");

export const saveProviderKey = (provider: string, apiKey: string) =>
  api.post<ProviderInfo>("/models/keys", { provider, api_key: apiKey });

export interface ValidateKeyResult {
  ok: boolean;
  provider: string;
  detail: string;
  status_code: number | null;
}

export const validateProviderKey = (provider: string, apiKey?: string) =>
  api.post<ValidateKeyResult>("/models/keys/validate", {
    provider,
    api_key: apiKey || undefined,
  });

// ---- ollama -----------------------------------------------------------------

export interface OllamaModel {
  name: string;
  size: number | null;
  modified_at: string | null;
  parameter_size: string | null;
}

export interface OllamaStatus {
  running: boolean;
  base_url: string;
  models: OllamaModel[];
  detail: string | null;
}

export const getOllama = () => api.get<OllamaStatus>("/models/ollama");

export type PullEvent =
  | { type: "progress"; status: string; completed?: number; total?: number }
  | { type: "done"; model?: string }
  | { type: "error"; message: string };

/** Stream Ollama pull progress over SSE. Resolves when the stream ends. */
export async function streamPull(
  model: string,
  onEvent: (e: PullEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/studio/models/ollama/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
    signal,
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
      if (line) onEvent(JSON.parse(line.slice(6)) as PullEvent);
    }
  }
}
