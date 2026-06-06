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
