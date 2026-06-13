// Typed client for the Studio privacy/governance surface (/api/studio/privacy/*).
//
// Wraps the consent ledger, right-to-erasure, and the signed audit bundle —
// the machinery is server-side; this file is only shapes + thin fetchers.

import { api } from "./api";

export interface PrivacySubject {
  subject_id: string;
  consent_records: number;
  purposes: number;
  data_records: number;
  erased: boolean;
  last_activity: string | null;
}

export interface SubjectsResponse {
  governed: boolean;
  subjects: PrivacySubject[];
}

export interface ConsentEntry {
  subject_id: string;
  purpose: string;
  state: string;
  effect: string; // what the PDP does with this state: allow | ephemeral | deny
  version: number; // position in the (subject, purpose) version chain
  actor: string;
  source: string;
  basis: string | null;
  expires_at: string | null;
  recorded_at: string;
}

export interface ConsentsResponse {
  governed: boolean;
  items: ConsentEntry[];
}

export interface EraseResult {
  subject_id: string;
  consents_withdrawn: string[];
  data_records: number;
  crypto_shredded: boolean;
  tombstone_id: string | null;
  erased_at: string | null;
}

export interface AuditBundleExport {
  exported_at: string;
  algorithm: string;
  record_count: number;
  record_kinds: string[];
  bundle: Record<string, unknown>;
}

export interface VerifyCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface VerifyResult {
  ok: boolean;
  algorithm: string;
  checks: VerifyCheck[];
}

export const getPrivacySubjects = () =>
  api.get<SubjectsResponse>("/privacy/subjects");

export const getPrivacyConsents = (subject?: string) =>
  api.get<ConsentsResponse>(
    `/privacy/consents${subject ? `?subject=${encodeURIComponent(subject)}` : ""}`,
  );

export const erasePrivacySubject = (subject_id: string, confirm: string) =>
  api.post<EraseResult>("/privacy/erase", { subject_id, confirm });

export const exportAuditBundle = () =>
  api.post<AuditBundleExport>("/privacy/audit/export", {});

export const verifyAuditBundle = (payload: unknown) =>
  api.post<VerifyResult>("/privacy/audit/verify", payload);

// ---- Security log (seclog) viewer (/api/studio/seclog) -------------------
//
// Read-only mirror of `himmy seclog`: recent security events (auth/authz
// decisions, denied tools, approvals, policy blocks) off the SAME durable
// .himmy/spine.db the CLI reads — so the rows match the CLI exactly.

export interface SeclogEvent {
  event_id: string;
  event_type: string;
  outcome: string; // allow | deny
  created_at: string;
  actor: string; // the principal subject, "-" when unknown
  resource: string | null;
  action: string | null;
  method: string | null;
  path: string | null;
  detail: string;
}

export interface SeclogResponse {
  items: SeclogEvent[];
  total: number;
  types: string[]; // distinct event_types in the window (filter options)
}

export const getSeclog = (opts?: { limit?: number; type?: string }) => {
  const params = new URLSearchParams();
  if (opts?.limit) params.set("limit", String(opts.limit));
  if (opts?.type) params.set("type", opts.type);
  const qs = params.toString();
  return api.get<SeclogResponse>(`/seclog${qs ? `?${qs}` : ""}`);
};
