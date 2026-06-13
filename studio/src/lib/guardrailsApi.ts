// Typed client for the Studio guardrails viewer (/api/studio/guardrails).
//
// Read-only: the built-in guardrail catalog (the same rows `himmy guardrails`
// prints) plus recent firings read from the Studio run store's cognition trace.
// The machinery is server-side; this file is only shapes + a thin fetcher.

import { api } from "./api";

export interface GuardrailInfo {
  name: string;
  description: string;
}

export interface GuardrailFiring {
  run_id: string;
  agent_name: string | null;
  created_at: string;
  seq: number;
  name: string; // guardrail family: pii | injection | grounding | …
  action: string; // blocked | redacted | flagged
  stage: string | null; // input | output
  detail: string;
  blocked: boolean;
  redacted: boolean;
  flags: string[];
  reasons: string[];
}

export interface GuardrailsResponse {
  builtins: GuardrailInfo[];
  firings: GuardrailFiring[];
  firing_count: number;
  runs_scanned: number;
}

export const getGuardrails = () =>
  api.get<GuardrailsResponse>("/guardrails");
