// Typed client for the Studio learning surface (/api/studio/learning).
//
// Read-only "what Himmy learned": per-tool reputation + recent reorders, mirroring the
// CLI `himmy learned`. Shares the fetch plumbing in api.ts.

import { api } from "./api";

/** The active operator override verdict on a tool (Sprint 2 "teach & correct"). */
export type LearningOverride = "pinned" | "ignored" | "reset";

export interface LearningTool {
  tool_name: string;
  score: number;
  operational_score: number;
  completed: number;
  failed: number;
  has_min_samples: boolean;
  flaky: boolean;
  outcome_score: number | null;
  outcome_samples: number;
  /** Active operator override, or null for pure auto-learning. */
  override: LearningOverride | null;
}

export interface LearningReorder {
  when: string | null;
  tools_reordered: number;
  deprioritised_tools: string[];
}

export interface LearningReport {
  floor: number;
  outcome_weight: number;
  /** Whether the per-workspace 👍/👎 trust gate is ON (blending recorded outcomes). */
  trust_feedback: boolean;
  flaky_count: number;
  has_data: boolean;
  tools: LearningTool[];
  recent_reorders: LearningReorder[];
}

/** One pin / mute / reset / clear action the operator can take on a tool. */
export type OverrideAction = "pin" | "mute" | "reset" | "clear";

export const getLearning = () => api.get<LearningReport>("/learning");

/**
 * Pin / mute / reset / clear a tool, then get the refreshed report back.
 * Best-effort on the server: a swallowed store error still returns the report.
 */
export const setOverride = (tool_name: string, action: OverrideAction) =>
  api.post<LearningReport>("/learning/overrides", { tool_name, action });

/** Flip the per-workspace 👍/👎 trust gate, then get the refreshed report back. */
export const setTrustFeedback = (enabled: boolean) =>
  api.post<LearningReport>("/learning/trust-feedback", { enabled });
