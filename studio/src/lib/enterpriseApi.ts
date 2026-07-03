// Typed client for the Enterprise Edition governance console (/v1/enterprise/overview).
//
// Unlike the rest of Studio (which talks to /api/studio/*), the console reads the
// tenant-scoped governance rollup off the core /v1 BFF. It is an ENTERPRISE feature: a
// community install answers 402 with an upgrade message — surfaced here as a discriminated
// `community` result so the page can render an upsell instead of an error toast.

export interface CostBucket {
  key: string;
  cost: number;
  calls: number;
  input_tokens: number;
  output_tokens: number;
}

export interface AgentRow {
  agent_id: string;
  name: string;
  description: string;
  created_at: string;
}

export interface AgentsSummary {
  total: number;
  agents: AgentRow[];
}

export interface RunsSummary {
  total: number;
  by_status: Record<string, number>;
}

export interface RbacSummary {
  roles: string[];
  permission_counts: Record<string, number>;
}

export interface HealthSummary {
  edition: string;
  backend: string;
  durable: boolean;
  ready: boolean;
}

export interface SecurityEventRow {
  event_id: string;
  event_type: string;
  outcome: string;
  created_at: string;
  actor: string | null;
  resource: string | null;
  action: string | null;
  detail: string;
}

export interface EnterpriseOverview {
  workspace_id: string | null;
  agents: AgentsSummary;
  runs: RunsSummary;
  cost_total: number;
  cost_by_agent: CostBucket[];
  cost_by_model: CostBucket[];
  recent_events: SecurityEventRow[];
  rbac: RbacSummary;
  health: HealthSummary;
}

/** Either the enterprise rollup, or a `community` upsell state (402), or an error. */
export type OverviewResult =
  | { kind: "ok"; overview: EnterpriseOverview }
  | { kind: "community"; message: string }
  | { kind: "error"; status: number; message: string };

/**
 * Fetch the tenant governance rollup. A 402 (community, no license) is returned as a
 * `community` result carrying the upgrade message rather than thrown, so the page can
 * render the upsell. Any other non-2xx is an `error` result.
 */
export async function getEnterpriseOverview(
  workspaceId?: string,
): Promise<OverviewResult> {
  const qs = workspaceId
    ? `?workspace_id=${encodeURIComponent(workspaceId)}`
    : "";
  let res: Response;
  try {
    res = await fetch(`/v1/enterprise/overview${qs}`, {
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    return { kind: "error", status: 0, message: String((e as Error)?.message || e) };
  }
  if (res.status === 402) {
    let message = "Enterprise Edition feature";
    try {
      const body = await res.json();
      if (body?.detail) message = body.detail;
    } catch {
      /* non-JSON */
    }
    return { kind: "community", message };
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) message = body.detail;
    } catch {
      /* non-JSON */
    }
    return { kind: "error", status: res.status, message };
  }
  const overview = (await res.json()) as EnterpriseOverview;
  return { kind: "ok", overview };
}
