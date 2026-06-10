// Typed client for the Studio lineage graph surface (/api/studio/lineage/*).
//
// The graph endpoint returns a bounded BFS subgraph of the entity registry
// around an entity (record id / stable id / thread id) or a run id; the
// entity endpoint returns one record in full for the side detail panel.

import { api } from "./api";

export interface LineageGraphNode {
  id: string;
  kind: string;
  label: string;
  created_at: string;
  stable_id: string;
  version: number;
}

export interface LineageGraphEdge {
  from: string;
  to: string;
  relation: string;
}

export interface LineageGraphT {
  root_id: string;
  nodes: LineageGraphNode[];
  edges: LineageGraphEdge[];
  truncated: boolean;
  node_count: number;
  edge_count: number;
}

export interface LineageLinkSummary {
  relation: string;
  other_id: string;
  other_kind: string | null;
  other_label: string | null;
}

export interface LineageEntityDetail {
  id: string;
  stable_id: string;
  kind: string;
  version: number;
  created_at: string;
  label: string;
  payload: Record<string, unknown>;
  metadata: Record<string, unknown>;
  links_in: LineageLinkSummary[];
  links_out: LineageLinkSummary[];
}

export function getLineageGraph(q: {
  entityId?: string;
  runId?: string;
  depth?: number;
}): Promise<LineageGraphT> {
  const p = new URLSearchParams();
  if (q.entityId) p.set("entity_id", q.entityId);
  if (q.runId) p.set("run_id", q.runId);
  if (q.depth) p.set("depth", String(q.depth));
  return api.get<LineageGraphT>(`/lineage/graph?${p.toString()}`);
}

export const getLineageEntity = (id: string) =>
  api.get<LineageEntityDetail>(`/lineage/entity/${encodeURIComponent(id)}`);
