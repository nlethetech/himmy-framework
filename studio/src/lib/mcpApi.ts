// Typed client for the Studio MCP surface (/api/studio/mcp/*):
// the persisted server registry, test-connections (listing only), the cached
// tool lists, and agent attach/detach.

import { api } from "./api";

// ---- shapes ----------------------------------------------------------------

export interface McpToolInfo {
  name: string;
  description: string;
}

export interface McpTestStatus {
  ok: boolean;
  at: string;
  error: string | null;
}

/** One registry entry. `env_keys` are secret NAMES only — values live in the
 *  secrets backend and are resolved server-side at test/attach time. */
export interface McpServer {
  name: string;
  transport: string; // "stdio"
  command: string;
  args: string[];
  env_keys: string[];
  cwd: string | null;
  prefix: string;
  requires_approval: boolean;
  enabled: boolean;
  created_at: string;
  updated_at: string;
  last_test: McpTestStatus | null;
  tools: McpToolInfo[]; // cached from the last successful test
  tools_at: string | null;
}

export interface McpServerInput {
  name: string;
  transport?: string;
  command: string;
  args?: string[];
  env_keys?: string[];
  cwd?: string | null;
  prefix?: string;
  requires_approval?: boolean;
  enabled?: boolean;
}

export interface McpTestResult {
  ok: boolean;
  at: string;
  error: string | null;
  tools: McpToolInfo[];
  server_info: Record<string, unknown>;
}

export interface McpAgentAttachment {
  path: string;
  name: string;
  attached: boolean;
}

// ---- calls -------------------------------------------------------------------

const enc = encodeURIComponent;

export const listMcpServers = () =>
  api.get<{ servers: McpServer[] }>("/mcp/servers");

export const createMcpServer = (body: McpServerInput) =>
  api.post<McpServer>("/mcp/servers", body);

export const updateMcpServer = (name: string, body: McpServerInput) =>
  api.put<McpServer>(`/mcp/servers/${enc(name)}`, body);

export const deleteMcpServer = (name: string) =>
  api.del<void>(`/mcp/servers/${enc(name)}`);

/** Launch briefly + list tools. Always resolves with a verdict (ok/error). */
export const testMcpServer = (name: string) =>
  api.post<McpTestResult>(`/mcp/servers/${enc(name)}/test`, {});

export const getMcpServerTools = (name: string) =>
  api.get<{ tools: McpToolInfo[]; tested_at: string | null }>(
    `/mcp/servers/${enc(name)}/tools`,
  );

export const getMcpServerAgents = (name: string) =>
  api.get<{ agents: McpAgentAttachment[] }>(`/mcp/servers/${enc(name)}/agents`);

export const attachMcpServer = (
  server: string,
  agentPath: string,
  attach: boolean,
) =>
  api.post<{ server: string; agent_path: string; attached: boolean }>(
    "/mcp/attach",
    { server, agent_path: agentPath, attach },
  );

// ---- presentation helpers ------------------------------------------------------

/** The ledger status annotation: 'untested' / 'N tools' / 'failed: …'. */
export function mcpStatus(s: McpServer): { text: string; err: boolean } {
  if (!s.enabled) return { text: "disabled", err: false };
  if (!s.last_test) return { text: "untested", err: false };
  if (s.last_test.ok)
    return {
      text: `${s.tools.length} tool${s.tools.length === 1 ? "" : "s"}`,
      err: false,
    };
  return { text: `failed: ${s.last_test.error ?? "unknown error"}`, err: true };
}
