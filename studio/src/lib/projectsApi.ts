// Typed client for Studio Projects (/api/studio/projects) — sustained work
// gets a home: a named group of chats with an optional default knowledge base
// and default agent. Backed by the same store as saved chats.

import { api, type ChatSessionSummary } from "./api";

export interface ProjectSummary {
  id: string;
  name: string;
  description: string;
  kb_id: string | null;
  agent_path: string | null;
  created_at: string;
  updated_at: string;
  chat_count: number;
}

// A chat row as returned inside a project (the chats endpoints now carry
// project_id too; the base ChatSessionSummary type predates it).
export type ProjectChatRow = ChatSessionSummary & {
  project_id?: string | null;
};

export interface ProjectDetail extends ProjectSummary {
  chats: ProjectChatRow[];
  kb_name: string | null;
  agent_name: string | null;
}

export interface ProjectCreateBody {
  name: string;
  description?: string;
  kb_id?: string | null;
  agent_path?: string | null;
}

// PATCH semantics: only the keys present are applied; an explicit null
// clears kb_id / agent_path (name can change but never clears).
export type ProjectUpdateBody = Partial<{
  name: string;
  description: string;
  kb_id: string | null;
  agent_path: string | null;
}>;

export const listProjects = () => api.get<ProjectSummary[]>("/projects");

export const getProject = (id: string) =>
  api.get<ProjectDetail>(`/projects/${encodeURIComponent(id)}`);

export const createProject = (body: ProjectCreateBody) =>
  api.post<ProjectSummary>("/projects", body);

export const updateProject = (id: string, changes: ProjectUpdateBody) =>
  api.patch<ProjectSummary>(`/projects/${encodeURIComponent(id)}`, changes);

export const deleteProject = (id: string) =>
  api.del<{ ok: boolean }>(`/projects/${encodeURIComponent(id)}`);

export const assignChat = (projectId: string, chatId: string) =>
  api.post<{ ok: boolean }>(
    `/projects/${encodeURIComponent(projectId)}/assign`,
    { chat_id: chatId },
  );

export const unassignChat = (projectId: string, chatId: string) =>
  api.post<{ ok: boolean }>(
    `/projects/${encodeURIComponent(projectId)}/unassign`,
    { chat_id: chatId },
  );
