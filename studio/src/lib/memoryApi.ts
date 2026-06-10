// Typed client for the Studio memory surface (/api/studio/memory*).
//
// List/add/forget/recall ride the main Studio router; edit (PATCH) lands on
// the dedicated memory router. All share the same fetch plumbing in api.ts.

import { api } from "./api";

export interface MemoryItem {
  memory_id: string;
  subject_id: string;
  kind: string;
  text: string;
  created_at: string;
}

export interface MemoryHit {
  memory_id: string;
  text: string;
  similarity: number;
}

export const MEMORY_TEXT_MAX = 20_000;

export const listMemorySubjects = () => api.get<string[]>("/memory/subjects");

export const listMemories = (subject: string) =>
  api.get<MemoryItem[]>(`/memory?subject=${encodeURIComponent(subject)}`);

export const addMemory = (text: string, subject_id: string) =>
  api.post<MemoryItem>("/memory", { text, subject_id });

export const updateMemory = (id: string, text: string) =>
  api.patch<MemoryItem>(`/memory/${encodeURIComponent(id)}`, { text });

export const forgetMemory = (id: string) =>
  api.del<{ ok: boolean }>(`/memory/${encodeURIComponent(id)}`);

export const recallMemory = (query: string, subject_id: string) =>
  api.post<MemoryHit[]>("/memory/recall", { query, subject_id, top_k: 5 });
