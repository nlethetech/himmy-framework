/* Command-palette sources, ranking and actions — pure logic, no JSX, so the
   palette component stays thin. Three sections feed the palette:

     navigate — every route (workspace, customizable apps, Settings sections)
     actions  — new chat (plain + per-agent), theme, sidebar, onboarding replay
     chats    — full-transcript hits (the component owns the async search; the
                entries are built through `chatEntry` here)

   Ranking is intentionally simple and predictable: exact > prefix >
   word-boundary > substring > keyword > subsequence. */

import { APP_DEFS } from "./apps";
import { resetFirstRun } from "../onboarding/firstRun";
import type { AgentSummary, ChatSessionSummary } from "./api";

export const QUERY_MAX = 200; // input bound — nobody searches with a novel
export const SECTION_LIMIT = 9; // per-section cap while filtering
export const CHAT_HIT_LIMIT = 8; // transcript hits shown in the palette
export const SEQUENCE_WINDOW_MS = 1000; // 'g' then 'h'/'a' must land within 1s

export type PaletteSection = "navigate" | "actions" | "chats";

/** What an entry needs to act: the router's navigate + where we are now. */
export interface PaletteContext {
  nav: (to: string) => void;
  pathname: string;
}

export interface PaletteEntry {
  id: string;
  section: PaletteSection;
  label: string;
  /** right-aligned mono annotation: path, model, timestamp… */
  meta: string;
  /** extra match terms beyond the label (never displayed) */
  keywords?: string;
  /** transcript snippet (chats only) — rendered with mark.hl */
  snippet?: string;
  perform: (ctx: PaletteContext) => void;
}

/* ---- navigation ---------------------------------------------------------- */

interface NavDef {
  label: string;
  to: string;
  keywords?: string;
}

// Every reachable screen: the workspace, the customizable apps (all of them —
// the palette reaches even apps hidden from the sidebar), the surfaces grouped
// under Settings (Build + Advanced), and the rest of the route map.
const NAV_DEFS: NavDef[] = [
  { label: "Home", to: "/", keywords: "start overview today" },
  { label: "Chat", to: "/chat", keywords: "conversation talk message" },
  { label: "Chats", to: "/chats", keywords: "history search conversations transcripts" },
  { label: "Approvals", to: "/approvals", keywords: "pending review confirm" },
  { label: "Activity", to: "/activity", keywords: "runs history log" },
  ...APP_DEFS.map((a) => ({ label: a.label, to: a.to, keywords: "app" })),
  { label: "Missions", to: "/missions", keywords: "long running goals" },
  { label: "Routines", to: "/routines", keywords: "schedule recurring cron" },
  { label: "Projects", to: "/projects", keywords: "workspaces" },
  { label: "MCP", to: "/mcp", keywords: "servers protocol tools" },
  { label: "Settings", to: "/settings", keywords: "preferences configuration" },
  { label: "Privacy", to: "/privacy", keywords: "settings audit consent subjects" },
  // Settings → Build
  { label: "Agents", to: "/agents", keywords: "settings build create edit specs builder" },
  { label: "Tools", to: "/tools", keywords: "settings build capability packs" },
  { label: "Cookbook", to: "/cookbook", keywords: "settings build recipes templates" },
  { label: "Models", to: "/models", keywords: "settings build providers routing ollama keys" },
  { label: "Compare", to: "/compare", keywords: "settings build side by side benchmark" },
  { label: "Connections", to: "/connections", keywords: "settings build email telegram google" },
  // Settings → Advanced
  { label: "Teams", to: "/advanced/teams", keywords: "settings advanced multi-agent orchestration" },
  { label: "Workflows", to: "/advanced/workflows", keywords: "settings advanced graph runs steps" },
  { label: "Knowledge", to: "/advanced/knowledge", keywords: "settings advanced rag documents bases" },
  { label: "Library", to: "/library", keywords: "knowledge documents rag" },
  { label: "Memory", to: "/advanced/memory", keywords: "settings advanced durable recall" },
  { label: "Evaluation", to: "/advanced/eval", keywords: "settings advanced test suites scores" },
  { label: "Lineage", to: "/advanced/lineage", keywords: "settings advanced entity audit trail" },
  { label: "Doctor", to: "/advanced/doctor", keywords: "settings advanced health checks diagnostics" },
  { label: "Appearance", to: "/theme", keywords: "settings advanced theme accent colors" },
];

export function navigationEntries(): PaletteEntry[] {
  return NAV_DEFS.map((d) => ({
    id: "nav:" + d.to,
    section: "navigate" as const,
    label: d.label,
    meta: d.to,
    keywords: d.keywords,
    perform: ({ nav }: PaletteContext) => nav(d.to),
  }));
}

/* ---- actions -------------------------------------------------------------- */

/**
 * Open a fresh conversation, optionally pre-picking an agent. Chat consumes
 * its deep-link params (?agent=) once on mount, so a same-route navigation
 * from /chat would neither remount nor reset the open conversation — in that
 * one case we do a real navigation (the same pattern Settings uses for the
 * onboarding replay), which is deterministic and cheap for a local-first SPA.
 */
export function openNewChat(ctx: PaletteContext, agentPath?: string): void {
  const target = agentPath
    ? `/chat?agent=${encodeURIComponent(agentPath)}`
    : "/chat";
  if (ctx.pathname === "/chat") {
    window.location.assign(target);
    return;
  }
  ctx.nav(target);
}

/**
 * Theme lives in AppShell's useTheme hook; driving its own footer toggle keeps
 * the ☾/☀ glyph in sync. If the button is ever absent, flip the attribute +
 * storage directly (identical effect — the CSS keys off data-theme).
 */
export function toggleTheme(): void {
  const btn = document.querySelector<HTMLButtonElement>("button.theme-toggle");
  if (btn) {
    btn.click();
    return;
  }
  const root = document.documentElement;
  const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
  root.setAttribute("data-theme", next);
  try {
    localStorage.setItem("himmy-theme", next);
  } catch {
    /* private mode — the visual flip still happened */
  }
}

/** Same pattern for the sidebar rail: drive AppShell's own collapse button. */
export function toggleSidebar(): void {
  const btn = document.querySelector<HTMLButtonElement>("button.sb-collapse");
  if (btn) {
    btn.click();
    return;
  }
  try {
    const key = "himmy.sb.collapsed";
    const next = localStorage.getItem(key) === "1" ? "0" : "1";
    localStorage.setItem(key, next); // applies on the next shell mount
  } catch {
    /* storage disabled — nothing to toggle against */
  }
}

/** Exactly what Settings → "Replay onboarding tour" does. */
export function replayOnboarding(): void {
  resetFirstRun();
  window.location.assign("/");
}

export function actionEntries(agents: AgentSummary[]): PaletteEntry[] {
  const entries: PaletteEntry[] = [
    {
      id: "act:new-chat",
      section: "actions",
      label: "New chat",
      meta: "n",
      keywords: "start compose conversation create",
      perform: (ctx) => openNewChat(ctx),
    },
  ];
  for (const a of agents) {
    if (a.error) continue; // unparseable specs can't start a chat
    entries.push({
      id: "act:new-chat:" + a.path,
      section: "actions",
      label: `New chat with ${a.name}`,
      meta: a.model || a.path,
      keywords: `agent ${a.path} ${a.description ?? ""}`,
      perform: (ctx) => openNewChat(ctx, a.path),
    });
  }
  entries.push(
    {
      id: "act:theme",
      section: "actions",
      label: "Toggle theme",
      meta: "dark · light",
      keywords: "appearance dark light mode switch",
      perform: () => toggleTheme(),
    },
    {
      id: "act:sidebar",
      section: "actions",
      label: "Collapse sidebar",
      meta: "toggle",
      keywords: "expand rail navigation hide show",
      perform: () => toggleSidebar(),
    },
    {
      id: "act:onboarding",
      section: "actions",
      label: "Replay onboarding",
      meta: "wizard + tour",
      keywords: "first run tutorial welcome help tour",
      perform: () => replayOnboarding(),
    },
  );
  return entries;
}

/* ---- chats ---------------------------------------------------------------- */

/**
 * Open a saved transcript at the ?hl= deep link — the exact link Chats.tsx
 * builds. Chat consumes ?session=/?hl= once on mount, so when the palette is
 * already on /chat a plain navigate would not remount and the session would
 * never load; in that one case we do a real navigation (mirrors openNewChat).
 */
export function openChatSession(
  ctx: PaletteContext,
  sessionId: string,
  query: string,
): void {
  const target = `/chat?session=${sessionId}&hl=${encodeURIComponent(query)}`;
  if (ctx.pathname === "/chat") {
    window.location.assign(target);
    return;
  }
  ctx.nav(target);
}

/** A transcript hit → palette entry opening the ?hl= deep link (Chats.tsx). */
export function chatEntry(
  session: ChatSessionSummary,
  snippet: string,
  query: string,
): PaletteEntry {
  return {
    id: "chat:" + session.id,
    section: "chats",
    label: session.title,
    meta: session.updated_at.slice(0, 16).replace("T", " "),
    snippet,
    perform: (ctx) => openChatSession(ctx, session.id, query),
  };
}

/* ---- ranking --------------------------------------------------------------- */

function isSubsequence(needle: string, hay: string): boolean {
  let i = 0;
  for (const ch of hay) {
    if (ch === needle[i]) i += 1;
    if (i === needle.length) return true;
  }
  return needle.length === 0;
}

/** Higher is better; null means "does not match". `q` must be lowercased. */
export function scoreEntry(entry: PaletteEntry, q: string): number | null {
  if (q.length === 0) return 0;
  const label = entry.label.toLowerCase();
  if (label === q) return 100;
  if (label.startsWith(q)) return 90;
  const boundary = label.split(/[\s/–·-]+/).some((w) => w.startsWith(q));
  if (boundary) return 70;
  if (label.includes(q)) return 50;
  const kw = (entry.keywords ?? "").toLowerCase();
  if (kw.includes(q)) return 40;
  if (q.length >= 3 && isSubsequence(q, label)) return 15;
  return null;
}

/** Filter + stable-sort one section by score. Empty query keeps the curated order. */
export function rankEntries(entries: PaletteEntry[], q: string): PaletteEntry[] {
  if (q.length === 0) return entries;
  return entries
    .map((entry, i) => ({ entry, i, score: scoreEntry(entry, q) }))
    .filter((r): r is { entry: PaletteEntry; i: number; score: number } => r.score !== null)
    .sort((a, b) => b.score - a.score || a.i - b.i)
    .map((r) => r.entry);
}

/* ---- snippets (mirrors Chats.tsx so hits look identical) ------------------- */

export function snippetAround(text: string, q: string, radius = 80): string | null {
  const idx = text.toLowerCase().indexOf(q);
  if (idx === -1) return null;
  const start = Math.max(0, idx - radius);
  const end = Math.min(text.length, idx + q.length + radius);
  return (
    (start > 0 ? "…" : "") +
    text.slice(start, end).replace(/\s+/g, " ") +
    (end < text.length ? "…" : "")
  );
}

export interface MarkPart {
  s: string;
  hit: boolean;
}

/** Split `text` so every occurrence of `q` (lowercased) can render as mark.hl. */
export function markParts(text: string, q: string): MarkPart[] {
  if (q.length === 0) return [{ s: text, hit: false }];
  const lower = text.toLowerCase();
  const parts: MarkPart[] = [];
  let pos = 0;
  for (;;) {
    const idx = lower.indexOf(q, pos);
    if (idx === -1) break;
    if (idx > pos) parts.push({ s: text.slice(pos, idx), hit: false });
    parts.push({ s: text.slice(idx, idx + q.length), hit: true });
    pos = idx + q.length;
  }
  if (pos < text.length) parts.push({ s: text.slice(pos), hit: false });
  return parts;
}
