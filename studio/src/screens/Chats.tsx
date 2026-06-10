import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  listChats,
  getChat,
  renameChat,
  deleteChat,
  type ChatSessionSummary,
  type ChatSessionDetail,
} from "../lib/api";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { ChatIcon, XIcon, PlusIcon, SearchIcon } from "../components/icons";

/* Chats + Search, merged: one screen that lists every conversation and turns
   into full-transcript search the moment you type. Opening a hit deep-links
   into the chat with every occurrence highlighted (?hl=). */

interface Hit {
  session: ChatSessionSummary;
  snippet: string;
}

function snippetAround(text: string, q: string, radius = 80): string | null {
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

function Marked({ text, q }: { text: string; q: string }) {
  const lower = text.toLowerCase();
  const parts: { s: string; hit: boolean }[] = [];
  let pos = 0;
  for (;;) {
    const idx = lower.indexOf(q, pos);
    if (idx === -1) break;
    if (idx > pos) parts.push({ s: text.slice(pos, idx), hit: false });
    parts.push({ s: text.slice(idx, idx + q.length), hit: true });
    pos = idx + q.length;
  }
  if (pos < text.length) parts.push({ s: text.slice(pos), hit: false });
  return (
    <>
      {parts.map((p, i) =>
        p.hit ? (
          <mark className="hl" key={i}>
            {p.s}
          </mark>
        ) : (
          <span key={i}>{p.s}</span>
        ),
      )}
    </>
  );
}

export default function Chats() {
  const [chats, setChats] = useState<ChatSessionSummary[] | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const detailCache = useRef(new Map<string, ChatSessionDetail>());
  const nav = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  const load = () => listChats().then(setChats).catch(() => setChats([]));
  useEffect(() => {
    load();
    inputRef.current?.focus();
  }, []);

  const q = useMemo(() => query.trim().toLowerCase(), [query]);

  // Debounced full-transcript search (titles + messages, cached per session).
  useEffect(() => {
    if (q.length < 2 || chats === null) {
      setHits(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(async () => {
      setSearching(true);
      const found: Hit[] = [];
      for (const s of chats) {
        let detail = detailCache.current.get(s.id);
        if (!detail) {
          try {
            detail = await getChat(s.id);
            detailCache.current.set(s.id, detail);
          } catch {
            continue;
          }
        }
        if (cancelled) return;
        const transcript = detail.messages.map((m) => m.text).join("\n");
        const fromBody = snippetAround(transcript, q);
        if (fromBody !== null || s.title.toLowerCase().includes(q)) {
          found.push({ session: s, snippet: fromBody ?? s.title });
        }
      }
      if (!cancelled) {
        setHits(found);
        setSearching(false);
      }
    }, 220);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [q, chats]);

  const openHit = (id: string) =>
    nav(`/chat?session=${id}&hl=${encodeURIComponent(query.trim())}`);

  const rename = async (c: ChatSessionSummary) => {
    const title = window.prompt("Rename chat", c.title);
    if (title && title.trim()) {
      await renameChat(c.id, title.trim());
      load();
    }
  };
  const remove = async (id: string) => {
    await deleteChat(id);
    detailCache.current.delete(id);
    load();
  };

  return (
    <>
      <Topbar
        title="Chats"
        sub="every conversation — type to search inside them"
        actions={
          <button className="btn btn-primary" onClick={() => nav("/chat")}>
            <PlusIcon /> New chat
          </button>
        }
      />
      <div className="search-page">
        <div className="search-box">
          <SearchIcon className="ico" />
          <input
            ref={inputRef}
            className="search-input"
            placeholder="Search every conversation…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && hits && hits.length > 0) {
                openHit(hits[0].session.id);
              }
            }}
          />
        </div>

        {q.length >= 2 ? (
          hits === null || searching ? (
            <div className="search-hint">Searching…</div>
          ) : hits.length === 0 ? (
            <div className="search-hint">No matches.</div>
          ) : (
            <>
              <div className="search-count">
                {hits.length} conversation{hits.length === 1 ? "" : "s"}
              </div>
              {hits.map((h) => (
                <button
                  key={h.session.id}
                  className="search-hit"
                  onClick={() => openHit(h.session.id)}
                >
                  <span className="search-hit-title">
                    <span>{h.session.title}</span>
                    <span className="search-hit-date">
                      {h.session.updated_at.slice(0, 16).replace("T", " ")}
                    </span>
                  </span>
                  <span className="search-hit-snippet">
                    <Marked text={h.snippet} q={q} />
                  </span>
                </button>
              ))}
            </>
          )
        ) : chats === null ? null : chats.length === 0 ? (
          <EmptyState icon={<ChatIcon />} title="No saved chats yet">
            Start a conversation in <b>Chat</b> — it's saved here automatically
            and you can pick it back up anytime.
          </EmptyState>
        ) : (
          <div className="chats-list" style={{ marginTop: 18 }}>
            {chats.map((c) => (
              <div
                className="chats-row"
                key={c.id}
                onClick={() => nav(`/chat?session=${c.id}`)}
                role="button"
                tabIndex={0}
              >
                <ChatIcon />
                <div className="chats-main">
                  <span className="chats-title">{c.title}</span>
                  <span className="chats-meta mono">
                    {c.message_count} msg
                    {c.agent_path ? ` · ${shortPath(c.agent_path)}` : ""}
                    {c.provider ? ` · ${c.provider}` : ""}
                  </span>
                </div>
                <span className="chats-date mono">{shortDate(c.updated_at)}</span>
                <button
                  className="chats-btn"
                  title="Rename"
                  onClick={(e) => {
                    e.stopPropagation();
                    rename(c);
                  }}
                >
                  ✎
                </button>
                <button
                  className="chats-btn"
                  title="Delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    remove(c.id);
                  }}
                >
                  <XIcon />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function shortPath(p: string): string {
  const parts = p.split("/");
  return parts[parts.length - 1] || p;
}
function shortDate(iso: string): string {
  return iso.slice(0, 16).replace("T", " ");
}
