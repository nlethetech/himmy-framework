import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  listChats,
  getChat,
  type ChatSessionSummary,
  type ChatSessionDetail,
} from "../lib/api";
import { Topbar } from "../components/Page";
import { SearchIcon } from "../components/icons";

interface Hit {
  session: ChatSessionSummary;
  snippet: string; // raw text around the first match (marked at render time)
}

/* Case-insensitive snippet around the first occurrence of `q` in `text`. */
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

/* Render a snippet with every query occurrence wrapped in <mark class="hl">. */
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

export default function Search() {
  const [query, setQuery] = useState("");
  const [sessions, setSessions] = useState<ChatSessionSummary[] | null>(null);
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const detailCache = useRef(new Map<string, ChatSessionDetail>());
  const nav = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listChats().then(setSessions).catch(() => setSessions([]));
    inputRef.current?.focus();
  }, []);

  const q = useMemo(() => query.trim().toLowerCase(), [query]);

  // Debounced search over titles + full transcripts (fetched once, cached).
  useEffect(() => {
    if (q.length < 2 || sessions === null) {
      setHits(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(async () => {
      setSearching(true);
      const found: Hit[] = [];
      for (const s of sessions) {
        let detail = detailCache.current.get(s.id);
        if (!detail) {
          try {
            detail = await getChat(s.id);
            detailCache.current.set(s.id, detail);
          } catch {
            continue; // a session that fails to load is skipped, not fatal
          }
        }
        if (cancelled) return;
        const transcript = detail.messages.map((m) => m.text).join("\n");
        const fromBody = snippetAround(transcript, q);
        const inTitle = s.title.toLowerCase().includes(q);
        if (fromBody !== null || inTitle) {
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
  }, [q, sessions]);

  const open = (id: string) =>
    nav(`/chat?session=${id}&hl=${encodeURIComponent(query.trim())}`);

  return (
    <>
      <Topbar title="Search" sub="across saved conversations" />
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
                open(hits[0].session.id);
              }
            }}
          />
        </div>

        {q.length < 2 ? (
          <div className="search-hint">
            Type at least two characters. Matches open the conversation with
            every occurrence highlighted.
          </div>
        ) : hits === null || searching ? (
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
                onClick={() => open(h.session.id)}
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
        )}
      </div>
    </>
  );
}
