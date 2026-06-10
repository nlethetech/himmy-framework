/* ⌘K command palette + the keyboard-first layer.
   Mounted once at App level (inside ToastProvider, above AppShell) so the
   shortcuts work on every screen:

     ⌘K / Ctrl+K  open / close the palette
     n            new chat
     /            focus the chat composer (only on /chat)
     g h · g a    go Home · go Activity (two-key sequence, 1s window)

   None of them fire while focus sits in an input/textarea/contenteditable,
   and the single-key ones stand down whenever another overlay (.scrim,
   onboarding) owns the keyboard. The palette itself is one hairline panel
   over an ink scrim: search-box row on top, mono micro-label sections
   (NAVIGATE / ACTIONS / CHATS), run-line result rows, mono shortcut footer. */

import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  api,
  listChats,
  getChat,
  type AgentSummary,
  type ChatSessionSummary,
  type ChatSessionDetail,
} from "../lib/api";
import { SearchIcon } from "./icons";
import {
  actionEntries,
  chatEntry,
  markParts,
  navigationEntries,
  openNewChat,
  rankEntries,
  snippetAround,
  CHAT_HIT_LIMIT,
  QUERY_MAX,
  SECTION_LIMIT,
  SEQUENCE_WINDOW_MS,
  type PaletteContext,
  type PaletteEntry,
} from "../lib/paletteActions";
import "../styles/palette.css";

const NAV_ENTRIES = navigationEntries();

function isEditable(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  if (t.isContentEditable) return true;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

/** True when a modal / picker / onboarding overlay should own the keyboard. */
function overlayOpen(): boolean {
  return document.querySelector(".scrim, .ob-overlay, .tour-pop") !== null;
}

function Marked({ text, q }: { text: string; q: string }) {
  return (
    <>
      {markParts(text, q).map((p, i) =>
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

interface ChatHit {
  session: ChatSessionSummary;
  snippet: string;
}

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [chats, setChats] = useState<ChatSessionSummary[] | null>(null);
  const [hits, setHits] = useState<ChatHit[] | null>(null);
  const [searching, setSearching] = useState(false);

  // transcripts cached per session id, invalidated when updated_at moves
  const detailCache = useRef(new Map<string, { updated: string; detail: ChatSessionDetail }>());
  const listRef = useRef<HTMLDivElement>(null);
  const lastFocus = useRef<HTMLElement | null>(null);

  const navigate = useNavigate();
  const { pathname } = useLocation();

  // refs so the once-registered global listener always sees current values
  const openRef = useRef(open);
  openRef.current = open;
  const pathRef = useRef(pathname);
  pathRef.current = pathname;
  const navRef = useRef(navigate);
  navRef.current = navigate;

  const openPalette = () => {
    lastFocus.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setQuery("");
    setHits(null);
    setSearching(false);
    setActive(0);
    setOpen(true);
  };

  // `restore` is false when an entry ran — focus belongs to the new screen then.
  const closePalette = (restore = true) => {
    setOpen(false);
    if (restore && lastFocus.current && lastFocus.current.isConnected) {
      lastFocus.current.focus();
    }
    lastFocus.current = null;
  };

  /* ---- global shortcuts: registered once, capture phase ------------------ */
  useEffect(() => {
    let seq: { at: number } | null = null; // pending 'g' of a g-sequence
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      const mod = e.metaKey || e.ctrlKey;

      // ⌘K / Ctrl+K — toggle. Honors the contract: never inside an editor
      // (the palette's own input handles its ⌘K-to-close itself).
      if (mod && !e.altKey && (e.key === "k" || e.key === "K")) {
        if (isEditable(e.target)) return;
        e.preventDefault();
        if (openRef.current) closePalette();
        else if (!overlayOpen()) openPalette();
        return;
      }

      if (openRef.current) return; // the palette input owns the keyboard
      if (mod || e.altKey || e.repeat) return;
      if (isEditable(e.target)) return;
      if (overlayOpen()) return;

      const k = e.key.toLowerCase();

      // two-key sequence: 'g' then 'h'/'a' within the window
      if (seq && Date.now() - seq.at <= SEQUENCE_WINDOW_MS) {
        seq = null;
        if (k === "h") {
          e.preventDefault();
          navRef.current("/");
          return;
        }
        if (k === "a") {
          e.preventDefault();
          navRef.current("/activity");
          return;
        }
        // any other key falls through and is judged on its own
      } else {
        seq = null;
      }
      if (k === "g" && !e.shiftKey) {
        seq = { at: Date.now() };
        return;
      }

      if (k === "n" && !e.shiftKey) {
        e.preventDefault();
        openNewChat({ nav: (to) => navRef.current(to), pathname: pathRef.current });
        return;
      }
      if (e.key === "/" && pathRef.current === "/chat") {
        const ta = document.querySelector<HTMLTextAreaElement>("textarea.composer-input");
        if (ta) {
          e.preventDefault();
          ta.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---- sources: fetched on open, kept warm between opens ----------------- */
  useEffect(() => {
    if (!open) return;
    let alive = true;
    api
      .get<AgentSummary[]>("/agents")
      .then((a) => alive && setAgents(a))
      .catch(() => {
        /* backend away — base actions still work */
      });
    listChats()
      .then((c) => alive && setChats(c))
      .catch(() => alive && setChats((prev) => prev ?? []));
    return () => {
      alive = false;
    };
  }, [open]);

  const q = useMemo(() => query.trim().toLowerCase(), [query]);

  /* ---- debounced full-transcript search (the Chats.tsx machinery) -------- */
  useEffect(() => {
    if (!open || q.length < 2 || chats === null) {
      setHits(null);
      setSearching(false);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(async () => {
      setSearching(true);
      const found: ChatHit[] = [];
      for (const s of chats) {
        if (found.length >= CHAT_HIT_LIMIT) break;
        const cached = detailCache.current.get(s.id);
        let detail = cached && cached.updated === s.updated_at ? cached.detail : undefined;
        if (!detail) {
          try {
            detail = await getChat(s.id);
            detailCache.current.set(s.id, { updated: s.updated_at, detail });
          } catch {
            continue; // one bad session never sinks the search
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
  }, [open, q, chats]);

  /* ---- merge + rank ------------------------------------------------------- */
  const navResults = useMemo(() => {
    const ranked = rankEntries(NAV_ENTRIES, q);
    return q ? ranked.slice(0, SECTION_LIMIT) : ranked;
  }, [q]);
  const actResults = useMemo(() => {
    const ranked = rankEntries(actionEntries(agents), q);
    return q ? ranked.slice(0, SECTION_LIMIT) : ranked;
  }, [agents, q]);
  const chatResults = useMemo(
    () => (q.length >= 2 && hits ? hits.map((h) => chatEntry(h.session, h.snippet, query.trim())) : []),
    [hits, q, query],
  );

  const chatsPending = q.length >= 2 && (hits === null || searching);
  const groups = useMemo(() => {
    const g: { label: string; items: PaletteEntry[] }[] = [];
    if (navResults.length > 0) g.push({ label: "Navigate", items: navResults });
    if (actResults.length > 0) g.push({ label: "Actions", items: actResults });
    if (q.length >= 2 && (chatResults.length > 0 || chatsPending)) {
      g.push({ label: "Chats", items: chatResults });
    }
    return g;
  }, [navResults, actResults, chatResults, chatsPending, q]);
  const flat = useMemo(() => groups.flatMap((g) => g.items), [groups]);

  useEffect(() => setActive(0), [q]);
  const eff = flat.length === 0 ? -1 : Math.min(active, flat.length - 1);

  // keep the active row in view while arrowing through a long list
  useEffect(() => {
    listRef.current
      ?.querySelector('[data-active="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [eff]);

  const run = (entry: PaletteEntry) => {
    const ctx: PaletteContext = { nav: navigate, pathname };
    closePalette(false);
    entry.perform(ctx);
  };

  const onInputKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      closePalette();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      closePalette();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => (flat.length === 0 ? 0 : Math.min(i + 1, flat.length - 1)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (eff >= 0 && flat[eff]) run(flat[eff]);
    } else if (e.key === "Tab") {
      e.preventDefault(); // focus stays on the query — no trap left on close
    }
  };

  if (!open) return null;

  let idx = -1; // running index across groups for keyboard selection
  return (
    <div
      className="cmdk-scrim"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) closePalette();
      }}
    >
      <div className="cmdk-panel" role="dialog" aria-modal="true" aria-label="Command palette">
        <div className="cmdk-inputrow">
          <SearchIcon className="ico" />
          <input
            autoFocus
            className="cmdk-input"
            role="combobox"
            aria-expanded="true"
            aria-controls="cmdk-list"
            aria-activedescendant={eff >= 0 ? flat[eff].id : undefined}
            aria-autocomplete="list"
            placeholder="Search pages, actions, chats…"
            maxLength={QUERY_MAX}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKey}
          />
          {q.length > 0 && <span className="cmdk-count">{flat.length}</span>}
        </div>

        <div className="cmdk-list" id="cmdk-list" role="listbox" ref={listRef}>
          {flat.length === 0 && !chatsPending ? (
            <div className="cmdk-empty">No matches.</div>
          ) : (
            groups.map((g) => (
              <div key={g.label}>
                <div className="cmdk-sec">{g.label}</div>
                {g.items.map((entry) => {
                  idx += 1;
                  const isActive = idx === eff;
                  const i = idx;
                  return (
                    <button
                      key={entry.id}
                      id={entry.id}
                      type="button"
                      role="option"
                      aria-selected={isActive}
                      data-active={isActive || undefined}
                      className={"cmdk-row" + (isActive ? " active" : "")}
                      onMouseEnter={() => setActive(i)}
                      onClick={() => run(entry)}
                    >
                      <span className="cmdk-row-main">
                        <span className="cmdk-row-label">
                          <Marked text={entry.label} q={q} />
                        </span>
                        <span className="cmdk-row-meta">{entry.meta}</span>
                      </span>
                      {entry.snippet !== undefined && (
                        <span className="cmdk-snippet">
                          <Marked text={entry.snippet} q={q} />
                        </span>
                      )}
                    </button>
                  );
                })}
                {g.label === "Chats" && chatsPending && (
                  <div className="cmdk-quiet">searching transcripts…</div>
                )}
              </div>
            ))
          )}
        </div>

        <div className="cmdk-foot">
          <span>↑↓ navigate · ↵ open · esc close</span>
          <span>⌘k palette · n new chat · / composer · g h home · g a activity</span>
        </div>
      </div>
    </div>
  );
}
