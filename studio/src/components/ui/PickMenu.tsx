import { useEffect, useMemo, useRef, useState } from "react";

/* A quiet replacement for the composer's native <select>: a flat mono trigger
   that opens an upward hairline menu with grouped options. Closes on outside
   click and Escape; the active option carries the mark. */

export interface PickOption {
  value: string;
  label: string;
  meta?: string;
}
export interface PickGroup {
  label?: string;
  options: PickOption[];
}

export function PickMenu({
  value,
  groups,
  onChange,
  placeholder = "Choose…",
  title,
  searchable = false,
}: {
  value: string;
  groups: PickGroup[];
  onChange: (v: string) => void;
  placeholder?: string;
  title?: string;
  /** Adds a filter input at the top — for long catalogs (300+ options). */
  searchable?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      if (searchable) searchRef.current?.focus();
    }
  }, [open, searchable]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const flat = groups.flatMap((g) => g.options);
  const current = flat.find((o) => o.value === value);
  const q = query.trim().toLowerCase();
  const shown = useMemo(
    () =>
      !q
        ? groups
        : groups
            .map((g) => ({
              ...g,
              options: g.options.filter(
                (o) =>
                  o.label.toLowerCase().includes(q) ||
                  o.value.toLowerCase().includes(q),
              ),
            }))
            .filter((g) => g.options.length > 0),
    [groups, q],
  );

  return (
    <div className="pick" ref={rootRef}>
      <button
        type="button"
        className={"pick-trigger" + (open ? " open" : "")}
        onClick={() => setOpen((o) => !o)}
        title={title}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="pick-label">{current?.label ?? placeholder}</span>
        <span className="pick-chev" aria-hidden>
          ›
        </span>
      </button>

      {open && (
        <div className="pick-menu" role="listbox">
          {searchable && (
            <input
              ref={searchRef}
              className="pick-search"
              placeholder="Filter…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const first = shown.flatMap((g) => g.options)[0];
                  if (first) {
                    onChange(first.value);
                    setOpen(false);
                  }
                }
              }}
            />
          )}
          {searchable && shown.length === 0 && (
            <div className="pick-group">no matches</div>
          )}
          {shown.map((g, gi) => (
            <div key={gi}>
              {g.label && <div className="pick-group">{g.label}</div>}
              {g.options.map((o) => (
                <button
                  type="button"
                  key={o.value}
                  role="option"
                  aria-selected={o.value === value}
                  className={"pick-opt" + (o.value === value ? " on" : "")}
                  onClick={() => {
                    onChange(o.value);
                    setOpen(false);
                  }}
                >
                  <span className="pick-opt-label">{o.label}</span>
                  {o.meta && <span className="pick-opt-meta">{o.meta}</span>}
                  {o.value === value && (
                    <span className="pick-opt-mark" aria-hidden>
                      ✓
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
