import { useEffect, useRef, useState } from "react";

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
}: {
  value: string;
  groups: PickGroup[];
  onChange: (v: string) => void;
  placeholder?: string;
  title?: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

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
          {groups.map((g, gi) => (
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
