import { useCallback, useEffect, useMemo, useState } from "react";
import { markTourDone } from "./firstRun";
import "../styles/onboarding.css";

/* Five coach-mark stops over the real sidebar: a hairline ring on the anchor
   and a small popover beside it. Resilient — stops whose anchor is missing
   (collapsed sidebar, no apps enabled) are silently dropped; if an anchor
   disappears mid-tour we advance past it. Escape or "skip" ends the tour. */

interface Stop {
  selector: string;
  title: string;
  body: string;
}

const STOPS: Stop[] = [
  {
    selector: ".sb-newchat",
    title: "New Chat",
    body: "Start a fresh conversation with any agent or team.",
  },
  {
    selector: '.sb-item[data-label="Chats"]',
    title: "Chats",
    body: "Every conversation is saved here — search inside all of them as you type.",
  },
  {
    selector: '.sb-item[data-label="Approvals"]',
    title: "Approvals",
    body: "Risky actions pause here and wait for your sign-off before they run.",
  },
  {
    selector: ".sb-section",
    title: "Apps",
    body: "Email, calendar, notes, research — workspaces your agents share with you.",
  },
  {
    selector: ".sb-foot .sb-gear",
    title: "Settings",
    body: "Models, connections, agents, and everything else lives here.",
  },
];

const POP_W = 248;
const POP_H = 132; // estimate for clamping only

export default function Tour({ onDone }: { onDone: () => void }) {
  // Resolve once at mount: stops whose anchor exists right now.
  const stops = useMemo(
    () => STOPS.filter((s) => document.querySelector(s.selector) !== null),
    [],
  );
  const [idx, setIdx] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);

  const finish = useCallback(() => {
    markTourDone();
    onDone();
  }, [onDone]);

  // Measure the current anchor; skip forward past any that vanished.
  useEffect(() => {
    if (stops.length === 0) {
      finish();
      return;
    }
    let i = idx;
    let el: Element | null = null;
    while (i < stops.length) {
      el = document.querySelector(stops[i].selector);
      if (el) break;
      i++;
    }
    if (!el || i >= stops.length) {
      finish();
      return;
    }
    if (i !== idx) {
      setIdx(i);
      return;
    }
    el.scrollIntoView({ block: "nearest" });
    const measure = () => setRect(el!.getBoundingClientRect());
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [idx, stops, finish]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        finish();
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (idx + 1 >= stops.length) finish();
        else setIdx(idx + 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [idx, stops.length, finish]);

  if (stops.length === 0 || !rect || idx >= stops.length) return null;
  const stop = stops[idx];
  const last = idx === stops.length - 1;

  const popLeft = Math.min(rect.right + 14, window.innerWidth - POP_W - 12);
  const popTop = Math.max(
    12,
    Math.min(rect.top - 4, window.innerHeight - POP_H - 12),
  );

  return (
    <>
      <div
        className="tour-ring"
        style={{
          top: rect.top - 4,
          left: rect.left - 4,
          width: rect.width + 8,
          height: rect.height + 8,
        }}
        aria-hidden
      />
      <div
        className="tour-pop"
        style={{ top: popTop, left: popLeft, width: POP_W }}
        role="dialog"
        aria-label={`Tour: ${stop.title}`}
      >
        <div className="tour-count mono">
          <span>
            {idx + 1} / {stops.length}
          </span>
          <span>tour</span>
        </div>
        <div className="tour-title">{stop.title}</div>
        <div className="tour-body">{stop.body}</div>
        <div className="tour-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => (last ? finish() : setIdx(idx + 1))}
          >
            {last ? "Done" : "Next"}
          </button>
          {!last && (
            <button type="button" className="ob-skip" onClick={finish}>
              skip tour
            </button>
          )}
        </div>
      </div>
    </>
  );
}
