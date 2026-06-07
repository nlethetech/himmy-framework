import { useEffect, useMemo, useState } from "react";
import {
  listCalendar,
  addCalendar,
  deleteCalendar,
  type CalendarEvent,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { Modal } from "../components/ui/Modal";
import { PlusIcon, XIcon } from "../components/icons";
import { useToast } from "../components/ui/Toast";

const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function ymd(y: number, m: number, d: number): string {
  return `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}

export default function Calendar() {
  const today = useMemo(() => new Date(), []);
  const [year, setYear] = useState(today.getFullYear());
  const [month, setMonth] = useState(today.getMonth()); // 0-11
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [adding, setAdding] = useState<string | null>(null); // date being added to
  const toast = useToast();

  const monthKey = `${year}-${String(month + 1).padStart(2, "0")}`;
  const load = () => listCalendar(monthKey).then(setEvents).catch(() => setEvents([]));
  useEffect(() => {
    load();
  }, [monthKey]);

  const byDate = useMemo(() => {
    const m: Record<string, CalendarEvent[]> = {};
    for (const e of events) (m[e.date] ??= []).push(e);
    return m;
  }, [events]);

  // Build the month grid (Mon-first), including leading/trailing blanks.
  const firstDow = (new Date(year, month, 1).getDay() + 6) % 7; // Mon=0
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const cells: (number | null)[] = [
    ...Array(firstDow).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ];
  while (cells.length % 7 !== 0) cells.push(null);

  const todayStr = ymd(today.getFullYear(), today.getMonth(), today.getDate());

  const move = (delta: number) => {
    const d = new Date(year, month + delta, 1);
    setYear(d.getFullYear());
    setMonth(d.getMonth());
  };

  const remove = async (id: string) => {
    await deleteCalendar(id);
    load();
  };

  return (
    <>
      <Topbar
        title="Calendar"
        sub="Plan and track events"
        actions={
          <div className="row gap6">
            <button className="btn" onClick={() => move(-1)}>
              ‹
            </button>
            <span className="cal-month mono">
              {MONTHS[month]} {year}
            </span>
            <button className="btn" onClick={() => move(1)}>
              ›
            </button>
            <button
              className="btn"
              onClick={() => {
                setYear(today.getFullYear());
                setMonth(today.getMonth());
              }}
            >
              Today
            </button>
          </div>
        }
      />
      <Page>
        <div className="cal-grid-head">
          {DOW.map((d) => (
            <div className="cal-dow" key={d}>
              {d}
            </div>
          ))}
        </div>
        <div className="cal-grid">
          {cells.map((d, i) => {
            const date = d ? ymd(year, month, d) : "";
            const evs = d ? byDate[date] ?? [] : [];
            return (
              <div
                className={
                  "cal-cell" +
                  (!d ? " empty" : "") +
                  (date === todayStr ? " today" : "")
                }
                key={i}
                onDoubleClick={() => d && setAdding(date)}
              >
                {d && (
                  <>
                    <div className="cal-day">
                      <span>{d}</span>
                      <button
                        className="cal-add"
                        title="Add event"
                        onClick={() => setAdding(date)}
                      >
                        <PlusIcon className="ico" />
                      </button>
                    </div>
                    <div className="cal-events">
                      {evs.map((e) => (
                        <div className="cal-event" key={e.id} title={e.notes}>
                          {e.time && <span className="cal-time">{e.time}</span>}
                          <span className="cal-event-title">{e.title}</span>
                          <button className="cal-del" onClick={() => remove(e.id)}>
                            <XIcon />
                          </button>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>
      </Page>

      {adding && (
        <AddEvent
          date={adding}
          onClose={() => setAdding(null)}
          onAdded={() => {
            setAdding(null);
            load();
            toast.show("Event added", "ok");
          }}
        />
      )}
    </>
  );
}

function AddEvent({
  date,
  onClose,
  onAdded,
}: {
  date: string;
  onClose: () => void;
  onAdded: () => void;
}) {
  const [title, setTitle] = useState("");
  const [time, setTime] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const toast = useToast();

  const save = async () => {
    if (!title.trim()) return;
    setSaving(true);
    try {
      await addCalendar({ date, title: title.trim(), time: time || null, notes });
      onAdded();
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      title={`New event — ${date}`}
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saving ? <span className="spinner" /> : "Add"}
          </button>
        </>
      }
    >
      <div className="form-grid">
        <label className="field">
          <span className="field-label">Title</span>
          <input
            className="input"
            value={title}
            autoFocus
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && save()}
          />
        </label>
        <label className="field">
          <span className="field-label">
            Time<span className="field-opt">optional · HH:MM</span>
          </span>
          <input
            className="input"
            placeholder="09:00"
            value={time}
            onChange={(e) => setTime(e.target.value)}
          />
        </label>
        <label className="field">
          <span className="field-label">
            Notes<span className="field-opt">optional</span>
          </span>
          <textarea
            className="input"
            rows={3}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </label>
      </div>
    </Modal>
  );
}
