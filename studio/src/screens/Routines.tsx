import { useEffect, useMemo, useState } from "react";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { PickMenu } from "../components/ui/PickMenu";
import { useToast } from "../components/ui/Toast";
import { CheckIcon } from "../components/icons";
import { api, listConnections, type AgentSummary } from "../lib/api";
import { relativeTime } from "../lib/format";
import {
  createRoutine,
  deleteRoutine,
  describeSchedule,
  listRoutines,
  runRoutineNow,
  updateRoutine,
  type DeliverKind,
  type Routine,
  type RoutineDraft,
} from "../lib/routinesApi";
import "../styles/routines.css";

/* Routines — the agent that acts without being asked. Ledger rows: name on
   the left, schedule + delivery + last run in mono on the right; failures are
   red annotations. The create/edit form is the same ledger grammar: a
   micro-label per row, hairline-under inputs, no boxes. */

type FormState = {
  id: string | null; // null = creating
  name: string;
  agent_path: string;
  prompt: string;
  kind: "daily" | "every";
  at: string;
  hours: string;
  deliver: DeliverKind;
};

const BLANK: FormState = {
  id: null,
  name: "",
  agent_path: "",
  prompt: "",
  kind: "daily",
  at: "07:00",
  hours: "6",
  deliver: "none",
};

function statusNote(r: Routine): { text: string; bad: boolean } | null {
  if (r.running || r.last_status === "running")
    return { text: "running…", bad: false };
  if (r.last_status === "error" || r.last_status === "timeout")
    return { text: r.last_status === "timeout" ? "timed out" : "failed", bad: true };
  if (r.last_status === "awaiting_approval")
    return { text: "needs approval", bad: true };
  if (r.last_delivery) return { text: "delivery failed", bad: true };
  return null;
}

export default function Routines() {
  const toast = useToast();
  const [routines, setRoutines] = useState<Routine[] | null>(null);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [delivers, setDelivers] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [formErr, setFormErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [busy, setBusy] = useState<Set<string>>(new Set());

  const load = () => {
    setError(null);
    listRoutines()
      .then(setRoutines)
      .catch((e) => setError(String(e.message ?? e)));
  };

  useEffect(() => {
    load();
    api
      .get<AgentSummary[]>("/agents")
      .then((a) => setAgents(a.filter((x) => !x.error)))
      .catch(() => setAgents([]));
    listConnections()
      .then((cs) =>
        setDelivers(
          new Set(cs.filter((c) => c.configured).map((c) => c.type)),
        ),
      )
      .catch(() => setDelivers(new Set()));
  }, []);

  const agentGroups = useMemo(
    () => [
      {
        label: "agents",
        options: agents.map((a) => ({
          value: a.path,
          label: a.name,
          meta: a.path,
        })),
      },
    ],
    [agents],
  );

  const deliverGroups = useMemo(() => {
    const opts = [{ value: "none", label: "in app", meta: "notifications" }];
    if (delivers.has("telegram"))
      opts.push({ value: "telegram", label: "Telegram", meta: "message" });
    if (delivers.has("email"))
      opts.push({ value: "email", label: "Email", meta: "to yourself" });
    // an edited routine may point at a channel that lost its connection —
    // keep its value pickable, honestly annotated
    const current = form?.deliver;
    if (current && current !== "none" && !delivers.has(current))
      opts.push({ value: current, label: current, meta: "not configured" });
    return [{ options: opts }];
  }, [delivers, form?.deliver]);

  const openCreate = () => {
    setFormErr(null);
    setForm({ ...BLANK, agent_path: agents[0]?.path ?? "" });
  };
  const openEdit = (r: Routine) => {
    setFormErr(null);
    setForm({
      id: r.id,
      name: r.name,
      agent_path: r.agent_path,
      prompt: r.prompt,
      kind: r.schedule.kind,
      at: r.schedule.at ?? "07:00",
      hours: String(r.schedule.hours ?? 6),
      deliver: r.deliver,
    });
  };

  const save = async () => {
    if (!form) return;
    const hours = parseInt(form.hours, 10);
    if (!form.name.trim()) return setFormErr("give the routine a name");
    if (!form.agent_path) return setFormErr("pick an agent to run");
    if (!form.prompt.trim()) return setFormErr("what should it be asked?");
    if (form.kind === "daily" && !/^\d{2}:\d{2}$/.test(form.at))
      return setFormErr("pick a time of day");
    if (form.kind === "every" && (!Number.isFinite(hours) || hours < 1 || hours > 168))
      return setFormErr("hours must be between 1 and 168");
    const draft: RoutineDraft = {
      name: form.name.trim(),
      agent_path: form.agent_path,
      prompt: form.prompt.trim(),
      schedule:
        form.kind === "daily"
          ? { kind: "daily", at: form.at }
          : { kind: "every", hours },
      deliver: form.deliver,
    };
    setSaving(true);
    setFormErr(null);
    try {
      if (form.id) await updateRoutine(form.id, draft);
      else await createRoutine(draft);
      toast.show(form.id ? "Routine updated" : "Routine scheduled", "ok");
      setForm(null);
      load();
    } catch (e) {
      setFormErr(String((e as Error).message ?? e));
    } finally {
      setSaving(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await deleteRoutine(id);
      toast.show("Routine deleted", "ok");
      setForm(null);
      load();
    } catch (e) {
      toast.show(String((e as Error).message ?? e), "err");
    }
  };

  const toggle = async (r: Routine) => {
    try {
      await updateRoutine(r.id, { enabled: !r.enabled });
      load();
    } catch (e) {
      toast.show(String((e as Error).message ?? e), "err");
    }
  };

  const runNow = async (r: Routine) => {
    setBusy((b) => new Set(b).add(r.id));
    try {
      const out = await runRoutineNow(r.id);
      if (out.last_status === "ok") toast.show(`${r.name} ran`, "ok");
      else if (out.last_status === "awaiting_approval")
        toast.show(`${r.name} paused — needs approval`, "info");
      else toast.show(out.last_error || `${r.name} failed`, "err");
      load();
    } catch (e) {
      toast.show(String((e as Error).message ?? e), "err");
    } finally {
      setBusy((b) => {
        const next = new Set(b);
        next.delete(r.id);
        return next;
      });
    }
  };

  const kindGroups = [
    {
      options: [
        { value: "daily", label: "daily at", meta: "once a day" },
        { value: "every", label: "every", meta: "N hours" },
      ],
    },
  ];

  const meta = (r: Routine) =>
    [
      describeSchedule(r.schedule),
      r.deliver !== "none" ? `via ${r.deliver}` : null,
      r.last_run_at ? `ran ${relativeTime(r.last_run_at)}` : "never run",
    ]
      .filter(Boolean)
      .join(" · ");

  return (
    <>
      <Topbar
        title="Routines"
        sub="scheduled work the agent does without being asked"
        actions={
          routines && routines.length > 0 && !form ? (
            <button className="btn" onClick={openCreate}>
              + New routine
            </button>
          ) : undefined
        }
      />
      <div className="home-col rt-col">
        {error && (
          <div className="rt-error mono">
            {error}{" "}
            <button className="rt-line-act" onClick={load}>
              retry
            </button>
          </div>
        )}

        {form && (
          <section className="home-sec">
            <div className="home-sec-head">
              <span>{form.id ? "Edit routine" : "New routine"}</span>
              <button className="rt-head-act" onClick={() => setForm(null)}>
                cancel
              </button>
            </div>
            <div className="rt-row">
              <span className="rt-row-label">name</span>
              <input
                className="rt-input"
                placeholder="Morning brief"
                maxLength={200}
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div className="rt-row">
              <span className="rt-row-label">agent</span>
              <PickMenu
                value={form.agent_path}
                groups={agentGroups}
                searchable
                placeholder={agents.length ? "Pick an agent…" : "no agents found"}
                onChange={(v) => setForm({ ...form, agent_path: v })}
              />
            </div>
            <div className="rt-row rt-row-top">
              <span className="rt-row-label">prompt</span>
              <textarea
                className="rt-textarea"
                rows={3}
                maxLength={100000}
                placeholder="Check my calendar and tasks, write me a brief."
                value={form.prompt}
                onChange={(e) => setForm({ ...form, prompt: e.target.value })}
              />
            </div>
            <div className="rt-row">
              <span className="rt-row-label">schedule</span>
              <PickMenu
                value={form.kind}
                groups={kindGroups}
                onChange={(v) =>
                  setForm({ ...form, kind: v as "daily" | "every" })
                }
              />
              {form.kind === "daily" ? (
                <input
                  className="rt-input short mono"
                  type="time"
                  value={form.at}
                  onChange={(e) => setForm({ ...form, at: e.target.value })}
                />
              ) : (
                <>
                  <input
                    className="rt-input short mono"
                    type="number"
                    min={1}
                    max={168}
                    value={form.hours}
                    onChange={(e) => setForm({ ...form, hours: e.target.value })}
                  />
                  <span className="rt-unit mono">hours</span>
                </>
              )}
            </div>
            <div className="rt-row">
              <span className="rt-row-label">deliver</span>
              <PickMenu
                value={form.deliver}
                groups={deliverGroups}
                onChange={(v) => setForm({ ...form, deliver: v as DeliverKind })}
              />
              {form.deliver === "none" && (
                <span className="rt-unit mono">
                  results stay in Activity + notifications
                </span>
              )}
            </div>
            <div className="rt-actions">
              <button className="btn btn-primary" onClick={save} disabled={saving}>
                {saving ? "Saving…" : form.id ? "Save changes" : "Schedule it"}
              </button>
              {formErr && <span className="rt-form-err">{formErr}</span>}
              {form.id && (
                <button
                  className="rt-head-act rt-delete"
                  onClick={() => remove(form.id!)}
                >
                  delete routine
                </button>
              )}
            </div>
          </section>
        )}

        {routines === null && !error ? (
          <div className="rt-quiet mono">loading routines…</div>
        ) : routines && routines.length === 0 && !form ? (
          <EmptyState icon="✦" title="Runs without being asked">
            Every morning at 7: check my calendar and tasks, message me a
            brief. Set it once — the agent does the rest, on schedule, with
            the same safety rails as any run.
            <div className="rt-empty-act">
              <button className="btn btn-primary" onClick={openCreate}>
                Create your first routine
              </button>
            </div>
          </EmptyState>
        ) : routines && routines.length > 0 ? (
          <section className="home-sec">
            <div className="home-sec-head">
              <span>Scheduled</span>
              <span>
                {routines.filter((r) => r.enabled).length} of {routines.length} on
              </span>
            </div>
            {routines.map((r) => {
              const note = statusNote(r);
              const isBusy = busy.has(r.id) || r.running;
              return (
                <div className="run-line rt-line" key={r.id}>
                  <button
                    className={"task-check" + (r.enabled ? " on" : "")}
                    title={r.enabled ? "Disable" : "Enable"}
                    onClick={() => toggle(r)}
                    type="button"
                  >
                    <CheckIcon />
                  </button>
                  <button
                    className={
                      "rt-name run-line-prompt" + (r.enabled ? "" : " off")
                    }
                    title={r.prompt}
                    onClick={() => openEdit(r)}
                    type="button"
                  >
                    {r.name}
                  </button>
                  {note && (
                    <span
                      className={"rt-note mono" + (note.bad ? " bad" : "")}
                      title={r.last_error ?? r.last_delivery ?? undefined}
                    >
                      {note.text}
                    </span>
                  )}
                  <span className="run-line-meta mono">{meta(r)}</span>
                  <button
                    className="rt-line-act"
                    onClick={() => runNow(r)}
                    disabled={isBusy}
                    type="button"
                  >
                    {isBusy ? "running…" : "run now"}
                  </button>
                </div>
              );
            })}
          </section>
        ) : null}
      </div>
    </>
  );
}
