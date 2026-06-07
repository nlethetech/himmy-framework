import { useEffect, useState } from "react";
import {
  listTasks,
  addTask,
  setTaskDone,
  deleteTask,
  type Task,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { PlusIcon, XIcon, CheckIcon } from "../components/icons";

export default function Tasks() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [draft, setDraft] = useState("");
  const [due, setDue] = useState("");

  const load = () => listTasks().then(setTasks).catch(() => setTasks([]));
  useEffect(() => {
    load();
  }, []);

  const add = async () => {
    if (!draft.trim()) return;
    await addTask(draft.trim(), due || null);
    setDraft("");
    setDue("");
    load();
  };
  const toggle = async (t: Task) => {
    await setTaskDone(t.id, !t.done);
    load();
  };
  const remove = async (id: string) => {
    await deleteTask(id);
    load();
  };

  const open = tasks.filter((t) => !t.done);
  const done = tasks.filter((t) => t.done);

  return (
    <>
      <Topbar title="Tasks" sub="A shared to-do board you and your agents both use" />
      <Page>
        <div className="card card-pad" style={{ maxWidth: 640 }}>
          <div className="row gap6">
            <input
              className="input"
              placeholder="What needs doing?"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && add()}
              style={{ flex: 1 }}
            />
            <input
              className="input"
              type="text"
              placeholder="due (YYYY-MM-DD)"
              value={due}
              onChange={(e) => setDue(e.target.value)}
              style={{ width: 150 }}
            />
            <button className="btn btn-primary" onClick={add}>
              <PlusIcon /> Add
            </button>
          </div>

          {tasks.length === 0 ? (
            <EmptyState icon={<CheckIcon />} title="No tasks yet">
              Add a task above. Agents with the <code>tasks</code> capability can add
              and complete these too.
            </EmptyState>
          ) : (
            <div className="task-list">
              {open.map((t) => (
                <TaskRow key={t.id} t={t} onToggle={toggle} onRemove={remove} />
              ))}
              {done.length > 0 && (
                <div className="task-done-head">Done ({done.length})</div>
              )}
              {done.map((t) => (
                <TaskRow key={t.id} t={t} onToggle={toggle} onRemove={remove} />
              ))}
            </div>
          )}
        </div>
      </Page>
    </>
  );
}

function TaskRow({
  t,
  onToggle,
  onRemove,
}: {
  t: Task;
  onToggle: (t: Task) => void;
  onRemove: (id: string) => void;
}) {
  return (
    <div className={"task-row" + (t.done ? " done" : "")}>
      <button
        className={"task-check" + (t.done ? " on" : "")}
        onClick={() => onToggle(t)}
        aria-label="toggle"
      >
        {t.done && <CheckIcon />}
      </button>
      <span className="task-title">{t.title}</span>
      {t.due && <span className="task-due mono">{t.due}</span>}
      <button className="task-del" onClick={() => onRemove(t.id)}>
        <XIcon />
      </button>
    </div>
  );
}
