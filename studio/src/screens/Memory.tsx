import { useEffect, useState } from "react";
import {
  listMemorySubjects,
  listMemories,
  addMemory,
  updateMemory,
  forgetMemory,
  recallMemory,
  MEMORY_TEXT_MAX,
  type MemoryItem,
  type MemoryHit,
} from "../lib/memoryApi";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { Modal } from "../components/ui/Modal";
import { PickMenu } from "../components/ui/PickMenu";
import { MemoryIcon } from "../components/icons";
import { relativeTime } from "../lib/format";
import { useToast } from "../components/ui/Toast";
import "../styles/memory.css";

/* Memory as a single centered ledger: one column, one section. The section
   header flips between MEMORIES (browsing a subject) and RECALL (a semantic
   query is active). Rows are run-lines — text flex-left, subject + date mono
   right, quiet ✎ / ✕ actions on hover. Editing happens inline; deletion is an
   erasure right and says so before it happens. */

const errMsg = (e: unknown) =>
  e instanceof Error ? e.message : "Something went wrong";

export default function Memory() {
  const [subjects, setSubjects] = useState<string[]>([]);
  const [subject, setSubject] = useState("default");
  const [items, setItems] = useState<MemoryItem[] | null>(null);
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<MemoryHit[] | null>(null);
  const [recalling, setRecalling] = useState(false);

  const [editId, setEditId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [saving, setSaving] = useState(false);

  const [toDelete, setToDelete] = useState<MemoryItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  const toast = useToast();

  const loadSubjects = () =>
    listMemorySubjects()
      .then((s) => {
        setSubjects(s);
        setSubject((cur) => (s.includes(cur) ? cur : (s[0] ?? "default")));
      })
      .catch(() => setSubjects(["default"]));
  const loadItems = (s: string) =>
    listMemories(s)
      .then(setItems)
      .catch((e) => {
        setItems([]);
        toast.show(errMsg(e), "err");
      });

  useEffect(() => {
    loadSubjects();
  }, []);
  useEffect(() => {
    setItems(null);
    setHits(null);
    setQuery("");
    setEditId(null);
    loadItems(subject);
  }, [subject]);

  const add = async () => {
    const text = draft.trim();
    if (!text) return;
    try {
      await addMemory(text, subject);
      setDraft("");
      toast.show("Remembered", "ok");
      loadItems(subject);
      loadSubjects();
    } catch (e) {
      toast.show(errMsg(e), "err");
    }
  };

  const runRecall = async () => {
    const q = query.trim();
    if (!q) return;
    setRecalling(true);
    try {
      setHits(await recallMemory(q, subject));
    } catch (e) {
      toast.show(errMsg(e), "err");
    } finally {
      setRecalling(false);
    }
  };

  const beginEdit = (m: MemoryItem) => {
    setEditId(m.memory_id);
    setEditText(m.text);
  };
  const saveEdit = async () => {
    const text = editText.trim();
    if (!editId || !text || saving) return;
    setSaving(true);
    try {
      const updated = await updateMemory(editId, text);
      setItems(
        (cur) =>
          cur?.map((m) => (m.memory_id === updated.memory_id ? updated : m)) ??
          cur,
      );
      setEditId(null);
      toast.show("Saved", "ok");
    } catch (e) {
      toast.show(errMsg(e), "err");
    } finally {
      setSaving(false);
    }
  };

  const confirmDelete = async () => {
    if (!toDelete || deleting) return;
    setDeleting(true);
    try {
      const r = await forgetMemory(toDelete.memory_id);
      if (!r.ok) toast.show("Already gone", "info");
      else toast.show("Erased", "ok");
      setToDelete(null);
      loadItems(subject);
      loadSubjects();
    } catch (e) {
      toast.show(errMsg(e), "err");
    } finally {
      setDeleting(false);
    }
  };

  const showRecall = hits !== null;
  const loaded = items !== null;

  return (
    <>
      <Topbar title="Memory" sub="What your agents remember" />
      <div className="home-col">
        <div className="mm-controls">
          <PickMenu
            value={subject}
            groups={[
              {
                label: "Subject",
                options: (subjects.length ? subjects : [subject]).map((s) => ({
                  value: s,
                  label: s,
                })),
              },
            ]}
            onChange={setSubject}
            title="Filter memories by subject"
          />
          <input
            className="input"
            placeholder="Recall — ask what this subject remembers…"
            value={query}
            maxLength={2000}
            onChange={(e) => {
              setQuery(e.target.value);
              if (!e.target.value.trim()) setHits(null);
            }}
            onKeyDown={(e) => e.key === "Enter" && runRecall()}
          />
        </div>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>
              {showRecall
                ? `Recall · ${hits.length} hit${hits.length === 1 ? "" : "s"}`
                : recalling
                  ? "Recalling…"
                  : `Memories${loaded ? ` · ${items.length}` : ""}`}
            </span>
            {showRecall ? (
              <button
                className="mm-clear"
                onClick={() => {
                  setHits(null);
                  setQuery("");
                }}
              >
                clear →
              </button>
            ) : (
              <span>{subject}</span>
            )}
          </div>

          {showRecall ? (
            hits.length === 0 ? (
              <div className="mm-quiet">
                Nothing relevant found for “{query.trim()}”.
              </div>
            ) : (
              hits.map((h) => (
                <div className="run-line" key={h.memory_id}>
                  <span className="run-line-prompt">{h.text}</span>
                  <span className="mm-sim" title="Similarity to your query">
                    {h.similarity.toFixed(2)}
                  </span>
                </div>
              ))
            )
          ) : !loaded ? (
            <div className="mm-quiet">Loading…</div>
          ) : items.length === 0 ? (
            <EmptyState icon={<MemoryIcon />} title="Nothing remembered yet">
              Facts you add here are what your agents recall when they act for{" "}
              <b>{subject}</b>. Remember the first one below.
            </EmptyState>
          ) : (
            items.map((m) =>
              editId === m.memory_id ? (
                <div className="mm-edit" key={m.memory_id}>
                  <textarea
                    className="input"
                    value={editText}
                    maxLength={MEMORY_TEXT_MAX}
                    autoFocus
                    onChange={(e) => setEditText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Escape") setEditId(null);
                      if (e.key === "Enter" && (e.metaKey || e.ctrlKey))
                        saveEdit();
                    }}
                  />
                  <div className="mm-edit-foot">
                    <span className="mm-edit-hint">
                      {m.subject_id} · {relativeTime(m.created_at)}
                    </span>
                    <button className="btn" onClick={() => setEditId(null)}>
                      Cancel
                    </button>
                    <button
                      className="btn btn-primary"
                      disabled={saving || !editText.trim()}
                      onClick={saveEdit}
                    >
                      {saving ? "Saving…" : "Save"}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="run-line mm-row" key={m.memory_id}>
                  <span className="run-line-prompt" title={m.text}>
                    {m.text}
                  </span>
                  <span className="run-line-meta mono">
                    {m.subject_id} · {relativeTime(m.created_at)}
                  </span>
                  <span className="mm-acts">
                    <button
                      className="mm-act"
                      title="Edit"
                      aria-label="Edit memory"
                      onClick={() => beginEdit(m)}
                    >
                      ✎
                    </button>
                    <button
                      className="mm-act danger"
                      title="Erase"
                      aria-label="Erase memory"
                      onClick={() => setToDelete(m)}
                    >
                      ✕
                    </button>
                  </span>
                </div>
              ),
            )
          )}

          <div className="mm-add">
            <input
              className="input"
              placeholder={`Remember something for ${subject}…`}
              value={draft}
              maxLength={MEMORY_TEXT_MAX}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && add()}
            />
            <button
              className="btn btn-primary"
              disabled={!draft.trim()}
              onClick={add}
            >
              Add
            </button>
          </div>
        </section>
      </div>

      {toDelete && (
        <Modal
          title="Erase this memory"
          onClose={() => (deleting ? undefined : setToDelete(null))}
          footer={
            <>
              <button className="btn" onClick={() => setToDelete(null)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                disabled={deleting}
                onClick={confirmDelete}
              >
                {deleting ? "Erasing…" : "Erase forever"}
              </button>
            </>
          }
        >
          <blockquote className="mm-quote">{toDelete.text}</blockquote>
          <p className="mm-erasure">
            Deleting is an erasure: this record is permanently removed from{" "}
            <b>{toDelete.subject_id}</b>’s memory. Your agents will never
            recall it again, and it cannot be recovered.
          </p>
        </Modal>
      )}
    </>
  );
}
