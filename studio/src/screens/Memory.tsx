import { useEffect, useState } from "react";
import {
  listMemorySubjects,
  listMemories,
  addMemory,
  forgetMemory,
  recallMemory,
  type MemoryItem,
  type MemoryHit,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { MemoryIcon, PlusIcon, XIcon } from "../components/icons";
import { relativeTime } from "../lib/format";
import { useToast } from "../components/ui/Toast";

export default function Memory() {
  const [subjects, setSubjects] = useState<string[]>([]);
  const [subject, setSubject] = useState("default");
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<MemoryHit[] | null>(null);
  const toast = useToast();

  const loadSubjects = () =>
    listMemorySubjects().then((s) => {
      setSubjects(s);
      setSubject((cur) => (s.includes(cur) ? cur : s[0] ?? "default"));
    });
  const loadItems = (s: string) => listMemories(s).then(setItems).catch(() => setItems([]));

  useEffect(() => {
    loadSubjects();
  }, []);
  useEffect(() => {
    loadItems(subject);
    setHits(null);
  }, [subject]);

  const add = async () => {
    if (!draft.trim()) return;
    await addMemory(draft.trim(), subject);
    setDraft("");
    toast.show("Remembered", "ok");
    loadItems(subject);
    loadSubjects();
  };
  const remove = async (id: string) => {
    await forgetMemory(id);
    loadItems(subject);
  };
  const runRecall = async () => {
    if (!query.trim()) return;
    setHits(await recallMemory(query.trim(), subject));
  };

  return (
    <>
      <Topbar title="Memory" sub="What your agents remember" />
      <Page>
        <div className="mem-bar">
          <label className="field" style={{ margin: 0 }}>
            <span className="field-label">Subject</span>
            <select
              className="input"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
            >
              {subjects.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="mem-cols">
          <div className="card card-pad">
            <span className="section-title">Add a memory</span>
            <div className="row gap6">
              <input
                className="input"
                placeholder="A fact this subject should remember…"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && add()}
              />
              <button className="btn btn-primary" onClick={add}>
                <PlusIcon /> Add
              </button>
            </div>

            <div className="mem-list">
              {items.length === 0 ? (
                <div className="dim" style={{ padding: 8 }}>
                  No memories for <b>{subject}</b> yet.
                </div>
              ) : (
                items.map((m) => (
                  <div className="mem-row" key={m.memory_id}>
                    <span className="mem-text">{m.text}</span>
                    <span className="mem-meta">{relativeTime(m.created_at)}</span>
                    <button
                      className="mem-del"
                      onClick={() => remove(m.memory_id)}
                      title="Forget"
                    >
                      <XIcon />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="card card-pad">
            <span className="section-title">Recall (semantic search)</span>
            <div className="row gap6">
              <input
                className="input"
                placeholder="Ask what this subject remembers…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runRecall()}
              />
              <button className="btn" onClick={runRecall}>
                Recall
              </button>
            </div>
            {hits === null ? (
              <EmptyState icon={<MemoryIcon />} title="Test recall">
                Run a query to see what the agent would pull from memory, ranked by
                similarity.
              </EmptyState>
            ) : hits.length === 0 ? (
              <div className="dim" style={{ padding: 8 }}>
                Nothing relevant found.
              </div>
            ) : (
              <div className="mem-hits">
                {hits.map((h, i) => (
                  <div className="mem-hit" key={i}>
                    <span className="cite-sim">
                      <span
                        className="cite-sim-fill"
                        style={{ width: Math.round(h.similarity * 100) + "%" }}
                      />
                    </span>
                    <span>{h.text}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </Page>
    </>
  );
}
