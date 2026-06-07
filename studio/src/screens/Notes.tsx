import { useEffect, useState } from "react";
import { listNotes, upsertNote, deleteNote, type Note } from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { Markdown } from "../components/Markdown";
import { EmptyState } from "../components/ui/EmptyState";
import { BookIcon, PlusIcon } from "../components/icons";
import { relativeTime } from "../lib/format";
import { useToast } from "../components/ui/Toast";

const BLANK: Note = { id: "", title: "", body: "", updated_at: "" };

export default function Notes() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [sel, setSel] = useState<Note | null>(null);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [preview, setPreview] = useState(false);
  const toast = useToast();

  const load = (selectId?: string) =>
    listNotes().then((n) => {
      setNotes(n);
      if (selectId) {
        const found = n.find((x) => x.id === selectId);
        if (found) open(found);
      }
    });
  useEffect(() => {
    load();
  }, []);

  const open = (n: Note) => {
    setSel(n);
    setTitle(n.title);
    setBody(n.body);
    setPreview(false);
  };
  const create = () => {
    setSel({ ...BLANK });
    setTitle("");
    setBody("");
    setPreview(false);
  };
  const save = async () => {
    if (!sel) return;
    const saved = await upsertNote({
      id: sel.id || undefined,
      title: title.trim() || "Untitled",
      body,
    });
    toast.show("Saved", "ok");
    await load(saved.id);
  };
  const remove = async () => {
    if (!sel?.id) return;
    await deleteNote(sel.id);
    setSel(null);
    load();
  };

  return (
    <>
      <Topbar
        title="Notes"
        sub="Markdown notes your agents can read & write"
        actions={
          <button className="btn btn-primary" onClick={create}>
            <PlusIcon /> New note
          </button>
        }
      />
      <div className="notes-layout">
        <aside className="notes-list">
          {notes.length === 0 ? (
            <div className="dim" style={{ padding: 12, fontSize: 13 }}>
              No notes yet.
            </div>
          ) : (
            notes.map((n) => (
              <button
                key={n.id}
                className={"notes-row" + (sel?.id === n.id ? " active" : "")}
                onClick={() => open(n)}
              >
                <span className="notes-row-title">{n.title || "Untitled"}</span>
                <span className="notes-row-time">{relativeTime(n.updated_at)}</span>
              </button>
            ))
          )}
        </aside>

        <div className="notes-editor">
          {!sel ? (
            <EmptyState icon={<BookIcon />} title="Pick a note">
              Select a note to edit, or create one. Agents with the{" "}
              <code>notes</code> capability read and write these same notes.
            </EmptyState>
          ) : (
            <Page>
              <input
                className="input notes-title"
                placeholder="Title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
              <div className="row spread" style={{ margin: "8px 0" }}>
                <div className="row gap6">
                  <button
                    className={"btn" + (!preview ? " btn-primary" : "")}
                    onClick={() => setPreview(false)}
                  >
                    Write
                  </button>
                  <button
                    className={"btn" + (preview ? " btn-primary" : "")}
                    onClick={() => setPreview(true)}
                  >
                    Preview
                  </button>
                </div>
                <div className="row gap6">
                  {sel.id && (
                    <button className="btn ghost" onClick={remove}>
                      Delete
                    </button>
                  )}
                  <button className="btn btn-primary" onClick={save}>
                    Save
                  </button>
                </div>
              </div>
              {preview ? (
                <div className="card card-pad md">
                  <Markdown>{body || "_Nothing to preview_"}</Markdown>
                </div>
              ) : (
                <textarea
                  className="input notes-body"
                  placeholder="Write in markdown…"
                  value={body}
                  onChange={(e) => setBody(e.target.value)}
                />
              )}
            </Page>
          )}
        </div>
      </div>
    </>
  );
}
