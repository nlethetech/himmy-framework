import { useEffect, useState } from "react";
import {
  listKbs,
  createKb,
  ingestText,
  searchKb,
  deleteKb,
  type KbInfo,
  type KbHit,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { BookIcon, PlusIcon } from "../components/icons";
import { useToast } from "../components/ui/Toast";

export default function Knowledge() {
  const [kbs, setKbs] = useState<KbInfo[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [doc, setDoc] = useState("");
  const [title, setTitle] = useState("");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<KbHit[] | null>(null);
  const toast = useToast();

  const load = () =>
    listKbs().then((l) => {
      setKbs(l);
      setSel((cur) => cur ?? l[0]?.kb_id ?? null);
    });
  useEffect(() => {
    load();
  }, []);

  const make = async () => {
    if (!newName.trim()) return;
    const kb = await createKb(newName.trim());
    setNewName("");
    setSel(kb.kb_id);
    toast.show(`Created “${kb.name}”`, "ok");
    load();
  };
  const ingest = async () => {
    if (!sel || !doc.trim()) return;
    await ingestText(sel, doc.trim(), title.trim() || undefined);
    setDoc("");
    setTitle("");
    toast.show("Ingested", "ok");
    setHits(null);
    load();
  };
  const run = async () => {
    if (!sel || !query.trim()) return;
    setHits(await searchKb(sel, query.trim()));
  };
  const remove = async (id: string) => {
    await deleteKb(id);
    if (sel === id) setSel(null);
    load();
  };

  const current = kbs.find((k) => k.kb_id === sel);

  return (
    <>
      <Topbar title="Knowledge" sub="Give agents documents to ground on (RAG)" />
      <Page>
        <div className="kb-head">
          <div className="row gap6">
            <input
              className="input"
              placeholder="New knowledge base name…"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && make()}
              style={{ maxWidth: 260 }}
            />
            <button className="btn btn-primary" onClick={make}>
              <PlusIcon /> New KB
            </button>
          </div>
          {kbs.length > 0 && (
            <select
              className="input"
              value={sel ?? ""}
              onChange={(e) => setSel(e.target.value)}
              style={{ maxWidth: 260 }}
            >
              {kbs.map((k) => (
                <option key={k.kb_id} value={k.kb_id}>
                  {k.name} · {k.documents} doc{k.documents === 1 ? "" : "s"}
                </option>
              ))}
            </select>
          )}
        </div>

        {!current ? (
          <EmptyState icon={<BookIcon />} title="No knowledge base yet">
            Create a knowledge base, paste in some text, then test retrieval. Agents
            with the <b>Knowledge</b> capability can search it via <code>kb_search</code>.
          </EmptyState>
        ) : (
          <div className="mem-cols">
            <div className="card card-pad">
              <div className="row spread">
                <span className="section-title">Ingest into “{current.name}”</span>
                <button className="btn ghost" onClick={() => remove(current.kb_id)}>
                  Delete KB
                </button>
              </div>
              <input
                className="input"
                placeholder="Title (optional)"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                style={{ marginBottom: 8 }}
              />
              <textarea
                className="input"
                rows={8}
                placeholder="Paste document text to ingest…"
                value={doc}
                onChange={(e) => setDoc(e.target.value)}
              />
              <div className="mt8">
                <button className="btn btn-primary" onClick={ingest} disabled={!doc.trim()}>
                  Ingest
                </button>
              </div>
            </div>

            <div className="card card-pad">
              <span className="section-title">Test retrieval</span>
              <div className="row gap6">
                <input
                  className="input"
                  placeholder="Search this knowledge base…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && run()}
                />
                <button className="btn" onClick={run}>
                  Search
                </button>
              </div>
              {hits === null ? (
                <div className="dim" style={{ padding: 8 }}>
                  Run a query to see the chunks an agent would retrieve.
                </div>
              ) : hits.length === 0 ? (
                <div className="dim" style={{ padding: 8 }}>
                  No matching chunks.
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
        )}
      </Page>
    </>
  );
}
