import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  listKbs,
  createKb,
  ingestText,
  searchKb,
  deleteKb,
  type KbInfo,
  type KbHit,
} from "../lib/api";
import {
  uploadKbFile,
  listKbDocuments,
  validateFile,
  ACCEPT_ATTR,
  type KbDocument,
} from "../lib/knowledgeApi";
import { Topbar } from "../components/Page";
import { PickMenu } from "../components/ui/PickMenu";
import { useToast } from "../components/ui/Toast";
import "../styles/knowledge.css";

/* Knowledge as a single centered ledger column: KB picker over a hairline,
   a dashed drop zone for files (pdf/txt/md/csv), the documents in the KB,
   and an ask box whose grounded chunks render as ledger rows. */

interface UploadRow {
  id: number;
  name: string;
  status: "uploading" | "done" | "error";
  progress: number;
  chunks?: number;
  characters?: number;
  error?: string;
}

const errMsg = (e: unknown): string =>
  e instanceof Error ? e.message : "request failed";

/** Wrap query-term matches in <mark class="hl"> (red's third job). */
function highlight(text: string, query: string): ReactNode {
  const tokens = Array.from(
    new Set(
      query
        .toLowerCase()
        .split(/\s+/)
        .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
        .filter((t) => t.length >= 2),
    ),
  );
  if (tokens.length === 0) return text;
  const parts = text.split(new RegExp(`(${tokens.join("|")})`, "gi"));
  return parts.map((p, i) =>
    i % 2 === 1 ? (
      <mark className="hl" key={i}>
        {p}
      </mark>
    ) : (
      p
    ),
  );
}

export default function Knowledge() {
  const toast = useToast();
  const [kbs, setKbs] = useState<KbInfo[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [newName, setNewName] = useState("");

  const [uploads, setUploads] = useState<UploadRow[]>([]);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const seq = useRef(0);

  const [docs, setDocs] = useState<KbDocument[] | null>(null);
  const [docsAvailable, setDocsAvailable] = useState(true);

  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [pasteTitle, setPasteTitle] = useState("");
  const [ingesting, setIngesting] = useState(false);

  const [query, setQuery] = useState("");
  const [askedQuery, setAskedQuery] = useState("");
  const [hits, setHits] = useState<KbHit[] | null>(null);
  const [asking, setAsking] = useState(false);

  const loadKbs = useCallback(
    () =>
      listKbs()
        .then((l) => {
          setKbs(l);
          setSel((cur) =>
            cur && l.some((k) => k.kb_id === cur) ? cur : (l[0]?.kb_id ?? null),
          );
        })
        .catch(() => setKbs([])), // quiet — the empty state explains itself
    [],
  );

  const loadDocs = useCallback((kbId: string) => {
    listKbDocuments(kbId)
      .then((d) => {
        setDocs(d);
        setDocsAvailable(true);
      })
      .catch(() => {
        setDocs([]);
        setDocsAvailable(false);
      });
  }, []);

  useEffect(() => {
    loadKbs();
  }, [loadKbs]);

  useEffect(() => {
    setDocs(null);
    setHits(null);
    setAskedQuery("");
    setUploads([]);
    if (sel) loadDocs(sel);
  }, [sel, loadDocs]);

  const refresh = useCallback(() => {
    loadKbs();
    if (sel) loadDocs(sel);
  }, [loadKbs, loadDocs, sel]);

  const make = async () => {
    const name = newName.trim().slice(0, 120);
    if (!name) return;
    try {
      const kb = await createKb(name);
      setNewName("");
      setSel(kb.kb_id);
      toast.show(`Created “${kb.name}”`, "ok");
      loadKbs();
    } catch (e) {
      toast.show(errMsg(e), "err");
    }
  };

  const removeCurrent = async () => {
    if (!sel) return;
    try {
      await deleteKb(sel);
      setSel(null);
      loadKbs();
    } catch (e) {
      toast.show(errMsg(e), "err");
    }
  };

  const patchUpload = (id: number, patch: Partial<UploadRow>) =>
    setUploads((rows) =>
      rows.map((r) => (r.id === id ? { ...r, ...patch } : r)),
    );

  const startUpload = (kbId: string, file: File) => {
    const id = ++seq.current;
    const invalid = validateFile(file);
    if (invalid) {
      setUploads((rows) => [
        ...rows,
        { id, name: file.name, status: "error", progress: 0, error: invalid },
      ]);
      return;
    }
    setUploads((rows) => [
      ...rows,
      { id, name: file.name, status: "uploading", progress: 0 },
    ]);
    uploadKbFile(kbId, file, (p) => patchUpload(id, { progress: p }))
      .then((r) => {
        patchUpload(id, {
          status: "done",
          chunks: r.chunks,
          characters: r.characters,
        });
        refresh();
      })
      .catch((e) => patchUpload(id, { status: "error", error: errMsg(e) }));
  };

  const handleFiles = (files: FileList | null) => {
    if (!sel || !files || files.length === 0) return;
    Array.from(files).forEach((f) => startUpload(sel, f));
  };

  const ingestPaste = async () => {
    if (!sel || !pasteText.trim() || ingesting) return;
    setIngesting(true);
    try {
      await ingestText(
        sel,
        pasteText.trim(),
        pasteTitle.trim().slice(0, 300) || undefined,
      );
      setPasteText("");
      setPasteTitle("");
      toast.show("Ingested", "ok");
      refresh();
    } catch (e) {
      toast.show(errMsg(e), "err");
    } finally {
      setIngesting(false);
    }
  };

  const ask = async () => {
    const q = query.trim().slice(0, 2000);
    if (!sel || !q || asking) return;
    setAsking(true);
    try {
      setHits(await searchKb(sel, q));
      setAskedQuery(q);
    } catch (e) {
      setHits(null);
      toast.show(errMsg(e), "err");
    } finally {
      setAsking(false);
    }
  };

  const current = kbs.find((k) => k.kb_id === sel);

  return (
    <>
      <Topbar
        title="Knowledge"
        sub="Upload documents, then ask grounded questions (RAG)"
      />
      <div className="home-col kn-col">
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Knowledge base</span>
            {current && (
              <button className="kn-head-act" onClick={removeCurrent}>
                delete “{current.name}”
              </button>
            )}
          </div>
          <div className="kn-kb-row">
            {kbs.length > 0 && (
              <PickMenu
                value={sel ?? ""}
                groups={[
                  {
                    options: kbs.map((k) => ({
                      value: k.kb_id,
                      label: k.name,
                      meta: `${k.documents} doc${k.documents === 1 ? "" : "s"}`,
                    })),
                  },
                ]}
                onChange={setSel}
                placeholder="Choose a knowledge base…"
                title="Knowledge base"
              />
            )}
            <span className="kn-grow" />
            <input
              className="input"
              placeholder="New knowledge base…"
              maxLength={120}
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && make()}
            />
            <button
              className={"btn" + (kbs.length === 0 ? " btn-primary" : "")}
              onClick={make}
              disabled={!newName.trim()}
            >
              Create
            </button>
          </div>
        </section>

        {!current ? (
          <div className="empty-state">
            <div className="kn-empty-title">No knowledge base yet</div>
            <div className="kn-empty-sub">
              Create one above, drop in files or paste text, then ask grounded
              questions. Agents with the Knowledge capability search the same
              store via kb_search.
            </div>
          </div>
        ) : (
          <>
            <section className="home-sec">
              <div className="home-sec-head">
                <span>Add documents</span>
                <button
                  className="kn-head-act"
                  onClick={() => setPasteOpen((o) => !o)}
                >
                  {pasteOpen ? "hide paste" : "paste text instead"}
                </button>
              </div>

              <button
                type="button"
                className={"kn-drop" + (dragging ? " drag" : "")}
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  handleFiles(e.dataTransfer.files);
                }}
              >
                <span>Drop files here, or click to browse</span>
                <span className="kn-drop-hint">
                  pdf · txt · md · csv — up to 15 MB each
                </span>
              </button>
              <input
                ref={fileRef}
                type="file"
                accept={ACCEPT_ATTR}
                multiple
                hidden
                onChange={(e) => {
                  handleFiles(e.target.files);
                  e.target.value = "";
                }}
              />

              {uploads.map((u) => (
                <div
                  className={"kn-up" + (u.status === "error" ? " err" : "")}
                  key={u.id}
                >
                  <span className="kn-up-name">{u.name}</span>
                  {u.status === "error" && <span className="pill err">failed</span>}
                  <span className="kn-up-meta">
                    {u.status === "uploading"
                      ? `uploading ${Math.round(u.progress * 100)}%`
                      : u.status === "done"
                        ? `${u.chunks} chunk${u.chunks === 1 ? "" : "s"} · ${(
                            u.characters ?? 0
                          ).toLocaleString()} chars`
                        : u.error}
                  </span>
                </div>
              ))}

              {pasteOpen && (
                <div className="kn-paste">
                  <input
                    className="input"
                    placeholder="Title (optional)"
                    maxLength={300}
                    value={pasteTitle}
                    onChange={(e) => setPasteTitle(e.target.value)}
                  />
                  <textarea
                    className="input"
                    rows={6}
                    maxLength={1000000}
                    placeholder="Paste document text to ingest…"
                    value={pasteText}
                    onChange={(e) => setPasteText(e.target.value)}
                  />
                  <div className="kn-paste-actions">
                    <button
                      className="btn btn-primary"
                      onClick={ingestPaste}
                      disabled={!pasteText.trim() || ingesting}
                    >
                      {ingesting ? "Ingesting…" : "Ingest"}
                    </button>
                  </div>
                </div>
              )}
            </section>

            <section className="home-sec">
              <div className="home-sec-head">
                <span>Documents</span>
                {docs !== null && docsAvailable && <span>{docs.length}</span>}
              </div>
              {docs === null ? (
                <div className="kn-quiet-line">Loading…</div>
              ) : !docsAvailable ? (
                <div className="kn-quiet-line">
                  Document listing isn’t available on this knowledge backend —
                  uploads still land in the KB and are searchable below.
                </div>
              ) : docs.length === 0 ? (
                <div className="kn-quiet-line">
                  Nothing here yet — drop a file above or paste text.
                </div>
              ) : (
                docs.map((d) => (
                  <div className="run-line" key={d.document_id}>
                    <span className="run-line-prompt">
                      {d.title || d.source_uri || d.document_id}
                    </span>
                    <span className="run-line-meta mono">
                      {d.source_uri ? `${d.source_uri} · ` : ""}
                      {d.chunks} chunk{d.chunks === 1 ? "" : "s"} ·{" "}
                      {d.characters.toLocaleString()} chars
                    </span>
                  </div>
                ))
              )}
            </section>

            <section className="home-sec">
              <div className="home-sec-head">
                <span>Ask</span>
              </div>
              <div className="kn-ask-row">
                <input
                  className="input"
                  placeholder={`Ask “${current.name}”…`}
                  maxLength={2000}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && ask()}
                />
                <button
                  className="btn btn-primary"
                  onClick={ask}
                  disabled={!query.trim() || asking}
                >
                  {asking ? "Asking…" : "Ask"}
                </button>
              </div>
              {asking ? (
                <div className="kn-quiet-line">retrieving…</div>
              ) : hits === null ? (
                <div className="kn-quiet-line">
                  Ask a question to see the grounded chunks an agent would
                  retrieve.
                </div>
              ) : hits.length === 0 ? (
                <div className="kn-quiet-line">
                  No grounded chunks for “{askedQuery}”.
                </div>
              ) : (
                hits.map((h, i) => (
                  <div className="kn-hit" key={i}>
                    <div className="kn-hit-head">
                      <span className="kn-hit-src">
                        {h.source_uri ?? "pasted text"}
                      </span>
                      <span>{h.similarity.toFixed(2)}</span>
                    </div>
                    <div className="kn-hit-text">
                      {highlight(h.text, askedQuery)}
                    </div>
                  </div>
                ))
              )}
            </section>
          </>
        )}
      </div>
    </>
  );
}
