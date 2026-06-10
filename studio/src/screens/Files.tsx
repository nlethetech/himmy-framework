import { useCallback, useEffect, useState } from "react";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { RefreshIcon } from "../components/icons";
import { useToast } from "../components/ui/Toast";
import { relativeTime } from "../lib/format";
import {
  downloadSandboxFile,
  formatBytes,
  listSandboxFiles,
  type FileListResponse,
} from "../lib/artifactsApi";
import "../styles/chatsurfaces.css";

// Files — the agent file-sandbox as a ledger: every file under fs_root (what
// write_file produced, what read_file can see), newest first, one quiet
// download per row. Single centered column; no boxes.

export default function Files() {
  const [data, setData] = useState<FileListResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyPath, setBusyPath] = useState<string | null>(null);
  const toast = useToast();

  const load = useCallback(() => {
    setLoading(true);
    listSandboxFiles(500)
      .then((d) => {
        setData(d);
        setErr(null);
      })
      .catch((e) => setErr(String((e as Error).message ?? e)))
      .finally(() => setLoading(false));
  }, []);
  useEffect(load, [load]);

  const get = async (path: string) => {
    if (busyPath) return;
    setBusyPath(path);
    try {
      await downloadSandboxFile(path);
    } catch (e: unknown) {
      const status = (e as { status?: number })?.status;
      toast.show(
        status === 404
          ? "File is gone — it may have been moved or deleted."
          : String((e as Error).message ?? e),
        "err",
      );
    } finally {
      setBusyPath(null);
    }
  };

  const items = data?.items ?? [];

  return (
    <>
      <Topbar
        title="Files"
        sub="the agent's file sandbox"
        actions={
          <button className="btn" onClick={load} disabled={loading}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      {!loading && !err && data && items.length === 0 ? (
        <EmptyState icon="✦" title="The sandbox is empty">
          When an agent writes a file — a report, an export, a draft — it lands
          here, ready to download.
        </EmptyState>
      ) : (
        <div className="home-col">
          <section className="home-sec">
            <div className="home-sec-head">
              <span>
                Sandbox
                {data && items.length > 0
                  ? ` · ${data.total} file${data.total === 1 ? "" : "s"}`
                  : ""}
              </span>
              {data && (
                <span className="mono" title={data.root}>
                  fs_root
                </span>
              )}
            </div>
            {err ? (
              <div className="files-quiet err">{err}</div>
            ) : loading && !data ? (
              <div className="files-quiet">Reading the sandbox…</div>
            ) : (
              <>
                {items.map((f) => {
                  const dir = f.path.includes("/")
                    ? f.path.slice(0, f.path.lastIndexOf("/") + 1)
                    : "";
                  return (
                    <div className="files-row" key={f.path}>
                      <span className="files-name" title={f.path}>
                        {dir && <span className="files-dir">{dir}</span>}
                        {f.name}
                      </span>
                      <span className="files-size mono">
                        {formatBytes(f.size)}
                      </span>
                      <span className="files-meta mono" title={f.mtime}>
                        {relativeTime(f.mtime)}
                      </span>
                      <button
                        className="afc-get mono"
                        onClick={() => get(f.path)}
                        disabled={busyPath !== null}
                        type="button"
                      >
                        {busyPath === f.path ? "…" : "download"}
                      </button>
                    </div>
                  );
                })}
                {data?.truncated && (
                  <div className="files-quiet mono">
                    showing {items.length} of {data.total} — newest first
                  </div>
                )}
                {data && <div className="files-root mono">{data.root}</div>}
              </>
            )}
          </section>
        </div>
      )}
    </>
  );
}
