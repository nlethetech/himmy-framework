import { useState } from "react";
import {
  downloadSandboxFile,
  formatBytes,
  type Artifact,
} from "../lib/artifactsApi";

// A flat hairline ledger block under an answer: the files this turn wrote to
// the sandbox, each with name (sans), size + path (mono dim) and a download
// action wired to /api/studio/files/download. A vanished file degrades to an
// inline "missing" note instead of navigating away.

function ArtifactRow({ artifact }: { artifact: Artifact }) {
  const [state, setState] = useState<"idle" | "busy" | "missing" | "err">(
    "idle",
  );

  const get = async () => {
    if (state === "busy") return;
    setState("busy");
    try {
      await downloadSandboxFile(artifact.path);
      setState("idle");
    } catch (e: unknown) {
      const status = (e as { status?: number })?.status;
      setState(status === 404 ? "missing" : "err");
    }
  };

  return (
    <div className="afc-row">
      <span className="afc-name">{artifact.name}</span>
      <span className="afc-meta mono">
        {artifact.bytes != null && `${formatBytes(artifact.bytes)} · `}
        {artifact.path}
      </span>
      {state === "missing" ? (
        <span className="afc-state mono err">missing</span>
      ) : state === "err" ? (
        <button className="afc-get mono err" onClick={get} type="button">
          retry
        </button>
      ) : (
        <button
          className="afc-get mono"
          onClick={get}
          disabled={state === "busy"}
          type="button"
        >
          {state === "busy" ? "…" : "download"}
        </button>
      )}
    </div>
  );
}

export function ArtifactCard({ artifacts }: { artifacts: Artifact[] }) {
  if (artifacts.length === 0) return null;
  return (
    <div className="afc">
      <div className="afc-head">
        {artifacts.length === 1 ? "artifact" : `artifacts · ${artifacts.length}`}
      </div>
      {artifacts.map((a) => (
        <ArtifactRow artifact={a} key={a.path} />
      ))}
    </div>
  );
}
