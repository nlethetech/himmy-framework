import { useEffect, useMemo, useState } from "react";
import {
  listModels,
  compare,
  type ModelProvider,
  type CompareTarget,
  type CompareResult,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { SendIcon, DoctorIcon } from "../components/icons";

function key(t: CompareTarget) {
  return `${t.provider}::${t.model}`;
}

export default function Compare() {
  const [providers, setProviders] = useState<ModelProvider[]>([]);
  const [picked, setPicked] = useState<CompareTarget[]>([]);
  const [prompt, setPrompt] = useState("");
  const [system, setSystem] = useState("");
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<CompareResult[] | null>(null);

  useEffect(() => {
    listModels()
      .then(setProviders)
      .catch(() => setProviders([]));
  }, []);

  const allTargets = useMemo<CompareTarget[]>(
    () =>
      providers
        .filter((p) => p.available)
        .flatMap((p) => p.models.map((m) => ({ provider: p.provider, model: m.name }))),
    [providers],
  );

  const pickedKeys = new Set(picked.map(key));
  const toggle = (t: CompareTarget) => {
    if (pickedKeys.has(key(t))) {
      setPicked((p) => p.filter((x) => key(x) !== key(t)));
    } else if (picked.length < 6) {
      setPicked((p) => [...p, t]);
    }
  };

  const run = async () => {
    if (!prompt.trim() || picked.length === 0) return;
    setRunning(true);
    setResults(null);
    try {
      const r = await compare(prompt.trim(), picked, system.trim() || null);
      setResults(r);
    } catch {
      setResults([]);
    } finally {
      setRunning(false);
    }
  };

  return (
    <>
      <Topbar
        title="Compare"
        sub="Run one prompt across several models, side by side"
        actions={
          <button
            className="btn btn-primary"
            onClick={run}
            disabled={running || !prompt.trim() || picked.length === 0}
          >
            <SendIcon /> {running ? "Running…" : `Run · ${picked.length}`}
          </button>
        }
      />
      <Page>
        <div className="cmp-grid">
          <div className="card card-pad cmp-setup">
            <label className="cmp-label">Prompt</label>
            <textarea
              className="textarea"
              rows={5}
              placeholder="Ask the same thing of every model…"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <label className="cmp-label">System (optional)</label>
            <textarea
              className="textarea"
              rows={2}
              placeholder="Shared system instruction"
              value={system}
              onChange={(e) => setSystem(e.target.value)}
            />
            <label className="cmp-label">
              Models <span className="cmp-dim">({picked.length}/6)</span>
            </label>
            {allTargets.length === 0 ? (
              <div className="cmp-dim">
                No models available. Start Ollama or install the Claude CLI.
              </div>
            ) : (
              <div className="cmp-models">
                {allTargets.map((t) => (
                  <button
                    key={key(t)}
                    className={"cmp-chip" + (pickedKeys.has(key(t)) ? " on" : "")}
                    onClick={() => toggle(t)}
                  >
                    <span className="cmp-chip-prov">{t.provider}</span>
                    {t.model}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="cmp-results">
            {!results && !running && (
              <EmptyState icon={<DoctorIcon />} title="No comparison yet">
                Pick up to 6 models, write a prompt, and hit Run.
              </EmptyState>
            )}
            {running && <div className="cmp-dim">Running {picked.length} models…</div>}
            {results && results.length === 0 && (
              <div className="cmp-dim">No results.</div>
            )}
            {results && results.length > 0 && (
              <div className="cmp-cols">
                {results.map((r) => (
                  <ResultCard key={`${r.provider}::${r.model}`} r={r} />
                ))}
              </div>
            )}
          </div>
        </div>
      </Page>
    </>
  );
}

function ResultCard({ r }: { r: CompareResult }) {
  return (
    <div className={"card cmp-card" + (r.ok ? "" : " err")}>
      <div className="cmp-card-head">
        <span className="cmp-chip-prov">{r.provider}</span>
        <span className="cmp-card-model">{r.model}</span>
      </div>
      <div className="cmp-card-body mono">
        {r.ok ? r.output : <span className="cmp-error">{r.error}</span>}
      </div>
      {r.ok && (
        <div className="cmp-card-meta mono">
          {r.latency_ms != null && <span>{Math.round(r.latency_ms)}ms</span>}
          {(r.input_tokens != null || r.output_tokens != null) && (
            <span>
              {r.input_tokens ?? 0}→{r.output_tokens ?? 0} tok
            </span>
          )}
          {r.cost != null && r.cost > 0 && <span>${r.cost.toFixed(4)}</span>}
        </div>
      )}
    </div>
  );
}
