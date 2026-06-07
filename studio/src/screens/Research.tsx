import { useRef, useState } from "react";
import { streamResearch, type RunEvent } from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { Markdown } from "../components/Markdown";
import { EmptyState } from "../components/ui/EmptyState";
import { CognitionTrace, type CogStep } from "../components/Cognition";
import { SearchIcon, SendIcon } from "../components/icons";

const PROVIDERS = [
  { value: "", label: "Auto" },
  { value: "claude-cli", label: "Claude CLI" },
  { value: "ollama", label: "Ollama" },
  { value: "pydantic-ai", label: "Cloud (key)" },
];

const EXAMPLES = [
  "Compare drip vs sprinkler irrigation for a small orchard",
  "What's the current state of solid-state batteries?",
  "Best practices for integrated duck-rice farming",
];

export default function Research() {
  const [query, setQuery] = useState("");
  const [provider, setProvider] = useState("");
  const [busy, setBusy] = useState(false);
  const [steps, setSteps] = useState<CogStep[]>([]);
  const [report, setReport] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sources = steps
    .filter((s) => s.kind === "tool")
    .map((s) => s as Extract<CogStep, { kind: "tool" }>);

  const run = async () => {
    const q = query.trim();
    if (!q || busy) return;
    setBusy(true);
    setStarted(true);
    setSteps([]);
    setReport("");
    setError(null);
    const ac = new AbortController();
    abortRef.current = ac;

    const onEvent = (e: RunEvent) => {
      switch (e.type) {
        case "tool":
          setSteps((s) => [
            ...s,
            {
              kind: "tool",
              name: e.name,
              agent: e.agent,
              args: e.args ?? null,
              result: e.result ?? null,
              outcome: e.outcome ?? null,
              read_only: e.read_only ?? null,
              latency_ms: e.latency_ms ?? null,
            },
          ]);
          break;
        case "reason":
          setSteps((s) => [...s, { kind: "reason", agent: e.agent, text: e.text }]);
          break;
        case "grounding":
          setSteps((s) => [
            ...s,
            {
              kind: "grounding",
              agent: e.agent,
              source: e.source,
              query: e.query,
              citations: e.citations,
            },
          ]);
          break;
        case "token":
          setReport((r) => r + e.delta);
          break;
        case "message":
          setReport(e.text);
          break;
        case "done":
          if (e.output_text) setReport(e.output_text);
          break;
        case "error":
          setError(e.message);
          break;
      }
    };

    try {
      await streamResearch({ query: q, provider: provider || null }, onEvent, ac.signal);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setBusy(false);
  };

  return (
    <>
      <Topbar
        title="Deep Research"
        sub="Investigate a question across the web and get a sourced report"
        actions={
          <select
            className="composer-pick"
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            disabled={busy}
            title="Inference provider"
          >
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        }
      />
      <Page>
        <div className="card card-pad rsch-input">
          <textarea
            className="textarea"
            rows={3}
            placeholder="What do you want researched? Be specific."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
            }}
          />
          <div className="rsch-actions">
            <span className="rsch-hint">⌘/Ctrl + Enter to run</span>
            {busy ? (
              <button className="btn" onClick={stop}>
                Stop
              </button>
            ) : (
              <button
                className="btn btn-primary"
                onClick={run}
                disabled={!query.trim()}
              >
                <SearchIcon /> Research
              </button>
            )}
          </div>
        </div>

        {!started ? (
          <EmptyState icon={<SearchIcon />} title="Ask a research question">
            Deep Research plans angles, searches the web, reads sources, and writes a
            cited report.
            <div className="rsch-examples">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  className="rsch-example"
                  onClick={() => setQuery(ex)}
                >
                  {ex}
                </button>
              ))}
            </div>
          </EmptyState>
        ) : (
          <div className="rsch-grid">
            <div className="rsch-report card card-pad">
              {error && <div className="google-hint err">⚠ {error}</div>}
              {report ? (
                <Markdown>{report}</Markdown>
              ) : busy ? (
                <div className="rsch-working">
                  <span className="spinner" /> Researching — searching{" "}
                  {sources.length > 0 ? `(${sources.length} lookups)` : "…"}
                </div>
              ) : (
                <div className="cmp-dim">No report produced.</div>
              )}
            </div>
            <aside className="rsch-side">
              <div className="rsch-side-head">
                <SendIcon /> Trail{steps.length > 0 ? ` · ${steps.length}` : ""}
              </div>
              {steps.length === 0 ? (
                <div className="cmp-dim">Steps will appear as the agent works.</div>
              ) : (
                <CognitionTrace steps={steps} />
              )}
            </aside>
          </div>
        )}
      </Page>
    </>
  );
}
