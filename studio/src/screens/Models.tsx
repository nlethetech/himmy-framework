import { useEffect, useState } from "react";
import { listModels, type ModelProvider } from "../lib/api";
import { Topbar, Page, Loading } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { DoctorIcon, RefreshIcon } from "../components/icons";

export default function Models() {
  const [providers, setProviders] = useState<ModelProvider[] | null>(null);

  const load = () => listModels().then(setProviders).catch(() => setProviders([]));
  useEffect(() => {
    load();
  }, []);

  const total =
    providers?.reduce((n, p) => n + p.models.length, 0) ?? 0;

  return (
    <>
      <Topbar
        title="Models"
        sub="Providers and the models available to your agents"
        actions={
          <button className="btn" onClick={load}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      <Page>
        {providers === null ? (
          <Loading label="Detecting models…" />
        ) : total === 0 ? (
          <EmptyState icon={<DoctorIcon />} title="No models detected">
            Start Ollama (<code>ollama serve</code> then <code>ollama pull …</code>) or
            sign in to the Claude CLI, then refresh. Agents pick a provider/model in
            the composer or their spec.
          </EmptyState>
        ) : (
          providers.map((p) => (
            <div className="models-provider" key={p.provider}>
              <div className="models-provider-head">
                <span
                  className="dot"
                  style={{ color: p.available ? "var(--ok)" : "var(--text-dim)" }}
                />
                {p.provider}
                <span className="dim" style={{ fontSize: 12 }}>
                  {p.available ? "available" : "not detected"} · {p.models.length}{" "}
                  model{p.models.length === 1 ? "" : "s"}
                </span>
              </div>
              {p.models.length > 0 && (
                <div className="models-list">
                  {p.models.map((m) => (
                    <div className="models-row" key={m.name}>
                      <span className="models-name mono">{m.name}</span>
                      {m.accuracy != null && (
                        <span className="models-stat">
                          {Math.round(m.accuracy * 100)}% acc
                        </span>
                      )}
                      {m.latency != null && (
                        <span className="models-stat">
                          {m.latency < 1000
                            ? `${Math.round(m.latency)}ms`
                            : `${(m.latency / 1000).toFixed(1)}s`}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </Page>
    </>
  );
}
