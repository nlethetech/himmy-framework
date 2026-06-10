import { useCallback, useEffect, useRef, useState } from "react";
import { Topbar } from "../components/Page";
import { useToast } from "../components/ui/Toast";
import { RefreshIcon } from "../components/icons";
import { relativeTime } from "../lib/format";
import {
  getOllama,
  getProviders,
  saveProviderKey,
  streamPull,
  validateProviderKey,
  type OllamaStatus,
  type ProviderInfo,
  type ProvidersResponse,
} from "../lib/modelsApi";
import "../styles/models.css";

/* Models as a ledger — three sections on one centered column:
   PROVIDERS (what himmy can run on, and the keys that unlock the metered ones),
   LOCAL MODELS (the Ollama library + a pull lane), and ROUTING (how a brain is
   picked). Status is mono and quiet; red only for actions, activity, errors. */

interface KeyEditor {
  draft: string;
  busy: "save" | "validate" | null;
  verdict: { ok: boolean; detail: string } | null;
}

interface PullState {
  running: boolean;
  status: string;
  completed: number | null;
  total: number | null;
  error: string | null;
}

const IDLE_PULL: PullState = {
  running: false,
  status: "",
  completed: null,
  total: null,
  error: null,
};

function formatBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export default function Models() {
  const toast = useToast();
  const [prov, setProv] = useState<ProvidersResponse | null>(null);
  const [provErr, setProvErr] = useState<string | null>(null);
  const [ollama, setOllama] = useState<OllamaStatus | null>(null);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [editors, setEditors] = useState<Record<string, KeyEditor>>({});
  const [pullModel, setPullModel] = useState("");
  const [pull, setPull] = useState<PullState>(IDLE_PULL);
  const pullCtrl = useRef<AbortController | null>(null);

  const patchEditor = (name: string, patch: Partial<KeyEditor>) =>
    setEditors((e) => {
      const base: KeyEditor = e[name] ?? {
        draft: "",
        busy: null,
        verdict: null,
      };
      return { ...e, [name]: { ...base, ...patch } };
    });

  const loadProviders = useCallback(() => {
    setProvErr(null);
    getProviders()
      .then(setProv)
      .catch((e) => {
        setProv(null);
        setProvErr(errMessage(e));
      });
  }, []);

  const loadOllama = useCallback(() => {
    getOllama()
      .then(setOllama)
      .catch((e) =>
        setOllama({
          running: false,
          base_url: "",
          models: [],
          detail: errMessage(e),
        }),
      );
  }, []);

  useEffect(() => {
    loadProviders();
    loadOllama();
    return () => pullCtrl.current?.abort();
  }, [loadProviders, loadOllama]);

  const doSave = async (p: ProviderInfo) => {
    const draft = (editors[p.name]?.draft ?? "").trim();
    if (!draft) return;
    patchEditor(p.name, { busy: "save", verdict: null });
    try {
      await saveProviderKey(p.name, draft);
      patchEditor(p.name, { busy: null, draft: "" });
      setOpenKey(null);
      toast.show(`${p.label} key saved`, "ok");
      loadProviders();
    } catch (e) {
      patchEditor(p.name, {
        busy: null,
        verdict: { ok: false, detail: errMessage(e) },
      });
    }
  };

  const doValidate = async (p: ProviderInfo) => {
    const draft = (editors[p.name]?.draft ?? "").trim();
    patchEditor(p.name, { busy: "validate", verdict: null });
    try {
      const v = await validateProviderKey(p.name, draft || undefined);
      patchEditor(p.name, {
        busy: null,
        verdict: { ok: v.ok, detail: v.detail },
      });
    } catch (e) {
      patchEditor(p.name, {
        busy: null,
        verdict: { ok: false, detail: errMessage(e) },
      });
    }
  };

  const doPull = async () => {
    const model = pullModel.trim();
    if (!model || pull.running) return;
    setPull({ ...IDLE_PULL, running: true, status: "starting" });
    const ctrl = new AbortController();
    pullCtrl.current = ctrl;
    try {
      await streamPull(
        model,
        (e) => {
          if (e.type === "progress") {
            setPull((s) => ({
              ...s,
              status: e.status || s.status,
              completed: e.completed ?? s.completed,
              total: e.total ?? s.total,
            }));
          } else if (e.type === "done") {
            setPull(IDLE_PULL);
            setPullModel("");
            toast.show(`${model} pulled`, "ok");
            loadOllama();
            loadProviders();
          } else {
            setPull((s) => ({ ...s, running: false, error: e.message }));
          }
        },
        ctrl.signal,
      );
      setPull((s) => (s.running ? { ...s, running: false } : s));
    } catch (e) {
      if (!ctrl.signal.aborted)
        setPull((s) => ({ ...s, running: false, error: errMessage(e) }));
    }
  };

  const refresh = () => {
    loadProviders();
    loadOllama();
  };

  const ordered = prov
    ? [...prov.providers].sort((a, b) => a.cost_hint - b.cost_hint)
    : [];
  const chain = ordered.filter((p) => p.configured);
  const skipped = ordered.filter((p) => !p.configured);
  const pct =
    pull.completed != null && pull.total
      ? Math.min(100, (pull.completed / pull.total) * 100)
      : 0;

  return (
    <>
      <Topbar
        title="Models"
        sub="Providers, local models, and how himmy picks a brain"
        actions={
          <button className="btn" onClick={refresh}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      <div className="home-col mdl-col">
        {/* ---- PROVIDERS ---- */}
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Providers</span>
            {prov && !prov.secrets_writable && (
              <span className="mdl-meta">secrets read-only</span>
            )}
          </div>
          {provErr ? (
            <div className="home-empty">
              <span className="mdl-meta err">{provErr}</span>{" "}
              <button className="mdl-retry" onClick={loadProviders}>
                retry
              </button>
            </div>
          ) : prov === null ? (
            <div className="home-empty">detecting providers…</div>
          ) : (
            prov.providers.map((p) => {
              const ed = editors[p.name];
              const open = openKey === p.name;
              const busy = ed?.busy != null;
              const draft = ed?.draft ?? "";
              return (
                <div key={p.name}>
                  <div className="run-line">
                    <span className="run-line-prompt" title={p.hint ?? undefined}>
                      {p.label}
                    </span>
                    <span className="mdl-meta">{p.status}</span>
                    {p.kind === "key" && (
                      <button
                        className={"mdl-keybtn" + (open ? " active" : "")}
                        onClick={() => setOpenKey(open ? null : p.name)}
                      >
                        {open ? "close" : p.configured ? "edit key" : "set key"}
                      </button>
                    )}
                  </div>
                  {open && (
                    <>
                      <div className="mdl-editor">
                        <input
                          className="input"
                          type="password"
                          autoComplete="off"
                          maxLength={512}
                          placeholder={p.key_name ?? "API key"}
                          value={draft}
                          onChange={(e) =>
                            patchEditor(p.name, {
                              draft: e.target.value,
                              verdict: null,
                            })
                          }
                          onKeyDown={(e) => {
                            if (e.key === "Enter") doSave(p);
                          }}
                        />
                        <button
                          className="btn btn-primary"
                          disabled={
                            busy || !draft.trim() || !prov.secrets_writable
                          }
                          onClick={() => doSave(p)}
                        >
                          {ed?.busy === "save" ? "Saving…" : "Save"}
                        </button>
                        {p.validate_supported && (
                          <button
                            className="btn"
                            disabled={busy || (!draft.trim() && !p.configured)}
                            onClick={() => doValidate(p)}
                          >
                            {ed?.busy === "validate" ? "Checking…" : "Validate"}
                          </button>
                        )}
                      </div>
                      {!prov.secrets_writable && (
                        <div className="mdl-verdict err">
                          secrets backend is read-only — set
                          HIMMY_SECRETS=keychain or file to save keys
                        </div>
                      )}
                      {ed?.verdict && (
                        <div
                          className={
                            "mdl-verdict" + (ed.verdict.ok ? "" : " err")
                          }
                        >
                          {ed.verdict.detail}
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })
          )}
        </section>

        {/* ---- LOCAL MODELS ---- */}
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Local models</span>
            {ollama?.running && (
              <span className="mdl-meta">{ollama.base_url}</span>
            )}
          </div>
          {ollama === null ? (
            <div className="home-empty">checking ollama…</div>
          ) : !ollama.running ? (
            <div className="home-empty">
              {ollama.detail ??
                "Ollama isn't running — start it with `ollama serve`."}
            </div>
          ) : (
            <>
              {ollama.models.length === 0 ? (
                <div className="home-empty">
                  No local models yet — pull one below (try llama3.2).
                </div>
              ) : (
                ollama.models.map((m) => (
                  <div className="run-line" key={m.name}>
                    <span className="run-line-prompt mono">{m.name}</span>
                    <span className="mdl-meta">
                      {m.parameter_size ? `${m.parameter_size} · ` : ""}
                      {formatBytes(m.size)}
                      {m.modified_at
                        ? ` · ${relativeTime(m.modified_at)}`
                        : ""}
                    </span>
                  </div>
                ))
              )}
              <div className="mdl-pullrow">
                <input
                  className="input"
                  maxLength={200}
                  placeholder="model to pull — e.g. llama3.2 or qwen2.5:7b"
                  value={pullModel}
                  onChange={(e) => setPullModel(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") doPull();
                  }}
                  disabled={pull.running}
                />
                <button
                  className="btn btn-primary"
                  disabled={pull.running || !pullModel.trim()}
                  onClick={doPull}
                >
                  {pull.running ? "Pulling…" : "Pull"}
                </button>
              </div>
              {pull.running && (
                <div className="mdl-progress">
                  <div className="mdl-progress-track">
                    <div
                      className="mdl-progress-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="mdl-progress-label">
                    {pull.status || "working"}
                    {pull.total
                      ? ` · ${formatBytes(pull.completed)} / ${formatBytes(
                          pull.total,
                        )} · ${Math.round(pct)}%`
                      : ""}
                  </div>
                </div>
              )}
              {pull.error && (
                <div className="mdl-verdict err">{pull.error}</div>
              )}
            </>
          )}
        </section>

        {/* ---- ROUTING ---- */}
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Routing</span>
            {prov && (
              <span className="mdl-meta">
                default · {prov.default_provider}
              </span>
            )}
          </div>
          {prov === null ? (
            <div className="home-empty">
              {provErr ? "—" : "waiting on provider detection…"}
            </div>
          ) : (
            <>
              <div className="mdl-note">
                Runs try free local providers before metered APIs — cheapest
                first, falling over on errors. With nothing picked, the default
                is {prov.default_detail}. Any run can override the provider and
                model per request.
              </div>
              {chain.length === 0 ? (
                <div className="home-empty">
                  Nothing configured yet — every run falls back to the offline
                  stub. Detect a local provider or save a key above.
                </div>
              ) : (
                chain.map((p, i) => (
                  <div className="run-line" key={p.name}>
                    <span className="mdl-route-idx">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="run-line-prompt">{p.label}</span>
                    <span className="mdl-meta">
                      {p.cost_hint === 0 ? "free · local" : "metered API"}
                    </span>
                  </div>
                ))
              )}
              {skipped.map((p) => (
                <div className="run-line mdl-skipped" key={p.name}>
                  <span className="mdl-route-idx">··</span>
                  <span className="run-line-prompt">{p.label}</span>
                  <span className="mdl-meta">skipped · {p.status}</span>
                </div>
              ))}
            </>
          )}
        </section>
      </div>
    </>
  );
}
