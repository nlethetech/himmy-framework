import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  api,
  ApiError,
  type AgentSummary,
  type DoctorReport,
} from "../lib/api";
import { markDone, setPreferredProvider } from "./firstRun";
import {
  DEFAULT_FIRST_PROMPT,
  DEFAULT_TOOL_PACKS,
  TEMPLATES,
  type AgentTemplate,
} from "./templates";
import "../styles/onboarding.css";

/* The first-run wizard — a full-screen page on the ink canvas, not a dialog.
   Masthead typography, ledger rows, a mono step counter; skippable at every
   step (skip = done, it never nags). Enter advances, Escape skips. */

const TOTAL_STEPS = 4;

interface HealthInfo {
  status: string;
  version: string;
  secrets_writable: boolean;
  providers: { claude_cli: boolean; ollama: boolean };
}

interface ProviderRow {
  value: string;
  label: string;
  sub: string;
  detected: boolean;
  builtin?: boolean;
}

function slugify(s: string): string {
  return s
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

export default function Wizard({
  onDone,
}: {
  onDone: (opts: { tour: boolean }) => void;
}) {
  const nav = useNavigate();
  const [step, setStep] = useState(1);

  // step 2 — environment
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [healthErr, setHealthErr] = useState(false);
  const [cloudKey, setCloudKey] = useState<boolean | null>(null);
  const [provider, setProvider] = useState<string>("");

  // step 3 — first agent
  const [tpl, setTpl] = useState<AgentTemplate | null>(null);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [instructions, setInstructions] = useState("");
  const [creating, setCreating] = useState(false);
  const [createErr, setCreateErr] = useState<string | null>(null);
  const [created, setCreated] = useState<AgentSummary | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .get<HealthInfo>("/health")
      .then((h) => alive && setHealth(h))
      .catch(() => alive && setHealthErr(true));
    api
      .get<DoctorReport>("/doctor")
      .then((d) => alive && setCloudKey(d.keys.some((k) => k.present)))
      .catch(() => alive && setCloudKey(false));
    return () => {
      alive = false;
    };
  }, []);

  const rows: ProviderRow[] = [
    {
      value: "ollama",
      label: "Ollama",
      sub: "Local models on your machine — private, works offline.",
      detected: health?.providers.ollama ?? false,
    },
    {
      value: "claude-cli",
      label: "Claude CLI",
      sub: "Runs through the claude command and your subscription.",
      detected: health?.providers.claude_cli ?? false,
    },
    {
      value: "pydantic-ai",
      label: "Cloud key",
      sub: "An Anthropic or OpenAI API key, over the network.",
      detected: cloudKey === true,
    },
    {
      value: "stub",
      label: "Offline stub",
      sub: "Canned responses to try the interface — no model needed.",
      detected: true,
      builtin: true,
    },
  ];
  const checking = health === null && !healthErr;
  const anyDetected =
    (health?.providers.ollama ?? false) ||
    (health?.providers.claude_cli ?? false) ||
    cloudKey === true;

  const pickProvider = (v: string) => {
    setProvider(v);
    setPreferredProvider(v);
  };

  const skip = useCallback(() => {
    markDone();
    onDone({ tour: false });
  }, [onDone]);

  const leaveTo = useCallback(
    (to: string) => {
      markDone();
      nav(to);
      onDone({ tour: false });
    },
    [nav, onDone],
  );

  const startChat = useCallback(() => {
    markDone();
    try {
      // keep the sidebar expanded so the tour's anchors are all present
      localStorage.setItem("himmy.sb.collapsed", "0");
    } catch {
      /* ignore */
    }
    const params = new URLSearchParams();
    if (created) params.set("agent", created.path);
    params.set("q", tpl?.firstPrompt ?? DEFAULT_FIRST_PROMPT);
    nav("/chat?" + params.toString());
    onDone({ tour: true });
  }, [created, tpl, nav, onDone]);

  const applyTemplate = (t: AgentTemplate) => {
    setTpl(t);
    setName(t.name);
    setDesc(t.description);
    setInstructions(t.instructions.join("\n"));
    setCreateErr(null);
  };

  const create = useCallback(async () => {
    const agentName = name.trim();
    if (!agentName) {
      setCreateErr("Give your agent a name first.");
      return;
    }
    if (creating || created) return;
    setCreating(true);
    setCreateErr(null);
    const spec: Record<string, unknown> = { name: agentName };
    const d = desc.trim();
    if (d) spec.description = d;
    const lines = instructions
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
      .slice(0, 40);
    if (lines.length) spec.instructions = lines;
    // A first agent must be CAPABLE, not a bare prompt: bind the template's
    // tool packs (or the default set), durable memory, and the tool router so
    // it can actually research, fetch news, and keep the boards — instead of
    // hallucinating "[News Article 1]" placeholders.
    spec.tool_packs = tpl?.toolPacks ?? DEFAULT_TOOL_PACKS;
    spec.memory = true;
    spec.tool_router = true;
    const path = (slugify(agentName) || "my-agent") + ".agent.yaml";
    try {
      const saved = await api.put<AgentSummary>("/agents", {
        path,
        spec,
        overwrite: false,
      });
      setCreated(saved);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setCreateErr(
          `${path} already exists — pick a different name, or skip this step.`,
        );
      } else {
        setCreateErr(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setCreating(false);
    }
  }, [name, desc, instructions, creating, created, tpl]);

  const advance = useCallback(() => {
    if (step === 1 || step === 2) setStep(step + 1);
    else if (step === 3) {
      if (created) setStep(4);
      else if (name.trim()) void create();
    } else if (step === 4) startChat();
  }, [step, created, name, create, startChat]);

  // Escape skips the wizard; Enter advances (except inside the textarea,
  // and never steals Enter from buttons/links).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        skip();
        return;
      }
      if (e.key !== "Enter") return;
      const t = e.target as HTMLElement | null;
      const tag = t?.tagName ?? "";
      if (tag === "TEXTAREA" || tag === "BUTTON" || tag === "A") return;
      e.preventDefault();
      advance();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [advance, skip]);

  return (
    <div className="ob-overlay" role="dialog" aria-modal="true" aria-label="Welcome to Himmy">
      <div className="ob-col">
        <div className="ob-head">
          <span>
            Step {step} of {TOTAL_STEPS}
          </span>
          <button type="button" className="ob-skip" onClick={skip}>
            skip setup →
          </button>
        </div>

        {step === 1 && (
          <div className="ob-stagger" key="s1">
            <div className="ob-mast">
              <div className="ob-kicker">Welcome</div>
              <h1 className="ob-wordmark">
                <span className="mark">Himmy</span> Studio
              </h1>
            </div>
            <p className="ob-para">
              Himmy is your private agent workspace. You describe assistants in
              plain language; they run on your machine, talk to your tools, and
              pause for your approval before anything risky. Everything stays
              local unless you connect it.
            </p>
            <p className="ob-para dim">
              Four short steps: pick a brain, create your first agent, start
              talking. Skippable at any point.
            </p>
            <div className="ob-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setStep(2)}
              >
                Continue
              </button>
              <span className="ob-key-hint mono">enter ↵</span>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="ob-stagger" key="s2">
            <div className="ob-mast">
              <div className="ob-kicker">Choose your brain</div>
              <h1 className="ob-title">Where should agents think?</h1>
            </div>
            <p className="ob-para">
              Himmy works with whatever you have. Pick a default — you can
              switch per conversation later.
            </p>

            <section className="ob-sec">
              <div className="ob-sec-head">
                <span>Providers</span>
                <span>{checking ? "checking…" : anyDetected ? "detected on this machine" : "none detected"}</span>
              </div>
              {rows.map((r) => (
                <button
                  type="button"
                  key={r.value}
                  className={"ob-row" + (provider === r.value ? " on" : "")}
                  onClick={() => pickProvider(r.value)}
                >
                  <span className="ob-row-label">
                    {r.label}
                    <span className="ob-row-sub">{r.sub}</span>
                  </span>
                  <span className="ob-row-meta mono">
                    {provider === r.value
                      ? "default"
                      : r.builtin
                        ? "built-in"
                        : checking && !r.detected
                          ? "…"
                          : r.detected
                            ? "detected"
                            : "not found"}
                  </span>
                </button>
              ))}
            </section>

            {healthErr && (
              <p className="ob-err">
                Couldn't reach the backend to detect providers — you can still
                continue and set this up later.
              </p>
            )}
            {!checking && !anyDetected && !healthErr && (
              <p className="ob-quiet">
                Nothing detected. You can{" "}
                <button
                  type="button"
                  className="ob-link"
                  onClick={() => leaveTo("/models")}
                >
                  browse models
                </button>{" "}
                to install one later — for now, continue on the offline stub.
              </p>
            )}

            <div className="ob-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  if (!provider && !anyDetected) pickProvider("stub");
                  setStep(3);
                }}
              >
                Continue
              </button>
              <button type="button" className="ob-skip" onClick={() => setStep(3)}>
                decide later
              </button>
            </div>
          </div>
        )}

        {step === 3 && !created && (
          <div className="ob-stagger" key="s3">
            <div className="ob-mast">
              <div className="ob-kicker">Create your first agent</div>
              <h1 className="ob-title">Describe a helper.</h1>
            </div>

            <section className="ob-sec">
              <div className="ob-sec-head">
                <span>Start from a template</span>
                <span>optional</span>
              </div>
              {TEMPLATES.map((t) => (
                <button
                  type="button"
                  key={t.id}
                  className={"ob-row" + (tpl?.id === t.id ? " on" : "")}
                  onClick={() => applyTemplate(t)}
                >
                  <span className="ob-row-label">
                    {t.label}
                    <span className="ob-row-sub">{t.blurb}</span>
                  </span>
                  <span className="ob-row-meta mono">
                    {tpl?.id === t.id ? "applied" : "use"}
                  </span>
                </button>
              ))}
            </section>

            <section className="ob-sec">
              <div className="ob-sec-head">
                <span>Your agent</span>
              </div>
              <label className="field ob-field">
                <span className="lab">Name</span>
                <input
                  className="input"
                  value={name}
                  maxLength={60}
                  autoFocus
                  placeholder="e.g. Personal Assistant"
                  onChange={(e) => {
                    setName(e.target.value);
                    setCreateErr(null);
                  }}
                />
              </label>
              <label className="field ob-field">
                <span className="lab">What it's for</span>
                <input
                  className="input"
                  value={desc}
                  maxLength={160}
                  placeholder="One line — shows up in your agent list"
                  onChange={(e) => setDesc(e.target.value)}
                />
              </label>
              <label className="field ob-field">
                <span className="lab">Instructions — one per line</span>
                <textarea
                  className="textarea"
                  rows={5}
                  value={instructions}
                  maxLength={4000}
                  placeholder={"Be concise.\nAsk before anything irreversible."}
                  onChange={(e) => setInstructions(e.target.value)}
                />
              </label>
            </section>

            {createErr && <p className="ob-err">{createErr}</p>}

            <div className="ob-actions">
              <button
                type="button"
                className="btn btn-primary"
                disabled={!name.trim() || creating}
                onClick={() => void create()}
              >
                {creating ? "Creating…" : "Create agent"}
              </button>
              <button
                type="button"
                className="ob-skip"
                onClick={() => setStep(4)}
              >
                skip — create one later
              </button>
            </div>
          </div>
        )}

        {step === 3 && created && (
          <div className="ob-stagger" key="s3done">
            <div className="ob-mast">
              <div className="ob-kicker">Created</div>
              <h1 className="ob-title">{created.name} is ready.</h1>
            </div>
            <section className="ob-sec">
              <div className="ob-sec-head">
                <span>Your agent</span>
              </div>
              <div className="ob-row static">
                <span className="ob-row-label">
                  {created.name}
                  {created.description && (
                    <span className="ob-row-sub">{created.description}</span>
                  )}
                </span>
                <span className="ob-row-meta mono">{created.path}</span>
              </div>
            </section>
            <p className="ob-quiet">
              Saved to your agents folder — edit it any time under Settings →
              Agents.
            </p>
            <div className="ob-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setStep(4)}
              >
                Continue
              </button>
              <span className="ob-key-hint mono">enter ↵</span>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="ob-stagger" key="s4">
            <div className="ob-mast">
              <div className="ob-kicker">Ready</div>
              <h1 className="ob-title">
                {created ? `Say hello to ${created.name}.` : "You're set."}
              </h1>
            </div>
            <p className="ob-para">
              A quick tour of the workspace follows your first chat. Connect
              channels whenever you like — agents can reach you where you
              already are.
            </p>
            <section className="ob-sec">
              <div className="ob-sec-head">
                <span>Optional</span>
              </div>
              <button
                type="button"
                className="ob-row"
                onClick={() => leaveTo("/connections")}
              >
                <span className="ob-row-label">
                  Connect Telegram
                  <span className="ob-row-sub">
                    Chat with your agents from your phone.
                  </span>
                </span>
                <span className="ob-row-meta mono">→</span>
              </button>
              <button
                type="button"
                className="ob-row"
                onClick={() => leaveTo("/connections")}
              >
                <span className="ob-row-label">
                  Connect Email
                  <span className="ob-row-sub">
                    Let agents draft and summarize your inbox.
                  </span>
                </span>
                <span className="ob-row-meta mono">→</span>
              </button>
            </section>
            <div className="ob-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={startChat}
              >
                Start chatting
              </button>
              <span className="ob-key-hint mono">enter ↵</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
