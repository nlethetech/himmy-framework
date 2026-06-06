import { useEffect, useRef, useState } from "react";
import {
  api,
  listConnections,
  type AgentSummary,
  type AgentDetail,
  type AgentFields,
  type PackInfo,
  type SkillInfo,
  type ValidationResult,
  type DoctorReport,
  type ConnectionStatus,
  ApiError,
} from "../lib/api";
import { Topbar } from "../components/Page";
import { PlusIcon, CheckIcon, ChevronIcon } from "../components/icons";
import {
  CapabilityToggle,
  CAPABILITIES,
} from "../components/CapabilityToggle";

const PROVIDERS = ["", "stub", "ollama", "claude-cli", "pydantic-ai"];
const NEW = "__new__";

type Form = {
  name: string;
  description: string;
  role: string;
  instructions: string[];
  provider: string;
  model: string;
  skills: string[];
  tool_packs: string[];
  knowledge: string[];
  guardrails: string[];
  memory: boolean;
  language: string;
};

const BLANK: Form = {
  name: "",
  description: "",
  role: "",
  instructions: [],
  provider: "",
  model: "default",
  skills: [],
  tool_packs: [],
  knowledge: [],
  guardrails: [],
  memory: false,
  language: "en",
};

function toForm(spec: AgentFields): Form {
  const g = <T,>(k: string, d: T): T => (spec[k] === undefined ? d : (spec[k] as T));
  return {
    name: g("name", ""),
    description: g("description", ""),
    role: g("role", "") ?? "",
    instructions: g("instructions", []),
    provider: g("provider", "") ?? "",
    model: g("model", "default") ?? "default",
    skills: g("skills", []),
    tool_packs: g("tool_packs", []),
    knowledge: g("knowledge", []),
    guardrails: g("guardrails", []),
    memory: g("memory", false),
    language: g("language", "en"),
  };
}

function slug(s: string): string {
  return s.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export default function Builder() {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [packs, setPacks] = useState<PackInfo[]>([]);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [guardrails, setGuardrails] = useState<string[]>([]);
  const [conns, setConns] = useState<ConnectionStatus[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [form, setForm] = useState<Form>(BLANK);
  const [path, setPath] = useState("");
  const [hasAdvanced, setHasAdvanced] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);
  const pathEdited = useRef(false);

  const refreshAgents = () =>
    api.get<AgentSummary[]>("/agents").then(setAgents).catch(() => {});

  useEffect(() => {
    refreshAgents();
    api.get<PackInfo[]>("/tools").then(setPacks).catch(() => {});
    api.get<SkillInfo[]>("/skills").then(setSkills).catch(() => {});
    api
      .get<DoctorReport>("/doctor")
      .then((d) => setGuardrails(d.guardrails))
      .catch(() => {});
    listConnections().then(setConns).catch(() => {});
  }, []);

  // Debounced live validation.
  useEffect(() => {
    if (selected === null) return;
    const t = setTimeout(() => {
      api
        .post<ValidationResult>("/agents/validate", { ...specFromForm(form) })
        .then((r) => setErrors(r.errors))
        .catch(() => {});
    }, 350);
    return () => clearTimeout(t);
  }, [form, selected]);

  const openAgent = async (p: string) => {
    setSelected(p);
    pathEdited.current = true;
    try {
      const d = await api.get<AgentDetail>(`/agent?path=${encodeURIComponent(p)}`);
      setForm(toForm(d.spec));
      setPath(d.path);
      setHasAdvanced(d.has_advanced);
    } catch {
      /* ignore */
    }
  };

  const newAgent = () => {
    setSelected(NEW);
    setForm(BLANK);
    setPath("");
    setHasAdvanced(false);
    setErrors([]);
    pathEdited.current = false;
  };

  // Auto-name the file from the agent name while creating (until user edits path).
  useEffect(() => {
    if (selected === NEW && !pathEdited.current) {
      setPath(form.name ? `${slug(form.name)}.agent.yaml` : "");
    }
  }, [form.name, selected]);

  const set = <K extends keyof Form>(k: K, v: Form[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  const save = async (overwrite?: boolean) => {
    if (!path) {
      setToast({ msg: "A file path is required", ok: false });
      return;
    }
    // Editing an existing agent is an intentional overwrite; a new agent isn't.
    const allowOverwrite = overwrite ?? selected !== NEW;
    setSaving(true);
    try {
      const saved = await api.put<AgentSummary>("/agents", {
        path,
        spec: specFromForm(form),
        overwrite: allowOverwrite,
      });
      setToast({ msg: `Saved ${saved.name}`, ok: true });
      await refreshAgents();
      setSelected(saved.path);
      pathEdited.current = true;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        if (window.confirm(`${path} already exists. Overwrite it?`)) {
          await save(true);
          return;
        }
        setToast({ msg: "Save cancelled", ok: false });
      } else if (e instanceof ApiError && e.status === 422) {
        setToast({ msg: "Fix validation errors first", ok: false });
      } else {
        setToast({ msg: String(e instanceof Error ? e.message : e), ok: false });
      }
    } finally {
      setSaving(false);
      setTimeout(() => setToast(null), 2600);
    }
  };

  const canSave = selected !== null && !!path && !!form.name && errors.length === 0;

  return (
    <>
      <Topbar
        title="Agents"
        sub="build & edit — no code"
        actions={
          selected !== null && (
            <button className="btn btn-primary" onClick={() => save()} disabled={!canSave || saving}>
              {saving ? <span className="spinner" /> : <CheckIcon />} Save
            </button>
          )
        }
      />

      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", flex: 1, minHeight: 0 }}>
        {/* List pane */}
        <aside
          style={{
            borderRight: "1px solid var(--border)",
            overflow: "auto",
            padding: 14,
          }}
        >
          <button className="btn btn-primary" style={{ width: "100%" }} onClick={newAgent}>
            <PlusIcon /> New agent
          </button>
          <div className="stack gap6 mt16">
            {agents.map((a) => (
              <button
                key={a.path}
                className={"nav-link" + (selected === a.path ? " active" : "")}
                style={{ textAlign: "left", border: "none", cursor: "pointer", background: "none" }}
                onClick={() => openAgent(a.path)}
              >
                <span className="stack" style={{ overflow: "hidden" }}>
                  <span style={{ fontWeight: 600 }}>{a.name}</span>
                  <span className="dim mono" style={{ fontSize: 11 }}>
                    {a.path}
                  </span>
                </span>
              </button>
            ))}
            {agents.length === 0 && (
              <div className="dim" style={{ fontSize: 12, padding: 8 }}>
                No agents yet.
              </div>
            )}
          </div>
        </aside>

        {/* Editor pane */}
        <div
          style={{
            overflow: "auto",
            padding: 26,
            display: "flex",
            flexDirection: "column",
          }}
        >
          {selected === null ? (
            <div className="chat-hero" style={{ margin: "auto" }}>
              <div className="hero-avatar">✎</div>
              <div className="hero-title">Agent builder</div>
              <div className="hero-sub">
                Select an agent on the left to edit it, or create a new one — no
                YAML required.
              </div>
            </div>
          ) : (
            <div style={{ maxWidth: 720 }}>
              <Editor
                form={form}
                set={set}
                path={path}
                isNew={selected === NEW}
                onPath={(v) => {
                  pathEdited.current = true;
                  setPath(v);
                }}
                packs={packs}
                skills={skills}
                guardrails={guardrails}
                conns={conns}
              />
              {hasAdvanced && (
                <div className="banner mt16">
                  <div className="b-ico">i</div>
                  <div className="b-msg">
                    This agent has advanced fields (HTTP tools, MCP servers, …) that
                    aren't shown here. They're <b>preserved</b> when you save.
                  </div>
                </div>
              )}
              {errors.length > 0 && (
                <div className="banner mt16" style={{ borderColor: "var(--err)" }}>
                  <div className="b-ico" style={{ background: "var(--err-soft)", color: "var(--err)" }}>!</div>
                  <div>
                    <div className="b-title">Validation</div>
                    {errors.map((e, i) => (
                      <div className="b-msg mono" key={i}>{e}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {toast && (
        <div className={"toast " + (toast.ok ? "ok" : "err")}>
          {toast.ok ? <CheckIcon /> : "⚠"} {toast.msg}
        </div>
      )}
    </>
  );
}

// Strip empties so validation/save see a clean spec.
function specFromForm(f: Form): AgentFields {
  const out: AgentFields = { name: f.name, description: f.description };
  if (f.role) out.role = f.role;
  if (f.instructions.filter(Boolean).length)
    out.instructions = f.instructions.filter(Boolean);
  if (f.provider) out.provider = f.provider;
  if (f.model && f.model !== "default") out.model = f.model;
  if (f.skills.length) out.skills = f.skills;
  if (f.tool_packs.length) out.tool_packs = f.tool_packs;
  if (f.knowledge.filter(Boolean).length)
    out.knowledge = f.knowledge.filter(Boolean);
  if (f.guardrails.length) out.guardrails = f.guardrails;
  if (f.memory) out.memory = true;
  if (f.language && f.language !== "en") out.language = f.language;
  return out;
}

function Editor({
  form,
  set,
  path,
  isNew,
  onPath,
  packs,
  skills,
  guardrails,
  conns,
}: {
  form: Form;
  set: <K extends keyof Form>(k: K, v: Form[K]) => void;
  path: string;
  isNew: boolean;
  onPath: (v: string) => void;
  packs: PackInfo[];
  skills: SkillInfo[];
  guardrails: string[];
  conns: ConnectionStatus[];
}) {
  const [advanced, setAdvanced] = useState(false);
  const toggle = (k: "skills" | "tool_packs" | "guardrails", v: string) =>
    set(
      k,
      form[k].includes(v) ? form[k].filter((x) => x !== v) : [...form[k], v],
    );

  // Capabilities are a friendly layer over tool_packs: a capability is "on" when
  // all its packs are present; toggling adds/removes exactly those packs.
  const capOn = (packsNeeded: string[]) =>
    packsNeeded.every((p) => form.tool_packs.includes(p));
  const toggleCap = (packsNeeded: string[]) => {
    const on = capOn(packsNeeded);
    set(
      "tool_packs",
      on
        ? form.tool_packs.filter((p) => !packsNeeded.includes(p))
        : [...new Set([...form.tool_packs, ...packsNeeded])],
    );
  };
  const connConfigured = (type?: string) =>
    type ? (conns.find((c) => c.type === type)?.configured ?? false) : null;

  return (
    <div>
      <Field label="File path" hint="relative to the project — where the agent.yaml is written">
        <input
          className="input"
          value={path}
          disabled={!isNew}
          placeholder="my-agent.agent.yaml"
          onChange={(e) => onPath(e.target.value)}
        />
      </Field>

      {/* Identity */}
      <div className="grid grid-2">
        <Field label="Name">
          <input className="input" value={form.name} onChange={(e) => set("name", e.target.value)} />
        </Field>
        <Field label="Role" hint="optional">
          <input className="input" value={form.role} onChange={(e) => set("role", e.target.value)} />
        </Field>
      </div>

      <Field label="Description">
        <input
          className="input"
          value={form.description}
          onChange={(e) => set("description", e.target.value)}
        />
      </Field>

      {/* Capabilities — the headline */}
      <div className="builder-section-title">Capabilities</div>
      <div className="cap-list">
        {CAPABILITIES.map((cap) => (
          <CapabilityToggle
            key={cap.key}
            cap={cap}
            on={capOn(cap.packs)}
            connected={connConfigured(cap.needs)}
            onToggle={() => toggleCap(cap.packs)}
          />
        ))}
      </div>

      <Field label="Instructions" hint="one guideline per line">
        <ListEditor
          items={form.instructions}
          onChange={(v) => set("instructions", v)}
          placeholder="e.g. Always cite your source."
        />
      </Field>

      {/* Advanced settings (progressive disclosure) */}
      <button
        type="button"
        className={"adv-toggle" + (advanced ? " open" : "")}
        onClick={() => setAdvanced((o) => !o)}
      >
        <ChevronIcon className="chev" /> Advanced settings
      </button>

      {advanced && (
        <div className="adv-body">
          <div className="grid grid-2">
            <Field label="Provider">
              <select className="select" value={form.provider} onChange={(e) => set("provider", e.target.value)}>
                {PROVIDERS.map((p) => (
                  <option key={p} value={p}>
                    {p || "auto (default)"}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Model" hint="e.g. llama3.2, haiku">
              <input className="input" value={form.model} onChange={(e) => set("model", e.target.value)} />
            </Field>
          </div>

          <Field label="Skills" hint="bundled capabilities = tools + know-how">
            <div className="row wrap gap6">
              {skills.map((s) => (
                <span
                  key={s.name}
                  className={"chip" + (form.skills.includes(s.name) ? " on" : "")}
                  title={s.description}
                  onClick={() => toggle("skills", s.name)}
                >
                  {s.name}
                </span>
              ))}
            </div>
          </Field>

          <Field label="More tools" hint="raw tool packs (capabilities above cover the common ones)">
            <div className="row wrap gap6">
              {packs.map((p) => (
                <span
                  key={p.name}
                  className={"chip" + (form.tool_packs.includes(p.name) ? " on" : "")}
                  title={p.tools.join(", ")}
                  onClick={() => toggle("tool_packs", p.name)}
                >
                  {p.name}
                </span>
              ))}
            </div>
          </Field>

          <Field label="Knowledge" hint="files/dirs to ground the agent (auto-ingested → kb_search)">
            <ListEditor
              items={form.knowledge}
              onChange={(v) => set("knowledge", v)}
              placeholder="e.g. ./docs"
            />
          </Field>

          <div className="grid grid-2">
            <Field label="Guardrails">
              <div className="row wrap gap6">
                {guardrails.map((g) => (
                  <span
                    key={g}
                    className={"chip" + (form.guardrails.includes(g) ? " on" : "")}
                    onClick={() => toggle("guardrails", g)}
                  >
                    {g}
                  </span>
                ))}
              </div>
            </Field>
            <div>
              <Field label="Language">
                <select className="select" value={form.language} onChange={(e) => set("language", e.target.value)}>
                  <option value="en">English</option>
                  <option value="ne">Nepali (Devanagari)</option>
                </select>
              </Field>
              <label className="row gap10" style={{ cursor: "pointer", marginTop: 6 }}>
                <input
                  type="checkbox"
                  checked={form.memory}
                  onChange={(e) => set("memory", e.target.checked)}
                />
                <span>Auto-recall memory each run</span>
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="field">
      <span className="lab">{label}</span>
      {children}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

function ListEditor({
  items,
  onChange,
  placeholder,
}: {
  items: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
}) {
  const rows = items.length ? items : [""];
  const update = (i: number, v: string) => {
    const next = rows.slice();
    next[i] = v;
    onChange(next);
  };
  return (
    <div className="stack gap6">
      {rows.map((it, i) => (
        <div className="row gap6" key={i}>
          <input
            className="input"
            value={it}
            placeholder={placeholder}
            onChange={(e) => update(i, e.target.value)}
          />
          <button
            className="btn"
            onClick={() => onChange(rows.filter((_, j) => j !== i))}
            title="remove"
            style={{ padding: "8px 11px" }}
          >
            ✕
          </button>
        </div>
      ))}
      <button className="btn" style={{ alignSelf: "flex-start" }} onClick={() => onChange([...rows, ""])}>
        <PlusIcon /> Add
      </button>
    </div>
  );
}
