import { useCallback, useEffect, useState } from "react";
import { Topbar } from "../components/Page";
import { useToast } from "../components/ui/Toast";
import { PlusIcon, RefreshIcon } from "../components/icons";
import { relativeTime } from "../lib/format";
import {
  attachMcpServer,
  createMcpServer,
  deleteMcpServer,
  getMcpServerAgents,
  listMcpServers,
  mcpStatus,
  testMcpServer,
  updateMcpServer,
  type McpAgentAttachment,
  type McpServer,
  type McpServerInput,
} from "../lib/mcpApi";
import "../styles/mcp.css";

/* MCP as a ledger — one SERVERS section on the centered column. Each server is
   a .run-line (name · mono status) with quiet mono actions; tools, agents and
   the edit form open as indented hairline panels under the row. Red only for
   primary action, active state, and failures. */

interface ServerForm {
  name: string;
  command: string;
  args: string; // one per line
  envKeys: string; // whitespace/comma separated NAMES
  cwd: string;
  prefix: string;
  requiresApproval: boolean;
  enabled: boolean;
}

const EMPTY_FORM: ServerForm = {
  name: "",
  command: "",
  args: "",
  envKeys: "",
  cwd: "",
  prefix: "",
  requiresApproval: false,
  enabled: true,
};

type Panel = "tools" | "agents" | "edit";

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function toForm(s: McpServer): ServerForm {
  return {
    name: s.name,
    command: s.command,
    args: s.args.join("\n"),
    envKeys: s.env_keys.join(" "),
    cwd: s.cwd ?? "",
    prefix: s.prefix,
    requiresApproval: s.requires_approval,
    enabled: s.enabled,
  };
}

function toInput(f: ServerForm): McpServerInput {
  return {
    name: f.name.trim(),
    transport: "stdio",
    command: f.command.trim(),
    args: f.args
      .split("\n")
      .map((a) => a.trim())
      .filter(Boolean),
    env_keys: f.envKeys
      .split(/[\s,]+/)
      .map((k) => k.trim())
      .filter(Boolean),
    cwd: f.cwd.trim() || null,
    prefix: f.prefix.trim(),
    requires_approval: f.requiresApproval,
    enabled: f.enabled,
  };
}

export default function Mcp() {
  const toast = useToast();
  const [servers, setServers] = useState<McpServer[] | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  // which panel is open under each server row
  const [panels, setPanels] = useState<Record<string, Panel | undefined>>({});
  const [agents, setAgents] = useState<
    Record<string, McpAgentAttachment[] | "loading" | { error: string }>
  >({});
  const [testing, setTesting] = useState<Record<string, boolean>>({});
  const [attachBusy, setAttachBusy] = useState<Record<string, boolean>>({});
  const [armedRemove, setArmedRemove] = useState<string | null>(null);
  // the add/edit form: editing === server name being edited, "" === creating
  const [editing, setEditing] = useState<string | null>(null);
  const [form, setForm] = useState<ServerForm>(EMPTY_FORM);
  const [formErr, setFormErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(() => {
    setLoadErr(null);
    listMcpServers()
      .then((r) => setServers(r.servers))
      .catch((e) => {
        setServers(null);
        setLoadErr(errMessage(e));
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const setField = <K extends keyof ServerForm>(k: K, v: ServerForm[K]) => {
    setForm((f) => ({ ...f, [k]: v }));
    setFormErr(null);
  };

  const closeForm = useCallback(() => {
    setEditing((cur) => {
      if (cur) setPanels((p) => ({ ...p, [cur]: undefined }));
      return null;
    });
    setFormErr(null);
  }, []);

  const openAdd = () => {
    closeForm();
    setEditing("");
    setForm(EMPTY_FORM);
    setPanels({});
  };

  const openEdit = (s: McpServer) => {
    closeForm();
    setEditing(s.name);
    setForm(toForm(s));
    setFormErr(null);
    setPanels((p) => ({ ...p, [s.name]: "edit" }));
  };

  const saveForm = async () => {
    const input = toInput(form);
    if (!input.name) {
      setFormErr("A name is required.");
      return;
    }
    if (!input.command) {
      setFormErr("A command is required.");
      return;
    }
    setSaving(true);
    try {
      if (editing === "") await createMcpServer(input);
      else if (editing) await updateMcpServer(editing, input);
      toast.show(
        editing === "" ? `${input.name} registered` : `${input.name} saved`,
        "ok",
      );
      closeForm();
      load();
    } catch (e) {
      setFormErr(errMessage(e));
    } finally {
      setSaving(false);
    }
  };

  const loadAgents = useCallback((name: string) => {
    setAgents((a) => ({ ...a, [name]: "loading" }));
    getMcpServerAgents(name)
      .then((r) => setAgents((a) => ({ ...a, [name]: r.agents })))
      .catch((e) =>
        setAgents((a) => ({ ...a, [name]: { error: errMessage(e) } })),
      );
  }, []);

  const togglePanel = (name: string, panel: Panel) => {
    if (editing !== null) closeForm();
    const next = panels[name] === panel ? undefined : panel;
    setPanels((p) => ({ ...p, [name]: next }));
    if (next === "agents" && !Array.isArray(agents[name])) loadAgents(name);
  };

  const runTest = async (name: string) => {
    setTesting((t) => ({ ...t, [name]: true }));
    setPanels((p) => ({ ...p, [name]: "tools" }));
    try {
      const r = await testMcpServer(name);
      toast.show(
        r.ok ? `${name}: ${r.tools.length} tools` : `${name} failed`,
        r.ok ? "ok" : "err",
      );
    } catch (e) {
      toast.show(errMessage(e), "err");
    } finally {
      setTesting((t) => ({ ...t, [name]: false }));
      load();
    }
  };

  const toggleEnabled = async (s: McpServer) => {
    try {
      await updateMcpServer(s.name, {
        ...toInput(toForm(s)),
        enabled: !s.enabled,
      });
      load();
    } catch (e) {
      toast.show(errMessage(e), "err");
    }
  };

  const remove = async (name: string) => {
    if (armedRemove !== name) {
      setArmedRemove(name);
      return;
    }
    setArmedRemove(null);
    try {
      await deleteMcpServer(name);
      toast.show(`${name} removed`, "ok");
      load();
    } catch (e) {
      toast.show(errMessage(e), "err");
    }
  };

  const toggleAttach = async (server: string, agent: McpAgentAttachment) => {
    const key = `${server}:${agent.path}`;
    if (attachBusy[key]) return;
    setAttachBusy((b) => ({ ...b, [key]: true }));
    try {
      await attachMcpServer(server, agent.path, !agent.attached);
      setAgents((a) => {
        const cur = a[server];
        if (!Array.isArray(cur)) return a;
        return {
          ...a,
          [server]: cur.map((x) =>
            x.path === agent.path ? { ...x, attached: !agent.attached } : x,
          ),
        };
      });
      toast.show(
        `${server} ${agent.attached ? "detached from" : "attached to"} ${agent.name}`,
        "ok",
      );
    } catch (e) {
      toast.show(errMessage(e), "err");
    } finally {
      setAttachBusy((b) => ({ ...b, [key]: false }));
    }
  };

  const renderForm = () => (
    <div className="mcp-form">
      <div className="mcp-form-row">
        <span className="mcp-form-label">name</span>
        <input
          className="input"
          maxLength={64}
          placeholder="filesystem"
          value={form.name}
          disabled={editing !== ""}
          onChange={(e) => setField("name", e.target.value)}
        />
      </div>
      <div className="mcp-form-row">
        <span className="mcp-form-label">command</span>
        <input
          className="input"
          maxLength={256}
          placeholder="npx"
          value={form.command}
          onChange={(e) => setField("command", e.target.value)}
        />
      </div>
      <div className="mcp-form-row">
        <span className="mcp-form-label">args</span>
        <textarea
          className="input"
          rows={2}
          maxLength={16000}
          placeholder={"-y\n@modelcontextprotocol/server-filesystem\n/tmp"}
          value={form.args}
          onChange={(e) => setField("args", e.target.value)}
        />
      </div>
      <div className="mcp-form-hint">one argument per line</div>
      <div className="mcp-form-row">
        <span className="mcp-form-label">env keys</span>
        <input
          className="input"
          maxLength={4000}
          placeholder="GITHUB_TOKEN"
          value={form.envKeys}
          onChange={(e) => setField("envKeys", e.target.value)}
        />
      </div>
      <div className="mcp-form-hint">
        secret NAMES only — values come from the secrets backend, never stored
        here
      </div>
      <div className="mcp-form-row">
        <span className="mcp-form-label">cwd</span>
        <input
          className="input"
          maxLength={512}
          placeholder="(optional working directory)"
          value={form.cwd}
          onChange={(e) => setField("cwd", e.target.value)}
        />
      </div>
      <div className="mcp-form-row">
        <span className="mcp-form-label">tool prefix</span>
        <input
          className="input"
          maxLength={32}
          placeholder="fs_"
          value={form.prefix}
          onChange={(e) => setField("prefix", e.target.value)}
        />
      </div>
      <div className="mcp-form-flags">
        <label className="mcp-flag">
          <input
            type="checkbox"
            className="mcp-check"
            checked={form.requiresApproval}
            onChange={(e) => setField("requiresApproval", e.target.checked)}
          />
          requires approval
        </label>
        <label className="mcp-flag">
          <input
            type="checkbox"
            className="mcp-check"
            checked={form.enabled}
            onChange={(e) => setField("enabled", e.target.checked)}
          />
          enabled
        </label>
      </div>
      {formErr && <div className="mcp-form-err">{formErr}</div>}
      <div className="mcp-form-actions">
        <button
          className="btn btn-primary"
          disabled={saving}
          onClick={saveForm}
        >
          {saving ? "Saving…" : editing === "" ? "Register" : "Save"}
        </button>
        <button className="btn" disabled={saving} onClick={closeForm}>
          Cancel
        </button>
      </div>
    </div>
  );

  const renderTools = (s: McpServer) => {
    if (testing[s.name])
      return <div className="mcp-note">testing — launching {s.command}…</div>;
    if (s.last_test && !s.last_test.ok)
      return <div className="mcp-note err">{s.last_test.error}</div>;
    if (!s.tools.length)
      return (
        <div className="mcp-note">
          no tools cached — run a test to discover them
        </div>
      );
    return (
      <>
        {s.tools.map((t) => (
          <div className="mcp-tool-line" key={t.name}>
            <span className="mcp-tool-name">
              {s.prefix}
              {t.name}
            </span>
            <span className="mcp-tool-desc" title={t.description}>
              {t.description || "—"}
            </span>
          </div>
        ))}
        {s.tools_at && (
          <div className="mcp-note">tested {relativeTime(s.tools_at)}</div>
        )}
      </>
    );
  };

  const renderAgents = (s: McpServer) => {
    const state = agents[s.name];
    if (state === "loading" || state === undefined)
      return <div className="mcp-note">finding agents…</div>;
    if (!Array.isArray(state))
      return (
        <div className="mcp-note err">
          {state.error}{" "}
          <button className="mcp-retry" onClick={() => loadAgents(s.name)}>
            retry
          </button>
        </div>
      );
    if (!state.length)
      return (
        <div className="mcp-note">
          no agents found in this project — create one in the Builder first
        </div>
      );
    return state.map((a) => {
      const busy = !!attachBusy[`${s.name}:${a.path}`];
      return (
        <label className={"mcp-agent-line" + (busy ? " busy" : "")} key={a.path}>
          <input
            type="checkbox"
            className="mcp-check"
            checked={a.attached}
            disabled={busy}
            onChange={() => toggleAttach(s.name, a)}
          />
          <span className="mcp-agent-name">{a.name}</span>
          <span className="mcp-agent-path">{a.path}</span>
        </label>
      );
    });
  };

  const renderServer = (s: McpServer) => {
    const st = mcpStatus(s);
    const panel = panels[s.name];
    return (
      <div key={s.name}>
        <div className="run-line">
          <span
            className="run-line-prompt"
            title={[s.command, ...s.args].join(" ")}
          >
            {s.name}
          </span>
          <span className="mcp-meta">{s.transport}</span>
          <span className={"mcp-meta" + (st.err ? " err" : "")} title={st.text}>
            {st.text}
          </span>
          <button
            className="mcp-act"
            disabled={!!testing[s.name]}
            onClick={() => runTest(s.name)}
          >
            {testing[s.name] ? "testing…" : "test"}
          </button>
          <button
            className={"mcp-act" + (panel === "tools" ? " active" : "")}
            onClick={() => togglePanel(s.name, "tools")}
          >
            tools
          </button>
          <button
            className={"mcp-act" + (panel === "agents" ? " active" : "")}
            onClick={() => togglePanel(s.name, "agents")}
          >
            agents
          </button>
          <button
            className={"mcp-act" + (panel === "edit" ? " active" : "")}
            onClick={() => (panel === "edit" ? closeForm() : openEdit(s))}
          >
            edit
          </button>
          <button className="mcp-act" onClick={() => toggleEnabled(s)}>
            {s.enabled ? "disable" : "enable"}
          </button>
          <button
            className={
              "mcp-act danger" + (armedRemove === s.name ? " armed" : "")
            }
            onBlur={() => setArmedRemove(null)}
            onClick={() => remove(s.name)}
          >
            {armedRemove === s.name ? "sure?" : "remove"}
          </button>
        </div>
        {panel === "edit" && editing === s.name && renderForm()}
        {panel === "tools" && <div className="mcp-panel">{renderTools(s)}</div>}
        {panel === "agents" && (
          <div className="mcp-panel">{renderAgents(s)}</div>
        )}
      </div>
    );
  };

  const empty = servers !== null && servers.length === 0 && editing === null;

  return (
    <>
      <Topbar
        title="MCP"
        sub="External tool servers your agents can use"
        actions={
          <>
            <button className="btn" onClick={load}>
              <RefreshIcon /> Refresh
            </button>
            <button className="btn btn-primary" onClick={openAdd}>
              <PlusIcon /> Add server
            </button>
          </>
        }
      />
      <div className="home-col mcp-col">
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Servers</span>
            {servers !== null && servers.length > 0 && (
              <span className="mcp-meta">{servers.length} registered</span>
            )}
          </div>

          {editing === "" && renderForm()}

          {loadErr ? (
            <div className="home-empty">
              <span className="mcp-meta err">{loadErr}</span>{" "}
              <button className="mcp-retry" onClick={load}>
                retry
              </button>
            </div>
          ) : servers === null ? (
            <div className="home-empty">loading servers…</div>
          ) : empty ? (
            <div className="home-empty">
              <p className="mcp-empty-copy">
                MCP (Model Context Protocol) servers are small local programs —
                a filesystem server, GitHub, a browser — that advertise tools
                your agents can call. Register one here, test it to see its
                tools, then attach it to an agent: the server is launched for
                the lifetime of each run and its tools flow through the normal
                approval and audit pipeline.
              </p>
            </div>
          ) : (
            servers.map(renderServer)
          )}
        </section>
      </div>
    </>
  );
}
