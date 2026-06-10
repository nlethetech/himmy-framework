import { useEffect, useMemo, useRef, useState } from "react";
import { api, type PackInfo, type TeamSummary } from "../lib/api";
import {
  getTeamDetail,
  saveTeam,
  validateTeam,
  type TeamMemberDraft,
} from "../lib/teamsApi";
import { Topbar } from "../components/Page";
import { PickMenu, type PickGroup } from "../components/ui/PickMenu";
import { OrchestrationGraph, type CogStep } from "../components/Cognition";
import { useToast } from "../components/ui/Toast";
import "../styles/teams.css";

/* Team builder as a ledger: existing teams are rows, the builder is labelled
   form rows under micro-label sections, the topology is a live graph preview.
   Red carries exactly its three jobs: the entry marker / loaded team (active),
   validation errors, and the Save action. */

interface MemberDraft {
  uid: number;
  name: string;
  description: string;
  instructions: string[];
  role: string;
  provider: string; // "" = auto
  model: string;
  toolsText: string; // comma-separated tool names
  tool_packs: string[];
  handoffs: string[];
  delegates: string[];
}

const PROVIDERS = ["claude-cli", "ollama", "pydantic-ai", "openrouter", "stub"];

let uidSeq = 1;
const nextUid = () => uidSeq++;

function blankMember(name: string): MemberDraft {
  return {
    uid: nextUid(),
    name,
    description: "",
    instructions: [],
    role: "",
    provider: "",
    model: "",
    toolsText: "",
    tool_packs: [],
    handoffs: [],
    delegates: [],
  };
}

function freshDraft(): { name: string; entry: string; members: MemberDraft[] } {
  const lead = blankMember("lead");
  const helper = blankMember("helper");
  lead.delegates = ["helper"];
  return { name: "", entry: "lead", members: [lead, helper] };
}

function toApiMember(m: MemberDraft): TeamMemberDraft {
  return {
    name: m.name.trim(),
    description: m.description.trim(),
    instructions: m.instructions,
    role: m.role.trim() || null,
    provider: m.provider || null,
    model: m.model.trim() || "default",
    tools: m.toolsText
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean),
    tool_packs: m.tool_packs,
    handoffs: m.handoffs,
    delegates: m.delegates,
  };
}

function fromApiMember(m: TeamMemberDraft): MemberDraft {
  return {
    uid: nextUid(),
    name: m.name,
    description: m.description ?? "",
    instructions: m.instructions ?? [],
    role: m.role ?? "",
    provider: m.provider ?? "",
    model: m.model && m.model !== "default" ? m.model : "",
    toolsText: (m.tools ?? []).join(", "),
    tool_packs: m.tool_packs ?? [],
    handoffs: m.handoffs ?? [],
    delegates: m.delegates ?? [],
  };
}

// A removable-token list with a PickMenu to add from the remaining options.
function TokenPick({
  values,
  options,
  onChange,
}: {
  values: string[];
  options: string[];
  onChange: (next: string[]) => void;
}) {
  const remaining = options.filter((o) => o && !values.includes(o));
  return (
    <>
      {values.map((v) => (
        <button
          type="button"
          className="tb-token"
          key={v}
          title={`remove ${v}`}
          onClick={() => onChange(values.filter((x) => x !== v))}
        >
          {v} <span className="x">×</span>
        </button>
      ))}
      {remaining.length > 0 ? (
        <PickMenu
          value=""
          placeholder="add…"
          groups={[{ options: remaining.map((o) => ({ value: o, label: o })) }]}
          onChange={(v) => v && onChange([...values, v])}
        />
      ) : (
        values.length === 0 && <span className="tb-note">—</span>
      )}
    </>
  );
}

export default function Teams() {
  const [teams, setTeams] = useState<TeamSummary[] | null>(null);
  const [packs, setPacks] = useState<PackInfo[]>([]);
  const [draft, setDraft] = useState(freshDraft);
  const [editingPath, setEditingPath] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const toast = useToast();
  const busyRef = useRef(false);

  const refresh = () =>
    api
      .get<TeamSummary[]>("/teams")
      .then(setTeams)
      .catch(() => setTeams([]));

  useEffect(() => {
    refresh();
    api
      .get<PackInfo[]>("/tools")
      .then(setPacks)
      .catch(() => setPacks([]));
  }, []);

  const memberNames = draft.members.map((m) => m.name.trim()).filter(Boolean);

  const setMember = (uid: number, patch: Partial<MemberDraft>) =>
    setDraft((d) => ({
      ...d,
      members: d.members.map((m) => (m.uid === uid ? { ...m, ...patch } : m)),
    }));

  const renameMember = (uid: number, name: string) =>
    setDraft((d) => {
      const old = d.members.find((m) => m.uid === uid)?.name ?? "";
      const swap = (xs: string[]) => xs.map((x) => (x === old ? name : x));
      return {
        ...d,
        entry: d.entry === old ? name : d.entry,
        members: d.members.map((m) =>
          m.uid === uid
            ? { ...m, name }
            : { ...m, handoffs: swap(m.handoffs), delegates: swap(m.delegates) },
        ),
      };
    });

  const removeMember = (uid: number) =>
    setDraft((d) => {
      const gone = d.members.find((m) => m.uid === uid)?.name ?? "";
      const members = d.members
        .filter((m) => m.uid !== uid)
        .map((m) => ({
          ...m,
          handoffs: m.handoffs.filter((x) => x !== gone),
          delegates: m.delegates.filter((x) => x !== gone),
        }));
      return {
        ...d,
        members,
        entry: d.entry === gone ? (members[0]?.name ?? "") : d.entry,
      };
    });

  const addMember = () =>
    setDraft((d) => {
      let i = d.members.length + 1;
      while (d.members.some((m) => m.name === `member-${i}`)) i++;
      return { ...d, members: [...d.members, blankMember(`member-${i}`)] };
    });

  const startNew = () => {
    setDraft(freshDraft());
    setEditingPath(null);
    setErrors([]);
  };

  const edit = async (team: TeamSummary) => {
    try {
      const detail = await getTeamDetail(team.path);
      setDraft({
        name: detail.name,
        entry: detail.spec.entry,
        members: detail.spec.members.map(fromApiMember),
      });
      setEditingPath(detail.path);
      setErrors([]);
      if (detail.has_advanced)
        toast.show(
          "This team has advanced fields (MCP / tools module) — they are kept as-is on save.",
        );
    } catch (e) {
      toast.show((e as Error).message, "err");
    }
  };

  const save = async () => {
    if (busyRef.current) return;
    const local: string[] = [];
    if (!draft.name.trim()) local.push("Team name is required.");
    if (!draft.entry.trim()) local.push("Pick an entry member.");
    if (draft.members.some((m) => !m.name.trim()))
      local.push("Every member needs a name.");
    if (local.length > 0) {
      setErrors(local);
      return;
    }
    busyRef.current = true;
    setSaving(true);
    setErrors([]);
    try {
      const body = {
        name: draft.name.trim(),
        entry: draft.entry.trim(),
        members: draft.members.map(toApiMember),
      };
      const check = await validateTeam(body);
      if (!check.ok) {
        setErrors(check.errors);
        return;
      }
      const summary = await saveTeam({
        ...body,
        path: editingPath,
        overwrite: editingPath != null,
      });
      setEditingPath(summary.path);
      toast.show(`Saved ${summary.path}`, "ok");
      refresh();
    } catch (e) {
      const msg = (e as Error).message;
      setErrors([msg]);
      toast.show(msg, "err");
    } finally {
      busyRef.current = false;
      setSaving(false);
    }
  };

  // Topology preview: the entry's delegate edges light up.
  const entryMember = draft.members.find((m) => m.name.trim() === draft.entry);
  const graphSteps: CogStep[] = (entryMember?.delegates ?? []).map((w) => ({
    kind: "delegate" as const,
    worker: w,
  }));
  const graphMembers = useMemo(
    () =>
      draft.members
        .filter((m) => m.name.trim())
        .map((m) => ({ name: m.name.trim(), role: m.role || null })),
    [draft.members],
  );

  const entryGroups: PickGroup[] = [
    { options: memberNames.map((n) => ({ value: n, label: n })) },
  ];
  const providerGroups: PickGroup[] = [
    {
      options: [
        { value: "", label: "auto", meta: "project default" },
        ...PROVIDERS.map((p) => ({ value: p, label: p })),
      ],
    },
  ];
  const packNames = packs.map((p) => p.name);

  return (
    <>
      <Topbar title="Teams" sub="Build a multi-agent team — entry, members, topology" />
      <div className="home-col teams-col">
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Teams</span>
            <button type="button" className="tb-head-act" onClick={startNew}>
              + new team
            </button>
          </div>
          {teams === null ? (
            <div className="home-empty">Loading teams…</div>
          ) : teams.length === 0 ? (
            <div className="home-empty">
              No teams yet — build the first one below and Save.
            </div>
          ) : (
            teams.map((t) => (
              <button
                type="button"
                className={
                  "run-line team-line" + (t.path === editingPath ? " on" : "")
                }
                key={t.path}
                onClick={() => edit(t)}
                title={`edit ${t.path}`}
              >
                <span className="run-line-prompt">{t.name}</span>
                <span className="run-line-meta mono">
                  {t.entry} · {t.members.length} member
                  {t.members.length === 1 ? "" : "s"} · {t.path}
                </span>
              </button>
            ))
          )}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>{editingPath ? `Team — editing ${editingPath}` : "Team"}</span>
          </div>
          <div className="tb-row">
            <span className="tb-row-label">name</span>
            <div className="tb-row-main">
              <input
                className="tb-input mono"
                value={draft.name}
                maxLength={64}
                placeholder="research-duo"
                onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
              />
            </div>
          </div>
          <div className="tb-row">
            <span className="tb-row-label">entry</span>
            <div className="tb-row-main">
              {memberNames.length > 0 ? (
                <PickMenu
                  value={draft.entry}
                  groups={entryGroups}
                  onChange={(v) => setDraft((d) => ({ ...d, entry: v }))}
                  placeholder="pick the manager…"
                  title="The member that receives the request"
                />
              ) : (
                <span className="tb-note">name a member first</span>
              )}
            </div>
          </div>
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Members</span>
            <button type="button" className="tb-head-act" onClick={addMember}>
              + member
            </button>
          </div>
          {draft.members.map((m) => {
            const others = memberNames.filter((n) => n !== m.name.trim());
            return (
              <div className="tb-member" key={m.uid}>
                <div className="tb-member-head">
                  <input
                    className="tb-input mono"
                    value={m.name}
                    maxLength={80}
                    placeholder="member name"
                    onChange={(e) => renameMember(m.uid, e.target.value)}
                  />
                  {m.name.trim() === draft.entry && m.name.trim() !== "" && (
                    <span className="tb-member-entry">entry</span>
                  )}
                  <button
                    type="button"
                    className="tb-remove"
                    onClick={() => removeMember(m.uid)}
                  >
                    remove
                  </button>
                </div>
                <div className="tb-row">
                  <span className="tb-row-label">persona</span>
                  <div className="tb-row-main">
                    <input
                      className="tb-input"
                      value={m.description}
                      maxLength={4000}
                      placeholder="What this member does, in one sentence"
                      onChange={(e) =>
                        setMember(m.uid, { description: e.target.value })
                      }
                    />
                    <input
                      className="tb-input short"
                      value={m.role}
                      maxLength={120}
                      placeholder="role (optional)"
                      onChange={(e) => setMember(m.uid, { role: e.target.value })}
                    />
                  </div>
                </div>
                <div className="tb-row">
                  <span className="tb-row-label">model</span>
                  <div className="tb-row-main">
                    <PickMenu
                      value={m.provider}
                      groups={providerGroups}
                      onChange={(v) => setMember(m.uid, { provider: v })}
                      placeholder="auto"
                      title="Provider this member runs on"
                    />
                    <input
                      className="tb-input mono short"
                      value={m.model}
                      maxLength={120}
                      placeholder="model (default)"
                      onChange={(e) => setMember(m.uid, { model: e.target.value })}
                    />
                  </div>
                </div>
                <div className="tb-row">
                  <span className="tb-row-label">tools</span>
                  <div className="tb-row-main">
                    <TokenPick
                      values={m.tool_packs}
                      options={packNames}
                      onChange={(v) => setMember(m.uid, { tool_packs: v })}
                    />
                    <input
                      className="tb-input mono"
                      value={m.toolsText}
                      placeholder="extra tool names, comma-separated"
                      onChange={(e) =>
                        setMember(m.uid, { toolsText: e.target.value })
                      }
                    />
                  </div>
                </div>
                <div className="tb-row">
                  <span className="tb-row-label">delegates</span>
                  <div className="tb-row-main">
                    <TokenPick
                      values={m.delegates}
                      options={others}
                      onChange={(v) => setMember(m.uid, { delegates: v })}
                    />
                  </div>
                </div>
                <div className="tb-row">
                  <span className="tb-row-label">handoffs</span>
                  <div className="tb-row-main">
                    <TokenPick
                      values={m.handoffs}
                      options={others}
                      onChange={(v) => setMember(m.uid, { handoffs: v })}
                    />
                  </div>
                </div>
              </div>
            );
          })}
          {draft.members.length === 0 && (
            <div className="home-empty">No members — add one above.</div>
          )}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Topology</span>
          </div>
          {graphMembers.length >= 2 ? (
            <OrchestrationGraph
              members={graphMembers}
              entry={draft.entry}
              steps={graphSteps}
              active={null}
            />
          ) : (
            <div className="tb-graph-empty">
              Name at least two members to preview the delegation topology.
            </div>
          )}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Save</span>
          </div>
          {errors.length > 0 && (
            <ul className="tb-errors">
              {errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          )}
          <div className="tb-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={save}
              disabled={saving}
            >
              {saving ? "Saving…" : editingPath ? "Save changes" : "Save team"}
            </button>
            <span className="note">
              {editingPath
                ? `writes back to ${editingPath}`
                : draft.name.trim()
                  ? `writes ${draft.name.trim()}.team.yaml`
                  : "writes <name>.team.yaml"}
            </span>
          </div>
        </section>
      </div>
    </>
  );
}
