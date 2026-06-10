import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  api,
  ApiError,
  listChats,
  listKbs,
  type AgentSummary,
  type KbInfo,
} from "../lib/api";
import {
  assignChat,
  createProject,
  deleteProject,
  getProject,
  listProjects,
  unassignChat,
  updateProject,
  type ProjectChatRow,
  type ProjectDetail,
  type ProjectSummary,
} from "../lib/projectsApi";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { Modal } from "../components/ui/Modal";
import { PickMenu, type PickGroup } from "../components/ui/PickMenu";
import { useToast } from "../components/ui/Toast";
import { PlusIcon, XIcon } from "../components/icons";
import { relativeTime } from "../lib/format";
import "../styles/projects.css";

/* Projects — sustained work gets a home. The index is a single ledger of
   projects; a project page is a masthead over three ledger sections: the
   chats that live here, the default knowledge base, the default agent.
   No cards anywhere; rows ride the global .run-line pattern. */

const errText = (e: unknown) =>
  e instanceof ApiError ? e.message : "request failed";

function shortPath(p: string): string {
  const parts = p.split("/");
  return parts[parts.length - 1] || p;
}

export default function Projects() {
  const { id } = useParams<{ id: string }>();
  return id ? <ProjectPage id={id} /> : <ProjectIndex />;
}

/* ---- index: every project, one ledger --------------------------------- */

function ProjectIndex() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const nav = useNavigate();

  const load = useCallback(() => {
    setFailed(false);
    listProjects()
      .then(setProjects)
      .catch(() => {
        setProjects([]);
        setFailed(true);
      });
  }, []);
  useEffect(load, [load]);

  return (
    <>
      <Topbar
        title="Projects"
        sub="sustained work — grouped chats, knowledge, and a default agent"
        actions={
          <button className="btn btn-primary" onClick={() => setCreateOpen(true)}>
            <PlusIcon /> New project
          </button>
        }
      />
      <div className="home-col">
        {projects === null ? (
          <div className="prj-quiet">Loading…</div>
        ) : failed ? (
          <div className="prj-quiet">
            Couldn't load projects.{" "}
            <button className="prj-linkish" onClick={load}>
              retry
            </button>
          </div>
        ) : projects.length === 0 ? (
          <EmptyState
            title="No projects yet"
            action={
              <button
                className="btn btn-primary"
                onClick={() => setCreateOpen(true)}
              >
                <PlusIcon /> New project
              </button>
            }
          >
            A project keeps related chats together and pins a knowledge base
            and a default agent to the work.
          </EmptyState>
        ) : (
          <section className="home-sec">
            <div className="home-sec-head">
              <span>All projects</span>
              <span className="mono">{projects.length}</span>
            </div>
            {projects.map((p) => (
              <button
                key={p.id}
                className="run-line prj-row"
                onClick={() => nav(`/projects/${p.id}`)}
              >
                <span className="run-line-prompt">{p.name}</span>
                <span className="run-line-meta mono">
                  {p.chat_count} chat{p.chat_count === 1 ? "" : "s"} ·{" "}
                  {relativeTime(p.updated_at)}
                </span>
              </button>
            ))}
          </section>
        )}
      </div>

      {createOpen && (
        <CreateProjectModal
          onClose={() => setCreateOpen(false)}
          onCreated={(p) => nav(`/projects/${p.id}`)}
        />
      )}
    </>
  );
}

function CreateProjectModal({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (p: ProjectSummary) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const create = async () => {
    if (!name.trim() || busy) return;
    setBusy(true);
    try {
      const p = await createProject({
        name: name.trim(),
        description: description.trim(),
      });
      onCreated(p);
    } catch (e) {
      toast.show(errText(e), "err");
      setBusy(false);
    }
  };

  return (
    <Modal
      title="New project"
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            disabled={!name.trim() || busy}
            onClick={create}
          >
            Create
          </button>
        </>
      }
    >
      <div className="prj-form">
        <input
          className="input"
          placeholder="Project name"
          maxLength={120}
          value={name}
          autoFocus
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && create()}
        />
        <textarea
          className="input prj-desc-input"
          placeholder="What is this work about? (optional)"
          maxLength={2000}
          rows={3}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </div>
    </Modal>
  );
}

/* ---- detail: one project's home ---------------------------------------- */

function ProjectPage({ id }: { id: string }) {
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [missing, setMissing] = useState(false);
  const [failed, setFailed] = useState(false);
  const [kbs, setKbs] = useState<KbInfo[]>([]);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [looseChats, setLooseChats] = useState<ProjectChatRow[]>([]);
  const [renameOpen, setRenameOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const nav = useNavigate();
  const toast = useToast();

  const load = useCallback(() => {
    setMissing(false);
    setFailed(false);
    getProject(id)
      .then(setDetail)
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.status === 404) setMissing(true);
        else setFailed(true);
      });
    // catalogs are best-effort furniture; their failure never blocks the page
    listKbs()
      .then(setKbs)
      .catch(() => setKbs([]));
    api
      .get<AgentSummary[]>("/agents")
      .then(setAgents)
      .catch(() => setAgents([]));
    listChats()
      .then((all) =>
        setLooseChats(
          (all as ProjectChatRow[]).filter((c) => !c.project_id),
        ),
      )
      .catch(() => setLooseChats([]));
  }, [id]);
  useEffect(load, [load]);

  const act = async (fn: () => Promise<unknown>, okMsg?: string) => {
    try {
      await fn();
      if (okMsg) toast.show(okMsg, "ok");
      load();
    } catch (e) {
      toast.show(errText(e), "err");
    }
  };

  const newChatHere = () => {
    if (!detail) return;
    const agent = detail.agent_path
      ? `&agent=${encodeURIComponent(detail.agent_path)}`
      : "";
    nav(`/chat?project=${encodeURIComponent(detail.id)}${agent}`);
  };

  if (missing) {
    return (
      <>
        <Topbar title="Projects" />
        <EmptyState
          title="Project not found"
          action={
            <button className="btn" onClick={() => nav("/projects")}>
              ← All projects
            </button>
          }
        >
          It may have been deleted. Its chats, if any, are still under Chats.
        </EmptyState>
      </>
    );
  }
  if (failed) {
    return (
      <>
        <Topbar title="Projects" />
        <div className="home-col">
          <div className="prj-quiet">
            Couldn't load this project.{" "}
            <button className="prj-linkish" onClick={load}>
              retry
            </button>
          </div>
        </div>
      </>
    );
  }
  if (detail === null) {
    return (
      <>
        <Topbar title="Projects" />
        <div className="home-col">
          <div className="prj-quiet">Loading…</div>
        </div>
      </>
    );
  }

  // KB picker options: "" = none; include the stored id even when it no longer
  // resolves this session (the registry is session-scoped) so it stays visible.
  const kbGroups: PickGroup[] = [
    { options: [{ value: "", label: "— none" }] },
    {
      label: "knowledge bases",
      options: [
        ...kbs.map((k) => ({
          value: k.kb_id,
          label: k.name,
          meta: `${k.documents} doc${k.documents === 1 ? "" : "s"}`,
        })),
        ...(detail.kb_id && !kbs.some((k) => k.kb_id === detail.kb_id)
          ? [
              {
                value: detail.kb_id,
                label: detail.kb_name ?? detail.kb_id,
                meta: "not loaded",
              },
            ]
          : []),
      ],
    },
  ];

  const agentGroups: PickGroup[] = [
    { options: [{ value: "", label: "— none" }] },
    {
      label: "agents",
      options: [
        ...agents.map((a) => ({
          value: a.path,
          label: a.name,
          meta: shortPath(a.path),
        })),
        ...(detail.agent_path &&
        !agents.some((a) => a.path === detail.agent_path)
          ? [
              {
                value: detail.agent_path,
                label: detail.agent_name ?? shortPath(detail.agent_path),
                meta: "missing",
              },
            ]
          : []),
      ],
    },
  ];

  const attachGroups: PickGroup[] = [
    {
      label: "chats without a project",
      options: looseChats.map((c) => ({
        value: c.id,
        label: c.title,
        meta: `${c.message_count} msg`,
      })),
    },
  ];

  return (
    <>
      <Topbar
        title="Projects"
        sub={detail.name}
        actions={
          <button className="btn btn-primary" onClick={newChatHere}>
            <PlusIcon /> New chat in this project
          </button>
        }
      />
      <div className="home-col">
        <button className="prj-back mono" onClick={() => nav("/projects")}>
          ← all projects
        </button>

        <div className="masthead prj-masthead">
          <div className="masthead-date">
            STARTED {detail.created_at.slice(0, 10)}
          </div>
          <h2 className="masthead-greeting">{detail.name}</h2>
          {detail.description && (
            <p className="prj-desc">{detail.description}</p>
          )}
          <div className="prj-acts mono">
            <button className="prj-act" onClick={() => setRenameOpen(true)}>
              rename
            </button>
            <span aria-hidden>·</span>
            <button
              className="prj-act danger"
              onClick={() => setDeleteOpen(true)}
            >
              delete
            </button>
          </div>
        </div>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Chats</span>
            <span className="mono">{detail.chats.length}</span>
          </div>
          {detail.chats.length === 0 ? (
            <div className="home-empty">
              No chats here yet — start one above, or attach an existing chat
              below.
            </div>
          ) : (
            detail.chats.map((c) => (
              <div
                key={c.id}
                className="run-line prj-row"
                role="button"
                tabIndex={0}
                onClick={() => nav(`/chat?session=${c.id}`)}
                onKeyDown={(e) =>
                  e.key === "Enter" && nav(`/chat?session=${c.id}`)
                }
              >
                <span className="run-line-prompt">{c.title}</span>
                <span className="run-line-meta mono">
                  {c.message_count} msg · {relativeTime(c.updated_at)}
                </span>
                <button
                  className="prj-detach"
                  title="Remove from project (keeps the chat)"
                  onClick={(e) => {
                    e.stopPropagation();
                    void act(
                      () => unassignChat(detail.id, c.id),
                      "Chat removed from project",
                    );
                  }}
                >
                  <XIcon />
                </button>
              </div>
            ))
          )}
          {looseChats.length > 0 && (
            <div className="prj-attach-row">
              <span className="prj-attach-label mono">attach a chat</span>
              <PickMenu
                value=""
                groups={attachGroups}
                searchable
                placeholder="choose an existing chat…"
                title="Move an unassigned chat into this project"
                onChange={(chatId) =>
                  void act(
                    () => assignChat(detail.id, chatId),
                    "Chat added to project",
                  )
                }
              />
            </div>
          )}
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Knowledge</span>
          </div>
          <div className="prj-setting">
            <span className="prj-setting-label">
              Default knowledge base
              {detail.kb_id && detail.kb_name === null && (
                <span className="prj-note mono">
                  saved, but not loaded this session
                </span>
              )}
            </span>
            <PickMenu
              value={detail.kb_id ?? ""}
              groups={kbGroups}
              searchable
              placeholder="attach…"
              title="Knowledge base agents in this project should read"
              onChange={(v) =>
                void act(() => updateProject(detail.id, { kb_id: v || null }))
              }
            />
          </div>
        </section>

        <section className="home-sec">
          <div className="home-sec-head">
            <span>Agent</span>
          </div>
          <div className="prj-setting">
            <span className="prj-setting-label">Default agent</span>
            <PickMenu
              value={detail.agent_path ?? ""}
              groups={agentGroups}
              searchable
              placeholder="attach…"
              title="Preselected agent for new chats in this project"
              onChange={(v) =>
                void act(() =>
                  updateProject(detail.id, { agent_path: v || null }),
                )
              }
            />
          </div>
        </section>
      </div>

      {renameOpen && (
        <RenameModal
          current={detail.name}
          onClose={() => setRenameOpen(false)}
          onSave={(name) =>
            void act(() => updateProject(detail.id, { name }), "Renamed").then(
              () => setRenameOpen(false),
            )
          }
        />
      )}
      {deleteOpen && (
        <Modal
          title="Delete project"
          onClose={() => setDeleteOpen(false)}
          footer={
            <>
              <button className="btn" onClick={() => setDeleteOpen(false)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={async () => {
                  try {
                    await deleteProject(detail.id);
                    toast.show("Project deleted", "ok");
                    nav("/projects");
                  } catch (e) {
                    toast.show(errText(e), "err");
                  }
                }}
              >
                Delete
              </button>
            </>
          }
        >
          <p className="prj-confirm">
            Delete <b>{detail.name}</b>? Its {detail.chats.length} chat
            {detail.chats.length === 1 ? "" : "s"} will be kept — just no
            longer grouped.
          </p>
        </Modal>
      )}
    </>
  );
}

function RenameModal({
  current,
  onClose,
  onSave,
}: {
  current: string;
  onClose: () => void;
  onSave: (name: string) => void;
}) {
  const [name, setName] = useState(current);
  const ok = name.trim().length > 0 && name.trim() !== current;
  return (
    <Modal
      title="Rename project"
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            disabled={!ok}
            onClick={() => onSave(name.trim())}
          >
            Save
          </button>
        </>
      }
    >
      <input
        className="input"
        value={name}
        maxLength={120}
        autoFocus
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && ok && onSave(name.trim())}
      />
    </Modal>
  );
}
