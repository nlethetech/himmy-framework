import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  api,
  listRecipes,
  upsertRecipe,
  deleteRecipe,
  type Recipe,
  type AgentSummary,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { Modal } from "../components/ui/Modal";
import { EmptyState } from "../components/ui/EmptyState";
import { BookIcon, PlusIcon } from "../components/icons";
import { useToast } from "../components/ui/Toast";

export default function Cookbook() {
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [editing, setEditing] = useState<Recipe | null>(null);
  const nav = useNavigate();
  const toast = useToast();

  const load = () => listRecipes().then(setRecipes).catch(() => setRecipes([]));
  useEffect(() => {
    load();
    api.get<AgentSummary[]>("/agents").then(setAgents).catch(() => {});
  }, []);

  const run = (r: Recipe) =>
    nav(`/chat?agent=${encodeURIComponent(r.agent_path)}&q=${encodeURIComponent(r.prompt)}`);

  const remove = async (id: string) => {
    await deleteRecipe(id);
    load();
  };

  return (
    <>
      <Topbar
        title="Cookbook"
        sub="Saved agent + prompt recipes — one click to run"
        actions={
          <button
            className="btn btn-primary"
            onClick={() =>
              setEditing({
                id: "",
                name: "",
                agent_path: agents[0]?.path ?? "",
                prompt: "",
                notes: "",
                created_at: "",
              })
            }
          >
            <PlusIcon /> New recipe
          </button>
        }
      />
      <Page>
        {recipes.length === 0 ? (
          <EmptyState icon={<BookIcon />} title="No recipes yet">
            Save an agent + a ready-to-run prompt as a recipe, then run it with one
            click any time.
          </EmptyState>
        ) : (
          <div className="cook-grid">
            {recipes.map((r) => {
              const agent = agents.find((a) => a.path === r.agent_path);
              return (
                <div className="cook-card" key={r.id}>
                  <div className="cook-name">{r.name}</div>
                  <div className="cook-agent mono">
                    {agent?.name ?? r.agent_path ?? "no agent"}
                  </div>
                  {r.prompt && <div className="cook-prompt">{r.prompt}</div>}
                  {r.notes && <div className="cook-notes dim">{r.notes}</div>}
                  <div className="cook-actions">
                    <button
                      className="btn btn-primary"
                      onClick={() => run(r)}
                      disabled={!r.agent_path}
                    >
                      Run
                    </button>
                    <button className="btn" onClick={() => setEditing(r)}>
                      Edit
                    </button>
                    <button className="btn ghost" onClick={() => remove(r.id)}>
                      Delete
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Page>

      {editing && (
        <RecipeModal
          recipe={editing}
          agents={agents}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null);
            load();
            toast.show("Recipe saved", "ok");
          }}
        />
      )}
    </>
  );
}

function RecipeModal({
  recipe,
  agents,
  onClose,
  onSaved,
}: {
  recipe: Recipe;
  agents: AgentSummary[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState(recipe.name);
  const [agentPath, setAgentPath] = useState(recipe.agent_path);
  const [prompt, setPrompt] = useState(recipe.prompt);
  const [notes, setNotes] = useState(recipe.notes);
  const [saving, setSaving] = useState(false);
  const toast = useToast();

  const save = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await upsertRecipe({
        id: recipe.id || undefined,
        name: name.trim(),
        agent_path: agentPath,
        prompt,
        notes,
      });
      onSaved();
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      title={recipe.id ? "Edit recipe" : "New recipe"}
      onClose={onClose}
      width={540}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saving ? <span className="spinner" /> : "Save"}
          </button>
        </>
      }
    >
      <div className="form-grid">
        <label className="field">
          <span className="field-label">Name</span>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field">
          <span className="field-label">Agent</span>
          <select
            className="input"
            value={agentPath}
            onChange={(e) => setAgentPath(e.target.value)}
          >
            <option value="">— pick an agent —</option>
            {agents.map((a) => (
              <option key={a.path} value={a.path}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field-label">Prompt</span>
          <textarea
            className="input"
            rows={4}
            placeholder="The prompt to run…"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </label>
        <label className="field">
          <span className="field-label">
            Notes<span className="field-opt">optional</span>
          </span>
          <input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
      </div>
    </Modal>
  );
}
