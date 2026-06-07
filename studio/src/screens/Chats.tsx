import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  listChats,
  renameChat,
  deleteChat,
  type ChatSessionSummary,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { ChatIcon, XIcon, PlusIcon } from "../components/icons";

export default function Chats() {
  const [chats, setChats] = useState<ChatSessionSummary[] | null>(null);
  const nav = useNavigate();

  const load = () => listChats().then(setChats).catch(() => setChats([]));
  useEffect(() => {
    load();
  }, []);

  const rename = async (c: ChatSessionSummary) => {
    const title = window.prompt("Rename chat", c.title);
    if (title && title.trim()) {
      await renameChat(c.id, title.trim());
      load();
    }
  };
  const remove = async (id: string) => {
    await deleteChat(id);
    load();
  };

  return (
    <>
      <Topbar
        title="Chats"
        sub="Saved conversations — resume any of them"
        actions={
          <button className="btn btn-primary" onClick={() => nav("/chat")}>
            <PlusIcon /> New chat
          </button>
        }
      />
      <Page>
        {chats === null ? null : chats.length === 0 ? (
          <EmptyState icon={<ChatIcon />} title="No saved chats yet">
            Start a conversation in <b>Chat</b> — it's saved here automatically and
            you can pick it back up anytime.
          </EmptyState>
        ) : (
          <div className="chats-list">
            {chats.map((c) => (
              <div
                className="chats-row"
                key={c.id}
                onClick={() => nav(`/chat?session=${c.id}`)}
                role="button"
                tabIndex={0}
              >
                <ChatIcon />
                <div className="chats-main">
                  <span className="chats-title">{c.title}</span>
                  <span className="chats-meta mono">
                    {c.message_count} msg
                    {c.agent_path ? ` · ${shortPath(c.agent_path)}` : ""}
                    {c.provider ? ` · ${c.provider}` : ""}
                  </span>
                </div>
                <span className="chats-date mono">{shortDate(c.updated_at)}</span>
                <button
                  className="chats-btn"
                  title="Rename"
                  onClick={(e) => {
                    e.stopPropagation();
                    rename(c);
                  }}
                >
                  ✎
                </button>
                <button
                  className="chats-btn"
                  title="Delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    remove(c.id);
                  }}
                >
                  <XIcon />
                </button>
              </div>
            ))}
          </div>
        )}
      </Page>
    </>
  );
}

function shortPath(p: string): string {
  const parts = p.split("/");
  return parts[parts.length - 1] || p;
}
function shortDate(iso: string): string {
  return iso.slice(0, 16).replace("T", " ");
}
