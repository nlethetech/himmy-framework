import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  listMemorySubjects,
  listMemories,
  listKbs,
  type KbInfo,
} from "../lib/api";
import { Topbar, Page } from "../components/Page";
import { MemoryIcon, BookIcon } from "../components/icons";

export default function Brain() {
  const [subjects, setSubjects] = useState<string[]>([]);
  const [memCount, setMemCount] = useState(0);
  const [kbs, setKbs] = useState<KbInfo[]>([]);

  useEffect(() => {
    listMemorySubjects().then(async (subs) => {
      setSubjects(subs);
      let n = 0;
      for (const s of subs) n += (await listMemories(s)).length;
      setMemCount(n);
    });
    listKbs().then(setKbs).catch(() => {});
  }, []);

  return (
    <>
      <Topbar title="Brain" sub="Everything your agents know — memory & knowledge" />
      <Page>
        <div className="brain-grid">
          <Link to="/advanced/memory" className="brain-card">
            <div className="brain-head">
              <span className="brain-icon">
                <MemoryIcon />
              </span>
              <span>Memory</span>
            </div>
            <div className="brain-stat">
              {memCount} memor{memCount === 1 ? "y" : "ies"}
            </div>
            <div className="brain-sub dim">
              {subjects.length} subject{subjects.length === 1 ? "" : "s"} · facts the
              agent recalls across runs
            </div>
            <div className="brain-cta">Open Memory →</div>
          </Link>

          <Link to="/advanced/knowledge" className="brain-card">
            <div className="brain-head">
              <span className="brain-icon">
                <BookIcon />
              </span>
              <span>Knowledge</span>
            </div>
            <div className="brain-stat">
              {kbs.length} knowledge base{kbs.length === 1 ? "" : "s"}
            </div>
            <div className="brain-sub dim">
              {kbs.reduce((n, k) => n + k.documents, 0)} document(s) · grounded
              retrieval (RAG)
            </div>
            <div className="brain-cta">Open Knowledge →</div>
          </Link>
        </div>
      </Page>
    </>
  );
}
