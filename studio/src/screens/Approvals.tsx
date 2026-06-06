import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  listApprovals,
  getApproval,
  resolveApproval,
  type ApprovalSummary,
  type ApprovalDetail,
  type RunEvent,
} from "../lib/api";
import { Topbar, Page, Loading } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { BellIcon, RefreshIcon, CheckIcon, XIcon } from "../components/icons";
import { relativeTime } from "../lib/format";
import { useToast } from "../components/ui/Toast";

export default function Approvals() {
  const [list, setList] = useState<ApprovalSummary[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<ApprovalDetail | null>(null);
  const [working, setWorking] = useState(false);
  const toast = useToast();

  const load = () =>
    listApprovals()
      .then((l) => {
        setList(l);
        setSel((cur) => cur ?? l[0]?.checkpoint_id ?? null);
      })
      .catch(() => setList([]));
  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (!sel) {
      setDetail(null);
      return;
    }
    getApproval(sel).then(setDetail).catch(() => setDetail(null));
  }, [sel]);

  const resolve = async (approve: boolean) => {
    if (!sel) return;
    setWorking(true);
    let outcome = "";
    const onEvent = (e: RunEvent) => {
      if (e.type === "message") outcome = e.text;
      if (e.type === "error") outcome = "⚠ " + e.message;
    };
    try {
      await resolveApproval(sel, approve, onEvent);
      toast.show(
        approve ? "Approved — run resumed" : "Rejected",
        approve ? "ok" : "info",
      );
      if (outcome) toast.show(outcome.slice(0, 80), "info");
      setSel(null);
      setDetail(null);
      load();
    } catch (e) {
      toast.show((e as Error).message, "err");
    } finally {
      setWorking(false);
    }
  };

  return (
    <>
      <Topbar
        title="Approvals"
        sub={list ? `${list.length} pending` : "human-in-the-loop"}
        actions={
          <button className="btn" onClick={load}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      <Page>
        {list === null ? (
          <Loading label="Loading approvals…" />
        ) : list.length === 0 ? (
          <EmptyState icon={<BellIcon />} title="No pending approvals">
            Agents will pause here when they need your sign-off before doing
            something sensitive — like sending an email or spending money.
          </EmptyState>
        ) : (
          <div className="appr-grid">
            <div className="appr-list">
              {list.map((a) => (
                <button
                  key={a.checkpoint_id}
                  className={"appr-row" + (sel === a.checkpoint_id ? " active" : "")}
                  onClick={() => setSel(a.checkpoint_id)}
                >
                  <div className="appr-row-top">
                    <span className="appr-tool mono">{a.tools.join(", ")}</span>
                    <span className="meta">{relativeTime(a.created_at)}</span>
                  </div>
                  <span className="meta">{a.agent ?? "agent"}</span>
                </button>
              ))}
            </div>

            <div className="appr-detail">
              {!detail ? (
                <div className="dim">Select an approval to review.</div>
              ) : (
                <>
                  <div className="appr-head">
                    <div>
                      <div className="appr-title">
                        {detail.agent ?? "An agent"} wants to run{" "}
                        <span className="mono">{detail.tools.join(", ")}</span>
                      </div>
                      {detail.run_id && (
                        <Link
                          to={`/activity/${detail.run_id}`}
                          className="dim"
                          style={{ fontSize: 13 }}
                        >
                          ↗ view the run
                        </Link>
                      )}
                    </div>
                  </div>

                  {detail.pending_tool_calls.map((c, i) => (
                    <div className="appr-call" key={i}>
                      <div className="appr-call-name mono">{c.tool_name}</div>
                      <div className="appr-args">
                        {Object.entries(c.args).map(([k, v]) => (
                          <div className="appr-arg" key={k}>
                            <span className="appr-arg-k mono">{k}</span>
                            <span className="appr-arg-v mono">
                              {typeof v === "string" ? v : JSON.stringify(v)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}

                  {detail.thread_preview.length > 0 && (
                    <div className="appr-ctx">
                      <div className="section-title">Context</div>
                      {detail.thread_preview.map((m, i) => (
                        <div className="appr-ctx-msg" key={i}>
                          <span className="appr-ctx-role">{m.role}</span>
                          <span>{m.content}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="appr-actions">
                    <button
                      className="btn btn-primary"
                      onClick={() => resolve(true)}
                      disabled={working}
                    >
                      {working ? <span className="spinner" /> : <CheckIcon />} Approve
                    </button>
                    <button
                      className="btn ghost"
                      onClick={() => resolve(false)}
                      disabled={working}
                    >
                      <XIcon /> Reject
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </Page>
    </>
  );
}
