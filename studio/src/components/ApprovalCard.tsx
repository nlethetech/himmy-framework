import { useEffect, useState } from "react";
import { getApproval, type PendingToolView } from "../lib/api";

// Inline HITL approval inside a chat ledger entry. Shown when the stream emits
// approval_required: the pending tool calls (args redacted server-side) with
// Approve / Deny. Resolution itself lives in Chat (which owns the message the
// resumed stream renders into) — this card is the decision surface.

export type ApprovalStatus =
  | "pending"
  | "working"
  | "approved"
  | "denied"
  | "error";

export interface ApprovalState {
  checkpointId: string;
  tools: string[];
  status: ApprovalStatus;
  note?: string | null; // quiet outcome/error line once resolved
}

function argsLine(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("  ");
}

function PendingCall({ call }: { call: PendingToolView }) {
  const [open, setOpen] = useState(false);
  const line = argsLine(call.args);
  return (
    <div className="apc-call">
      <button
        type="button"
        className="apc-call-head"
        onClick={() => line && setOpen((o) => !o)}
        style={{ cursor: line ? "pointer" : "default" }}
      >
        <span className="apc-tool mono">{call.tool_name}</span>
        {line && !open && <span className="apc-args mono">{line}</span>}
        {line && <span className="apc-caret mono">{open ? "▾" : "▸"}</span>}
      </button>
      {open && (
        <div className="apc-call-body">
          {Object.entries(call.args).map(([k, v]) => (
            <div className="apc-kv" key={k}>
              <span className="apc-k mono">{k}</span>
              <span className="apc-v mono">
                {typeof v === "string" ? v : JSON.stringify(v)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ApprovalCard({
  approval,
  onResolve,
}: {
  approval: ApprovalState;
  onResolve: (approve: boolean) => void;
}) {
  const { checkpointId, tools, status, note } = approval;
  const [calls, setCalls] = useState<PendingToolView[] | null>(null);

  // The stream frame only carries tool names; the redacted args come from the
  // approvals detail. Best-effort — the names alone are still decidable.
  useEffect(() => {
    let alive = true;
    getApproval(checkpointId)
      .then((d) => alive && setCalls(d.pending_tool_calls))
      .catch(() => alive && setCalls(null));
    return () => {
      alive = false;
    };
  }, [checkpointId]);

  const resolved = status === "approved" || status === "denied";

  return (
    <div className="apc">
      <div className="apc-head">
        <span className={"apc-label" + (status === "pending" ? " hot" : "")}>
          {status === "error" ? "approval failed" : "approval required"}
        </span>
        <span className="apc-id mono">{checkpointId.slice(0, 8)}</span>
      </div>
      {calls && calls.length > 0 ? (
        calls.map((c, i) => <PendingCall call={c} key={i} />)
      ) : (
        <div className="apc-call">
          <div className="apc-call-head as-text">
            <span className="apc-tool mono">{tools.join(", ") || "tool"}</span>
          </div>
        </div>
      )}
      {(status === "pending" || status === "working") && (
        <div className="apc-actions">
          <button
            className="btn btn-primary"
            onClick={() => onResolve(true)}
            disabled={status === "working"}
            type="button"
          >
            Approve
          </button>
          <button
            className="btn"
            onClick={() => onResolve(false)}
            disabled={status === "working"}
            type="button"
          >
            Deny
          </button>
          {status === "working" && (
            <span className="apc-note mono">resuming…</span>
          )}
        </div>
      )}
      {resolved && (
        <div className="apc-note mono">
          {status === "approved" ? "approved — run resumed" : "denied"}
          {note ? ` · ${note}` : ""}
        </div>
      )}
      {status === "error" && (
        <div className="apc-note mono err">
          {note || "could not resolve"} — review it in the Approvals tab.
        </div>
      )}
    </div>
  );
}
