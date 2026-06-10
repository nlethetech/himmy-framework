import type { PlanStep } from "../lib/missionsApi";

// The plan-first checklist: a hairline ledger of the agent's own plan, updated
// in place as new `plan` frames arrive. Mono step numbers, statuses as quiet
// text annotations; the active step takes red (red's "active state" job).
export function PlanCard({ steps }: { steps: PlanStep[] }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className="plan-card">
      <div className="plan-head">plan</div>
      {steps.map((s, i) => (
        <div className={`plan-row ${s.status}`} key={i}>
          <span className="plan-num">{String(i + 1).padStart(2, "0")}</span>
          <span className="plan-title">{s.title}</span>
          <span className="plan-status">{s.status}</span>
        </div>
      ))}
    </div>
  );
}
