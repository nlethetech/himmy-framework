import { useEffect, useState } from "react";
import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import {
  getEnterpriseOverview,
  type CostBucket,
  type EnterpriseOverview,
  type OverviewResult,
} from "../lib/enterpriseApi";
import "../styles/enterprise.css";

function money(x: number): string {
  return `$${x.toFixed(x < 1 ? 4 : 2)}`;
}

/** A cost/calls breakdown table (top-N by cost) with a compact bar per row. */
function CostBreakdown({
  title,
  buckets,
  empty,
}: {
  title: string;
  buckets: CostBucket[];
  empty: string;
}) {
  const max = Math.max(1e-9, ...buckets.map((b) => b.cost));
  return (
    <section className="home-sec">
      <div className="home-sec-head">
        <span>{title}</span>
      </div>
      {buckets.length === 0 ? (
        <div className="ent-dim">{empty}</div>
      ) : (
        buckets.slice(0, 8).map((b) => (
          <div key={b.key} className="ent-row">
            <div className="ent-row-key">
              <span className="mono" title={b.key}>
                {b.key}
              </span>
              <div className="ent-row-sub">
                {b.calls} call{b.calls === 1 ? "" : "s"} ·{" "}
                {b.input_tokens + b.output_tokens} tok
              </div>
            </div>
            <div className="ent-row-val mono">{money(b.cost)}</div>
            <div
              style={{
                gridColumn: "1 / -1",
                height: 3,
                borderRadius: 2,
                marginTop: 2,
                background: "var(--accent)",
                opacity: 0.55,
                width: `${(b.cost / max) * 100}%`,
              }}
            />
          </div>
        ))
      )}
    </section>
  );
}

/** The community (no-license) upsell — the OSS core is untouched; the console is EE-only. */
function Upsell({ message }: { message: string }) {
  return (
    <div className="ent-upsell">
      <h2>Enterprise Edition</h2>
      <p>
        The governance &amp; cost console — cross-agent spend, run rollups, audit
        events and RBAC posture in one tenant-scoped view — is an Enterprise Edition
        feature. Your OSS install is fully functional; this surface is licensed.
      </p>
      <p className="mono ent-dim">{message}</p>
      <p>
        Install a license: <code>himmy license install &lt;key&gt;</code>
      </p>
    </div>
  );
}

function Overview({ ov }: { ov: EnterpriseOverview }) {
  return (
    <div className="home-col">
      <div className="ent-badges">
        <span className={"ent-badge ee"}>{ov.health.edition}</span>
        <span className="ent-badge">backend · {ov.health.backend}</span>
        <span className={"ent-badge " + (ov.health.ready ? "ok" : "warn")}>
          {ov.health.ready ? "ready" : "degraded"}
        </span>
        {ov.workspace_id && (
          <span className="ent-badge mono">tenant · {ov.workspace_id}</span>
        )}
      </div>

      {/* Top-line cost + scale cards. */}
      <div className="ent-cards">
        <div className="ent-card">
          <div className="ent-card-label">Total cost</div>
          <div className="ent-card-value accent">{money(ov.cost_total)}</div>
        </div>
        <div className="ent-card">
          <div className="ent-card-label">Runs</div>
          <div className="ent-card-value">{ov.runs.total}</div>
        </div>
        <div className="ent-card">
          <div className="ent-card-label">Agents</div>
          <div className="ent-card-value">{ov.agents.total}</div>
        </div>
        <div className="ent-card">
          <div className="ent-card-label">Roles</div>
          <div className="ent-card-value">{ov.rbac.roles.length}</div>
        </div>
      </div>

      {/* Runs by status. */}
      <section className="home-sec">
        <div className="home-sec-head">
          <span>Runs by status</span>
        </div>
        {Object.keys(ov.runs.by_status).length === 0 ? (
          <div className="ent-dim">No runs in this tenant yet.</div>
        ) : (
          <div className="ent-statuses">
            {Object.entries(ov.runs.by_status).map(([status, n]) => (
              <span key={status} className="ent-status">
                <span className="mono ent-dim">{status}</span>
                <b>{n}</b>
              </span>
            ))}
          </div>
        )}
      </section>

      {/* Cost per agent + per model. */}
      <div className="ent-grid2">
        <CostBreakdown
          title="Cost by agent"
          buckets={ov.cost_by_agent}
          empty="No attributed spend yet."
        />
        <CostBreakdown
          title="Cost by model"
          buckets={ov.cost_by_model}
          empty="No attributed spend yet."
        />
      </div>

      {/* Recent audit / security events. */}
      <section className="home-sec">
        <div className="home-sec-head">
          <span>Recent security events</span>
        </div>
        {ov.recent_events.length === 0 ? (
          <div className="ent-dim">No security events recorded.</div>
        ) : (
          ov.recent_events.map((e) => (
            <div key={e.event_id} className="ent-evt">
              <span className="mono ent-dim">
                {(e.created_at || "").slice(0, 19).replace("T", " ")}
              </span>
              <span
                className={
                  "mono " +
                  (e.outcome === "deny" ? "ent-evt-deny" : "ent-evt-allow")
                }
              >
                {e.event_type}
              </span>
              <span className="ent-dim">
                {e.actor || "-"} · {e.resource || "-"}:{e.action || "-"}
                {e.detail ? ` — ${e.detail}` : ""}
              </span>
            </div>
          ))
        )}
      </section>

      {/* Agents listing. */}
      <section className="home-sec">
        <div className="home-sec-head">
          <span>Agents</span>
        </div>
        {ov.agents.agents.length === 0 ? (
          <div className="ent-dim">No registered agents in this tenant.</div>
        ) : (
          ov.agents.agents.map((a) => (
            <div key={a.agent_id} className="ent-row">
              <div className="ent-row-key">
                <span>{a.name}</span>
                {a.description && (
                  <div className="ent-row-sub">{a.description}</div>
                )}
              </div>
              <span className="ent-row-sub mono">{a.agent_id.slice(0, 8)}</span>
            </div>
          ))
        )}
      </section>

      {/* RBAC posture. */}
      <section className="home-sec">
        <div className="home-sec-head">
          <span>RBAC policy</span>
        </div>
        <div className="ent-statuses">
          {ov.rbac.roles.map((r) => (
            <span key={r} className="ent-status">
              <span className="mono">{r}</span>
              <b className="ent-dim">{ov.rbac.permission_counts[r] ?? 0} perms</b>
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}

export default function Enterprise() {
  const [result, setResult] = useState<OverviewResult | null>(null);

  const load = () => {
    setResult(null);
    getEnterpriseOverview().then(setResult);
  };
  useEffect(load, []);

  return (
    <>
      <Topbar
        title="Enterprise"
        sub="tenant governance, cost and audit — the Enterprise Edition console"
      />
      {!result && <div className="home-col ent-dim">Loading…</div>}
      {result?.kind === "community" && <Upsell message={result.message} />}
      {result?.kind === "error" && (
        <div className="home-col">
          <EmptyState icon="!" title="Could not load the console">
            {result.message}{" "}
            <button className="rt-line-act" onClick={load}>
              retry
            </button>
          </EmptyState>
        </div>
      )}
      {result?.kind === "ok" && <Overview ov={result.overview} />}
    </>
  );
}
