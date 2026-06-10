import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { api, getLineage, type RunSummary, type LineageView } from "../lib/api";
import {
  getLineageGraph,
  getLineageEntity,
  type LineageGraphT,
  type LineageGraphNode,
  type LineageEntityDetail,
} from "../lib/lineageApi";
import { Topbar, Page } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";
import { PickMenu } from "../components/ui/PickMenu";
import { GlobeIcon } from "../components/icons";
import { fmtTokens, fmtUsd } from "../components/Usage";
import { relativeTime } from "../lib/format";
import "../styles/lineage.css";

/* Lineage: how the agent reached this conclusion. A hand-rolled SVG layered
   DAG over the entity registry (left half), a ledger detail panel for the
   selected record (right half), and below it the cognition-trace provenance
   for the chosen run. The flat list is kept because Studio runs are not
   entity-projected — for them the trace is the only provenance there is. */

// ---- topological column layout ---------------------------------------------

const NODE_W = 168;
const NODE_H = 40;
const COL_GAP = 84;
const ROW_GAP = 16;
const PAD = 24;

interface PlacedNode {
  n: LineageGraphNode;
  x: number;
  y: number;
}

interface Layout {
  placed: Map<string, PlacedNode>;
  width: number;
  height: number;
}

function layoutGraph(g: LineageGraphT): Layout {
  const ids = new Set(g.nodes.map((n) => n.id));
  const adj = new Map<string, string[]>();
  const touch = (a: string, b: string) => {
    if (!adj.has(a)) adj.set(a, []);
    adj.get(a)!.push(b);
  };
  for (const e of g.edges) {
    touch(e.from, e.to);
    touch(e.to, e.from);
  }
  // BFS hop distance from the root → column index.
  const layer = new Map<string, number>();
  if (ids.has(g.root_id)) layer.set(g.root_id, 0);
  let frontier = ids.has(g.root_id) ? [g.root_id] : [];
  while (frontier.length > 0) {
    const next: string[] = [];
    for (const id of frontier) {
      for (const nb of adj.get(id) ?? []) {
        if (!layer.has(nb) && ids.has(nb)) {
          layer.set(nb, (layer.get(id) ?? 0) + 1);
          next.push(nb);
        }
      }
    }
    frontier = next;
  }
  // Anything unreached (disconnected) lands in one extra column.
  const reached = [...layer.values()];
  const maxReached = reached.length > 0 ? Math.max(...reached) : 0;
  for (const n of g.nodes) {
    if (!layer.has(n.id)) layer.set(n.id, maxReached + 1);
  }
  // Group by column, preserving the server's BFS node order.
  const cols = new Map<number, LineageGraphNode[]>();
  for (const n of g.nodes) {
    const c = layer.get(n.id) ?? 0;
    if (!cols.has(c)) cols.set(c, []);
    cols.get(c)!.push(n);
  }
  const colIdx = [...cols.keys()].sort((a, b) => a - b);
  const maxRows = Math.max(1, ...[...cols.values()].map((v) => v.length));
  const height = PAD * 2 + maxRows * NODE_H + (maxRows - 1) * ROW_GAP;
  const placed = new Map<string, PlacedNode>();
  colIdx.forEach((c, i) => {
    const nodes = cols.get(c)!;
    const colH = nodes.length * NODE_H + (nodes.length - 1) * ROW_GAP;
    const y0 = PAD + (height - PAD * 2 - colH) / 2; // optically centered column
    nodes.forEach((n, r) => {
      placed.set(n.id, {
        n,
        x: PAD + i * (NODE_W + COL_GAP),
        y: y0 + r * (NODE_H + ROW_GAP),
      });
    });
  });
  const width = PAD * 2 + colIdx.length * NODE_W + (colIdx.length - 1) * COL_GAP;
  return { placed, width, height };
}

function edgePath(a: PlacedNode, b: PlacedNode): string {
  if (b.x > a.x) {
    const x1 = a.x + NODE_W;
    const y1 = a.y + NODE_H / 2;
    const x2 = b.x;
    const y2 = b.y + NODE_H / 2;
    const dx = (x2 - x1) / 2;
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }
  if (b.x < a.x) {
    const x1 = a.x;
    const y1 = a.y + NODE_H / 2;
    const x2 = b.x + NODE_W;
    const y2 = b.y + NODE_H / 2;
    const dx = (x1 - x2) / 2;
    return `M ${x1} ${y1} C ${x1 - dx} ${y1}, ${x2 + dx} ${y2}, ${x2} ${y2}`;
  }
  // same column — arc out to the right
  const x = a.x + NODE_W;
  const y1 = a.y + NODE_H / 2;
  const y2 = b.y + NODE_H / 2;
  return `M ${x} ${y1} C ${x + 44} ${y1}, ${x + 44} ${y2}, ${x} ${y2}`;
}

const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

// ---- the SVG graph -----------------------------------------------------------

function GraphSvg({
  graph,
  selected,
  onSelect,
}: {
  graph: LineageGraphT;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const lay = useMemo(() => layoutGraph(graph), [graph]);
  return (
    <div className="lng-canvas">
      <svg
        width={lay.width}
        height={lay.height}
        role="img"
        aria-label="lineage graph"
      >
        <defs>
          <marker
            id="lng-arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M0,0.5 L7.5,4 L0,7.5 z" className="lng-arrow" />
          </marker>
        </defs>
        {graph.edges.map((e, i) => {
          const a = lay.placed.get(e.from);
          const b = lay.placed.get(e.to);
          if (!a || !b) return null;
          const mx = (a.x + b.x) / 2 + NODE_W / 2;
          const my = (a.y + b.y) / 2 + NODE_H / 2 - 5;
          return (
            <g key={i}>
              <path className="lng-edge" d={edgePath(a, b)} markerEnd="url(#lng-arrow)" />
              <text className="lng-edge-label" x={mx} y={my} textAnchor="middle">
                {e.relation}
              </text>
            </g>
          );
        })}
        {[...lay.placed.values()].map(({ n, x, y }) => (
          <g
            key={n.id}
            className={"lng-node" + (selected === n.id ? " sel" : "")}
            transform={`translate(${x},${y})`}
            onClick={() => onSelect(n.id)}
          >
            <rect width={NODE_W} height={NODE_H} />
            <text className="lng-node-kind" x={8} y={14}>
              {clip(n.kind.toUpperCase(), 20)}
              {n.id === graph.root_id ? " · ROOT" : ""}
            </text>
            <text className="lng-node-label" x={8} y={29}>
              {clip(n.label, 23)}
            </text>
            <title>{`${n.kind} — ${n.label}`}</title>
          </g>
        ))}
      </svg>
    </div>
  );
}

// ---- the side detail panel ---------------------------------------------------

function fmtVal(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 1) ?? String(v);
  } catch {
    return String(v);
  }
}

function DetailPanel({
  detail,
  err,
  loading,
  onSelect,
}: {
  detail: LineageEntityDetail | null;
  err: string | null;
  loading: boolean;
  onSelect: (id: string) => void;
}) {
  if (loading) return <div className="lng-note">loading record…</div>;
  if (err) return <div className="lng-note err">{err}</div>;
  if (!detail) return <div className="lng-note">Select a node to inspect it.</div>;
  const payload = Object.entries(detail.payload);
  const metadata = Object.entries(detail.metadata);
  return (
    <>
      <div className="lng-sec-head">
        <span>Record</span>
        <span className="lng-head-meta">
          v{detail.version} · {relativeTime(detail.created_at)}
        </span>
      </div>
      <div className="lng-row">
        <span className="lng-row-key">kind</span>
        <span className="lng-row-val mono">{detail.kind}</span>
      </div>
      <div className="lng-row">
        <span className="lng-row-key">record id</span>
        <span className="lng-row-val mono">{detail.id}</span>
      </div>
      <div className="lng-row">
        <span className="lng-row-key">stable id</span>
        <span className="lng-row-val mono">{detail.stable_id}</span>
      </div>

      {payload.length > 0 && (
        <>
          <div className="lng-sec-head" style={{ marginTop: 22 }}>
            <span>Payload</span>
          </div>
          {payload.map(([k, v]) => (
            <div className="lng-row" key={k}>
              <span className="lng-row-key">{clip(k, 14)}</span>
              <span className="lng-row-val">{clip(fmtVal(v), 600)}</span>
            </div>
          ))}
        </>
      )}

      {metadata.length > 0 && (
        <>
          <div className="lng-sec-head" style={{ marginTop: 22 }}>
            <span>Metadata</span>
          </div>
          {metadata.map(([k, v]) => (
            <div className="lng-row" key={k}>
              <span className="lng-row-key">{clip(k, 14)}</span>
              <span className="lng-row-val mono">{clip(fmtVal(v), 300)}</span>
            </div>
          ))}
        </>
      )}

      {(detail.links_in.length > 0 || detail.links_out.length > 0) && (
        <>
          <div className="lng-sec-head" style={{ marginTop: 22 }}>
            <span>Links</span>
            <span className="lng-head-meta">
              {detail.links_in.length} in · {detail.links_out.length} out
            </span>
          </div>
          {detail.links_out.map((ln, i) => (
            <button
              type="button"
              className="lng-row click"
              key={"o" + i}
              onClick={() => onSelect(ln.other_id)}
            >
              <span className="lng-row-key">{ln.relation} →</span>
              <span className="lng-row-val">
                {ln.other_label ?? clip(ln.other_id, 18)}
              </span>
              <span className="lng-row-meta">{ln.other_kind ?? "?"}</span>
            </button>
          ))}
          {detail.links_in.map((ln, i) => (
            <button
              type="button"
              className="lng-row click"
              key={"i" + i}
              onClick={() => onSelect(ln.other_id)}
            >
              <span className="lng-row-key">← {ln.relation}</span>
              <span className="lng-row-val">
                {ln.other_label ?? clip(ln.other_id, 18)}
              </span>
              <span className="lng-row-meta">{ln.other_kind ?? "?"}</span>
            </button>
          ))}
        </>
      )}
    </>
  );
}

// ---- the flat cognition-trace provenance (kept: Studio runs live here) -------

function TraceLedger({ view }: { view: LineageView }) {
  return (
    <section className="lng-sec">
      <div className="lng-sec-head">
        <span>Provenance — cognition trace</span>
        <Link to={`/activity/${view.run_id}`}>open the full run →</Link>
      </div>
      <div className="lng-row">
        <span className="lng-row-key">answer</span>
        <span className="lng-row-val">{clip(view.answer || "(no answer)", 400)}</span>
        <span className="lng-row-meta">
          {fmtTokens(view.tokens)} tok
          {view.cost > 0 ? ` · ${fmtUsd(view.cost)}` : ""}
        </span>
      </div>
      <div className="lng-row">
        <span className="lng-row-key">agent</span>
        <span className="lng-row-val">{view.agent ?? "agent"}</span>
        <span className="lng-row-meta">
          {view.models.length > 0 ? view.models.join(", ") : (view.provider ?? "auto")}
        </span>
      </div>
      <div className="lng-row">
        <span className="lng-row-key">prompt</span>
        <span className="lng-row-val">{clip(view.prompt || "(no prompt)", 400)}</span>
      </div>
      {view.delegates.map((d, i) => (
        <div className="lng-row" key={"d" + i}>
          <span className="lng-row-key">delegated</span>
          <span className="lng-row-val">{d.worker}</span>
          {d.task && <span className="lng-row-meta">{clip(d.task, 60)}</span>}
        </div>
      ))}
      {view.tools.map((t, i) => (
        <div className="lng-row" key={"t" + i}>
          <span className="lng-row-key">tool</span>
          <span className="lng-row-val mono">
            {t.name}
            {t.result ? `  ·  ${clip(t.result, 100)}` : ""}
          </span>
          <span className="lng-row-meta">
            {t.read_only === false ? "write" : "read"}
          </span>
        </div>
      ))}
      {view.evidence.map((e, i) => (
        <div className="lng-row" key={"e" + i}>
          <span className="lng-row-key">evidence</span>
          <span className="lng-row-val">{clip(e.text, 300)}</span>
          <span className="lng-row-meta">
            {e.source}
            {e.source_uri ? ` · ${e.source_uri.split("/").pop()}` : ""}
          </span>
        </div>
      ))}
    </section>
  );
}

// ---- the screen ----------------------------------------------------------------

export default function Lineage() {
  const [params, setParams] = useSearchParams();
  const entityId = params.get("entity") ?? "";
  const runId = params.get("run") ?? "";

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [query, setQuery] = useState(entityId);
  const [graph, setGraph] = useState<LineageGraphT | null>(null);
  const [graphErr, setGraphErr] = useState<string | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [view, setView] = useState<LineageView | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<LineageEntityDetail | null>(null);
  const [detailErr, setDetailErr] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    api
      .get<{ items: RunSummary[] }>("/runs?limit=50")
      .then((r) => setRuns(r.items))
      .catch(() => setRuns([]));
  }, []);

  // graph: entity takes precedence; a run id resolves through its thread.
  useEffect(() => {
    setGraph(null);
    setGraphErr(null);
    setSelected(null);
    setDetail(null);
    setDetailErr(null);
    setQuery(entityId || runId);
    if (!entityId && !runId) return;
    setGraphLoading(true);
    getLineageGraph(entityId ? { entityId } : { runId })
      .then((g) => {
        setGraph(g);
        setSelected(g.root_id);
      })
      .catch((e) => setGraphErr(String(e.message ?? e)))
      .finally(() => setGraphLoading(false));
  }, [entityId, runId]);

  // the flat cognition-trace provenance, when a run is in play
  useEffect(() => {
    if (!runId) {
      setView(null);
      return;
    }
    getLineage(runId)
      .then(setView)
      .catch(() => setView(null));
  }, [runId]);

  // side panel record
  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    setDetailLoading(true);
    setDetailErr(null);
    getLineageEntity(selected)
      .then(setDetail)
      .catch((e) => setDetailErr(String(e.message ?? e)))
      .finally(() => setDetailLoading(false));
  }, [selected]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) {
      setParams({});
      return;
    }
    // a known run id traces through the run; anything else is an entity id
    setParams(runs.some((r) => r.id === q) ? { run: q } : { entity: q });
  };

  const runOptions = [
    {
      label: "Recent runs",
      options: runs.map((r) => ({
        value: r.id,
        label: `${r.agent_name ?? "agent"} — ${clip(r.prompt || "(no prompt)", 40)}`,
        meta: relativeTime(r.created_at),
      })),
    },
  ];

  const hasTarget = Boolean(entityId || runId);

  return (
    <>
      <Topbar title="Lineage" sub="How the agent reached this conclusion" />
      <Page>
        <div className="lng-col">
          <section className="lng-sec">
            <div className="lng-sec-head">
              <span>Trace</span>
              {graph && (
                <span className="lng-head-meta">
                  {graph.node_count} nodes · {graph.edge_count} edges
                  {graph.truncated ? " · truncated" : ""}
                </span>
              )}
            </div>
            <form className="lng-controls" onSubmit={submit}>
              <input
                className="input"
                value={query}
                maxLength={200}
                placeholder="entity id, thread id, or run id…"
                onChange={(e) => setQuery(e.target.value)}
              />
              <button className="btn btn-primary" type="submit">
                trace
              </button>
              <PickMenu
                value={runId}
                groups={runOptions}
                onChange={(v) => setParams({ run: v })}
                placeholder="Pick a run…"
                title="Trace a recent run"
              />
            </form>
          </section>

          {!hasTarget ? (
            <EmptyState icon={<GlobeIcon />} title="Trace a conclusion">
              Enter an entity or run id — or pick a recent run — to see its
              provenance graph: the thread, the persona it used, the prompt,
              and the evidence it was built from.
            </EmptyState>
          ) : graphLoading ? (
            <div className="lng-note">tracing…</div>
          ) : graph ? (
            <section className="lng-sec">
              <div className="lng-halves">
                <div className="lng-graph-pane">
                  <GraphSvg
                    graph={graph}
                    selected={selected}
                    onSelect={setSelected}
                  />
                </div>
                <div className="lng-detail-pane">
                  <DetailPanel
                    detail={detail}
                    err={detailErr}
                    loading={detailLoading}
                    onSelect={setSelected}
                  />
                </div>
              </div>
            </section>
          ) : graphErr ? (
            <section className="lng-sec">
              <div className="lng-sec-head">
                <span>Graph</span>
              </div>
              <div className="lng-note">
                {graphErr}
                {runId && view
                  ? " The cognition-trace provenance below is the full story for this run."
                  : ""}
              </div>
            </section>
          ) : null}

          {view && <TraceLedger view={view} />}
        </div>
      </Page>
    </>
  );
}
