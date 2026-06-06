import { useEffect, useState } from "react";
import { listConnections, type ConnectionStatus } from "../lib/api";
import { Topbar, Page, Loading, ErrorState } from "../components/Page";
import { ConnectionCard } from "../components/ConnectionCard";
import { EmptyState } from "../components/ui/EmptyState";
import { PlugIcon, RefreshIcon } from "../components/icons";

export default function Connections() {
  const [conns, setConns] = useState<ConnectionStatus[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    listConnections()
      .then((c) => {
        setConns(c);
        setError(null);
      })
      .catch((e) => setError(String(e.message ?? e)));
  };
  useEffect(load, []);

  const anyWritable = conns?.some((c) => c.writable);
  const connectedCount = conns?.filter((c) => c.configured).length ?? 0;

  return (
    <>
      <Topbar
        title="Connections"
        sub="Connect your accounts so agents can act for you"
        actions={
          <button className="btn" onClick={load}>
            <RefreshIcon /> Refresh
          </button>
        }
      />
      <Page>
        {error ? (
          <ErrorState message={error} />
        ) : conns === null ? (
          <Loading label="Loading connections…" />
        ) : (
          <>
            {!anyWritable && (
              <div className="banner" style={{ marginBottom: 16 }}>
                <div className="b-ico" style={{ background: "var(--warn-soft)", color: "var(--warn)" }}>
                  !
                </div>
                <div>
                  <div className="b-title">Secrets backend is read-only</div>
                  <div className="b-msg">
                    Set <code>HIMMY_SECRETS=keychain</code> (macOS) or{" "}
                    <code>HIMMY_SECRETS=file</code> and restart Studio to store
                    connections from here.
                  </div>
                </div>
              </div>
            )}
            {connectedCount === 0 && anyWritable && (
              <EmptyState icon={<PlugIcon />} title="No connections yet">
                Connect an account below to let agents send email, message
                Telegram, and search the web for you.
              </EmptyState>
            )}
            <div className="conn-grid">
              {conns.map((c) => (
                <ConnectionCard key={c.type} conn={c} onChange={load} />
              ))}
            </div>
          </>
        )}
      </Page>
    </>
  );
}
